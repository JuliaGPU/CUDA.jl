# segmented sort
#
# CUB sorts small segments with merge sorts that compare keys with `<`, and large ones with
# a radix sort of their bits, so floats would be ordered differently depending on the size
# of their segment (and NaNs break the merge sorts). We therefore always sort keys as
# unsigned integers, encoding floats like `radix_sort!` does.
#
# CUB never touches elements outside the segments. Segments given by a single vector of
# offsets cover `offsets[1]:offsets[end]` (in CUB's zero-based terms), so that is the only
# range where results need copying back; kernels read its bounds on the device.

# The CUB segmented sort of `n` keys of type `K`, and values of type `V` unless that is
# `Nothing`, into segments given by `offsets` counted from `base`. Returns a function
# `cub_sort!(temp, keys, alt_keys, values, alt_values)` like `cub_radix_sort` does, and the
# size of the temporary storage it needs.
function cub_segmented_sort(::Type{K}, ::Type{V}, n::Int, offsets::DenseCuVector{I},
                            base::Int, rev::Bool, keys_ptr,
                            values_ptr) where {K,V,I<:Base.BitInteger64}
    nsegments = length(offsets) - 1
    # HostJIT sorts keys only if the values pointer is null, so never build from empty
    # arrays; segments of fewer than two elements need no sorting anyway
    (n > 1 && nsegments > 0) || return nothing, 0

    order = rev ? CCCL_DESCENDING : CCCL_ASCENDING
    begin_offsets = pointer_iterator(I, pointer(offsets))
    end_offsets = pointer_iterator(I, pointer(offsets) + sizeof(I))
    keys_it = pointer_iterator(K, keys_ptr)
    values_it = V === Nothing ? null_iterator() : pointer_iterator(V, values_ptr)
    build = cached_build((:segmented_sort, K, V, I, order),
                         cccl_device_segmented_sort_build_result_t) do result, major, minor, include, ctk, config
        cccl_device_segmented_sort_build_ex(result, order, keys_it, values_it,
                                            begin_offsets, end_offsets, major, minor,
                                            C_NULL, C_NULL, include, ctk, config)
    end

    # with 1-based offsets, CUB sorts arrays that start one element earlier
    cub(temp, nbytes, keys, alt_keys, values, alt_values, selector) =
        cccl_device_segmented_sort(build, temp, nbytes, keys, alt_keys, values, alt_values,
                                   n + base, nsegments, begin_offsets, end_offsets, true,
                                   selector, stream())
    nbytes = temp_size() do temp, nbytes
        cub(temp, nbytes, keys_it, keys_it, values_it, values_it, Ref{Cint}(0))
    end

    shifted(x) = pointer_iterator(eltype(x), pointer(x) - base * sizeof(eltype(x)))
    function cub_sort!(temp, keys::DenseCuArray{K}, alt_keys::DenseCuArray{K}, values,
                       alt_values)
        selector = Ref{Cint}(0)
        cub(temp, Ref{Csize_t}(nbytes), shifted(keys), shifted(alt_keys),
            values === nothing ? null_iterator() : shifted(values),
            alt_values === nothing ? null_iterator() : shifted(alt_values), selector)
        return selector[] == 1
    end
    return cub_sort!, nbytes
end

function check_offset_base(base)
    base in (0, 1) || throw(ArgumentError("offset_base must be 0 or 1, got $base"))
    return Int(base)
end

# Set `dst[i] = f(src[i])` for the elements `i` within the segments and `dst[i] = f(dst[i])`
# for those outside if `swapped`, or `dst[i] = f(dst[i])` everywhere otherwise.
function segment_result_kernel(dst, src, f, swapped::Bool, offsets, base::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(dst)
        lo, hi = @inbounds offsets[1] - base, offsets[end] - base
        within = swapped && lo < i <= hi
        @inbounds dst[i] = f(within ? src[i] : dst[i])
    end
    return
end

function segment_result!(dst, src, f, swapped, offsets, base)
    kernel = @cuda launch=false segment_result_kernel(dst, src, f, swapped, offsets, base)
    threads = min(length(dst), launch_configuration(kernel.fun).threads)
    kernel(dst, src, f, swapped, offsets, base; threads, blocks=cld(length(dst), threads))
end

struct DecodeFloat{T} end
(::DecodeFloat{T})(x) where {T} = decode_float(T, x)

unsupported_offsets(offsets) =
    offsets isa AbstractVector && eltype(offsets) <: Base.BitInteger64 ? nothing :
        "CCCL requires a vector of integers of at most 64 bits as offsets, got a " *
        "$(typeof(offsets))"

"""
    CCCL.segmented_sort!(keys, offsets; values=nothing, rev=false, offset_base=0,
                         workspace=nothing)

Sort each segment `keys[offsets[i]+1:offsets[i+1]]` of `keys` in place, optionally
permuting `values` along. `offsets` are zero-based as in CSR sparse matrices (see
[`CCCL.segmented_reduce!`](@ref)), or one-based (as CUSPARSE's `rowPtr`) with
`offset_base=1`, in which case segment `i` is `keys[offsets[i]:offsets[i+1]-1]`.

Keys can be of the same types as for [`CCCL.radix_sort!`](@ref), and are ordered the same
way: like `sort!` does, by `isless` (or reversed with `rev=true`). The sort is stable,
except that NaNs are ordered by their bits.

The offsets must be nondecreasing, and lie between `offset_base` and
`length(keys) + offset_base`; this is not checked. Elements outside the segments are left
untouched. Neither `values` nor `offsets` may overlap `keys` or each other.
"""
function segmented_sort!(keys::DenseCuArray, offsets::DenseCuVector;
                         values::Union{Nothing,DenseCuArray}=nothing, rev::Bool=false,
                         offset_base::Integer=0,
                         workspace::Union{Nothing,DenseCuVector{UInt8}}=nothing)
    segmented_sort_impl!(keys, offsets, values, rev, offset_base, workspace)
    return keys
end

workspace_size(::typeof(segmented_sort!), keys::DenseCuArray, offsets::DenseCuVector;
               values::Union{Nothing,DenseCuArray}=nothing, rev::Bool=false,
               offset_base::Integer=0) =
    segmented_sort_impl!(keys, offsets, values, rev, offset_base, SizeQuery())

unsupported(::typeof(segmented_sort!), keys, offsets; values=nothing, kwargs...) =
    reason(unsupported_arrays(keys, offsets, something(values, keys)),
           unsupported_offsets(offsets), unsupported_radix_key(eltype(keys)))

function segmented_sort_impl!(keys::DenseCuArray{T}, offsets::DenseCuVector, values, rev,
                              offset_base, workspace) where {T}
    values === nothing || length(values) == length(keys) ||
        throw(DimensionMismatch("keys and values lengths differ"))
    base = check_offset_base(offset_base)
    n = length(keys)
    V = values === nothing ? Nothing : eltype(values)
    K = radix_type(T)
    values_ptr = values === nothing ? C_NULL : pointer(values)
    cub_sort!, nbytes = cub_segmented_sort(K, V, n, offsets, base, rev, pointer(keys),
                                           values_ptr)
    cub_sort! === nothing && return workspace isa SizeQuery ? 0 : nothing

    alt_values = Buffer{V === Nothing ? UInt8 : V}(values === nothing ? 0 : n)
    with_scratch(workspace, nbytes, Buffer{K}(n), alt_values) do temp, alt_k, alt_v
        k = vec(reinterpret(K, keys))
        alt_v = values === nothing ? nothing : alt_v
        # encoding all keys is harmless: those outside the segments get decoded unchanged
        T <: RadixFloat && (k .= encode_float.(T, k))
        swapped = cub_sort!(temp, k, alt_k, values, alt_v)
        if T <: RadixFloat || swapped
            segment_result!(k, alt_k, T <: RadixFloat ? DecodeFloat{T}() : identity,
                            swapped, offsets, base)
        end
        if swapped && values !== nothing
            segment_result!(vec(values), alt_v, identity, true, offsets, base)
        end
    end
end

"""
    CCCL.segmented_sortperm!(ix, keys, offsets; rev=false, offset_base=0,
                             workspace=nothing)

Store in `ix` the permutation that sorts each segment of `keys` given by `offsets`, as
[`CCCL.segmented_sort!`](@ref) does. Like `sortperm(A; dims)`, `ix` holds linear indices
into `keys`, so `keys[ix]` is sorted within each segment; outside the segments,
`ix[i] == i`. The sort is stable. `ix` must not overlap `keys` or `offsets`.
"""
function segmented_sortperm!(ix::DenseCuArray, keys::DenseCuArray, offsets::DenseCuVector;
                             rev::Bool=false, offset_base::Integer=0,
                             workspace::Union{Nothing,DenseCuVector{UInt8}}=nothing)
    segmented_sortperm_impl!(ix, keys, offsets, rev, offset_base, workspace)
    return ix
end

workspace_size(::typeof(segmented_sortperm!), ix::DenseCuArray, keys::DenseCuArray,
               offsets::DenseCuVector; rev::Bool=false, offset_base::Integer=0) =
    segmented_sortperm_impl!(ix, keys, offsets, rev, offset_base, SizeQuery())

function unsupported(::typeof(segmented_sortperm!), ix, keys, offsets; kwargs...)
    reason(unsupported_arrays(ix, keys, offsets), unsupported_offsets(offsets),
           eltype(ix) <: Base.BitInteger ? nothing :
               "CCCL sortperms need integer indices, got $(eltype(ix))",
           unsupported_radix_key(eltype(keys)))
end

function segmented_sortperm_impl!(ix::DenseCuArray{I}, keys::DenseCuArray{T},
                                  offsets::DenseCuVector, rev, offset_base,
                                  workspace) where {I<:Integer,T}
    n = length(keys)
    length(ix) == n || throw(DimensionMismatch("index and keys lengths differ"))
    n <= typemax(I) || throw(ArgumentError("index type $I cannot hold $n indices"))
    base = check_offset_base(offset_base)
    R = radix_type(T)

    # as in `radix_sortperm!`, moving 32-bit indices around is cheaper
    J = sizeof(I) > 4 && n <= typemax(Int32) ? Int32 : I
    cub_sort!, nbytes = cub_segmented_sort(R, J, n, offsets, base, rev, pointer(keys),
                                           pointer(ix))
    if cub_sort! === nothing
        workspace isa SizeQuery && return 0
        vec(ix) .= 1:n
        return
    end

    buffers = (Buffer{R}(n), Buffer{R}(n), Buffer{J}(n), Buffer{J}(J == I ? 0 : n))
    with_scratch(workspace, nbytes, buffers...) do temp, sort_keys, alt_keys, alt_ix,
                                                   sort_ix
        sort_ix = J == I ? vec(ix) : sort_ix
        kernel = @cuda launch=false sortperm_init_kernel(sort_keys, sort_ix, keys)
        threads = min(n, launch_configuration(kernel.fun).threads)
        kernel(sort_keys, sort_ix, keys; threads, blocks=cld(n, threads))

        swapped = cub_sort!(temp, sort_keys, alt_keys, sort_ix, alt_ix)
        swapped && segment_result!(sort_ix, alt_ix, identity, true, offsets, base)
        J == I || (vec(ix) .= sort_ix)
    end
end

"""
    CCCL.segmented_sortperm(keys, offsets; rev=false, offset_base=0)

Return the permutation that sorts each segment of `keys` given by `offsets`; see
[`CCCL.segmented_sortperm!`](@ref).
"""
segmented_sortperm(keys::DenseCuArray, offsets::DenseCuVector; rev::Bool=false,
                   offset_base::Integer=0) =
    segmented_sortperm!(CuArray{Int}(undef, size(keys)), keys, offsets; rev, offset_base)
