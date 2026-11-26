# device-wide algorithms

function check_eltypes(op, T, xs...)
    for x in xs
        eltype(x) == T ||
            throw(ArgumentError("CCCL.$op requires matching element types, got $T and $(eltype(x))"))
    end
end


## reduce

"""
    CCCL.reduce!(op, out, x; init)

Reduce `x` using the binary operator `op`, starting from `init`, and store the result in
`out[1]`. `op` is applied to values of the element type of `out`, which must equal the one
of `x`. The reduction is deterministic from run to run.
"""
function reduce!(op, out::DenseCuArray{T}, x::DenseCuArray; init) where {T}
    check_eltypes(:reduce!, T, x)
    length(out) >= 1 || throw(ArgumentError("output array needs to hold at least one value"))
    operator = CCCL.operator(op, T, T, T)
    build = cached_build((:reduce, T, operator.name), cccl_device_reduce_build_result_t) do result, major, minor, include, ctk, config
        cccl_device_reduce_build_ex(result, iterator(x), iterator(out), cccl_op(operator),
                                    type_info(T), CCCL_VALUE_INIT, CCCL_RUN_TO_RUN,
                                    major, minor, C_NULL, C_NULL, include, ctk, config)
    end

    state = op_state(op)
    init_ref = Ref{T}(init)
    GC.@preserve state init_ref begin
        with_temp_storage() do temp, nbytes
            cccl_device_reduce(build, temp, nbytes, iterator(x), iterator(out), length(x),
                               cccl_op(operator, state), cccl_value(init_ref), stream())
        end
    end
    return out
end

"""
    CCCL.reduce(op, x; init)

Reduce `x` using the binary operator `op`, starting from `init`, and return the result.
"""
function reduce(op, x::DenseCuArray{T}; init) where {T}
    out = CuArray{T}(undef, 1)
    reduce!(op, out, x; init)
    return Array(out)[]
end


## scan

function scan!(op, y::DenseCuArray{T}, x::DenseCuArray, init, inclusive::Bool) where {T}
    check_eltypes(inclusive ? :inclusive_scan! : :exclusive_scan!, T, x)
    length(y) == length(x) || throw(DimensionMismatch("input and output lengths differ"))
    operator = CCCL.operator(op, T, T, T)
    init_kind = init === nothing ? CCCL_NO_INIT : CCCL_VALUE_INIT
    key = (:scan, T, operator.name, inclusive, init_kind)
    build = cached_build(key, cccl_device_scan_build_result_t) do result, major, minor, include, ctk, config
        cccl_device_scan_build_ex(result, iterator(x), iterator(y), cccl_op(operator),
                                  type_info(T), inclusive, init_kind,
                                  major, minor, C_NULL, C_NULL, include, ctk, config)
    end

    state = op_state(op)
    init_ref = Ref{T}(something(init, zero(T)))
    GC.@preserve state init_ref begin
        with_temp_storage() do temp, nbytes
            args = (build, temp, nbytes, iterator(x), iterator(y), length(x),
                    cccl_op(operator, state))
            if init === nothing
                cccl_device_inclusive_scan_no_init(args..., stream())
            elseif inclusive
                cccl_device_inclusive_scan(args..., cccl_value(init_ref), stream())
            else
                cccl_device_exclusive_scan(args..., cccl_value(init_ref), stream())
            end
        end
    end
    return y
end

"""
    CCCL.inclusive_scan!(op, y, x; init=nothing)

Store the inclusive scan (prefix reduction) of `x` with the binary operator `op` in `y`,
optionally starting from `init`.
"""
inclusive_scan!(op, y::DenseCuArray, x::DenseCuArray; init=nothing) =
    scan!(op, y, x, init, true)

"""
    CCCL.exclusive_scan!(op, y, x; init)

Store the exclusive scan of `x` with the binary operator `op` in `y`, so that `y[1] == init`.
"""
exclusive_scan!(op, y::DenseCuArray, x::DenseCuArray; init) = scan!(op, y, x, init, false)


## segmented reduce

"""
    CCCL.segmented_reduce!(op, out, x, offsets; init)

Reduce the segments `x[offsets[i]+1:offsets[i+1]]` of `x` (with zero-based `offsets`, as
in CSR sparse matrices) using `op`, starting from `init`, and store the results in `out`.
"""
function segmented_reduce!(op, out::DenseCuArray{T}, x::DenseCuArray,
                           offsets::DenseCuVector{I}; init) where {T,I<:Integer}
    check_eltypes(:segmented_reduce!, T, x)
    nsegments = length(offsets) - 1
    length(out) >= nsegments || throw(DimensionMismatch("output array is too small"))
    operator = CCCL.operator(op, T, T, T)
    begin_offsets = pointer_iterator(I, pointer(offsets))
    end_offsets = pointer_iterator(I, pointer(offsets) + sizeof(I))
    init_ref = Ref{T}(init)
    key = (:segmented_reduce, T, I, operator.name)
    build = cached_build(key, cccl_device_segmented_reduce_build_result_t) do result, major, minor, include, ctk, config
        GC.@preserve init_ref begin
            cccl_device_segmented_reduce_build_ex(result, iterator(x), iterator(out),
                                                  begin_offsets, end_offsets,
                                                  cccl_op(operator), cccl_value(init_ref),
                                                  major, minor, C_NULL, C_NULL, include,
                                                  ctk, config)
        end
    end

    state = op_state(op)
    GC.@preserve state init_ref begin
        with_temp_storage() do temp, nbytes
            cccl_device_segmented_reduce(build, temp, nbytes, iterator(x), iterator(out),
                                         nsegments, begin_offsets, end_offsets,
                                         cccl_op(operator, state), cccl_value(init_ref),
                                         stream())
        end
    end
    return out
end


## radix sort
#
# CUB orders keys by their bits. For integers that is `isless`, but for floating-point keys
# CUB puts NaNs with the sign bit set first and considers -0.0 equal to 0.0. We therefore
# give CUB every key as an integer of the same size, and restore Julia's order around it.

const RadixFloat = @static if isdefined(Core, :BFloat16)
    Union{Base.IEEEFloat,Core.BFloat16}
else
    Base.IEEEFloat
end

# the integer type CUB sorts keys of type `T` as
radix_type(::Type{T}) where {T<:Base.BitInteger64} = T
radix_type(::Type{Bool}) = UInt8
radix_type(::Type{Char}) = UInt32
radix_type(::Type{T}) where {T<:RadixFloat} =
    sizeof(T) == 2 ? UInt16 : sizeof(T) == 4 ? UInt32 : UInt64
radix_type(::Type{T}) where {T} =
    throw(ArgumentError("CCCL radix sorts do not support keys of type $T"))

float_inf(::Type{T}) where {T<:Base.IEEEFloat} = Base.exponent_mask(T)
@static if isdefined(Core, :BFloat16)
    float_inf(::Type{Core.BFloat16}) = 0x7f80
end

# An order-preserving bijection from the bits `u` of a float of type `T` to unsigned
# integers. Setting the sign bit of positive numbers and flipping all bits of negative ones
# orders everything but NaNs; subtracting the code of -Inf then wraps the negative NaNs
# around to the end, after the positive ones. NaNs end up ordered by their bits.
@inline function encode_float(::Type{T}, u::U) where {T,U<:Unsigned}
    top = ~(typemax(U) >> 1)
    v = iszero(u & top) ? u | top : ~u
    return v - ~(float_inf(T) | top)
end

@inline function decode_float(::Type{T}, v::U) where {T,U<:Unsigned}
    top = ~(typemax(U) >> 1)
    u = v + ~(float_inf(T) | top)
    return iszero(u & top) ? ~u : u ⊻ top
end

# Sort `keys` (and `values`) with CUB, using `alt_keys` and `alt_values` as scratch space.
# Returns whether the result ended up there instead of in `keys` and `values`.
function cub_radix_sort!(keys::DenseCuArray{K}, alt_keys::DenseCuArray{K},
                         values::Union{Nothing,DenseCuArray},
                         alt_values::Union{Nothing,DenseCuArray}, rev::Bool) where {K}
    # empty arrays have null pointers, which would make the build sort only keys
    length(keys) <= 1 && return false
    V = values === nothing ? Nothing : eltype(values)
    order = rev ? CCCL_DESCENDING : CCCL_ASCENDING
    values_it = values === nothing ? null_iterator() : iterator(values)
    alt_values_it = alt_values === nothing ? null_iterator() : iterator(alt_values)
    build = cached_build((:radix_sort, K, V, order),
                         cccl_device_radix_sort_build_result_t) do result, major, minor, include, ctk, config
        cccl_device_radix_sort_build_ex(result, order, iterator(keys), values_it, null_op,
                                        C_NULL, major, minor, C_NULL, C_NULL, include, ctk,
                                        config)
    end

    selector = Ref{Cint}(0)
    with_temp_storage() do temp, nbytes
        cccl_device_radix_sort(build, temp, nbytes, iterator(keys), iterator(alt_keys),
                               values_it, alt_values_it, null_op, length(keys), 0,
                               8 * sizeof(K), true, selector, stream())
    end
    return selector[] == 1
end

"""
    CCCL.radix_sort!(keys; values=nothing, rev=false)

Sort `keys` in place using a radix sort, optionally permuting `values` along. Keys can be
integers, `Bool`, `Char` or IEEE floating-point numbers (including `BFloat16`), and are
ordered like `sort!` does: by `isless`, so -0.0 comes before 0.0 and NaNs come last (first
when `rev=true`). The sort is stable, except that NaNs are ordered by their bits.
"""
function radix_sort!(keys::DenseCuArray{T}; values::Union{Nothing,DenseCuArray}=nothing,
                     rev::Bool=false) where {T}
    values === nothing || length(values) == length(keys) ||
        throw(DimensionMismatch("keys and values lengths differ"))
    R = radix_type(T)
    if T <: RadixFloat && values === nothing
        sort_float_keys!(vec(reinterpret(signed(R), keys)), T, rev)
    elseif T <: RadixFloat
        k = reinterpret(R, keys)
        k .= encode_float.(T, k)
        alt_keys = similar(k)
        alt_values = similar(values)
        if cub_radix_sort!(k, alt_keys, values, alt_values, rev)
            k .= decode_float.(T, alt_keys)
            copyto!(values, alt_values)
        else
            k .= decode_float.(T, k)
        end
        CUDACore.unsafe_free!(alt_keys)
        CUDACore.unsafe_free!(alt_values)
    else
        k = reinterpret(R, keys)
        alt_keys = similar(k)
        alt_values = values === nothing ? nothing : similar(values)
        if cub_radix_sort!(k, alt_keys, values, alt_values, rev)
            copyto!(k, alt_keys)
            values === nothing || copyto!(values, alt_values)
        end
        CUDACore.unsafe_free!(alt_keys)
        alt_values === nothing || CUDACore.unsafe_free!(alt_values)
    end
    return keys
end

# Without values, the order of equal keys is unobservable, so we can avoid transforming the
# keys before and after sorting them. Sorted as signed integers, floats come out as -0.0 to
# -Inf, then negative NaNs, then 0.0 to Inf, then positive NaNs (mirrored when reversed).
# Reversing the negative numbers in place fixes all but the negative NaNs, which are rare
# and which we move past the positive numbers with three more reversals.
function sort_float_keys!(keys::DenseCuVector{S}, ::Type{T}, rev::Bool) where {S<:Signed,T}
    alt_keys = similar(keys)
    cub_radix_sort!(keys, alt_keys, nothing, nothing, rev) && copyto!(keys, alt_keys)
    CUDACore.unsafe_free!(alt_keys)

    # each block finds the segments to reverse on its own, so needs three warps
    threads = 256
    sms = attribute(device(), CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    blocks = clamp(cld(length(keys) ÷ 2, threads), 1, 4 * sms)
    for phase in 1:2
        @cuda threads blocks float_order_kernel(keys, T, rev, phase)
    end
    return keys
end

function float_order_kernel(keys::CuDeviceVector{S}, ::Type{T}, rev::Bool,
                            phase::Int) where {S,T}
    n = length(keys)

    # sorted ascending, the keys are partitioned into negative numbers [0, m), negative
    # NaNs [m, z), positive numbers [z, q) and positive NaNs [q, n)
    counts = CuStaticSharedArray(Int, 3)
    warp = (threadIdx().x - 1i32) ÷ 32i32 + 1i32
    if warp <= 3
        inf = float_inf(T) % S
        bound = warp == 1 ? inf | typemin(S) : warp == 2 ? -one(S) : inf
        count = count_sorted_le(keys, bound, rev)
        laneid() == 1 && (@inbounds counts[warp] = count)
    end
    sync_threads()
    m, z, q = @inbounds counts[1], counts[2], counts[3]

    empty = (0, 0)
    a, b, c = if phase == 1
        (0, m), z > m ? (m, z) : empty, z > m ? (z, q) : empty
    else
        z > m ? (m, q) : empty, empty, empty
    end
    if rev
        a, b, c = map(((lo, hi),) -> (n - hi, n - lo), (a, b, c))
    end

    # swap pairs of elements, one segment after the other
    pa = (a[2] - a[1]) ÷ 2
    pb = (b[2] - b[1]) ÷ 2
    pc = (c[2] - c[1]) ÷ 2
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    stride = gridDim().x * blockDim().x
    while i < pa + pb + pc
        lo, hi, j = if i < pa
            a[1], a[2], i
        elseif i < pa + pb
            b[1], b[2], i - pa
        else
            c[1], c[2], i - pa - pb
        end
        x, y = lo + j + 1, hi - j
        @inbounds keys[x], keys[y] = keys[y], keys[x]
        i += stride
    end
    return
end

# The number of keys `<= bound` in `keys`, which are sorted (descending if `rev`), found
# by a warp narrowing down the range where that predicate flips 32-fold per step.
@inline function count_sorted_le(keys, bound, rev)
    n = length(keys)
    lane = laneid() - 1i32
    lo, hi = 0, n
    while lo < hi
        step = cld(hi - lo, 32)
        i = lo + (lane + 1) * step - 1
        pred = i < hi && ((@inbounds keys[i+1]) <= bound) != rev
        c = count_ones(vote_ballot_sync(FULL_MASK, pred))
        lo, hi = lo + c * step, min(hi, lo + (c + 1) * step - 1)
    end
    return rev ? n - lo : lo
end

# the key we sort to find the permutation: NaNs are all made equal, so they stay in order
@inline function sortperm_key(x::T) where {T}
    u = reinterpret(radix_type(T), x)
    T <: RadixFloat || return u
    top = ~(typemax(u) >> 1)
    return u & ~top > float_inf(T) ? typemax(u) : encode_float(T, u)
end

function sortperm_init_kernel(sort_keys, ix, keys)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(keys)
        @inbounds sort_keys[i] = sortperm_key(keys[i])
        @inbounds ix[i] = i
    end
    return
end

"""
    CCCL.radix_sortperm!(ix, keys; rev=false)

Store in `ix` the permutation that sorts `keys`, like `sortperm!(ix, keys; rev)`, using a
radix sort. The sort is stable, and supports the same key types as
[`CCCL.radix_sort!`](@ref).
"""
function radix_sortperm!(ix::DenseCuVector{I}, keys::DenseCuVector{T};
                         rev::Bool=false) where {I<:Integer,T}
    n = length(keys)
    length(ix) == n || throw(DimensionMismatch("index and keys lengths differ"))
    n <= typemax(I) || throw(ArgumentError("index type $I cannot hold $n indices"))
    R = radix_type(T)

    # moving 32-bit indices around is much cheaper, even with widening them afterwards
    J = sizeof(I) > 4 && n <= typemax(Int32) ? Int32 : I
    sort_ix = J == I ? ix : CuVector{J}(undef, n)

    sort_keys = CuVector{R}(undef, n)
    if n > 0
        kernel = @cuda launch=false sortperm_init_kernel(sort_keys, sort_ix, keys)
        threads = min(n, launch_configuration(kernel.fun).threads)
        kernel(sort_keys, sort_ix, keys; threads, blocks=cld(n, threads))
    end
    alt_keys = similar(sort_keys)
    alt_ix = similar(sort_ix)
    swapped = cub_radix_sort!(sort_keys, alt_keys, sort_ix, alt_ix, rev)
    CUDACore.unsafe_free!(sort_keys)
    CUDACore.unsafe_free!(alt_keys)

    if swapped || sort_ix !== ix
        ix .= swapped ? alt_ix : sort_ix
    end
    CUDACore.unsafe_free!(alt_ix)
    sort_ix === ix || CUDACore.unsafe_free!(sort_ix)
    return ix
end

"""
    CCCL.radix_sortperm(keys; rev=false)

Return the permutation that sorts `keys`, like `sortperm(keys; rev)`, using a radix sort.
"""
radix_sortperm(keys::DenseCuVector; rev::Bool=false) =
    radix_sortperm!(CuVector{Int}(undef, length(keys)), keys; rev)


## merge sort

"""
    CCCL.merge_sort!(keys; lt=isless, values=nothing)

Sort `keys` in place using a merge sort with the comparison function `lt`, optionally
permuting `values` along.
"""
function merge_sort!(keys::DenseCuArray{K}; lt=isless,
                     values::Union{Nothing,DenseCuArray}=nothing) where {K}
    values === nothing || length(values) == length(keys) ||
        throw(DimensionMismatch("keys and values lengths differ"))
    V = values === nothing ? Nothing : eltype(values)
    operator = CCCL.operator(lt, K, K, Bool)

    out_keys = similar(keys)
    out_values = values === nothing ? nothing : similar(values)
    values_it = values === nothing ? null_iterator() : iterator(values)
    out_values_it = out_values === nothing ? null_iterator() : iterator(out_values)
    build = cached_build((:merge_sort, K, V, operator.name),
                         cccl_device_merge_sort_build_result_t) do result, major, minor, include, ctk, config
        cccl_device_merge_sort_build_ex(result, iterator(keys), values_it, iterator(out_keys),
                                        out_values_it, cccl_op(operator), major, minor,
                                        C_NULL, C_NULL, include, ctk, config)
    end

    state = op_state(lt)
    GC.@preserve state begin
        with_temp_storage() do temp, nbytes
            cccl_device_merge_sort(build, temp, nbytes, iterator(keys), values_it,
                                   iterator(out_keys), out_values_it, length(keys),
                                   cccl_op(operator, state), stream())
        end
    end
    copyto!(keys, out_keys)
    CUDACore.unsafe_free!(out_keys)
    if values !== nothing
        copyto!(values, out_values)
        CUDACore.unsafe_free!(out_values)
    end
    return keys
end
