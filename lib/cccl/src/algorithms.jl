# device-wide algorithms
#
# Every in-place algorithm takes a `workspace` keyword with device memory for its scratch
# buffers (see `workspace_size`), and has an `unsupported` method telling whether it can
# handle a call at all.

function check_eltypes(op, T, xs...)
    for x in xs
        eltype(x) == T ||
            throw(ArgumentError("CCCL.$op requires matching element types, got $T and $(eltype(x))"))
    end
end

"""
    CCCL.workspace_size(f, args...; kwargs...) -> Int

The number of bytes of device memory that the call `f(args...; kwargs...)` of one of CCCL's
in-place algorithms needs as scratch space. Passing that much memory, as a `CuVector{UInt8}`,
to the call's `workspace` keyword makes it allocate no device memory of its own.

The size depends on the types and sizes of the arguments, not on their contents; computing
it builds the algorithm if that has not happened yet. A workspace may be reused by
consecutive calls, but not by calls that may run at the same time, and it must not overlap
the arguments.
"""
function workspace_size end

"""
    CCCL.unsupported(f, args...; kwargs...) -> Union{Nothing,String}

Why CCCL cannot perform the call `f(args...; kwargs...)`, with `f` one of its in-place
algorithms, or `nothing` if it can. This looks at the types of the arguments and at whether
CCCL is functional, and never builds anything. A supported call may still throw for invalid
arguments (e.g., mismatched sizes), or when the operator turns out to throw an exception
once it is compiled.
"""
unsupported(f, args...; kwargs...) = "CCCL does not provide $f"

reason(reasons...) = something(reasons..., Some(nothing))

function unsupported_arrays(arrays...)
    functional() || return "CCCL is not functional"
    for x in arrays
        x isa DenseCuArray || return "CCCL only supports dense CuArrays, got a $(typeof(x))"
        isbitstype(eltype(x)) ||
            return "CCCL only supports arrays of bits types, got $(eltype(x))"
    end
    return nothing
end

unsupported_eltypes(x, y) = eltype(x) == eltype(y) ? nothing :
    "CCCL requires matching element types, got $(eltype(x)) and $(eltype(y))"

unsupported_operator(op) = isbitstype(typeof(op)) ? nothing :
    "CCCL operators must be bits values; $(typeof(op)) captures non-bits state"


## reduce

"""
    CCCL.reduce!(op, out, x; init, workspace=nothing)

Reduce `x` using the binary operator `op`, starting from `init`, and store the result in
`out[1]`. `op` is applied to values of the element type of `out`, which must equal the one
of `x`. The reduction is deterministic from run to run.
"""
function reduce!(op, out::DenseCuArray, x::DenseCuArray; init,
                 workspace::Union{Nothing,DenseCuVector{UInt8}}=nothing)
    reduce_impl!(op, out, x, init, workspace)
    return out
end

workspace_size(::typeof(reduce!), op, out::DenseCuArray, x::DenseCuArray; init) =
    reduce_impl!(op, out, x, init, SizeQuery())

unsupported(::typeof(reduce!), op, out, x; kwargs...) =
    reason(unsupported_arrays(out, x), unsupported_eltypes(out, x), unsupported_operator(op))

function reduce_impl!(op, out::DenseCuArray{T}, x::DenseCuArray, init, workspace) where {T}
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
        cub(temp, nbytes) =
            cccl_device_reduce(build, temp, nbytes, iterator(x), iterator(out), length(x),
                               cccl_op(operator, state), cccl_value(init_ref), stream())
        nbytes = temp_size(cub)
        with_scratch(workspace, nbytes) do temp
            cub(temp, Ref{Csize_t}(nbytes))
        end
    end
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

# The initial value of a scan: none, a host value, or a device reference whose value CUB
# reads when the scan runs.
scan_init_kind(::Nothing) = CCCL_NO_INIT
scan_init_kind(::CUDACore.AbstractCuRef) = CCCL_FUTURE_VALUE_INIT
scan_init_kind(_) = CCCL_VALUE_INIT

unsupported_init(y, init, inclusive) = unsupported_init(eltype(y), typeof(init), inclusive)
unsupported_init(T, ::Type, inclusive) = nothing
function unsupported_init(T, R::Type{<:CUDACore.AbstractCuRef{S}}, inclusive) where {S}
    # CCCL passes the `cub::FutureValue` to `cub::DeviceScan::InclusiveScanInit`, which has
    # no overload for one and uses it as the value itself: every tile but the first then
    # adds the bits of the pointer.
    inclusive && return "CCCL cannot read the `init` of an inclusive scan from the " *
                        "device; use `exclusive_scan!` followed by `y .= op.(y, x)`, " *
                        "with `y` distinct from `x`"
    # references to elements of host arrays have host pointers
    R <: Union{CUDACore.CuRefValue,CUDACore.CuRefArray{<:Any,<:StridedCuArray}} ||
        return "CCCL requires a device `init` to refer to device memory, got a $R"
    S == T || return "CCCL requires a device `init` of the output's element type $T, got $S"
    return nothing
end

function scan_impl!(op, y::DenseCuArray{T}, x::DenseCuArray, init, inclusive::Bool,
                    workspace) where {T}
    check_eltypes(inclusive ? :inclusive_scan! : :exclusive_scan!, T, x)
    length(y) == length(x) || throw(DimensionMismatch("input and output lengths differ"))
    msg = unsupported_init(y, init, inclusive)
    msg === nothing || throw(ArgumentError(msg))
    operator = CCCL.operator(op, T, T, T)
    init_kind = scan_init_kind(init)
    key = (:scan, T, operator.name, inclusive, init_kind)
    build = cached_build(key, cccl_device_scan_build_result_t) do result, major, minor, include, ctk, config
        cccl_device_scan_build_ex(result, iterator(x), iterator(y), cccl_op(operator),
                                  type_info(T), inclusive, init_kind,
                                  major, minor, C_NULL, C_NULL, include, ctk, config)
    end

    state = op_state(op)
    init_ref = init isa Union{Nothing,CUDACore.AbstractCuRef} ? nothing : Ref{T}(init)
    GC.@preserve state init init_ref begin
        function cub(temp, nbytes)
            args = (build, temp, nbytes, iterator(x), iterator(y), length(x),
                    cccl_op(operator, state))
            if init === nothing
                cccl_device_inclusive_scan_no_init(args..., stream())
            elseif init isa CUDACore.AbstractCuRef
                init_it = pointer_iterator(T, Base.unsafe_convert(CuPtr{T}, init))
                cccl_device_exclusive_scan_future_value(args..., init_it, stream())
            elseif inclusive
                cccl_device_inclusive_scan(args..., cccl_value(init_ref), stream())
            else
                cccl_device_exclusive_scan(args..., cccl_value(init_ref), stream())
            end
        end
        nbytes = temp_size(cub)
        with_scratch(workspace, nbytes) do temp
            cub(temp, Ref{Csize_t}(nbytes))
        end
    end
end

"""
    CCCL.inclusive_scan!(op, y, x; init=nothing, workspace=nothing)

Store the inclusive scan (prefix reduction) of `x` with the binary operator `op` in `y`,
optionally starting from `init`. Unlike for [`CCCL.exclusive_scan!`](@ref), `init` cannot
be a device reference.
"""
function inclusive_scan!(op, y::DenseCuArray, x::DenseCuArray; init=nothing,
                         workspace::Union{Nothing,DenseCuVector{UInt8}}=nothing)
    scan_impl!(op, y, x, init, true, workspace)
    return y
end

workspace_size(::typeof(inclusive_scan!), op, y::DenseCuArray, x::DenseCuArray;
               init=nothing) = scan_impl!(op, y, x, init, true, SizeQuery())

"""
    CCCL.exclusive_scan!(op, y, x; init, workspace=nothing)

Store the exclusive scan of `x` with the binary operator `op` in `y`, so that `y[1] == init`.

`init` may also be a device reference of the element type of `y`, such as `CuRef(v)`, or
`CUDACore.CuRefArray(a, i)` for the element `a[i]` of a `CuArray`; `y[1]` is then its
value. That value is read on the device when the scan executes, so it can be the result of
work queued before on the same stream without synchronizing, and the scan can be captured
in a CUDA graph. It must not overlap `y`. For example, to continue the inclusive scan `y1`
of `x1` over `x2`:

```julia
CCCL.exclusive_scan!(+, y2, x2; init=CUDACore.CuRefArray(y1, length(y1)))
y2 .+= x2
```
"""
function exclusive_scan!(op, y::DenseCuArray, x::DenseCuArray; init,
                         workspace::Union{Nothing,DenseCuVector{UInt8}}=nothing)
    scan_impl!(op, y, x, init, false, workspace)
    return y
end

workspace_size(::typeof(exclusive_scan!), op, y::DenseCuArray, x::DenseCuArray; init) =
    scan_impl!(op, y, x, init, false, SizeQuery())

for (f, inclusive) in ((:inclusive_scan!, true), (:exclusive_scan!, false))
    @eval unsupported(::typeof($f), op, y, x; init=nothing, kwargs...) =
        reason(unsupported_arrays(y, x), unsupported_eltypes(y, x),
               unsupported_init(y, init, $inclusive), unsupported_operator(op))
end


## segmented reduce

"""
    CCCL.segmented_reduce!(op, out, x, offsets; init, workspace=nothing)

Reduce the segments `x[offsets[i]+1:offsets[i+1]]` of `x` (with zero-based `offsets`, as
in CSR sparse matrices) using `op`, starting from `init`, and store the results in `out`.
"""
function segmented_reduce!(op, out::DenseCuArray, x::DenseCuArray, offsets::DenseCuVector;
                           init, workspace::Union{Nothing,DenseCuVector{UInt8}}=nothing)
    segmented_reduce_impl!(op, out, x, offsets, init, workspace)
    return out
end

workspace_size(::typeof(segmented_reduce!), op, out::DenseCuArray, x::DenseCuArray,
               offsets::DenseCuVector; init) =
    segmented_reduce_impl!(op, out, x, offsets, init, SizeQuery())

function unsupported(::typeof(segmented_reduce!), op, out, x, offsets; kwargs...)
    reason(unsupported_arrays(out, x, offsets), unsupported_eltypes(out, x),
           eltype(offsets) <: Integer ? nothing :
               "CCCL requires integer offsets, got $(eltype(offsets))",
           unsupported_operator(op))
end

function segmented_reduce_impl!(op, out::DenseCuArray{T}, x::DenseCuArray,
                                offsets::DenseCuVector{I}, init,
                                workspace) where {T,I<:Integer}
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
        cub(temp, nbytes) =
            cccl_device_segmented_reduce(build, temp, nbytes, iterator(x), iterator(out),
                                         nsegments, begin_offsets, end_offsets,
                                         cccl_op(operator, state), cccl_value(init_ref),
                                         stream())
        nbytes = temp_size(cub)
        with_scratch(workspace, nbytes) do temp
            cub(temp, Ref{Csize_t}(nbytes))
        end
    end
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

const RadixKey = Union{Base.BitInteger64,Bool,Char,RadixFloat}

# the integer type CUB sorts keys of type `T` as
radix_type(::Type{T}) where {T<:Base.BitInteger64} = T
radix_type(::Type{Bool}) = UInt8
radix_type(::Type{Char}) = UInt32
radix_type(::Type{T}) where {T<:RadixFloat} =
    sizeof(T) == 2 ? UInt16 : sizeof(T) == 4 ? UInt32 : UInt64
radix_type(::Type{T}) where {T} = throw(ArgumentError(unsupported_radix_key(T)))

unsupported_radix_key(T) = T <: RadixKey ? nothing :
    "CCCL radix sorts do not support keys of type $T"

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

# The CUB radix sort of `n` keys of type `K`, and values of type `V` unless that is
# `Nothing`. Returns a function `cub_sort!(temp, keys, alt_keys, values, alt_values)`, which
# sorts the arrays using the others of the same type as scratch space and returns whether
# the result ended up there, and the size of the temporary storage it needs. The pointers
# are only used to describe the types to CUB, but must not be null (except for `n == 0`).
function cub_radix_sort(::Type{K}, ::Type{V}, n::Int, rev::Bool, keys_ptr,
                        values_ptr) where {K,V}
    # HostJIT sorts keys only if the values pointer is null, so we never build from empty
    # arrays: a keys-only build would be cached for sorts of pairs too
    n > 1 || return nothing, 0

    order = rev ? CCCL_DESCENDING : CCCL_ASCENDING
    keys_it = pointer_iterator(K, keys_ptr)
    values_it = V === Nothing ? null_iterator() : pointer_iterator(V, values_ptr)
    build = cached_build((:radix_sort, K, V, order),
                         cccl_device_radix_sort_build_result_t) do result, major, minor, include, ctk, config
        cccl_device_radix_sort_build_ex(result, order, keys_it, values_it, null_op,
                                        C_NULL, major, minor, C_NULL, C_NULL, include, ctk,
                                        config)
    end

    cub(temp, nbytes, keys, alt_keys, values, alt_values, selector) =
        cccl_device_radix_sort(build, temp, nbytes, keys, alt_keys, values, alt_values,
                               null_op, n, 0, 8 * sizeof(K), true, selector, stream())
    nbytes = temp_size() do temp, nbytes
        cub(temp, nbytes, keys_it, keys_it, values_it, values_it, Ref{Cint}(0))
    end

    function cub_sort!(temp, keys::DenseCuArray{K}, alt_keys::DenseCuArray{K}, values,
                       alt_values)
        selector = Ref{Cint}(0)
        cub(temp, Ref{Csize_t}(nbytes), iterator(keys), iterator(alt_keys),
            values === nothing ? null_iterator() : iterator(values),
            alt_values === nothing ? null_iterator() : iterator(alt_values), selector)
        return selector[] == 1
    end
    return cub_sort!, nbytes
end

"""
    CCCL.radix_sort!(keys; values=nothing, rev=false, workspace=nothing)

Sort `keys` in place using a radix sort, optionally permuting `values` along. Keys can be
integers, `Bool`, `Char` or IEEE floating-point numbers (including `BFloat16`), and are
ordered like `sort!` does: by `isless`, so -0.0 comes before 0.0 and NaNs come last (first
when `rev=true`). The sort is stable, except that NaNs are ordered by their bits.
"""
function radix_sort!(keys::DenseCuArray; values::Union{Nothing,DenseCuArray}=nothing,
                     rev::Bool=false, workspace::Union{Nothing,DenseCuVector{UInt8}}=nothing)
    radix_sort_impl!(keys, values, rev, workspace)
    return keys
end

workspace_size(::typeof(radix_sort!), keys::DenseCuArray;
               values::Union{Nothing,DenseCuArray}=nothing, rev::Bool=false) =
    radix_sort_impl!(keys, values, rev, SizeQuery())

unsupported(::typeof(radix_sort!), keys; values=nothing, kwargs...) =
    reason(unsupported_arrays(keys, something(values, keys)),
           unsupported_radix_key(eltype(keys)))

function radix_sort_impl!(keys::DenseCuArray{T}, values, rev, workspace) where {T}
    values === nothing || length(values) == length(keys) ||
        throw(DimensionMismatch("keys and values lengths differ"))
    n = length(keys)
    V = values === nothing ? Nothing : eltype(values)
    values_ptr = values === nothing ? C_NULL : pointer(values)
    R = radix_type(T)

    # without values, floats are sorted as signed integers and fixed up afterwards
    keys_only_float = T <: RadixFloat && values === nothing
    K = keys_only_float ? signed(R) : R
    cub_sort!, nbytes = cub_radix_sort(K, V, n, rev, pointer(keys), values_ptr)
    cub_sort! === nothing && return workspace isa SizeQuery ? 0 : nothing

    alt_values = Buffer{V === Nothing ? UInt8 : V}(values === nothing ? 0 : n)
    with_scratch(workspace, nbytes, Buffer{K}(n), alt_values) do temp, alt_k, alt_v
        k = vec(reinterpret(K, keys))
        alt_v = values === nothing ? nothing : alt_v
        if keys_only_float
            cub_sort!(temp, k, alt_k, nothing, nothing) && copyto!(k, alt_k)
            fix_float_order!(k, T, rev)
        elseif T <: RadixFloat
            k .= encode_float.(T, k)
            if cub_sort!(temp, k, alt_k, values, alt_v)
                k .= decode_float.(T, alt_k)
                copyto!(values, alt_v)
            else
                k .= decode_float.(T, k)
            end
        elseif cub_sort!(temp, k, alt_k, values, alt_v)
            copyto!(k, alt_k)
            values === nothing || copyto!(values, alt_v)
        end
    end
end

# Without values, the order of equal keys is unobservable, so we can avoid transforming the
# keys before and after sorting them. Sorted as signed integers, floats come out as -0.0 to
# -Inf, then negative NaNs, then 0.0 to Inf, then positive NaNs (mirrored when reversed).
# Reversing the negative numbers in place fixes all but the negative NaNs, which are rare
# and which we move past the positive numbers with three more reversals.
function fix_float_order!(keys::DenseCuVector{S}, ::Type{T}, rev::Bool) where {S<:Signed,T}
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
    CCCL.radix_sortperm!(ix, keys; rev=false, workspace=nothing)

Store in `ix` the permutation that sorts `keys`, like `sortperm!(ix, keys; rev)`, using a
radix sort. The sort is stable, and supports the same key types as
[`CCCL.radix_sort!`](@ref).
"""
function radix_sortperm!(ix::DenseCuVector, keys::DenseCuVector; rev::Bool=false,
                         workspace::Union{Nothing,DenseCuVector{UInt8}}=nothing)
    radix_sortperm_impl!(ix, keys, rev, workspace)
    return ix
end

workspace_size(::typeof(radix_sortperm!), ix::DenseCuVector, keys::DenseCuVector;
               rev::Bool=false) = radix_sortperm_impl!(ix, keys, rev, SizeQuery())

function unsupported(::typeof(radix_sortperm!), ix, keys; kwargs...)
    reason(unsupported_arrays(ix, keys),
           keys isa AbstractVector && ix isa AbstractVector ? nothing :
               "CCCL radix sortperms only support vectors",
           eltype(ix) <: Base.BitInteger ? nothing :
               "CCCL radix sortperms need integer indices, got $(eltype(ix))",
           unsupported_radix_key(eltype(keys)))
end

function radix_sortperm_impl!(ix::DenseCuVector{I}, keys::DenseCuVector{T}, rev,
                              workspace) where {I<:Integer,T}
    n = length(keys)
    length(ix) == n || throw(DimensionMismatch("index and keys lengths differ"))
    n <= typemax(I) || throw(ArgumentError("index type $I cannot hold $n indices"))
    R = radix_type(T)

    # moving 32-bit indices around is much cheaper, even with widening them afterwards
    J = sizeof(I) > 4 && n <= typemax(Int32) ? Int32 : I
    cub_sort!, nbytes = cub_radix_sort(R, J, n, rev, pointer(keys), pointer(ix))
    if cub_sort! === nothing
        workspace isa SizeQuery && return 0
        n == 1 && fill!(ix, one(I))
        return
    end

    buffers = (Buffer{R}(n), Buffer{R}(n), Buffer{J}(n), Buffer{J}(J == I ? 0 : n))
    with_scratch(workspace, nbytes, buffers...) do temp, sort_keys, alt_keys, alt_ix, sort_ix
        sort_ix = J == I ? ix : sort_ix
        kernel = @cuda launch=false sortperm_init_kernel(sort_keys, sort_ix, keys)
        threads = min(n, launch_configuration(kernel.fun).threads)
        kernel(sort_keys, sort_ix, keys; threads, blocks=cld(n, threads))

        swapped = cub_sort!(temp, sort_keys, alt_keys, sort_ix, alt_ix)
        if swapped || sort_ix !== ix
            ix .= swapped ? alt_ix : sort_ix
        end
    end
end

"""
    CCCL.radix_sortperm(keys; rev=false)

Return the permutation that sorts `keys`, like `sortperm(keys; rev)`, using a radix sort.
"""
radix_sortperm(keys::DenseCuVector; rev::Bool=false) =
    radix_sortperm!(CuVector{Int}(undef, length(keys)), keys; rev)


## merge sort

"""
    CCCL.merge_sort!(keys; lt=isless, values=nothing, workspace=nothing)

Sort `keys` in place using a merge sort with the comparison function `lt`, optionally
permuting `values` along.
"""
function merge_sort!(keys::DenseCuArray; lt=isless,
                     values::Union{Nothing,DenseCuArray}=nothing,
                     workspace::Union{Nothing,DenseCuVector{UInt8}}=nothing)
    merge_sort_impl!(keys, lt, values, workspace)
    return keys
end

workspace_size(::typeof(merge_sort!), keys::DenseCuArray; lt=isless,
               values::Union{Nothing,DenseCuArray}=nothing) =
    merge_sort_impl!(keys, lt, values, SizeQuery())

unsupported(::typeof(merge_sort!), keys; lt=isless, values=nothing, kwargs...) =
    reason(unsupported_arrays(keys, something(values, keys)), unsupported_operator(lt))

function merge_sort_impl!(keys::DenseCuArray{K}, lt, values, workspace) where {K}
    values === nothing || length(values) == length(keys) ||
        throw(DimensionMismatch("keys and values lengths differ"))
    n = length(keys)
    # like radix sorts, merge sorts would sort keys only if built from empty values
    n > 1 || return workspace isa SizeQuery ? 0 : nothing
    V = values === nothing ? Nothing : eltype(values)
    operator = CCCL.operator(lt, K, K, Bool)

    values_it = values === nothing ? null_iterator() : iterator(values)
    build = cached_build((:merge_sort, K, V, operator.name),
                         cccl_device_merge_sort_build_result_t) do result, major, minor, include, ctk, config
        cccl_device_merge_sort_build_ex(result, iterator(keys), values_it, iterator(keys),
                                        values_it, cccl_op(operator), major, minor,
                                        C_NULL, C_NULL, include, ctk, config)
    end

    state = op_state(lt)
    GC.@preserve state begin
        cub(temp, nbytes, out_keys, out_values) =
            cccl_device_merge_sort(build, temp, nbytes, iterator(keys), values_it, out_keys,
                                   out_values, n, cccl_op(operator, state), stream())
        nbytes = temp_size() do temp, nbytes
            cub(temp, nbytes, iterator(keys), values_it)
        end

        out_values = Buffer{V === Nothing ? UInt8 : V}(values === nothing ? 0 : n)
        with_scratch(workspace, nbytes, Buffer{K}(n), out_values) do temp, out_k, out_v
            out_v_it = values === nothing ? null_iterator() : iterator(out_v)
            cub(temp, Ref{Csize_t}(nbytes), iterator(out_k), out_v_it)
            copyto!(keys, out_k)
            values === nothing || copyto!(values, out_v)
        end
    end
end
