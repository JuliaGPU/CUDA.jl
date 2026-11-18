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

"""
    CCCL.radix_sort!(keys; values=nothing, rev=false)

Sort `keys` (of a primitive numeric type) in place using a radix sort, optionally
permuting `values` along.
"""
function radix_sort!(keys::DenseCuArray{K}; values::Union{Nothing,DenseCuArray}=nothing,
                     rev::Bool=false) where {K}
    values === nothing || length(values) == length(keys) ||
        throw(DimensionMismatch("keys and values lengths differ"))
    cccl_type(K) == CCCL_STORAGE &&
        throw(ArgumentError("CCCL.radix_sort! only supports primitive key types, got $K"))
    V = values === nothing ? Nothing : eltype(values)
    order = rev ? CCCL_DESCENDING : CCCL_ASCENDING
    values_it = values === nothing ? null_iterator() : iterator(values)
    build = cached_build((:radix_sort, K, V, order),
                         cccl_device_radix_sort_build_result_t) do result, major, minor, include, ctk, config
        cccl_device_radix_sort_build_ex(result, order, iterator(keys), values_it, null_op,
                                        C_NULL, major, minor, C_NULL, C_NULL, include, ctk,
                                        config)
    end

    # sort into alternate buffers, letting CUB pick where the result ends up
    alt_keys = similar(keys)
    alt_values = values === nothing ? nothing : similar(values)
    alt_values_it = alt_values === nothing ? null_iterator() : iterator(alt_values)
    selector = Ref{Cint}(0)
    with_temp_storage() do temp, nbytes
        cccl_device_radix_sort(build, temp, nbytes, iterator(keys), iterator(alt_keys),
                               values_it, alt_values_it, null_op, length(keys), 0,
                               8 * sizeof(K), true, selector, stream())
    end
    if selector[] == 1
        copyto!(keys, alt_keys)
        values === nothing || copyto!(values, alt_values)
    end
    CUDACore.unsafe_free!(alt_keys)
    alt_values === nothing || CUDACore.unsafe_free!(alt_values)
    return keys
end


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
