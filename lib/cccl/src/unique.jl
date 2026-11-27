# unique by key: compacting runs of consecutive equal keys


## implementation

overlaps(x::DenseCuArray, y::DenseCuArray) = !isempty(x) && !isempty(y) &&
    UInt(pointer(x)) < UInt(pointer(y)) + sizeof(y) &&
    UInt(pointer(y)) < UInt(pointer(x)) + sizeof(x)

# the types CUB can write the number of runs as
const RunCount = Base.BitInteger64

unsupported_count(count) = count === nothing ||
    (count isa AbstractVector && isconcretetype(eltype(count)) &&
     eltype(count) <: RunCount) ? nothing :
    "CCCL uniques need a vector of (at most 64-bit) integers as count"

function check_unique_arrays(f, outputs, inputs, count)
    for (out, in) in zip(outputs, inputs)
        check_eltypes(f, eltype(in), out)
        length(out) >= length(in) ||
            throw(DimensionMismatch("output vectors need to be as long as the inputs"))
    end
    written = count === nothing ? outputs : (outputs..., view(count, 1:1))
    for (i, out) in enumerate(written), other in (inputs..., written[i+1:end]...)
        overlaps(out, other) &&
            throw(ArgumentError("outputs must not overlap each other or the inputs"))
    end
end

# Select the first key of each run of `keys` (and its value, unless `values` is `nothing`)
# into `keys_out` (and `values_out`). Returns the number of runs, read back from the device,
# or `count` with the number in its first element.
function unique_impl!(keys_out::DenseCuVector, values_out, keys::DenseCuVector{K}, values,
                      eq, count, workspace) where {K}
    msg = unsupported_sizes(keys, something(values, keys))
    msg === nothing || throw(ArgumentError(msg))
    n = length(keys)
    I = count === nothing ? Int : eltype(count)
    if count !== nothing
        msg = unsupported_count(count)
        msg === nothing || throw(ArgumentError(msg))
        length(count) >= 1 || throw(ArgumentError("count needs to hold at least one value"))
        n <= typemax(I) || throw(ArgumentError("count type $I cannot hold $n"))
    end
    if values === nothing
        check_unique_arrays(:unique_consecutive!, (keys_out,), (keys,), count)
    else
        length(values) == length(keys) ||
            throw(DimensionMismatch("keys and values lengths differ"))
        check_unique_arrays(:unique_by_key!, (keys_out, values_out), (keys, values), count)
    end

    # never build from empty arrays, whose null pointers could select other variants
    if n == 0
        workspace isa SizeQuery && return 0
        count === nothing && return 0
        fill!(view(count, 1:1), zero(I))
        return count
    end

    # keys-only builds use other iterators than builds with values, even of `Nothing`
    V = values === nothing ? :keys_only : eltype(values)
    operator = CCCL.operator(eq, K, K, Bool)
    state = op_state(eq)
    # CUB only selects keys together with values, so give keys-only uniques zeros as values
    # and discard them afterwards
    values_in, values_out_it = if values === nothing
        zeros_iterator(UInt8), discard_iterator(UInt8)
    else
        iterator(values), iterator(values_out)
    end
    count_it(ptr) = pointer_iterator(I, ptr)

    GC.@preserve keys values keys_out values_out count state begin
        # builds only need a non-null pointer for the count
        build = cached_build((:unique_by_key, K, V, I, operator.name),
                             cccl_device_unique_by_key_build_result_t) do result, major, minor, include, ctk, config
            cccl_device_unique_by_key_build_ex(result, iterator(keys), values_in,
                                               iterator(keys_out), values_out_it,
                                               count_it(pointer(keys_out)),
                                               cccl_op(operator), major, minor, C_NULL,
                                               C_NULL, include, ctk, config)
        end

        cub(temp, nbytes, count_ptr) =
            cccl_device_unique_by_key(build, temp, nbytes, iterator(keys), values_in,
                                      iterator(keys_out), values_out_it, count_it(count_ptr),
                                      cccl_op(operator, state), n, stream())
        nbytes = temp_size((temp, nbytes) -> cub(temp, nbytes, C_NULL))

        if count === nothing
            with_scratch(workspace, nbytes, Buffer{I}(1)) do temp, c
                cub(temp, Ref{Csize_t}(nbytes), pointer(c))
                Int(CUDACore.@allowscalar c[1])
            end
        else
            size = with_scratch(workspace, nbytes) do temp
                cub(temp, Ref{Csize_t}(nbytes), pointer(count))
            end
            workspace isa SizeQuery ? size : count
        end
    end
end


## keys and values

"""
    CCCL.unique_by_key!(keys_out, values_out, keys, values;
                        eq=isequal, count=nothing, workspace=nothing) -> Int

Copy the first key of every run of consecutive equal `keys` to the front of `keys_out`,
and the value at the position of that key in `values` to `values_out`, keeping their order.
Return the number of runs. With the default `eq=isequal` and sorted keys, the keys selected
are those that `unique!` keeps. See [`CCCL.unique_consecutive!`](@ref) for keys without values.

Every key is compared with the one before it, as `eq(previous, key)`, so that a run is
a maximal range of keys that are each `eq` to their predecessor; for an equivalence
relation like the default, `isequal`, this means all keys of a run are equal (all NaNs
are, but -0.0 differs from 0.0). `eq` must return a `Bool`.

The outputs need to be at least as long as the inputs, and must not overlap each other,
the inputs or `count[1]`, nor may `count[1]` overlap the inputs. Their elements beyond the number of runs are left unspecified.

Returning the number of runs waits for the operation to finish. To avoid that, pass a
vector of integers (of at most 64 bits, and able to hold `length(keys)`) as `count`: the
number of runs is then stored in `count[1]`, on the device, and `count` is returned.
"""
function unique_by_key!(keys_out::DenseCuVector, values_out::DenseCuVector,
                        keys::DenseCuVector, values::DenseCuVector; eq=isequal,
                        count::Union{Nothing,DenseCuVector{<:Integer}}=nothing,
                        workspace::Union{Nothing,DenseCuVector{UInt8}}=nothing)
    unique_impl!(keys_out, values_out, keys, values, eq, count, workspace)
end

workspace_size(::typeof(unique_by_key!), keys_out::DenseCuVector, values_out::DenseCuVector,
               keys::DenseCuVector, values::DenseCuVector; eq=isequal,
               count::Union{Nothing,DenseCuVector{<:Integer}}=nothing) =
    unique_impl!(keys_out, values_out, keys, values, eq, count, SizeQuery())

"""
    CCCL.unique_by_key(keys, values; eq=isequal) -> (keys_out, values_out)

Return the first key of every run of consecutive equal `keys`, and the corresponding
elements of `values`, as new vectors. This waits for the operation to finish. See
[`CCCL.unique_by_key!`](@ref).
"""
function unique_by_key(keys::DenseCuVector, values::DenseCuVector; eq=isequal)
    keys_out, values_out = similar(keys), similar(values)
    n = unique_by_key!(keys_out, values_out, keys, values; eq)
    return resize!(keys_out, n), resize!(values_out, n)
end


## keys only

"""
    CCCL.unique_consecutive!(keys_out, keys; eq=isequal, count=nothing, workspace=nothing)
        -> Int

Copy the first key of every run of consecutive equal `keys` to the front of `keys_out`,
keeping their order, and return the number of runs. With the default `eq=isequal` and
sorted keys, the keys selected are those that `unique!` keeps, but `keys` is left
unchanged. Equality, the requirements on the arguments and
`count` are as for [`CCCL.unique_by_key!`](@ref).
"""
function unique_consecutive!(keys_out::DenseCuVector, keys::DenseCuVector; eq=isequal,
                             count::Union{Nothing,DenseCuVector{<:Integer}}=nothing,
                             workspace::Union{Nothing,DenseCuVector{UInt8}}=nothing)
    unique_impl!(keys_out, nothing, keys, nothing, eq, count, workspace)
end

workspace_size(::typeof(unique_consecutive!), keys_out::DenseCuVector, keys::DenseCuVector;
               eq=isequal, count::Union{Nothing,DenseCuVector{<:Integer}}=nothing) =
    unique_impl!(keys_out, nothing, keys, nothing, eq, count, SizeQuery())

"""
    CCCL.unique_consecutive(keys; eq=isequal) -> keys_out

Return the first key of every run of consecutive equal `keys` as a new vector. With the
default `eq=isequal` and sorted keys, this is `unique(keys)`. This waits for the operation
to finish. See [`CCCL.unique_consecutive!`](@ref).
"""
function unique_consecutive(keys::DenseCuVector; eq=isequal)
    keys_out = similar(keys)
    return resize!(keys_out, unique_consecutive!(keys_out, keys; eq))
end


## support

# HostJIT cannot represent elements without storage
unsupported_sizes(arrays...) = all(x -> sizeof(eltype(x)) > 0, arrays) ? nothing :
    "CCCL uniques do not support element types of size zero"

function unsupported_unique(arrays, eq, count)
    reason(unsupported_arrays(arrays...), unsupported_sizes(arrays...),
           all(x -> x isa AbstractVector, arrays) ? nothing :
               "CCCL uniques only support vectors",
           count === nothing ? nothing : unsupported_arrays(count),
           unsupported_count(count),
           unsupported_operator(eq))
end

unsupported(::typeof(unique_by_key!), keys_out, values_out, keys, values; eq=isequal,
            count=nothing, kwargs...) =
    reason(unsupported_unique((keys_out, values_out, keys, values), eq, count),
           unsupported_eltypes(keys_out, keys), unsupported_eltypes(values_out, values))

unsupported(::typeof(unique_consecutive!), keys_out, keys; eq=isequal, count=nothing,
            kwargs...) =
    reason(unsupported_unique((keys_out, keys), eq, count),
           unsupported_eltypes(keys_out, keys))
