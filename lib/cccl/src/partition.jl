# partitioning and stream compaction
#
# The C library offers one algorithm for these, CUB's three-way `DevicePartition::If`: two
# predicates split the input into three stable parts. Unlike what CUB's documentation says,
# the unselected part keeps its order too. Everything else is built on it, using custom
# output iterators to discard a part or to write it backwards from the end of an array.


## output iterators

# writes backwards, starting at the address in its state
struct ReverseAdvance{T} end
@inline function (::ReverseAdvance{T})(state::OpPtr, offset::OpPtr) where {T}
    store!(state, load(UInt, state) - load(Int, offset) * sizeof(T))
    return
end
struct ReverseStore{T} end
@inline function (::ReverseStore{T})(state::OpPtr, value::OpPtr) where {T}
    unsafe_store!(reinterpret(LLVMPtr{T,0}, load(UInt, state)), load(T, value), 1,
                  Val(Base.datatype_alignment(T)))
    return
end

# a vector to be written from its end; `state` holds the address of its last element
struct Reversed{A<:DenseCuVector}
    parent::A
    state::Base.RefValue{UInt}
end
Reversed(y::DenseCuVector) = Reversed(y, Ref{UInt}(0))

function reverse_iterator(r::Reversed{<:DenseCuVector{T}}) where {T}
    r.state[] = UInt(pointer(r.parent, length(r.parent)))
    custom_iterator(iterator_function(ReverseAdvance{T}), iterator_function(ReverseStore{T}),
                    T, r.state)
end

output_iterator(y::DenseCuVector, T) = iterator(y)
output_iterator(::Nothing, T) = discard_iterator(T)
output_iterator(r::Reversed, T) = reverse_iterator(r)

# what selects the build variant
output_kind(::DenseCuVector) = :pointer
output_kind(::Nothing) = :discard
output_kind(::Reversed) = :reverse

output_array(y) = y
output_array(r::Reversed) = r.parent

isreversed(y) = y isa Reversed


## checks

const PartitionCount = Union{Int32,Int64,UInt32,UInt64}

struct AlwaysFalse end
(::AlwaysFalse)(x) = false

function unsupported_predicate(f, T)
    R = Base.promote_op(f, T)
    reason(unsupported_operator(f),
           R === Bool ? nothing : "CCCL predicates must return a Bool, got $R for $T")
end

unsupported_counts(::Nothing) = nothing
unsupported_counts(counts) = reason(unsupported_arrays(counts),
    counts isa AbstractVector && eltype(counts) <: PartitionCount ? nothing :
        "CCCL partitions count into vectors of $PartitionCount, got a $(typeof(counts))")

function unsupported_partition(x, outputs, counts, predicates...)
    arrays = Base.filter(!isnothing, (x, outputs...))
    reason(unsupported_arrays(arrays...),
           all(y -> y isa Union{Nothing,AbstractVector}, outputs) ? nothing :
               "CCCL partitions write to vectors",
           map(y -> unsupported_eltypes(x, y), arrays)...,
           Base.elsize(Vector{eltype(x)}) == sizeof(eltype(x)) ? nothing :
               "CCCL partitions do not support $(eltype(x)), which arrays pad",
           unsupported_counts(counts),
           map(f -> unsupported_predicate(f, eltype(x)), predicates)...)
end

function memory_range(x::DenseCuArray)
    lo = UInt(pointer(x))
    return lo:(lo + sizeof(x) - 1)
end

function check_disjoint(arrays...)
    arrays = Base.filter(x -> x !== nothing && !isempty(x), arrays)
    for i in eachindex(arrays), j in 1:i-1
        isempty(intersect(memory_range(arrays[i]), memory_range(arrays[j]))) ||
            throw(ArgumentError("CCCL partitions require their arguments not to overlap"))
    end
end

function check_partition_args(x, outputs, counts, ncounts)
    T = eltype(x)
    Base.elsize(Vector{T}) == sizeof(T) ||
        throw(ArgumentError("CCCL partitions do not support $T, which arrays pad"))
    for y in (x, outputs..., counts)
        y === nothing || isempty(y) ||
            UInt(pointer(y)) % Base.datatype_alignment(eltype(y)) == 0 ||
            throw(ArgumentError("CCCL partitions require aligned arrays"))
    end
    n = length(x)
    for y in outputs
        y === nothing && continue
        length(y) >= n || throw(DimensionMismatch(
            "output vectors need to hold all $n input elements, got one of $(length(y))"))
    end
    if counts !== nothing
        problem = unsupported_counts(counts)
        problem === nothing || throw(ArgumentError(problem))
        length(counts) >= ncounts || throw(DimensionMismatch(
            "counts vector needs to hold $ncounts counts, got $(length(counts))"))
        n <= typemax(eltype(counts)) ||
            throw(ArgumentError("count type $(eltype(counts)) cannot hold $n"))
    end
end


## the CUB algorithm

# The CUB partition of `n` elements of type `T` into `outputs` (vectors, `nothing` to
# discard, or `Reversed` vectors) with counts of type `C`. Returns a function
# `run!(temp, x, cell)` partitioning `x`, with the counts written to the 2-element `cell`,
# and the size of the temporary storage it needs. `x` must not be empty.
function cub_partition(f1, f2, x::DenseCuArray{T}, outputs, ::Type{C}) where {T,C}
    op1 = CCCL.operator(f1, T, Bool)
    op2 = CCCL.operator(f2, T, Bool)
    out_its = map(y -> output_iterator(y, T), outputs)
    # only describes the type, so must not be null
    counts_it = pointer_iterator(C, pointer(x))
    key = (:three_way_partition, T, C, map(output_kind, outputs)..., op1.name, op2.name)
    build = cached_build(key, cccl_device_three_way_partition_build_result_t) do result, major, minor, include, ctk, config
        GC.@preserve outputs begin
            cccl_device_three_way_partition_build_ex(result, iterator(x), out_its...,
                                                     counts_it, cccl_op(op1), cccl_op(op2),
                                                     major, minor, C_NULL, C_NULL, include,
                                                     ctk, config)
        end
    end

    state1, state2 = op_state(f1), op_state(f2)
    function cub(temp, nbytes, input, cell)
        GC.@preserve state1 state2 outputs begin
            cccl_device_three_way_partition(build, temp, nbytes, input, out_its..., cell,
                                            cccl_op(op1, state1), cccl_op(op2, state2),
                                            length(x), stream())
        end
    end
    nbytes = temp_size() do temp, nbytes
        cub(temp, nbytes, iterator(x), counts_it)
    end
    run!(temp, input, cell) = GC.@preserve input cell begin
        cub(temp, Ref{Csize_t}(nbytes), iterator(input), iterator(cell))
    end
    return run!, nbytes
end

# Partition `x` with `f1` and `f2` into `outputs`, and write the first `ncounts` counts to
# `counts`, or return them if that is `nothing`. With `inplace`, `x` is copied to scratch
# memory first, so that it can be one of the outputs. `after(cell)` runs once CUB is done,
# with the device vector holding the counts.
function partition_impl!(f1, f2, x::DenseCuArray{T}, outputs, counts, ncounts, workspace;
                         inplace::Bool=false, after=Returns(nothing)) where {T}
    arrays = map(output_array, outputs)
    check_eltypes(:partition!, T, Base.filter(!isnothing, arrays)...)
    check_partition_args(x, arrays, counts, ncounts)
    # a reversed output shares its vector with another output by construction
    workspace isa SizeQuery ||
        check_disjoint((inplace ? () : (x,))..., Base.filter(!isreversed, outputs)...,
                       counts, workspace)
    if isempty(x)
        workspace isa SizeQuery && return 0
        counts === nothing && return ntuple(Returns(0), ncounts)
        fill!(view(counts, 1:ncounts), 0)
        return
    end

    C = counts === nothing ? Int : eltype(counts)
    # CUB always writes two counts, so needs a cell of its own unless we want both
    own_cell = counts === nothing || ncounts < 2
    run!, nbytes = cub_partition(f1, f2, x, outputs, C)
    buffers = (Buffer{C}(own_cell ? 2 : 0), Buffer{T}(inplace ? length(x) : 0))
    with_scratch(workspace, nbytes, buffers...) do temp, cell, copy
        cell = own_cell ? cell : counts
        input = inplace ? copyto!(copy, 1, x, 1, length(x)) : x
        run!(temp, input, cell)
        after(cell)
        if counts === nothing
            host = Array(cell)
            return ntuple(i -> Int(host[i]), ncounts)
        elseif own_cell
            copyto!(counts, 1, cell, 1, ncounts)
        end
        return
    end
end

const counts_doc = """
With `counts=nothing`, the count is copied to the host, which waits for the partition to
finish. Pass a device vector of integers (`Int32`, `Int64`, `UInt32` or `UInt64`) as
`counts` to have it written there instead, without synchronizing; the call then returns
`nothing`."""


## three-way partition

"""
    CCCL.three_way_partition!(f1, f2, y1, y2, y3, x; counts=nothing, workspace=nothing)
        -> (n1, n2)

Partition `x` into three parts, keeping the order of the elements within each: the
elements `v` for which `f1(v)` holds go to `y1[1:n1]`, those for which `f2(v)` holds of the
remaining ones to `y2[1:n2]`, and the others to `y3[1:length(x)-n1-n2]`. `f2` is only called
on elements for which `f1` is `false`.

The predicates must return a `Bool`, and are called an unspecified number of times. The
outputs are vectors of the element type of `x`, each long enough to hold all of `x`, and
must not overlap with `x` or each other. Elements beyond the counts are left unspecified.
Any of the outputs may be `nothing` to discard that part.

$counts_doc The counts are the first two elements of `counts` then, `n1` and `n2`.
"""
function three_way_partition!(f1, f2, y1, y2, y3, x::DenseCuArray;
                              counts::Union{Nothing,DenseCuVector}=nothing,
                              workspace::Union{Nothing,DenseCuVector{UInt8}}=nothing)
    check_predicates(x, f1, f2)
    partition_impl!(f1, f2, x, (y1, y2, y3), counts, 2, workspace)
end

workspace_size(::typeof(three_way_partition!), f1, f2, y1, y2, y3, x::DenseCuArray;
               counts::Union{Nothing,DenseCuVector}=nothing) =
    partition_impl!(f1, f2, x, (y1, y2, y3), counts, 2, SizeQuery())

unsupported(::typeof(three_way_partition!), f1, f2, y1, y2, y3, x;
            counts=nothing, kwargs...) =
    unsupported_partition(x, (y1, y2, y3), counts, f1, f2)

function check_predicates(x, predicates...)
    for f in predicates
        problem = unsupported_predicate(f, eltype(x))
        problem === nothing || throw(ArgumentError(problem))
    end
end


## stream compaction

"""
    CCCL.select!(f, y, x; counts=nothing, workspace=nothing) -> k

Copy the elements `v` of `x` for which `f(v)` holds to `y[1:k]`, in order. `f` must return a
`Bool`; `y` must be a vector of the element type of `x` that can hold all of it, and must
not overlap with `x`. Elements of `y` past `k` are left unchanged.

$counts_doc

To filter `x` in place, as `filter!` does, select into a scratch vector, copy its first `k`
elements back and `resize!` `x`. [`CCCL.filter`](@ref) returns a new vector.
"""
function select!(f, y::DenseCuVector, x::DenseCuArray;
                 counts::Union{Nothing,DenseCuVector}=nothing,
                 workspace::Union{Nothing,DenseCuVector{UInt8}}=nothing)
    check_predicates(x, f)
    k = partition_impl!(f, AlwaysFalse(), x, (y, nothing, nothing), counts, 1, workspace)
    return k === nothing ? nothing : k[1]
end

workspace_size(::typeof(select!), f, y::DenseCuVector, x::DenseCuArray;
               counts::Union{Nothing,DenseCuVector}=nothing) =
    partition_impl!(f, AlwaysFalse(), x, (y, nothing, nothing), counts, 1, SizeQuery())

unsupported(::typeof(select!), f, y, x; counts=nothing, kwargs...) =
    reason(unsupported_arrays(y), unsupported_partition(x, (y,), counts, f))

"""
    CCCL.filter(f, x) -> CuVector

Return the elements `v` of `x` for which `f(v)` holds, like `filter(f, x)`, as a vector.
See [`CCCL.select!`](@ref) for the requirements on `f`. This waits for the result to be
computed; the returned vector may keep the memory needed for all of `x`.
"""
function filter(f, x::DenseCuArray{T}) where {T}
    y = similar(x, T, length(x))
    k = select!(f, y, x)
    return resize!(y, k)
end


## two-way partition
#
# CUB writes the selected elements to the front of the output, and the others backwards
# from its end, as its own two-way partition does. Afterwards, we reverse those in place.

function reverse_tail_kernel(y::CuDeviceVector, cell::CuDeviceVector)
    n = length(y)
    k = Int(@inbounds cell[1])
    m = (n - k) ÷ 2
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= m
        a, b = k + i, n - i + 1
        @inbounds y[a], y[b] = y[b], y[a]
        i += stride
    end
    return
end

function reverse_tail!(y::DenseCuVector, cell::DenseCuVector)
    threads = 256
    sms = attribute(device(), CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    blocks = clamp(cld(length(y) ÷ 2, threads), 1, 4 * sms)
    @cuda threads blocks reverse_tail_kernel(y, cell)
    return
end

partition_outputs(y) = (y, nothing, Reversed(y))

"""
    CCCL.partition!(f, y, x; counts=nothing, workspace=nothing) -> k
    CCCL.partition!(f, x; counts=nothing, workspace=nothing) -> k

Partition `x` by the predicate `f`, keeping the order of the elements in each part: the
elements `v` for which `f(v)` holds go to `y[1:k]`, the others to `y[k+1:length(x)]`. `y`
must be a vector of the element type of `x` and of the same length, and must not overlap
with `x`. Without `y`, partition the vector `x` in place.

`f` must return a `Bool`, and is called an unspecified number of times.

$counts_doc
"""
function partition!(f, y::DenseCuVector, x::DenseCuArray;
                    counts::Union{Nothing,DenseCuVector}=nothing,
                    workspace::Union{Nothing,DenseCuVector{UInt8}}=nothing)
    check_predicates(x, f)
    partition_impl(f, y, x, counts, workspace, false)
end

function partition!(f, x::DenseCuVector; counts::Union{Nothing,DenseCuVector}=nothing,
                    workspace::Union{Nothing,DenseCuVector{UInt8}}=nothing)
    check_predicates(x, f)
    partition_impl(f, x, x, counts, workspace, true)
end

function partition_impl(f, y, x, counts, workspace, inplace)
    length(y) == length(x) ||
        throw(DimensionMismatch("input and output lengths differ"))
    k = partition_impl!(f, AlwaysFalse(), x, partition_outputs(y), counts, 1, workspace;
                        inplace, after=cell -> reverse_tail!(y, cell))
    return k isa Tuple ? k[1] : k
end

workspace_size(::typeof(partition!), f, y::DenseCuVector, x::DenseCuArray;
               counts::Union{Nothing,DenseCuVector}=nothing) =
    partition_impl(f, y, x, counts, SizeQuery(), false)
workspace_size(::typeof(partition!), f, x::DenseCuVector;
               counts::Union{Nothing,DenseCuVector}=nothing) =
    partition_impl(f, x, x, counts, SizeQuery(), true)

unsupported(::typeof(partition!), f, y, x; counts=nothing, kwargs...) =
    reason(unsupported_arrays(y), unsupported_partition(x, (y,), counts, f))
unsupported(::typeof(partition!), f, x; counts=nothing, kwargs...) =
    reason(x isa AbstractVector ? nothing : "CCCL partitions in place only vectors",
           unsupported_partition(x, (x,), counts, f))

"""
    CCCL.partition(f, x) -> (yes, no)

Return the elements `v` of `x` for which `f(v)` holds, and the others, as two vectors,
keeping their order. See [`CCCL.partition!`](@ref) for the requirements on `f`. This waits
for the result to be computed. Both vectors are views of one array of the length of `x`.
"""
function partition(f, x::DenseCuArray{T}) where {T}
    y = similar(x, T, length(x))
    k = partition!(f, y, x)
    return view(y, 1:k), view(y, k+1:length(x))
end
