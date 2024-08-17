export CuIterator

"""
    CuIterator([to], batches)

Create a `CuIterator` that iterates through the provided `batches` via `iterate`. Upon each
iteration, the current `batch` is copied to the GPU, and the previous iteration is marked as
freeable from GPU memory (via `unsafe_free!`).

The conversion to GPU memory is done recursively, using Adapt.jl, so that each batch can be
an array, an array of arrays, or more complex iterable objects. To customize the conversion,
an adaptor can be specified as the first argument, e.g., to change the element type:

```julia
julia> first(CuIterator([[1.]]))
1-element CuArray{Float64, 1, CUDA.DeviceMemory}:
 1.0

julia> first(CuIterator(CuArray{Float32}, [[1.]]))
1-element CuArray{Float32, 1, CUDA.DeviceMemory}:
 1.0
```

This abstraction is useful for batching data into GPU memory in a manner that allows old
iterations to potentially be freed (or marked as reusable) earlier than they otherwise would
via `CuArray`'s internal polling mechanism.
"""
mutable struct CuIterator{T,B}
    to::T
    batches::B
    previous::Any

    CuIterator(batches) = CuIterator(nothing, batches)
    CuIterator(to, batches) = new{typeof(to),typeof(batches)}(to, batches)
end

function Base.iterate(c::CuIterator, state...)
    item = iterate(c.batches, state...)
    isdefined(c, :previous) && adapt(CuIteratorFree(), c.previous)
    item === nothing && return nothing
    batch, next_state = item
    cubatch = adapt(c, batch)
    c.previous = cubatch
    return cubatch, next_state
end

Base.IteratorSize(::Type{CuIterator{T,B}}) where {T,B} = Base.IteratorSize(B)
Base.length(c::CuIterator) = length(c.batches)  # required for HasLength
Base.axes(c::CuIterator) = axes(c.batches)  # required for HasShape{N}

Base.IteratorEltype(::Type{CuIterator{T,B}}) where {T,B} = Base.IteratorEltype(B)
Base.eltype(c::CuIterator) = eltype(c.batches)  # required for HasEltype

# adaptor for uploading
Adapt.adapt_storage(c::CuIterator, x) = adapt(something(c.to, CuArray), x)
## unlike adapt(CuArray, x), recurse into arrays of arrays
function Adapt.adapt_storage(c::CuIterator, xs::AbstractArray{T}) where T
    to = something(c.to, CuArray)
    isbitstype(T) ? adapt(to, xs) : map(to, xs)
end

# adaptor for clean-up
struct CuIteratorFree end
Adapt.adapt_storage(::CuIteratorFree, x::CuArray) = unsafe_free!(x)
function Adapt.adapt_storage(::CuIteratorFree, xs::AbstractArray{T}) where T
    foreach(adapt(CuIteratorFree()), xs)
    xs
end

