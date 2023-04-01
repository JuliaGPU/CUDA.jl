export CuIterator

"""
    CuIterator(batches)

Return a `CuIterator` that can iterate through the provided `batches` via `Base.iterate`.

Upon each iteration, the current `batch` is copied to the GPU,
and the previous iteration is marked as freeable from GPU memory (via `unsafe_free!`).
Both of these use `adapt`, so that each `batch` can be an array, an array of arrays,
or a more complex object such as a nested set of NamedTuples, which is explored recursively.

This abstraction is useful for batching data into GPU memory in a manner that
allows old iterations to potentially be freed (or marked as reusable) earlier
than they otherwise would via CuArray's internal polling mechanism.
"""
mutable struct CuIterator{B}
    batches::B
    previous::Any
    CuIterator(batches) = new{typeof(batches)}(batches)
end

function Base.iterate(c::CuIterator, state...)
    item = iterate(c.batches, state...)
    isdefined(c, :previous) && adapt(CuIteratorFree, c.previous)
    item === nothing && return nothing
    batch, next_state = item
    cubatch = adapt(CuIterator, batch)
    c.previous = cubatch
    return cubatch, next_state
end

Base.IteratorSize(::Type{CuIterator{B}}) where {B} = Base.IteratorSize(B)
Base.length(c::CuIterator) = length(c.batches)  # required for HasLength
Base.axes(c::CuIterator) = axes(c.batches)  # required for HasShape{N}

Base.IteratorEltype(::Type{CuIterator{B}}) where {B} = Base.IteratorEltype(B)
Base.eltype(c::CuIterator) = eltype(c.batches)  # required for HasEltype

# This struct exists to control adapt for clean-up-afterwards step:
struct CuIteratorFree end
Adapt.adapt_storage(::Type{CuIteratorFree}, x::CuArray) = unsafe_free!(x)

# We re-purpose struct CuIterator for the matching transfer-before-use step,
# mostly fall back to adapt(CuArray, x) which recurses into Tuples etc:
Adapt.adapt_storage(::Type{<:CuIterator}, x) = adapt(CuArray, x)

# But unlike adapt(CuArray, x), returse into arrays of arrays:
function Adapt.adapt_storage(::Type{<:CuIterator}, xs::AbstractArray{T}) where T
    isbitstype(T) ? adapt(CuArray, xs) : map(adapt(CuArray), xs)
end
function Adapt.adapt_storage(::Type{CuIteratorFree}, xs::AbstractArray{T}) where T
    foreach(adapt(CuIteratorFree), xs)
    xs
end
