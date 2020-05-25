export CuIterator

"""
    CuIterator(batches)

Return a `CuIterator` that can iterate through the provided `batches` via `Base.iterate`.

Upon each iteration, the current `batch` is adapted to the GPU (via `map(x -> adapt(CuArray, x), batch)`)
and the previous iteration is marked as freeable from GPU memory (via `unsafe_free!`).

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
    isdefined(c, :previous) && foreach(unsafe_free!, c.previous)
    item === nothing && return nothing
    batch, next_state = item
    cubatch = map(x -> adapt(CuArray, x), batch)
    c.previous = cubatch
    return cubatch, next_state
end
