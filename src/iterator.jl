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
