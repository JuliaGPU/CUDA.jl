struct CuIterator{B,P}
    batches::B
    pool::Vector{P}
    pool_limit::Int
end

CuIterator(batches, pool=Any[]) = CuIterator(batches, pool, 10)

pool_key(x) = eltype(x) => size(x)

function Base.iterate(c::CuIterator, state...)
    item = iterate(c.batches, state...)
    if item === nothing
        map(batch -> map(unsafe_free!, batch), c.pool)
        empty!(c.pool)
        return nothing
    end
    batch, next_state = item
    i = findfirst(allocated -> pool_key.(allocated) == pool_key.(batch), c.pool)
    if i === nothing
        cubatch = map(x -> adapt(CuArray, x), batch)
        push!(c.pool, cubatch)
    else
        cubatch = map(copyto!, c.pool[i], batch)
    end
    length(c.pool) > c.pool_limit && foreach(unsafe_free!, popfirst!(c.pool))
    return cubatch, next_state
end
