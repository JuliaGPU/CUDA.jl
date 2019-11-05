mutable struct CuIterator{B}
    batches::B
    initial_pool_size::Int
    pool::Mem.DeviceBuffer
    function CuIterator(batches, initial_pool_size=0)
        return new{typeof(batches)}(batches, initial_pool_size)
    end
end

function Base.iterate(c::CuIterator, state...)
    item = iterate(c.batches, state...)
    if item === nothing
        isdefined(c, :pool) && Mem.free(c.pool)
        return nothing
    end
    batch, next_state = item
    required_pool_size = sum(sizeof, batch)
    if isempty(state)
        c.initial_pool_size = max(required_pool_size, c.initial_pool_size)
        c.pool = Mem.alloc(Mem.DeviceBuffer, c.initial_pool_size)
    elseif required_pool_size > sizeof(c.pool)
        Mem.free(c.pool)
        c.initial_pool_size = required_pool_size
        c.pool = Mem.alloc(Mem.DeviceBuffer, required_pool_size)
    end
    pool = c.pool
    offset = 0
    cubatch = map(batch) do array
        @assert array isa AbstractArray
        ptr = Base.unsafe_convert(CuPtr{eltype(array)}, pool.ptr + offset)
        cuarray = unsafe_wrap(CuArray, ptr, size(array); own=false);
        copyto!(cuarray, array)
        offset += sizeof(array)
        return cuarray
    end
    return cubatch, next_state
end
