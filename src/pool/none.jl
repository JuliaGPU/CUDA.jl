# dummy allocator that passes through any requests, calling into the GC if that fails.

Base.@kwdef struct NoPool <: AbstractPool
    stream_ordered::Bool
end

function alloc(pool::NoPool, sz; stream::CuStream)
    block = nothing
    for phase in 1:4
        if phase == 2
            GC.gc(false)
        elseif phase == 3
            GC.gc(true)
        elseif phase == 4 && pool.stream_ordered
            device_synchronize()
        end

        block = actual_alloc(sz, phase==3; pool.stream_ordered, stream)
        block === nothing || break
    end

    return block
end

function free(pool::NoPool, block; stream::CuStream)
    actual_free(block; pool.stream_ordered, stream)
    return
end

reclaim(pool::NoPool, target_bytes::Int=typemax(Int)) = return 0

cached_memory(pool::NoPool) = 0
