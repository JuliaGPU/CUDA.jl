# dummy allocator that passes through any requests, calling into the GC if that fails.

Base.@kwdef struct NoPool <: AbstractPool
    device::CuDevice
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

function reclaim(pool::NoPool, target_bytes::Int=typemax(Int))
    if pool.stream_ordered
        # TODO: respect target_bytes
        device_synchronize()
        trim(memory_pool(pool.device))
    else
        return 0
    end
end

cached_memory(pool::NoPool) = 0
