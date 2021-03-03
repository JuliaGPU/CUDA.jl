# dummy allocator that passes through any requests, calling into the GC if that fails.

using .PoolUtils

Base.@kwdef struct NoPool <: AbstractPool
    dev::CuDevice
    stream_ordered::Bool
end

function alloc(pool::NoPool, sz)
    block = nothing
    for phase in 1:3
        if phase == 2
            @pool_timeit "$phase.0 gc (incremental)" GC.gc(false)
        elseif phase == 3
            @pool_timeit "$phase.0 gc (full)" GC.gc(true)
        end

        @pool_timeit "$phase.1 alloc" begin
            block = actual_alloc(pool.dev, sz, phase==3; pool.stream_ordered)
        end
        block === nothing || break
    end

    return block
end

function free(pool::NoPool, block)
    actual_free(pool.dev, block; pool.stream_ordered)
    return
end

reclaim(pool::NoPool, target_bytes::Int=typemax(Int)) = return 0

cached_memory(pool::NoPool) = 0
