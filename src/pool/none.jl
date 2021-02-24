module NoPool

# dummy allocator that passes through any requests, calling into the GC if that fails.

using ..CUDA
import ..CUDA: actual_alloc, actual_free

using ..PoolUtils

init() = return

function alloc(dev, sz)
    block = nothing
    for phase in 1:3
        if phase == 2
            @pool_timeit "$phase.0 gc (incremental)" GC.gc(false)
        elseif phase == 3
            @pool_timeit "$phase.0 gc (full)" GC.gc(true)
        end

        @pool_timeit "$phase.1 alloc" begin
            block = actual_alloc(dev, sz, phase==3)
        end
        block === nothing || break
    end

    return block
end

function free(dev, block)
    actual_free(dev, block)
    return
end

reclaim(dev, target_bytes::Int=typemax(Int)) = return 0

cached_memory(dev=device()) = 0

end
