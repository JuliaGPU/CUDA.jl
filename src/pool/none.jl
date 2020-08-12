module DummyPool

# dummy allocator that passes through any requests, calling into the GC if that fails.

using ..CUDA
using ..CUDA: @pool_timeit, @safe_lock, @safe_lock_spin, NonReentrantLock, PerDevice, initialize!, actual_alloc, actual_free

using Base: @lock

init() = return

function alloc(sz, dev=device())
    block = nothing
    for phase in 1:3
        if phase == 2
            @pool_timeit "$phase.0 gc (incremental)" GC.gc(false)
        elseif phase == 3
            @pool_timeit "$phase.0 gc (full)" GC.gc(true)
        end

        @pool_timeit "$phase.1 alloc" begin
            block = actual_alloc(dev, sz)
        end
        block === nothing || break
    end

    return block
end

function free(block, dev=device())
    actual_free(dev, block)
    return
end

reclaim(target_bytes::Int=typemax(Int)) = return 0
cached_memory() = 0

end
