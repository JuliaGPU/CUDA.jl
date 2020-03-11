module DummyPool

# dummy allocator that passes through any requests, calling into the GC if that fails.

using ..CuArrays: @pool_timeit, actual_alloc, actual_free

using CUDAdrv

# use a macro-version of Base.lock to avoid closures
using Base: @lock

const pool_lock = ReentrantLock()

const allocated = Dict{CuPtr{Nothing},Int}()

init() = return

function alloc(sz)
    ptr = nothing
    for phase in 1:3
        if phase == 2
            @pool_timeit "$phase.0 gc (incremental)" GC.gc(false)
        elseif phase == 3
            @pool_timeit "$phase.0 gc (full)" GC.gc(true)
        end

        @pool_timeit "$phase.1 alloc" begin
            ptr = actual_alloc(sz)
        end
        ptr === nothing || break
    end

    if ptr !== nothing
        @lock pool_lock begin
            allocated[ptr] = sz
        end
        return ptr
    else
        return nothing
    end
end

function free(ptr)
    @lock pool_lock begin
        sz = allocated[ptr]
        delete!(allocated, ptr)
    end
    actual_free(ptr)
    return
end

reclaim(target_bytes::Int=typemax(Int)) = return 0

used_memory() = isempty(allocated) ? 0 : @lock pool_lock sum(sizeof, values(allocated))

cached_memory() = 0

end
