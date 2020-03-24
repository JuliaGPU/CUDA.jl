module DummyPool

# dummy allocator that passes through any requests, calling into the GC if that fails.

using ..CuArrays: @pool_timeit, actual_alloc, actual_free

using CUDAdrv

using Base: @lock
using Base.Threads: SpinLock

const allocated_lock = SpinLock()
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
        @lock allocated_lock begin
            allocated[ptr] = sz
        end
        return ptr
    else
        return nothing
    end
end

function free(ptr)
    @lock allocated_lock begin
        sz = allocated[ptr]
        delete!(allocated, ptr)
    end
    actual_free(ptr)
    return
end

reclaim(target_bytes::Int=typemax(Int)) = return 0

used_memory() = @lock allocated_lock mapreduce(sizeof, +, values(allocated); init=0)

cached_memory() = 0

end
