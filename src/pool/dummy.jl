module DummyPool

# dummy allocator that passes through any requests, calling into the GC if that fails.

using ..CUDA
using ..CUDA: @pool_timeit, @safe_lock, @safe_lock_spin, NonReentrantLock

using Base: @lock

const allocated_lock = NonReentrantLock()
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
            ptr = CUDA.actual_alloc(sz)
        end
        ptr === nothing || break
    end

    if ptr !== nothing
        @safe_lock allocated_lock begin
            allocated[ptr] = sz
        end
        return ptr
    else
        return nothing
    end
end

function free(ptr)
    @safe_lock_spin allocated_lock begin
        sz = allocated[ptr]
        delete!(allocated, ptr)
    end
    CUDA.actual_free(ptr)
    return
end

reclaim(target_bytes::Int=typemax(Int)) = return 0

used_memory() = @safe_lock allocated_lock mapreduce(sizeof, +, values(allocated); init=0)

cached_memory() = 0

end
