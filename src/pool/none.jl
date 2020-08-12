module DummyPool

# dummy allocator that passes through any requests, calling into the GC if that fails.

using ..CUDA
using ..CUDA: @pool_timeit, @safe_lock, @safe_lock_spin, NonReentrantLock, PerDevice, initialize!, actual_alloc, actual_free

using Base: @lock

const allocated_lock = NonReentrantLock()
const allocated = PerDevice{Dict{CuPtr,Block}}() do dev
    Dict{CuPtr,Block}()
end

function init()
    initialize!(allocated, ndevices())
end

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

    if block !== nothing
        @safe_lock allocated_lock begin
            allocated[dev][ptr] = block
        end
        return pointer(block)
    else
        return nothing
    end
end

function free(ptr, dev=device())
    block = @safe_lock_spin allocated_lock begin
        block = allocated[dev][ptr]
        delete!(allocated[dev], ptr)
        block
    end

    actual_free(dev, block)
    return
end

reclaim(target_bytes::Int=typemax(Int)) = return 0

used_memory(dev=device()) = @safe_lock allocated_lock begin
    mapreduce(sizeof, +, values(allocated[dev]); init=0)
end

cached_memory() = 0

end
