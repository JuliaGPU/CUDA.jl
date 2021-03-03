# simple scan into a list of free buffers

using .PoolUtils


## tunables

# how much larger a buf can be to fullfil an allocation request.
# lower values improve efficiency, but increase pressure on the underlying GC.
function max_oversize(sz)
    if sz <= 2^20       # 1 MiB
        # small buffers are fine no matter
        return typemax(Int)
    elseif sz <= 2^20   # 32 MiB
        return 2^20
    else
        return 2^22
    end
end


## pooling

Base.@kwdef struct SimplePool <: AbstractPool
    dev::CuDevice
    stream_ordered::Bool

    lock::ReentrantLock = ReentrantLock()
    cache::Set{Block} = Set{Block}()

    freed_lock::NonReentrantLock = NonReentrantLock()
    freed::Vector{Block} = Vector{Block}()
end

function scan(pool::SimplePool, sz)
    @lock pool.lock for block in pool.cache
        if sz <= sizeof(block) <= max_oversize(sz)
            delete!(pool.cache, block)
            return block
        end
    end
    return
end

function repopulate(pool::SimplePool)
    blocks = @lock pool.freed_lock begin
        isempty(pool.freed) && return
        blocks = Set(pool.freed)
        empty!(pool.freed)
        blocks
    end

    @lock pool.lock begin
        for block in blocks
            @assert !in(block, pool.cache)
            push!(pool.cache, block)
        end
    end

    return
end

function reclaim(pool::SimplePool, sz::Int=typemax(Int))
    repopulate(pool)

    @lock pool.lock begin
        freed_bytes = 0
        while freed_bytes < sz && !isempty(pool.cache)
            block = pop!(pool.cache)
            freed_bytes += sizeof(block)
            actual_free(pool.dev, block; pool.stream_ordered)
        end
        return freed_bytes
    end
end

function alloc(pool::SimplePool, sz)
    block = nothing
    for phase in 1:3
        if phase == 2
            @pool_timeit "$phase.0 gc (incremental)" GC.gc(false)
        elseif phase == 3
            @pool_timeit "$phase.0 gc (full)" GC.gc(true)
        end

        @pool_timeit "$phase.1 repopulate" repopulate(pool)

        @pool_timeit "$phase.2 scan" begin
            block = scan(pool, sz)
        end
        block === nothing || break

        @pool_timeit "$phase.3 alloc" begin
            block = actual_alloc(pool.dev, sz; pool.stream_ordered)
        end
        block === nothing || break

        @pool_timeit "$phase.4 reclaim + alloc" begin
            reclaim(pool, sz)
            block = actual_alloc(pool.dev, sz, phase==3; pool.stream_ordered)
        end
        block === nothing || break
    end

    return block
end

function free(pool::SimplePool, block)
    # we don't do any work here to reduce pressure on the GC (spending time in finalizers)
    # and to simplify locking (preventing concurrent access during GC interventions)
    @spinlock pool.freed_lock begin
        push!(pool.freed, block)
    end
end

function cached_memory(pool::SimplePool)
    sz = @lock pool.freed_lock mapreduce(sizeof, +, pool.freed; init=0)
    sz += @lock pool.lock mapreduce(sizeof, +, pool.cache; init=0)
    return sz
end
