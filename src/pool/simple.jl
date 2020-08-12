module SimplePool

# simple scan into a list of free buffers

using ..CUDA
using ..CUDA: @pool_timeit, @safe_lock, @safe_lock_spin, NonReentrantLock, PerDevice, initialize!, Block

using Base: @lock


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


## block of memory

@inline function actual_alloc(dev, sz)
    ptr = CUDA.actual_alloc(dev, sz)
    block = ptr === nothing ? nothing : Block(ptr, sz)
end

function actual_free(dev, block::Block)
    CUDA.actual_free(dev, pointer(block), sizeof(block))
    return
end


## pooling

const pool_lock = ReentrantLock()
const pool = PerDevice{Set{Block}}() do dev
    Set{Block}()
end

const freed_lock = NonReentrantLock()
const freed = PerDevice{Vector{Block}}() do dev
    Vector{Block}()
end

function scan(dev, sz)
    @lock pool_lock for block in pool[dev]
        if sz <= sizeof(block) <= max_oversize(sz)
            delete!(pool[dev], block)
            return block
        end
    end
    return
end

function repopulate(dev)
    blocks = @lock freed_lock begin
        isempty(freed[dev]) && return
        blocks = Set(freed[dev])
        empty!(freed[dev])
        blocks
    end

    @lock pool_lock begin
        for block in blocks
            @assert !in(block, pool[dev])
            push!(pool[dev], block)
        end
    end

    return
end

function reclaim(sz::Int=typemax(Int), dev=device())
    repopulate(dev)

    @lock pool_lock begin
        freed_bytes = 0
        while freed_bytes < sz && !isempty(pool[dev])
            block = pop!(pool[dev])
            freed_bytes += sizeof(block)
            actual_free(dev, block)
        end
        return freed_bytes
    end
end

function pool_alloc(dev, sz)
    block = nothing
    for phase in 1:3
        if phase == 2
            @pool_timeit "$phase.0 gc (incremental)" GC.gc(false)
        elseif phase == 3
            @pool_timeit "$phase.0 gc (full)" GC.gc(true)
        end

        @pool_timeit "$phase.1 repopulate" repopulate(dev)

        @pool_timeit "$phase.2 scan" begin
            block = scan(dev, sz)
        end
        block === nothing || break

        @pool_timeit "$phase.3 alloc" begin
            block = actual_alloc(dev, sz)
        end
        block === nothing || break

        @pool_timeit "$phase.4 reclaim + alloc" begin
            reclaim(sz, dev)
            block = actual_alloc(dev, sz)
        end
        block === nothing || break
    end

    return block
end

function pool_free(dev, block)
    # we don't do any work here to reduce pressure on the GC (spending time in finalizers)
    # and to simplify locking (preventing concurrent access during GC interventions)
    @safe_lock_spin freed_lock begin
        push!(freed[dev], block)
    end
end


## interface

const pool_allocated_lock = NonReentrantLock()
const allocated = PerDevice{Dict{CuPtr,Block}}() do dev
    Dict{CuPtr,Block}()
end

function init()
    initialize!(pool, ndevices())
    initialize!(freed, ndevices())
    initialize!(allocated, ndevices())
end

function alloc(sz, dev=device())
    block = pool_alloc(dev, sz)
    if block !== nothing
        ptr = pointer(block)
        @safe_lock pool_allocated_lock begin
            allocated[dev][ptr] = block
        end
        return ptr
    else
        return nothing
    end
end

function free(ptr, dev=device())
    block = @safe_lock_spin pool_allocated_lock begin
        block = allocated[dev][ptr]
        delete!(allocated[dev], ptr)
        block
    end
    pool_free(dev, block)
    return
end

function used_memory(dev=device())
    @safe_lock pool_allocated_lock begin
        mapreduce(sizeof, +, values(allocated[dev]); init=0)
    end
end

function cached_memory(dev=device())
    sz = @safe_lock freed_lock mapreduce(sizeof, +, freed[dev]; init=0)
    sz += @lock pool_lock mapreduce(sizeof, +, pool[dev]; init=0)
    return sz
end

end
