module SimplePool

# simple scan into a list of free buffers

using ..CuArrays
using ..CuArrays: @pool_timeit

using CUDAdrv

using Base: @lock
using Base.Threads: SpinLock

const pool_lock = ReentrantLock()


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

struct Block
    ptr::CuPtr{Nothing}
    sz::Int
end

Base.pointer(block::Block) = block.ptr
Base.sizeof(block::Block) = block.sz

@inline function actual_alloc(sz)
    ptr = CuArrays.actual_alloc(sz)
    block = ptr === nothing ? nothing : Block(ptr, sz)
end

function actual_free(block::Block)
    CuArrays.actual_free(pointer(block))
    return
end


## pooling

const available = Set{Block}()
const allocated = Dict{CuPtr{Nothing},Block}()

const freed = Vector{Block}()
const freed_lock = SpinLock()

function scan(sz)
    @lock pool_lock for block in available
        if sz <= sizeof(block) <= max_oversize(sz)
            delete!(available, block)
            return block
        end
    end
    return
end

function repopulate()
    blocks = @lock freed_lock begin
        isempty(freed) && return
        blocks = Set(freed)
        empty!(freed)
        blocks
    end

    @lock pool_lock begin
        for block in blocks
            @assert !in(block, available)
            push!(available, block)
        end
    end

    return blocks
end

function reclaim(sz::Int=typemax(Int))
    repopulate()

    @lock pool_lock begin
        freed_bytes = 0
        while freed_bytes < sz && !isempty(available)
            block = pop!(available)
            freed_bytes += sizeof(block)
            actual_free(block)
        end
        return freed_bytes
    end
end

function pool_alloc(sz)
    block = nothing
    for phase in 1:3
        if phase == 2
            @pool_timeit "$phase.0 gc (incremental)" GC.gc(false)
        elseif phase == 3
            @pool_timeit "$phase.0 gc (full)" GC.gc(true)
        end

        @pool_timeit "$phase.1 repopulate" repopulate()

        @pool_timeit "$phase.2 scan" begin
            block = scan(sz)
        end
        block === nothing || break

        @pool_timeit "$phase.3 alloc" begin
            block = actual_alloc(sz)
        end
        block === nothing || break

        @pool_timeit "$phase.4 reclaim + alloc" begin
            reclaim(sz)
            block = actual_alloc(sz)
        end
        block === nothing || break
    end

    return block
end

function pool_free(block)
    # we don't do any work here to reduce pressure on the GC (spending time in finalizers)
    # and to simplify locking (preventing concurrent access during GC interventions)
    @lock freed_lock begin
        push!(freed, block)
    end
end


## interface

init() = return

function alloc(sz)
    block = pool_alloc(sz)
    if block !== nothing
        ptr = pointer(block)
        @lock pool_lock begin
            allocated[ptr] = block
        end
        return ptr
    else
        return nothing
    end
end

function free(ptr)
    block = @lock pool_lock begin
        block = allocated[ptr]
        delete!(allocated, ptr)
        block
    end
    pool_free(block)
    return
end

used_memory() = @lock pool_lock mapreduce(sizeof, +, values(allocated); init=0)

cached_memory() = @lock pool_lock mapreduce(sizeof, +, union(available, freed); init=0)

end
