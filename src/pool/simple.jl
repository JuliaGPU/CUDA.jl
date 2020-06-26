module SimplePool

# simple scan into a list of free buffers

using ..CUDA
using ..CUDA: @pool_timeit, @safe_lock, @safe_lock_spin, NonReentrantLock, isvalid

using DataStructures

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

struct Block
    ptr::CuPtr{Nothing}
    sz::Int
end

Base.pointer(block::Block) = block.ptr
Base.sizeof(block::Block) = block.sz

@inline function actual_alloc(ctx, sz)
    @assert isvalid(ctx)
    ptr = context!(ctx) do
        CUDA.actual_alloc(sz)
    end
    block = ptr === nothing ? nothing : Block(ptr, sz)
end

function actual_free(ctx, block::Block)
    isvalid(ctx) || return
    context!(ctx) do
        CUDA.actual_free(pointer(block))
    end
    return
end


## pooling

const pool_lock = ReentrantLock()
const pool = DefaultDict{CuContext,Set{Block}}(()->Set{Block}())

const freed = DefaultDict{CuContext,Vector{Block}}(()->Vector{Block}())
const freed_lock = NonReentrantLock()

function scan(ctx, sz)
    @lock pool_lock for block in pool[ctx]
        if sz <= sizeof(block) <= max_oversize(sz)
            delete!(pool[ctx], block)
            return block
        end
    end
    return
end

function repopulate(ctx)
    blocks = @lock freed_lock begin
        isempty(freed[ctx]) && return
        blocks = Set(freed[ctx])
        empty!(freed[ctx])
        blocks
    end

    @lock pool_lock begin
        for block in blocks
            @assert !in(block, pool[ctx])
            push!(pool[ctx], block)
        end
    end

    return
end

function reclaim(sz::Int=typemax(Int), ctx=context())
    repopulate(ctx)

    @lock pool_lock begin
        freed_bytes = 0
        while freed_bytes < sz && !isempty(pool[ctx])
            block = pop!(pool[ctx])
            freed_bytes += sizeof(block)
            actual_free(ctx, block)
        end
        return freed_bytes
    end
end

function pool_alloc(ctx, sz)
    block = nothing
    for phase in 1:3
        if phase == 2
            @pool_timeit "$phase.0 gc (incremental)" GC.gc(false)
        elseif phase == 3
            @pool_timeit "$phase.0 gc (full)" GC.gc(true)
        end

        @pool_timeit "$phase.1 repopulate" repopulate(ctx)

        @pool_timeit "$phase.2 scan" begin
            block = scan(ctx, sz)
        end
        block === nothing || break

        @pool_timeit "$phase.3 alloc" begin
            block = actual_alloc(ctx, sz)
        end
        block === nothing || break

        @pool_timeit "$phase.4 reclaim + alloc" begin
            reclaim(sz, ctx)
            block = actual_alloc(ctx, sz)
        end
        block === nothing || break
    end

    return block
end

function pool_free(ctx, block)
    # we don't do any work here to reduce pressure on the GC (spending time in finalizers)
    # and to simplify locking (preventing concurrent access during GC interventions)
    @safe_lock_spin freed_lock begin
        push!(freed[ctx], block)
    end
end


## interface

const allocated_lock = NonReentrantLock()
const allocated = Dict{CuPtr{Nothing},Tuple{CuContext,Block}}()

init() = return

function alloc(sz, ctx=context())
    block = pool_alloc(ctx, sz)
    if block !== nothing
        ptr = pointer(block)
        @safe_lock allocated_lock begin
            allocated[ptr] = (ctx,block)
        end
        return ptr
    else
        return nothing
    end
end

function free(ptr)
    ctx, block = @safe_lock_spin allocated_lock begin
        ctx, block = allocated[ptr]
        delete!(allocated, ptr)
        ctx, block
    end
    pool_free(ctx, block)
    return
end

used_memory(ctx=context()) = @safe_lock allocated_lock mapreduce(sizeof, +, values(allocated[ctx]); init=0)

function cached_memory(ctx=context())
    sz = @safe_lock freed_lock mapreduce(sizeof, +, freed[ctx]; init=0)
    sz += @lock pool_lock mapreduce(sizeof, +, pool[ctx]; init=0)
    return sz
end

end
