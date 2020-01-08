module SimplePool

# simple scan into a list of free buffers

using ..CuArrays
using ..CuArrays: @pool_timeit

using CUDAdrv


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

function scan(sz)
    for block in available
        if sz <= sizeof(block) <= max_oversize(sz)
            delete!(available, block)
            return block
        end
    end
    return
end

function reclaim(sz::Int=typemax(Int))
    freed = 0
    while freed < sz && !isempty(available)
        block = pop!(available)
        freed += sizeof(block)
        actual_free(block)
    end
    return freed
end

function pool_alloc(sz)
    block = nothing
    for phase in 1:3
        if phase == 2
            @pool_timeit "$phase.0 gc (incremental)" GC.gc(false)
        elseif phase == 3
            @pool_timeit "$phase.0 gc (full)" GC.gc(true)
        end

        @pool_timeit "$phase.1 scan" begin
            block = scan(sz)
        end
        block === nothing || break

        @pool_timeit "$phase.2 alloc" begin
            block = actual_alloc(sz)
        end
        block === nothing || break

        @pool_timeit "$phase.3 reclaim + alloc" begin
            reclaim(sz)
            block = actual_alloc(sz)
        end
        block === nothing || break
    end

    return block
end

function pool_free(block)
    @assert !in(block, available)
    push!(available, block)
end


## interface

init() = return

function alloc(sz)
    block = pool_alloc(sz)
    if block !== nothing
        ptr = pointer(block)
        allocated[ptr] = block
        return ptr
    else
        return nothing
    end
end

function free(ptr)
    block = allocated[ptr]
    delete!(allocated, block)
    pool_free(block)
    return
end

used_memory() = isempty(allocated) ? 0 : sum(sizeof, values(allocated))

cached_memory() = isempty(available) ? 0 : sum(sizeof, available)

dump() = return

end
