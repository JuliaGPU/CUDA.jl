module SplittingPool

# scan into a sorted list of free buffers, splitting buffers along the way

using ..CUDA
using ..CUDA: @pool_timeit, @safe_lock, @safe_lock_spin, NonReentrantLock

using DataStructures

using Base: @lock


## tunables

# pool boundaries
const SMALL_CUTOFF   = 2^20    # 1 MiB
const LARGE_CUTOFF   = 2^25    # 32 MiB

# memory size rounding
const SMALL_ROUNDOFF = 2^9     # 512 bytes
const LARGE_ROUNDOFF = 2^17    # 128 KiB
const HUGE_ROUNDOFF  = 2^20    # 1 MiB

# maximum overhead (unused space) when selecting a buffer
# small and large buffers with large overhead will be split, but huge buffers never are
const SMALL_OVERHEAD = typemax(Int)
const LARGE_OVERHEAD = typemax(Int)
const HUGE_OVERHEAD  = 0


## block of memory

using Printf

@enum BlockState begin
    INVALID
    AVAILABLE
    ALLOCATED
    FREED
end

const block_id = Threads.Atomic{UInt}(1)

# TODO: it would be nice if this could be immutable, since that's what SortedSet requires
mutable struct Block
    ptr::Union{Nothing,CuPtr{Nothing}}  # base allocation
    sz::Int                             # size into it
    off::Int                            # offset into it

    state::BlockState
    prev::Union{Nothing,Block}
    next::Union{Nothing,Block}

    id::UInt

    Block(ptr, sz; off=0, state=INVALID, prev=nothing, next=nothing,
          id=Threads.atomic_add!(block_id, UInt(1))) =
        new(ptr, sz, off, state, prev, next, id)
end

Base.sizeof(block::Block) = block.sz
Base.pointer(block::Block) = block.ptr + block.off

iswhole(block::Block) = block.prev === nothing && block.next === nothing


## block utilities

function Base.show(io::IO, block::Block)
    fields = [@sprintf("#%d", block.id)]
    push!(fields, @sprintf("%s at %p", Base.format_bytes(sizeof(block)), pointer(block)))
    push!(fields, "$(block.state)")
    block.prev !== nothing && push!(fields, @sprintf("prev=Block(#%d)", block.prev.id))
    block.next !== nothing && push!(fields, @sprintf("next=Block(#%d)", block.next.id))

    print(io, "Block(", join(fields, ", "), ")")
end

# split a block at size `sz`, returning the newly created block
function split!(block, sz)
    @assert sz < block.sz "Cannot split a $block at too-large offset $sz"
    split = Block(block.ptr, sizeof(block) - sz; off = block.off + sz)
    block.sz = sz

    # update links
    split.prev = block
    split.next = block.next
    if block.next !== nothing
        block.next.prev = split
    end
    block.next = split

    return split
end

# merge a sequence of blocks that starts at `head` and ends with `tail` (inclusive)
function merge!(head, tail)
    head.next = tail.next
    if tail.next !== nothing
        tail.next.prev = head
    end
    head.sz = tail.sz + (tail.off - head.off)

    return head
end

@inline function actual_alloc(sz)
    ptr = CUDA.actual_alloc(sz)
    block = ptr === nothing ? nothing : Block(ptr, sz)
end

function actual_free(block::Block)
    @assert iswhole(block) "Cannot free $block: block is not whole"
    if block.state != AVAILABLE
        error("Cannot free $block: block is not available")
    else
        CUDA.actual_free(block.ptr)
        block.state = INVALID
    end
    return
end


## pooling

const pool_lock = ReentrantLock()

function scan!(pool, sz, max_overhead=typemax(Int))
    max_sz = Base.max(sz + max_overhead, max_overhead)   # protect against overflow
    @lock pool_lock begin
        # get the first entry that is sufficiently large
        i = searchsortedfirst(pool, Block(nothing, sz; id=0))
        if i != pastendsemitoken(pool)
            block = deref((pool,i))
            @assert sizeof(block) >= sz
            if sz <= max_sz
                delete!((pool,i))   # FIXME: this allocates
                return block
            end
        end

        return nothing
    end
end

# compact sequences of blocks into larger ones.
# looks up possible sequences based on each block in the input set.
# destroys the input set.
# returns the net difference in amount of blocks.
function incremental_compact!(blocks)
    compacted = 0
    @lock pool_lock begin
        while !isempty(blocks)
            block = pop!(blocks)
            @assert block.state == AVAILABLE
            pool = get_pool(sizeof(block))
            @assert in(block, pool)

            # find the head of a sequence
            head = block
            while head.prev !== nothing && head.prev.state == AVAILABLE
                head = head.prev
                @assert in(head, pool)
            end
            szclass = size_class(sizeof(head))

            if head.next !== nothing && head.next.state == AVAILABLE
                delete!(pool, head)

                # find the tail (from the head, removing blocks as we go)
                tail = head.next
                while true
                    @assert szclass === size_class(sizeof(tail)) "block $tail should not have been split to a different pool than $head"
                    delete!(pool, tail)    # FIXME: allocates
                    delete!(blocks, tail)
                    tail.state = INVALID
                    compacted += 1
                    if tail.next !== nothing && tail.next.state == AVAILABLE
                        tail = tail.next
                    else
                        break
                    end
                end

                # compact
                head = merge!(head, tail)
                @assert !in(head, pool) "$head should not be in the pool"
                @assert head.state == AVAILABLE
                @assert szclass === size_class(sizeof(head)) "compacted $head should not end up in a different pool"
                push!(pool, head)
            end
        end
    end
    return compacted
end

function reclaim!(pool, sz=typemax(Int))
    freed = 0

    @lock pool_lock begin
        # mark non-split blocks
        candidates = Block[]
        for block in pool
            if iswhole(block)
                push!(candidates, block)
            end
        end

        # free them
        for block in candidates
            delete!(pool, block)
            freed += sizeof(block)
            actual_free(block)
            freed >= sz && break
        end
    end

    return freed
end

# repopulate the pools from the list of freed blocks
function repopulate()
    blocks = @safe_lock freed_lock begin
        isempty(freed) && return
        blocks = Set(freed)
        empty!(freed)
        blocks
    end

    @lock pool_lock begin
        for block in blocks
            pool = get_pool(sizeof(block))
            @assert !in(block, pool) "$block should not be in the pool"
            @assert block.state == FREED "$block should have been marked freed"
            block.state = AVAILABLE
            push!(pool, block) # FIXME: allocates
        end

        incremental_compact!(blocks)
    end

    return
end

const SMALL = 1
const LARGE = 2
const HUGE  = 3

# sorted containers need unique keys, which the size of a block isn't.
# mix in the block address to keep the key sortable, but unique.
unique_sizeof(block::Block) = (UInt128(sizeof(block))<<64) | UInt64(block.id)
const UniqueIncreasingSize = Base.By(unique_sizeof)

const pool_small = SortedSet{Block}(UniqueIncreasingSize)
const pool_large = SortedSet{Block}(UniqueIncreasingSize)
const pool_huge  = SortedSet{Block}(UniqueIncreasingSize)

const freed = Vector{Block}()
const freed_lock = NonReentrantLock()

function size_class(sz)
    if sz <= SMALL_CUTOFF
        SMALL
    elseif SMALL_CUTOFF < sz <= LARGE_CUTOFF
        LARGE
    else
        HUGE
    end
end

@inline function get_pool(sz)
    szclass = size_class(sz)
    if szclass == SMALL
        return pool_small
    elseif szclass == LARGE
        return pool_large
    elseif szclass == HUGE
        return pool_huge
    end
end

function pool_alloc(sz)
    szclass = size_class(sz)

    # round off the block size
    req_sz = sz
    roundoff = if szclass == SMALL
        SMALL_ROUNDOFF
    elseif szclass == LARGE
        LARGE_ROUNDOFF
    elseif szclass == HUGE
        HUGE_ROUNDOFF
    end
    sz = cld(sz, roundoff) * roundoff
    szclass = size_class(sz)

    # select a pool
    pool = get_pool(sz)

    # determine the maximum scan overhead
    max_overhead = if szclass == SMALL
        SMALL_OVERHEAD
    elseif szclass == LARGE
        LARGE_OVERHEAD
    elseif szclass == HUGE
        HUGE_OVERHEAD
    end

    block = nothing
    for phase in 1:3
        if phase == 2
            @pool_timeit "$phase.0 gc (incremental)" GC.gc(false)
        elseif phase == 3
            @pool_timeit "$phase.0 gc (full)" GC.gc(true)
        end

        @pool_timeit "$phase.1 repopulate" repopulate()

        @pool_timeit "$phase.2 scan" begin
            block = scan!(pool, sz, max_overhead)
        end
        block === nothing || break

        @pool_timeit "$phase.3 alloc" begin
            block = actual_alloc(sz)
        end
        block === nothing || break

        # we're out of memory, try freeing up some memory. this is a fairly expensive
        # operation, so start with the largest pool that is likely to free up much memory
        # without requiring many calls to free.
        for pool in (pool_huge, pool_large, pool_small)
            @pool_timeit "$phase.4a reclaim" reclaim!(pool, sz)
            @pool_timeit "$phase.4b alloc" block = actual_alloc(sz)
            block === nothing || break
        end
        block === nothing || break

        # last-ditch effort, reclaim everything
        @pool_timeit "$phase.5a reclaim" begin
            reclaim!(pool_huge)
            reclaim!(pool_large)
            reclaim!(pool_small)
        end
        @pool_timeit "$phase.5b alloc" block = actual_alloc(sz)
    end

    if block !== nothing
        @assert size_class(sizeof(block)) == szclass "Cannot satisfy an allocation in pool $szclass with a buffer in pool $(size_class(sizeof(block)))"

        # split the block if that would yield one from the same pool
        # (i.e. don't split off chunks that would be way smaller)
        # TODO: avoid creating unaligned blocks here (doesn't currently seem to happen
        #       because of the roundoff, but we should take care anyway)
        remainder = sizeof(block) - sz
        if szclass != HUGE && remainder > 0 && size_class(remainder) == szclass
            split = split!(block, sz)
            split.state = AVAILABLE
            @lock pool_lock begin
                push!(pool, split)
            end
        end
    end

    return block
end

function pool_free(block)
    # we don't do any work here to reduce pressure on the GC (spending time in finalizers)
    # and to simplify locking (preventing concurrent access during GC interventions)
    block.state = FREED
    @safe_lock_spin freed_lock begin
        push!(freed, block)
    end
end


## interface

const allocated_lock = NonReentrantLock()
const allocated = Dict{CuPtr{Nothing},Block}()

init() = return

function alloc(sz)
    block = pool_alloc(sz)
    if block !== nothing
        block.state = ALLOCATED
        ptr = pointer(block)
        @safe_lock allocated_lock begin
            @assert !haskey(allocated, ptr) "Newly-allocated block $block is already allocated"
            allocated[ptr] = block
        end
        return ptr
    else
        return nothing
    end
end

function free(ptr)
    block = @safe_lock_spin allocated_lock begin
        block = allocated[ptr]
        delete!(allocated, ptr)
        block
    end
    block.state == ALLOCATED || error("Cannot free a $(block.state) block")
    pool_free(block)
    return
end

function reclaim(sz::Int=typemax(Int))
    repopulate()

    freed_sz = 0
    for pool in (pool_huge, pool_large, pool_small)
        freed_sz >= sz && break
        freed_sz += reclaim!(pool, sz-freed_sz)
    end
    return freed_sz
end

used_memory() = @safe_lock allocated_lock mapreduce(sizeof, +, values(allocated); init=0)

function cached_memory()
    sz = @safe_lock freed_lock mapreduce(sizeof, +, freed; init=0)
    @lock pool_lock for pool in (pool_small, pool_large, pool_huge)
        sz += mapreduce(sizeof, +, pool; init=0)
    end
    return sz
end

end
