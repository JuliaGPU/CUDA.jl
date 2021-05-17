# scan into a sorted list of free buffers, splitting buffers along the way


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


## block utilities

# split a block at size `sz`, returning the newly created block
function split!(block, sz)
    @assert sz < block.sz "Cannot split a $block at too-large offset $sz"
    split = Block(block.buf, sizeof(block) - sz; off = block.off + sz)
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


## pooling

const SMALL = 1
const LARGE = 2
const HUGE  = 3

# sorted containers need unique keys, which the size of a block isn't.
# mix in the block address to keep the key sortable, but unique.
unique_sizeof(block::Block) = (UInt128(sizeof(block))<<64) | UInt64(pointer(block))
const UniqueIncreasingSize = Base.By(unique_sizeof)

Base.@kwdef struct SplitPool <: AbstractPool
    stream_ordered::Bool

    lock::ReentrantLock = ReentrantLock()

    small::SortedSet{Block} = SortedSet{Block}(UniqueIncreasingSize)
    large::SortedSet{Block} = SortedSet{Block}(UniqueIncreasingSize)
    huge::SortedSet{Block}  = SortedSet{Block}(UniqueIncreasingSize)

    freed::Vector{Block} = Vector{Block}()
    freed_lock::NonReentrantLock = NonReentrantLock()
end

function pool_scan(pool::SplitPool, cache, sz, max_overhead=typemax(Int))
    max_sz = Base.max(sz + max_overhead, max_overhead)   # protect against overflow
    @lock pool.lock begin
        # get the first entry that is sufficiently large
        i = searchsortedfirst(cache, Block(Mem.DeviceBuffer(CU_NULL, 0), sz))
        if i != pastendsemitoken(cache)
            block = deref((cache,i))
            @assert sizeof(block) >= sz
            if sz <= max_sz
                delete!((cache,i))   # FIXME: this allocates
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
function pool_compact(pool::SplitPool, blocks)
    compacted = 0
    @lock pool.lock begin
        while !isempty(blocks)
            block = pop!(blocks)
            @assert block.state == AVAILABLE
            cache = get_cache(pool, sizeof(block))
            @assert in(block, cache)

            # find the head of a sequence
            head = block
            while head.prev !== nothing && head.prev.state == AVAILABLE
                head = head.prev
                @assert in(head, cache)
            end
            szclass = size_class(sizeof(head))

            if head.next !== nothing && head.next.state == AVAILABLE
                delete!(cache, head)

                # find the tail (from the head, removing blocks as we go)
                tail = head.next
                while true
                    @assert szclass === size_class(sizeof(tail)) "block $tail should not have been split to a different pool than $head"
                    delete!(cache, tail)    # FIXME: allocates
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
                @assert !in(head, cache) "$head should not be in the pool"
                @assert head.state == AVAILABLE
                @assert szclass === size_class(sizeof(head)) "compacted $head should not end up in a different pool"
                push!(cache, head)
            end
        end
    end
    return compacted
end

function reclaim_single(pool::SplitPool, cache, sz=typemax(Int))
    freed = 0

    @lock pool.lock begin
        # mark non-split blocks
        candidates = Block[]
        for block in cache
            if iswhole(block)
                push!(candidates, block)
            end
        end

        # free them
        for block in candidates
            delete!(cache, block)
            freed += sizeof(block)
            actual_free(block; pool.stream_ordered)
            freed >= sz && break
        end
    end

    return freed
end

# repopulate the pools from the list of pool.freed blocks
function pool_repopulate(pool::SplitPool)
    blocks = @lock pool.freed_lock begin
        isempty(pool.freed) && return
        blocks = Set(pool.freed)
        empty!(pool.freed)
        blocks
    end

    @lock pool.lock begin
        for block in blocks
            cache = get_cache(pool, sizeof(block))
            @assert !in(block, cache) "$block should not be in the pool"
            @assert block.state == FREED "$block should have been marked pool.freed"
            block.state = AVAILABLE
            push!(cache, block) # FIXME: allocates
        end

        pool_compact(pool, blocks)
    end

    return
end

function size_class(sz)
    if sz <= SMALL_CUTOFF
        SMALL
    elseif SMALL_CUTOFF < sz <= LARGE_CUTOFF
        LARGE
    else
        HUGE
    end
end

@inline function get_cache(pool::SplitPool, sz)
    szclass = size_class(sz)
    if szclass == SMALL
        return pool.small
    elseif szclass == LARGE
        return pool.large
    elseif szclass == HUGE
        return pool.huge
    else
        error("unreachable")
    end
end

function alloc(pool::SplitPool, sz; stream::CuStream)
    szclass = size_class(sz)

    # round off the block size
    req_sz = sz
    roundoff = if szclass == SMALL
        SMALL_ROUNDOFF
    elseif szclass == LARGE
        LARGE_ROUNDOFF
    elseif szclass == HUGE
        HUGE_ROUNDOFF
    else
        error("unreachable")
    end
    sz = cld(sz, roundoff) * roundoff
    szclass = size_class(sz)

    # select a pool
    cache = get_cache(pool, sz)

    # determine the maximum scan overhead
    max_overhead = if szclass == SMALL
        SMALL_OVERHEAD
    elseif szclass == LARGE
        LARGE_OVERHEAD
    elseif szclass == HUGE
        HUGE_OVERHEAD
    else
        error("unreachable")
    end

    block = nothing
    for phase in 1:3
        if phase == 2
            GC.gc(false)
        elseif phase == 3
            GC.gc(true)
        end

        pool_repopulate(pool)

        block = pool_scan(pool, cache, sz, max_overhead)
        block === nothing || break

        block = actual_alloc(sz; pool.stream_ordered)
        block === nothing || break

        # we're out of memory, try freeing up some memory. this is a fairly expensive
        # operation, so start with the largest pool that is likely to free up much memory
        # without requiring many calls to free.
        for cache in (pool.huge, pool.large, pool.small)
            reclaim_single(pool, cache, sz)
            block = actual_alloc(sz; pool.stream_ordered)
            block === nothing || break
        end
        block === nothing || break

        # last-ditch effort, reclaim everything
        reclaim_single(pool, pool.huge)
        reclaim_single(pool, pool.large)
        reclaim_single(pool, pool.small)
        block = actual_alloc(sz, phase==3; pool.stream_ordered)
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
            @lock pool.lock begin
                push!(cache, split)
            end
        end
    end

    if block !== nothing
        block.state = ALLOCATED
    end
    return block
end

function free(pool::SplitPool, block; stream::CuStream)
    # we don't do any work here to reduce pressure on the GC (spending time in finalizers)
    # and to simplify locking (preventing concurrent access during GC interventions)
    block.state == ALLOCATED || error("Cannot free a $(block.state) block")
    block.state = FREED
    @spinlock pool.freed_lock begin
        push!(pool.freed, block)
    end
end

function reclaim(pool::SplitPool, sz::Int=typemax(Int))
    pool_repopulate(pool)

    freed_sz = 0
    for cache in (pool.huge, pool.large, pool.small)
        freed_sz >= sz && break
        freed_sz += reclaim_single(pool, cache, sz-freed_sz)
    end
    return freed_sz
end

function cached_memory(pool::SplitPool)
    sz = @lock pool.freed_lock mapreduce(sizeof, +, pool.freed; init=0)
    @lock pool.lock for cache in (pool.small, pool.large, pool.huge)
        sz += mapreduce(sizeof, +, cache; init=0)
    end
    return sz
end
