# scan into a sorted list of free buffers, splitting buffers along the way

using DataStructures


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

const pool_lock = ReentrantLock()

function pool_scan(dev, pool, sz, max_overhead=typemax(Int))
    max_sz = Base.max(sz + max_overhead, max_overhead)   # protect against overflow
    @lock pool_lock begin
        # get the first entry that is sufficiently large
        i = searchsortedfirst(pool, Block(Mem.DeviceBuffer(CU_NULL, 0), sz))
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
function pool_compact(dev, blocks)
    compacted = 0
    @lock pool_lock begin
        while !isempty(blocks)
            block = pop!(blocks)
            @assert block.state == AVAILABLE
            pool = get_pool(dev, sizeof(block))
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

function pool_reclaim_single(dev, pool, sz=typemax(Int))
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
            actual_free(dev, block)
            freed >= sz && break
        end
    end

    return freed
end

# repopulate the pools from the list of freed blocks
function pool_repopulate(dev)
    blocks = @lock freed_lock begin
        isempty(freed[dev]) && return
        blocks = Set(freed[dev])
        empty!(freed[dev])
        blocks
    end

    @lock pool_lock begin
        for block in blocks
            pool = get_pool(dev, sizeof(block))
            @assert !in(block, pool) "$block should not be in the pool"
            @assert block.state == FREED "$block should have been marked freed"
            block.state = AVAILABLE
            push!(pool, block) # FIXME: allocates
        end

        pool_compact(dev, blocks)
    end

    return
end

const SMALL = 1
const LARGE = 2
const HUGE  = 3

# sorted containers need unique keys, which the size of a block isn't.
# mix in the block address to keep the key sortable, but unique.
unique_sizeof(block::Block) = (UInt128(sizeof(block))<<64) | UInt64(pointer(block))
const UniqueIncreasingSize = Base.By(unique_sizeof)

const pool_small = PerDevice{SortedSet{Block}}((dev)->SortedSet{Block}(UniqueIncreasingSize))
const pool_large = PerDevice{SortedSet{Block}}((dev)->SortedSet{Block}(UniqueIncreasingSize))
const pool_huge  = PerDevice{SortedSet{Block}}((dev)->SortedSet{Block}(UniqueIncreasingSize))

const freed = PerDevice{Vector{Block}}((dev)->Vector{Block}())
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

@inline function get_pool(dev, sz)
    szclass = size_class(sz)
    if szclass == SMALL
        return pool_small[dev]
    elseif szclass == LARGE
        return pool_large[dev]
    elseif szclass == HUGE
        return pool_huge[dev]
    end
end

function pool_alloc(dev, sz)
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
    pool = get_pool(dev, sz)

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

        @pool_timeit "$phase.1 repopulate" pool_repopulate(dev)

        @pool_timeit "$phase.2 scan" begin
            block = pool_scan(dev, pool, sz, max_overhead)
        end
        block === nothing || break

        @pool_timeit "$phase.3 alloc" begin
            block = actual_alloc(dev, sz)
        end
        block === nothing || break

        # we're out of memory, try freeing up some memory. this is a fairly expensive
        # operation, so start with the largest pool that is likely to free up much memory
        # without requiring many calls to free.
        for pool in (pool_huge[dev], pool_large[dev], pool_small[dev])
            @pool_timeit "$phase.4a reclaim" pool_reclaim_single(dev, pool, sz)
            @pool_timeit "$phase.4b alloc" block = actual_alloc(dev, sz)
            block === nothing || break
        end
        block === nothing || break

        # last-ditch effort, reclaim everything
        @pool_timeit "$phase.5a reclaim" begin
            pool_reclaim_single(dev, pool_huge[dev])
            pool_reclaim_single(dev, pool_large[dev])
            pool_reclaim_single(dev, pool_small[dev])
        end
        @pool_timeit "$phase.5b alloc" block = actual_alloc(dev, sz)
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

    if block !== nothing
        block.state = ALLOCATED
    end
    return block
end

function pool_free(dev, block)
    # we don't do any work here to reduce pressure on the GC (spending time in finalizers)
    # and to simplify locking (preventing concurrent access during GC interventions)
    block.state == ALLOCATED || error("Cannot free a $(block.state) block")
    block.state = FREED
    @spinlock freed_lock begin
        push!(freed[dev], block)
    end
end

function pool_init()
    initialize!(freed, ndevices())

    initialize!(pool_small, ndevices())
    initialize!(pool_large, ndevices())
    initialize!(pool_huge, ndevices())
end

function pool_reclaim(dev, sz::Int=typemax(Int))
    pool_repopulate(dev)

    freed_sz = 0
    for pool in (pool_huge[dev], pool_large[dev], pool_small[dev])
        freed_sz >= sz && break
        freed_sz += pool_reclaim_single(dev, pool, sz-freed_sz)
    end
    return freed_sz
end

function cached_memory(dev=device())
    sz = @lock freed_lock mapreduce(sizeof, +, freed[dev]; init=0)
    @lock pool_lock for pool in (pool_small[dev], pool_large[dev], pool_huge[dev])
        sz += mapreduce(sizeof, +, pool; init=0)
    end
    return sz
end
