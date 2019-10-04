module SplittingPool

# scan into a sorted list of free buffers, splitting buffers along the way

using ..CuArrays
using ..CuArrays: @pool_timeit

using DataStructures

using CUDAdrv

# use a macro-version of Base.lock to avoid closures
if VERSION >= v"1.3.0-DEV.555"
    using Base: @lock
else
    macro lock(l, expr)
        quote
            temp = $(esc(l))
            lock(temp)
            try
                $(esc(expr))
            finally
                unlock(temp)
            end
        end
    end
end


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

const block_id = Ref(UInt(0))

# TODO: it would be nice if this could be immutable, since that's what SortedSet requires
mutable struct Block
    ptr::Union{Nothing,CuPtr{Nothing}}  # base allocation
    sz::Int                             # size into it
    off::Int                            # offset into it

    state::BlockState
    prev::Union{Nothing,Block}
    next::Union{Nothing,Block}

    id::UInt

    Block(ptr, sz; off=0, state=INVALID,
          prev=nothing, next=nothing, id=block_id[]+=1) =
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
    ptr = CuArrays.actual_alloc(sz)
    block = ptr === nothing ? nothing : Block(ptr, sz)
end

function actual_free(block::Block)
    @assert iswhole(block) "Cannot free $block: block is not whole"
    if block.state != AVAILABLE
        error("Cannot free $block: block is not available")
    else
        CuArrays.actual_free(block.ptr)
        block.state = INVALID
    end
    return
end


## pooling

using Base.Threads: SpinLock
const pool_lock = SpinLock()   # protect against deletion from freelists

const scan_lower_bound = Block(nothing, 0; id=0)
function scan!(blocks, sz, max_overhead=typemax(Int))
    max_sz = max(sz + max_overhead, max_overhead)   # protect against overflow
    @lock pool_lock begin
        # semantically, the following code iterates and selects a block:
        #   for block in blocks
        #   if sz <= sizeof(block) <= max_sz
        #       delete!(blocks, block)
        #       return block
        #   end
        #   return nothing
        # but since we know the sorted set is backed by a balanced tree, we can do better

        # get the entry right before first sufficiently large one
        scan_lower_bound.sz = sz    # prevent allocations
        i, exact = findkey(blocks.bt, scan_lower_bound)
        @assert !exact  # block id bits are zero, so this match can't be exact

        if i == DataStructures.endloc(blocks.bt)
            # last entry, none is following
            return nothing
        else
            # a valid entry, make sure it isn't too large
            i = DataStructures.nextloc0(blocks.bt, i)
            block = blocks.bt.data[i].k
            @assert sz <= sizeof(block)
            if sz > max_sz
                return nothing
            else
                delete!(blocks.bt, i)   # FIXME: this allocates
                return block
            end
        end
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
            available = get_available(sizeof(block))

            # find the head of a sequence
            head = block
            while head.prev !== nothing && head.prev.state == AVAILABLE
                head = head.prev
            end

            if head.next !== nothing && head.next.state == AVAILABLE
                delete!(available, head)

                # find the tail (from the head, removing blocks as we go)
                tail = head.next
                while true
                    delete!(available, tail)    # FIXME: allocates
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
                @assert !in(head, available) "$head should not be in the pool"
                @assert head.state == AVAILABLE
                push!(available, head)
            end
        end
    end
    return compacted
end

function reclaim!(blocks, sz=typemax(Int))
    freed = 0
    candidates = Block[]

    @lock pool_lock begin
        # mark non-split blocks
        for block in blocks
            if iswhole(block)
                push!(candidates, block)
            end
        end

        # free them
        for block in candidates
            delete!(blocks, block)
            freed += sizeof(block)
            actual_free(block)
            freed >= sz && break
        end
    end

    return freed
end

# repopulate the "available" pools from a list of freed blocks
function repopulate(blocks)
    @lock pool_lock begin
        for block in blocks
            available = get_available(sizeof(block))
            @assert !in(block, available) "$block should not be in the pool"
            @assert block.state == FREED "$block should have been marked freed"
            block.state = AVAILABLE
            push!(available, block) # FIXME: allocates
        end
    end
end

const SMALL = 1
const LARGE = 2
const HUGE  = 3

# sorted containers need unique keys, which the size of a block isn't.
# mix in the block address to keep the key sortable, but unique.
# the size is shifted 24 bits, and as many identifier bits are
# mixed in, supporting 16777216 unique allocations of up to 1 TiB.
unique_sizeof(block::Block) = (UInt64(sizeof(block))<<24) | (UInt64(block.id) & (2<<24-1))
const UniqueIncreasingSize = Base.By(unique_sizeof)

const available_small = SortedSet{Block}(UniqueIncreasingSize)
const available_large = SortedSet{Block}(UniqueIncreasingSize)
const available_huge  = SortedSet{Block}(UniqueIncreasingSize)
const allocated = Dict{CuPtr{Nothing},Block}()
const freed = Vector{Block}()

function size_class(sz)
    if sz <= SMALL_CUTOFF
        SMALL
    elseif SMALL_CUTOFF < sz <= LARGE_CUTOFF
        LARGE
    else
        HUGE
    end
end

@inline function get_available(sz)
    szclass = size_class(sz)
    if szclass == SMALL
        return available_small
    elseif szclass == LARGE
        return available_large
    elseif szclass == HUGE
        return available_huge
    end
end

function pool_alloc(sz)
    szclass = size_class(sz)

    # round of the block size
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
    available = get_available(sz)

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
            @pool_timeit "$phase.0 gc (incremental)" GC.gc(VERSION >= v"1.4.0-DEV.257" ? GC.Incremental : false)
        elseif phase == 3
            @pool_timeit "$phase.0 gc (full)" GC.gc(VERSION >= v"1.4.0-DEV.257" ? GC.Full : true)
        end

        if !isempty(freed)
            # `freed` may be modified concurrently, so take a copy
            blocks = Set(freed)
            empty!(freed)
            @pool_timeit "$phase.1a repopulate" repopulate(blocks)
            @pool_timeit "$phase.1b compact" incremental_compact!(blocks)
        end

        @pool_timeit "$phase.2 scan" begin
            block = scan!(available, sz, max_overhead)
        end
        block === nothing || break

        @pool_timeit "$phase.3 alloc" begin
            block = actual_alloc(sz)
        end
        block === nothing || break

        # we're out of memory, try freeing up some memory. this is a fairly expensive
        # operation, so start with the largest pool that is likely to free up much memory
        # without requiring many calls to free.
        for available in (available_huge, available_large, available_small)
            @pool_timeit "$phase.4a reclaim" reclaim!(available, sz)
            @pool_timeit "$phase.4b alloc" block = actual_alloc(sz)
            block === nothing || break
        end
        block === nothing || break

        # last-ditch effort, reclaim everything
        @pool_timeit "$phase.5a reclaim" begin
            reclaim!(available_huge)
            reclaim!(available_large)
            reclaim!(available_small)
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
            push!(available, split)
        end
    end

    return block
end

function pool_free(block)
    # we don't do any work here to reduce pressure on the GC (spending time in finalizers)
    # and to simplify locking (and prevent concurrent access during GC interventions)
    block.state = FREED
    push!(freed, block)
end


## interface

init() = return

function deinit()
    @assert isempty(allocated) "Cannot deinitialize memory pool with outstanding allocations"

    repopulate(freed)
    incremental_compact!(Set(freed))
    empty!(freed)

    for available in (available_small, available_large, available_huge)
        while !isempty(available)
            block = pop!(available)
            actual_free(block)
        end
    end

    return
end

function alloc(sz)
    block = pool_alloc(sz)
    if block !== nothing
        ptr = pointer(block)
        @assert !haskey(allocated, ptr) "Newly-allocated block $block is already allocated"
        block.state = ALLOCATED
        allocated[ptr] = block
        return ptr
    else
        return nothing
    end
end

function free(ptr)
    block = allocated[ptr]
    block.state == ALLOCATED || error("Cannot free a $(block.state) block")
    delete!(allocated, ptr)
    pool_free(block)
    return
end

used_memory() = mapreduce(sizeof, +, values(allocated); init=0)

cached_memory() = mapreduce(sizeof, +, union(available_small, available_large, available_huge); init=0)

function dump()
    println("Allocated blocks: $((Base.format_bytes(used_memory())))")
    for block in sort(collect(values(allocated)); by=sizeof)
        println(" - ", block)
    end

    println("Available, but fragmented blocks: $((Base.format_bytes(cached_memory())))")
    for block in available_small
        println(" - small ", block)
    end
    for block in available_large
        println(" - large ", block)
    end
    for block in available_huge
        println(" - huge ", block)
    end
end

end
