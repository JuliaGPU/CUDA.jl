module SplittingPool

# scan into a sorted list of free buffers, splitting buffers along the way

import ..@pool_timeit, ..actual_alloc, ..actual_free

using DataStructures

using CUDAdrv


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

# TODO: it would be nice if this could be immutable, since that's what OrderedSet requires
mutable struct Block
    buf::Mem.Buffer     # base allocation
    sz::Integer         # size into it
    off::Integer        # offset into it

    state::BlockState
    prev::Union{Nothing,Block}
    next::Union{Nothing,Block}

    id::UInt

    Block(buf, sz=sizeof(buf), off=0, state=INVALID, prev=nothing, next=nothing) =
        new(buf, sz, off, state, prev, next, block_id[]+=1)
end

Base.sizeof(block::Block) = block.sz
Base.pointer(block::Block) = pointer(block.buf) + block.off

Base.convert(::Type{Mem.Buffer}, block::Block) = similar(block.buf, pointer(block), sizeof(block))

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
    @assert sz < block.sz
    split = Block(block.buf, sizeof(block) - sz, block.off + sz)
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

# merge a sequence of blocks `blocks`
function merge!(head, blocks...)
    for block in blocks
        @assert head.next === block
        head.sz += block.sz

        # update links
        tail = block.next
        head.next = tail
        if tail !== nothing
            tail.prev = head
        end
    end

    return head
end

function actual_free(block::Block)
    @assert iswhole(block) "Cannot free a split block"
    if block.state != AVAILABLE
        error("Cannot free a $(block.state) block")
    else
        actual_free(block.buf)
        block.state = INVALID
    end
    return
end


## pooling

using Base.Threads
const pool_lock = SpinLock()   # protect against deletion from freelists

function scan!(blocks, sz, max_overhead=typemax(Int))
    max_sz = max(sz + max_overhead, max_overhead)   # protect against overflow
    lock(pool_lock) do
        for block in blocks
            if sz <= sizeof(block) <= max_sz
                delete!(blocks, block)
                return block
            end
        end
        return nothing
    end
end

function incremental_compact!(blocks)
    # we mutate the list of blocks, so take a copy
    blocks = Set(blocks)

    compacted = 0
    lock(pool_lock) do
        while !isempty(blocks)
            block = pop!(blocks)
            szclass = size_class(sizeof(block))
            available = (available_small, available_large, available_huge)[szclass]

            # get the first unallocated node in a chain
            head = block
            while head.prev !== nothing && head.prev.state == AVAILABLE
                head = head.prev
            end

            # construct a chain of unallocated blocks
            chain = [head]
            let block = head
                while block.next !== nothing && block.next.state == AVAILABLE
                    block = block.next
                    @assert block in available
                    push!(chain, block)
                end
            end

            # compact the chain into a single block
            if length(chain) > 1
                for block in chain
                    delete!(available, block)
                    delete!(blocks, block)
                end
                block = merge!(chain...)
                @assert !in(block, available) "Collision in the available memory pool"
                push!(available, block)
                compacted += length(chain) - 1
            end
        end
    end
    return compacted
end

# TODO: partial reclaim on ordered list?
function reclaim!(blocks, sz)
    freed = 0
    candidates = []
    lock(pool_lock) do
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
        end
    end

    return freed
end

function repopulate!(blocks)
    lock(pool_lock) do
        for block in blocks
            szclass = size_class(sizeof(block))
            available = (available_small, available_large, available_huge)[szclass]
            @assert !in(block, available) "Collision in the available memory pool"
            @assert block.state == FREED
            block.state = AVAILABLE
            push!(available, block)
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
const allocated = Dict{Mem.Buffer,Block}()
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

function pool_alloc(sz)
    szclass = size_class(sz)

    # round of the block size
    req_sz = sz
    roundoff = (SMALL_ROUNDOFF, LARGE_ROUNDOFF, HUGE_ROUNDOFF)[szclass]
    sz = cld(sz, roundoff) * roundoff
    szclass = size_class(sz)

    # select a pool
    available = (available_small, available_large, available_huge)[szclass]

    # determine the maximum scan overhead
    max_overhead = (SMALL_OVERHEAD, LARGE_OVERHEAD, HUGE_OVERHEAD)[szclass]

    block = nothing
    for phase in 1:3
        if phase == 2
           @pool_timeit "$phase.0 gc(false)" GC.gc(false)
        elseif phase == 3
           @pool_timeit "$phase.0 gc(true)" GC.gc(true)
        end

        if !isempty(freed)
            @pool_timeit "$phase.1 repopulate + compact" begin
                # `freed` may be modified concurrently, so take a copy
                blocks = copy(freed)
                empty!(freed)

                repopulate!(blocks)
                incremental_compact!(blocks)
            end
        end

        @pool_timeit "$phase.2 scan" begin
            block = scan!(available, sz, max_overhead)
        end
        block === nothing || break

        @pool_timeit "$phase.3 alloc" begin
            buf = actual_alloc(sz)
            block = buf === nothing ? nothing : Block(buf)
        end
        block === nothing || break

        @pool_timeit "$phase.4 reclaim + alloc" begin
            reclaim!(available_small, sz)
            reclaim!(available_large, sz)
            reclaim!(available_huge, sz)
            buf = actual_alloc(sz)
            block = buf === nothing ? nothing : Block(buf)
        end
        block === nothing || break
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

        block.state = ALLOCATED
    end

    return block
end

function pool_free(block)
    # we don't do any work here to reduce pressure on the GC (spending time in finalizers)
    # and to simplify locking (and prevent concurrent access during GC interventions)
    block.state == ALLOCATED || error("Cannot free a $(block.state) block")
    block.state = FREED
    push!(freed, block)
end


## interface

init() = return

function deinit()
    @assert isempty(allocated) "Cannot deinitialize memory pool with outstanding allocations"

    repopulate!(freed)
    incremental_compact!(freed)
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
        buf = convert(Mem.Buffer, block)
        @assert !haskey(allocated, buf)
        allocated[buf] = block
        return buf
    else
        return nothing
    end
end

function free(buf)
    block = allocated[buf]
    delete!(allocated, buf)
    pool_free(block)
    return
end

used_memory() = mapreduce(sizeof, +, values(allocated); init=0)

cached_memory() = mapreduce(sizeof, +, union(available_small, available_large, available_huge); init=0)

function dump()
    println("Allocated buffers: $((Base.format_bytes(used_memory())))")
    for block in sort(collect(values(allocated)); by=sizeof)
        println(" - ", block)
    end

    println("Available, but fragmented buffers: $((Base.format_bytes(cached_memory())))")
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
