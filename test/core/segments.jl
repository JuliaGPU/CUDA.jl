# the bookkeeping of the caching allocator, against synthetic streams and fences

using Random
using CUDACore: Arena, Block, BlockTree, Fence, Segment, GrowthCredit,
                take_block!, add_segment!, new_fence!, free_block!, poll_fences!,
                evict_segments!, unhold_fences!, segment_released!,
                segment_release_failed!, take_credit!, return_credit!,
                block_capacity, segment_class, segment_size, smallest_fit,
                SMALL_SEGMENTS, MEDIUM_SEGMENTS, LARGE_SEGMENTS,
                COMPLETED_INDEX, LANE_INDEX, NOT_INDEXED, RESIDENT, OVERSIZE

# a stream executes the work submitted to it in order. its generation is bumped when it's
# recycled, which only happens once all of its work has completed.
mutable struct TestEvent
    done::Bool
end
struct TestLane
    stream::Int
    generation::Int
end
const test_generations = Dict{Int,Int}()
CUDACore.fence_completed(e::TestEvent) = e.done
CUDACore.lane_retired(l::TestLane) = test_generations[l.stream] != l.generation

const TestArena = Arena{TestEvent,TestLane,Int}
const MiB = 1 << 20
const KiB = 1 << 10

lane(stream) = TestLane(stream, test_generations[stream])

@testset "size classes" begin
    @test block_capacity(1) == 512
    @test block_capacity(512) == 512
    @test block_capacity(513) == 1024

    @test segment_class(MiB) == SMALL_SEGMENTS
    @test segment_class(MiB + 512) == MEDIUM_SEGMENTS
    @test segment_class(10MiB - 512) == MEDIUM_SEGMENTS
    @test segment_class(10MiB) == LARGE_SEGMENTS

    @test segment_size(512) == 2MiB
    @test segment_size(MiB) == 2MiB
    @test segment_size(MiB + 512) == 20MiB
    @test segment_size(10MiB) == 10MiB
    @test segment_size(10MiB + 512) == 12MiB
end

@testset "splitting and fitting" begin
    test_generations[1] = 0
    arena = TestArena()
    addr = UInt(1) << 32

    # small requests split their segment, down to the alignment
    b = add_segment!(arena, addr, 2MiB, 100)
    @test b.size == 512
    @test b.next.size == 2MiB - 512
    @test arena.used == 512 && arena.requested == 100 && arena.resident == 2MiB

    # medium ones only split off remainders larger than a small request
    m = add_segment!(arena, addr + 2MiB, 20MiB, 8MiB)
    @test m.size == 8MiB && m.next.size == 12MiB
    m2 = take_block!(arena, 9MiB + 512KiB, lane(1))
    @test m2.size == 9MiB + 512KiB && m2.next.size == 2MiB + 512KiB
    m3 = take_block!(arena, 2MiB, lane(1))
    @test m3.size == 2MiB + 512KiB && m3.next === nothing

    # small requests aren't served from medium segments
    @test take_block!(arena, MiB + 512, lane(1)) === nothing
    @test take_block!(arena, 512, lane(1)).segment === b.segment

    # oversize blocks are only used for requests within an eighth of their size, whole
    big = add_segment!(arena, addr + 64MiB, 128MiB, 128MiB)
    free_block!(arena, big.addr, nothing)
    @test take_block!(arena, 100MiB, lane(1)) === nothing
    b2 = take_block!(arena, 112MiB, lane(1))
    @test b2 === big && b2.size == 128MiB && b2.next === nothing

    # large blocks below that are split when the rest is large enough
    l = add_segment!(arena, addr + 256MiB, 60MiB, 60MiB)
    free_block!(arena, l.addr, nothing)
    l2 = take_block!(arena, 40MiB, lane(1))
    @test l2.size == 40MiB && l2.next.size == 20MiB
    free_block!(arena, l2.addr, nothing)
    @test l.size == 60MiB && l.next === nothing  # coalesced again
    l3 = take_block!(arena, 55MiB, lane(1))
    @test l3.size == 60MiB  # (the rest is too small to split off)
end

@testset "block index" begin
    # a treap: in-order traversal sorted by (size, address), heap-ordered priorities
    rng = Xoshiro(1)
    seg = Segment(0, 0, SMALL_SEGMENTS, RESIDENT, nothing, 0, 0, 0, nothing)
    B = Block{Fence{TestEvent,TestLane,Int}}
    t = BlockTree{B}()
    blocks = [B(seg, UInt(i), rand(rng, 1:16) * 512, rand(rng, UInt64)) for i in 1:500]
    foreach(b -> push!(t, b), blocks)
    function check(t)
        seen = B[]
        foreach(b -> push!(seen, b), t)
        ok = issorted(seen; lt=CUDACore.before)
        heap(n) = n === nothing ||
                  ((n.left === nothing || n.left.priority <= n.priority) &&
                   (n.right === nothing || n.right.priority <= n.priority) &&
                   heap(n.left) && heap(n.right))
        return ok && heap(t.root) ? length(seen) : -1
    end
    @test check(t) == 500
    for b in blocks[1:2:end]
        delete!(t, b)
    end
    @test check(t) == 250
    for sz in 0:512:17*512
        expected = filter(b -> b.size >= sz, blocks[2:2:end])
        best = isempty(expected) ? nothing :
               first(sort(expected; lt=CUDACore.before))
        @test smallest_fit(t, sz) === best
    end
end

# an arena together with simulated streams that execute the work submitted to them in order
mutable struct Model
    arena::TestArena
    rng::Xoshiro
    streams::Vector{Vector{TestEvent}}  # work that hasn't completed yet, in order
    allocations::Dict{UInt,Tuple{Int,Int}}  # live: address => (stream, bytes)
    # earlier uses of memory: (first, last address, stream, event recorded after the use)
    uses::Vector{Tuple{UInt,UInt,Int,TestEvent}}
    next_addr::UInt
    seq::Int
    credit::GrowthCredit
    limit::Int
    released::Vector{Segment}
end
function Model(seed; nstreams=3, limit=0)
    for s in 1:nstreams
        test_generations[s] = 0
    end
    Model(TestArena(), Xoshiro(seed), [TestEvent[] for _ in 1:nstreams],
          Dict{UInt,Tuple{Int,Int}}(), Tuple{UInt,UInt,Int,TestEvent}[], UInt(1) << 40,
          0, GrowthCredit(), limit, Segment[])
end

function check_reuse(m::Model, b::Block, stream::Int)
    first, last = b.addr, b.addr + b.size - 1
    for (addr, (_, sz)) in m.allocations
        addr == b.addr && continue
        @assert addr + block_capacity(sz) - 1 < first || addr > last "overlapping allocations"
    end
    for (ufirst, ulast, ustream, event) in m.uses
        (ulast < first || ufirst > last) && continue
        # an earlier use that hasn't completed needs to be on the same stream
        @assert event.done || ustream == stream "reused memory before its last use completed"
    end
end

function alloc!(m::Model, stream::Int, sz::Int)
    b = take_block!(m.arena, sz, lane(stream))
    if b === nothing
        poll_fences!(m.arena)
        b = take_block!(m.arena, sz, lane(stream))
    end
    if b === nothing
        cap = block_capacity(sz)
        segsize = segment_size(cap)
        take_credit!(m.credit, segsize, m.limit) || return nothing
        base = m.next_addr
        m.next_addr += segsize + 4096
        b = add_segment!(m.arena, base, segsize, sz)
    end
    check_reuse(m, b, stream)
    m.allocations[b.addr] = (stream, sz)
    return b
end

function free!(m::Model, addr::UInt)
    stream, sz = pop!(m.allocations, addr)
    b = m.arena.live[addr]
    # the memory was used on its stream, so record a fence after that
    event = TestEvent(false)
    push!(m.streams[stream], event)
    push!(m.uses, (b.addr, b.addr + b.size - 1, stream, event))
    m.seq += 1
    fence = new_fence!(m.arena, event, lane(stream), stream, m.seq)
    free_block!(m.arena, addr, fence; now=UInt64(m.seq))
end

# complete some of the work submitted to a stream
function progress!(m::Model, stream::Int)
    queue = m.streams[stream]
    n = rand(m.rng, 0:length(queue))
    for event in queue[1:n]
        event.done = true
    end
    deleteat!(queue, 1:n)
end

# hand an idle stream to another task
function recycle!(m::Model, stream::Int)
    isempty(m.streams[stream]) || return
    test_generations[stream] += 1
end

function evict!(m::Model, bytes)
    segments, fences = evict_segments!(m.arena, bytes; now=UInt64(m.seq))
    # the release is ordered after the fences, i.e., it happens once they complete.
    # simulate that by waiting for them.
    for f in fences
        for (s, queue) in enumerate(m.streams)
            i = findfirst(e -> e === f.event, queue)
            i === nothing && continue
            foreach(e -> e.done = true, queue[1:i])
            deleteat!(queue, 1:i)
        end
    end
    unhold_fences!(m.arena, fences)
    for seg in segments
        if rand(m.rng) < 0.1
            # a failed release keeps its credit
            segment_release_failed!(m.arena, seg)
        else
            segment_released!(m.arena, seg)
            return_credit!(m.credit, seg.size)
        end
        push!(m.released, seg)
    end
    # every hold has been released
    for queue in values(m.arena.fences)
        f = queue.head
        while f !== nothing
            @assert f.holds == 0
            f = f.next
        end
    end
end

function queued_fences(arena, recorder)
    fences = []
    f = arena.fences[recorder].head
    while f !== nothing
        push!(fences, f)
        f = f.next
    end
    return fences
end

function check_invariants(m::Model)
    arena = m.arena
    used = requested = resident = 0
    for seg in arena.segments
        @assert seg.state == RESIDENT
        resident += seg.size
        b = seg.head
        @assert b.prev === nothing
        addr = seg.base
        nlive = 0
        while b !== nothing
            @assert b.segment === seg && b.addr == addr && b.size > 0
            @assert b.addr % 512 == 0
            if b.live
                @assert b.index == NOT_INDEXED && b.fence === nothing
                @assert arena.live[b.addr] === b
                used += b.size
                requested += b.requested
                nlive += 1
            else
                f = b.fence
                if f === nothing
                    @assert b.index == COMPLETED_INDEX
                else
                    @assert b.index == LANE_INDEX && any(x -> x === b, CUDACore.fence_blocks(f))
                    # otherwise it would never be found completed
                    @assert f.queued && !f.done
                    @assert any(x -> x === f, queued_fences(arena, f.recorder))
                end
                # free neighbours are merged where their fences allow it
                n = b.next
                if n !== nothing && !n.live
                    @assert !first(CUDACore.merged_fence(b, n)) "unmerged free blocks"
                end
            end
            addr += b.size
            b.next === nothing || @assert b.next.prev === b
            b = b.next
        end
        @assert addr == seg.base + seg.size "segment not covered by its blocks"
        @assert seg.nlive == nlive
        @assert (nlive == 0) == (seg in arena.wholly_free)
    end
    @assert length(arena.live) == length(m.allocations)
    @assert arena.used == used && arena.requested == requested
    @assert arena.resident == resident
    # credit covers every segment that hasn't been released
    @assert m.credit.reserved == arena.resident + arena.queued + arena.failed
    m.limit > 0 && @assert m.credit.reserved <= m.limit
    for seg in m.released
        @assert !(seg in arena.segments)
    end
    for recorder in keys(arena.fences)
        queue = queued_fences(arena, recorder)
        @assert !isempty(queue) && issorted(queue; by=f -> f.seq)
        @assert all(f -> f.queued && f.nblocks == length(CUDACore.fence_blocks(f)), queue)
        # fences that guard nothing aren't kept around for long
        @assert count(f -> f.nblocks == 0 && f.holds == 0, queue) <= 64
    end
    return true
end

function run_model(seed; steps=2000, limit=0)
    m = Model(seed; limit)
    sizes = [rand(m.rng, 1:4096) for _ in 1:20]
    append!(sizes, [rand(m.rng, MiB:12MiB) for _ in 1:5])
    append!(sizes, [rand(m.rng, 12MiB:80MiB) for _ in 1:2])
    for step in 1:steps
        r = rand(m.rng)
        stream = rand(m.rng, 1:length(m.streams))
        if r < 0.45
            alloc!(m, stream, rand(m.rng, sizes))
        elseif r < 0.8
            isempty(m.allocations) || free!(m, rand(m.rng, collect(keys(m.allocations))))
        elseif r < 0.9
            progress!(m, stream)
        elseif r < 0.93
            recycle!(m, stream)
        elseif r < 0.96
            poll_fences!(m.arena; exhaustive=rand(m.rng, Bool))
        else
            evict!(m, rand(m.rng, 0:64MiB))
        end
        check_invariants(m)
    end
    # once all work has completed and everything is freed, all memory can be returned
    foreach(addr -> free!(m, addr), collect(keys(m.allocations)))
    for queue in m.streams
        foreach(e -> e.done = true, queue)
        empty!(queue)
    end
    poll_fences!(m.arena; exhaustive=true)
    check_invariants(m)
    @assert all(seg -> seg.head.next === nothing && seg.head.fence === nothing,
                m.arena.segments)
    @assert isempty(m.arena.fences)
    evict!(m, typemax(Int))
    check_invariants(m)
    @assert isempty(m.arena.segments) && m.arena.resident == 0 && m.arena.queued == 0
    @assert isempty(m.arena.fences) && isempty(m.arena.lanes)
    return true
end

@testset "random operations" begin
    for seed in 1:20
        @test run_model(seed)
    end
    # with a limit on the memory that can be allocated
    for seed in 1:5
        @test run_model(seed; limit=64MiB)
    end
end

@testset "same-stream reuse" begin
    m = Model(42; nstreams=2)
    b = alloc!(m, 1, 4096)
    addr = b.addr
    free!(m, addr)
    # the block can be reused on its stream right away, but not on another one
    @test alloc!(m, 2, 4096).addr != addr
    @test alloc!(m, 1, 4096).addr == addr
    free!(m, addr)
    # until its fence has completed
    progress!(m, 1); foreach(e -> e.done = true, m.streams[1]); empty!(m.streams[1])
    poll_fences!(m.arena)
    @test alloc!(m, 2, 4096).addr == addr
    # or until its stream has been recycled
    free!(m, addr)
    foreach(e -> e.done = true, m.streams[2]); empty!(m.streams[2])
    recycle!(m, 2)
    poll_fences!(m.arena)
    @test alloc!(m, 1, 4096).addr == addr
    @test check_invariants(m)
end

@testset "coalescing" begin
    test_generations[1] = 0
    test_generations[2] = 0
    arena = TestArena()
    a = add_segment!(arena, UInt(0), 2MiB, 512)
    b = take_block!(arena, 512, lane(1))
    c = take_block!(arena, 512, lane(1))
    @test (a.addr, b.addr, c.addr) == (0, 512, 1024)
    ea, eb = TestEvent(false), TestEvent(false)
    # fences on different streams don't cover each other
    fa = new_fence!(arena, ea, lane(1), 1, 1)
    fb = new_fence!(arena, eb, lane(2), 2, 2)
    free_block!(arena, a.addr, fa)
    free_block!(arena, b.addr, fb)
    @test a.next === b && a.size == 512 && b.size == 512
    # a later fence on the same stream covers an earlier one
    ec = TestEvent(false)
    fc = new_fence!(arena, ec, lane(2), 2, 3)
    free_block!(arena, c.addr, fc)
    @test b.size == 1024 && b.fence === fc
    # but a completed block isn't merged with a pending one, which would restrict it
    @test b.next.size == 2MiB - 1536 && b.next.fence === nothing
    # once completed, everything merges
    eb.done = ec.done = true
    poll_fences!(arena)
    @test a.size == 512 && a.fence === fa
    ea.done = true
    poll_fences!(arena)
    @test a.size == 2MiB && a.fence === nothing
    @test length(arena.spare_events) == 3
end

@testset "eviction" begin
    test_generations[1] = 0
    arena = TestArena()
    a = add_segment!(arena, UInt(0), 2MiB, 512; now=UInt64(0))
    b = add_segment!(arena, UInt(4MiB), 20MiB, 2MiB; now=UInt64(0))
    e = TestEvent(false)
    f = new_fence!(arena, e, lane(1), 1, 1)
    free_block!(arena, a.addr, f; now=UInt64(10))
    # live segments aren't evicted, nor recently freed ones
    @test isempty(first(evict_segments!(arena, 100MiB; now=UInt64(15), min_age=10)))
    # pending ones are, along with their fence
    segments, fences = evict_segments!(arena, 1; now=UInt64(20), min_age=10)
    seg = only(segments)
    @test seg === a.segment && fences == [f]
    @test arena.queued == 2MiB && arena.resident == 20MiB
    # the fence is kept until the release has been ordered after it
    e.done = true
    poll_fences!(arena)
    @test isempty(arena.spare_events)
    unhold_fences!(arena, fences)
    @test arena.spare_events == [e]
    segment_released!(arena, seg)
    @test arena.queued == 0
    # an evicted segment is never reused
    @test take_block!(arena, 512, lane(1)) === nothing
end

@testset "growth credit" begin
    credit = GrowthCredit()
    @test take_credit!(credit, 60, 100)
    @test !take_credit!(credit, 50, 100)
    @test credit.reserved == 60
    return_credit!(credit, 60)
    # concurrent growth never exceeds the limit
    peak = Threads.Atomic{Int}(0)
    Threads.@threads for i in 1:10_000
        if take_credit!(credit, 7, 100)
            Threads.atomic_max!(peak, credit.reserved)
            return_credit!(credit, 7)
        end
    end
    @test peak[] <= 100
    @test credit.reserved == 0
end
