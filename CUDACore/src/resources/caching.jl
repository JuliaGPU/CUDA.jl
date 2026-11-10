## caching allocator over `cuMemAlloc`
#
# memory from CUDA's stream-ordered pool can't be shared with other processes through legacy
# CUDA IPC (`cuIpcGetMemHandle`), which is what CUDA-aware MPI implementations use within a
# node. this allocator caches `cuMemAlloc` segments instead, and suballocates blocks from
# them (see segments.jl for the bookkeeping). every allocation then lies within an
# IPC-capable allocation, and transports can keep the segments mapped.
#
# reusing a block never waits: on the stream that freed it right away, elsewhere once its
# fence has completed. segments are only returned to the driver once wholly free, in stream
# order after their fences on the disposal stream, which is synchronized later on to
# actually return the memory (see `record_free_events`). that happens:
# - when allocating would grow the cache beyond `RETENTION_FACTOR` times what's in use (or
#   the soft limit), for segments that have been wholly free for at least `MIN_FREE_AGE`;
# - before failing an allocation, for all of them, waiting for them to be returned;
# - when reclaiming memory, and when the device has been idle for a while (the idle trim,
#   down to the soft limit).

const RETENTION_FACTOR = 1.25
const MIN_FREE_AGE = 250_000_000  # ns

# the stream that memory was last used on, during a generation (see `release_stream`). work
# that is submitted to that stream later runs after that use. streams are compared by
# identity, as their handles are reused after destroying them.
struct Lane
  stream::CuStream
  ctx::CuContext
  generation::Int
end
Base.:(==)(a::Lane, b::Lane) =
  a.stream === b.stream && a.ctx == b.ctx && a.generation == b.generation
Base.hash(l::Lane, h::UInt) = hash(objectid(l.stream), hash(l.ctx, hash(l.generation, h)))

lane_retired(l::Lane) = l.generation != generation(l.stream)

# the stream an event is recorded on: either a stream object, or a disposal stream
struct Recorder
  stream::Union{Nothing,CuStream}
  handle::CUstream
  ctx::CuContext
end
Base.:(==)(a::Recorder, b::Recorder) =
  a.stream === b.stream && a.handle == b.handle && a.ctx == b.ctx
Base.hash(r::Recorder, h::UInt) =
  hash(objectid(r.stream), hash(r.handle, hash(r.ctx, h)))

# (callers relax the capture mode, as querying an event is prohibited while another thread
# captures in global mode)
function fence_completed(event::CuEvent)
  # (the work of a destroyed context has ended)
  isvalid(event.ctx) || return true
  unsafe_isdone(event)
end

const CachingArena = Arena{CuEvent,Lane,Recorder}
const CachingFence = Fence{CuEvent,Lane,Recorder}

# segments that are being returned to the driver. they're either freed in stream order on
# the disposal stream, and returned once that is synchronized (when `event`, recorded after
# the frees, has completed, or by synchronizing the stream itself), or freed with `cuMemFree`
# when reclaiming memory, after waiting for their fences. it's registered as soon as the
# segments are detached from the cache, but only finished once `ready`, i.e., once the task
# that detached them is done submitting their release.
mutable struct SegmentRelease
  const allocator::Any
  const ctx::CuContext
  const stream::CUstream
  segments::Vector{Segment}
  fences::Vector{CachingFence}
  freed::Bool                     # whether they've been freed in stream order
  event::Union{Nothing,CuEvent}
  Base.@atomic ready::Bool
  Base.@atomic done::Bool
end
SegmentRelease(a, ctx::CuContext, stream::CUstream) =
  SegmentRelease(a, ctx, stream, Segment[], CachingFence[], false, nothing, false, false)

mutable struct CachingAllocator <: DeviceAllocator
  const dev::CuDevice
  const limits::@NamedTuple{soft::UInt, hard::UInt}
  const lock::ReentrantLock
  const arenas::Dict{CuContext,CachingArena}
  const credit::GrowthCredit
  const releases::Vector{SegmentRelease}  # protected by `lock`
  # segments that were allocated, but couldn't be added to the cache: (ctx, base, size)
  const orphans::Vector{Tuple{CuContext,UInt,Int}}  # protected by `lock`
  const events::Dict{CuContext,Vector{CuEvent}}  # spare events, protected by `events_lock`
  const events_lock::ReentrantLock
  retention::Int
  # failures to inject, for testing: `:alloc`, `:metadata`, `:release`, `:release_event`
  const faults::Set{Symbol}
  # (for testing: where releasing segments waits before freeing them)
  release_barrier::Union{Nothing,Base.Event}

  Base.@atomic hits::Int
  Base.@atomic misses::Int
  Base.@atomic segments_allocated::Int
  Base.@atomic segments_released::Int
end

function CachingAllocator(dev::CuDevice; limits=memory_limits(), register::Bool=true)
  a = CachingAllocator(dev, limits, ReentrantLock(), Dict{CuContext,CachingArena}(),
                       GrowthCredit(), SegmentRelease[], Tuple{CuContext,UInt,Int}[],
                       Dict{CuContext,Vector{CuEvent}}(), ReentrantLock(), Int(limits.soft),
                       Set{Symbol}(), nothing, 0, 0, 0, 0)
  # (registering makes reclaiming memory return wholly free segments, see `purge!`)
  register && register_reclaimable!(CacheReclaimer(a))
  start_pool_cleanup()
  return a
end

name(::CachingAllocator) = "caching (cached `cuMemAlloc` segments)"

struct CacheReclaimer <: Reclaimable
  allocator::CachingAllocator
end
purge!(r::CacheReclaimer) = purge!(r.allocator)

function arena!(a::CachingAllocator, ctx::CuContext)
  get!(CachingArena, a.arenas, ctx)
end

used(a::CachingAllocator) = @lock a.lock sum(arena -> arena.used, values(a.arenas); init=0)
reserved(a::CachingAllocator) = Base.@atomic a.credit.reserved

set_retention!(a::CachingAllocator, bytes) = (a.retention = bytes; nothing)

# (internal, for testing)
inject_fault!(a::CachingAllocator, fault::Symbol) = (@lock a.lock push!(a.faults, fault); a)
function take_fault!(a::CachingAllocator, fault::Symbol)
  isempty(a.faults) && return false
  @lock a.lock begin
    fault in a.faults || return false
    delete!(a.faults, fault)
    return true
  end
end


## events

function take_event!(a::CachingAllocator, ctx::CuContext)
  event = @lock a.events_lock begin
    events = get(a.events, ctx, nothing)
    events === nothing || isempty(events) ? nothing : pop!(events)
  end
  return @something event context!(() -> CuEvent(EVENT_DISABLE_TIMING), ctx)
end

# put the events of fences that aren't needed anymore back (holding `a.lock`)
function recycle_events!(a::CachingAllocator, arena::CachingArena)
  isempty(arena.spare_events) && return
  @lock a.events_lock for event in arena.spare_events
    # (events of a destroyed context have been destroyed along with it)
    isvalid(event.ctx) && push!(get!(Vector{CuEvent}, a.events, event.ctx), event)
  end
  empty!(arena.spare_events)
  return
end

# fences are ordered per stream by when they were recorded
const fence_seq = Threads.Atomic{UInt64}(0)


## allocation

@inline function try_allocate(a::CachingAllocator, sz, state)
  pool_mark!(a.dev, true)
  ctx = state.context
  lane = Lane(state.stream, ctx, generation(state.stream))
  b = @lock a.lock begin
    arena = arena!(a, ctx)
    b = take_block!(arena, sz, lane)
    recycle_events!(a, arena)
    b
  end
  if b !== nothing
    Base.@atomic a.hits += 1
    return cached_memory(a, ctx, b, sz)
  end
  return allocate_slow(a, sz, ctx, lane)
end

@noinline function allocate_slow(a::CachingAllocator, sz, ctx::CuContext, lane::Lane)
  Base.@atomic a.misses += 1

  # blocks whose fences have completed may fit
  b = take_polled!(a, ctx, sz, lane; exhaustive=true)
  b === nothing || return cached_memory(a, ctx, b, sz)

  cap = block_capacity(sz)
  segsize = segment_size(cap)
  # releases that have finished return their credit
  finish_releases!(a)
  # keep the cache from growing much beyond what's in use
  limit_retention!(a, cap, segsize)

  b = grow!(a, ctx, segsize, sz)
  b === nothing || return cached_memory(a, ctx, b, sz)

  # out of memory or credit: look harder before failing. this waits for the GPU, but
  # nothing is held while doing so, and the alternative is the reclaim ladder.
  b = take_polled!(a, ctx, sz, lane; exhaustive=true)
  b === nothing || return cached_memory(a, ctx, b, sz)
  if active_captures[] == 0
    evict!(a; min_age=0, bytes=typemax(Int))
    finish_releases!(a; blocking=true)
    b = take_polled!(a, ctx, sz, lane; exhaustive=true)
    b === nothing || return cached_memory(a, ctx, b, sz)
  end
  b = grow!(a, ctx, segsize, sz)
  if b === nothing && segsize > cap
    # near the limit, a segment of just the requested size may still fit
    b = grow!(a, ctx, cap, sz)
  end
  b === nothing && return nothing
  return cached_memory(a, ctx, b, sz)
end

@inline function cached_memory(a::CachingAllocator, ctx::CuContext, b::Block, sz)
  annotating() && annotate_allocation(b.segment.payload::Ptr{Cvoid}, b.addr, sz)
  return DeviceMemory(a.dev, ctx, reinterpret(CuPtr{Cvoid}, b.addr), sz, CACHED_MEMORY)
end

function take_polled!(a::CachingAllocator, ctx::CuContext, sz, lane::Lane; exhaustive::Bool)
  @lock a.lock begin
    arena = arena!(a, ctx)
    # (querying events is prohibited while another thread captures in global mode)
    relaxed_capture_mode(() -> poll_fences!(arena; exhaustive))
    recycle_events!(a, arena)
    take_block!(arena, sz, lane)
  end
end

# allocate a segment of `segsize` bytes, and a block for `sz` bytes from it
function grow!(a::CachingAllocator, ctx::CuContext, segsize::Int, sz)
  take_credit!(a.credit, segsize, a.limits.hard) || return nothing
  ref = Ref{CUdeviceptr}()
  res = try
    if take_fault!(a, :alloc)
      ERROR_OUT_OF_MEMORY
    else
      # (allocating is prohibited while another thread captures in global mode)
      relaxed_capture_mode(() -> unchecked_cuMemAlloc_v2(ref, segsize))
    end
  catch
    return_credit!(a.credit, segsize)
    rethrow()
  end
  if res != SUCCESS
    return_credit!(a.credit, segsize)
    res == ERROR_OUT_OF_MEMORY && return nothing
    throw_api_error(res)
  end
  base = UInt(ref[])
  # the memory now exists, so its credit is only returned once it has been freed
  try
    take_fault!(a, :metadata) && error("Injected failure")
    payload = annotating() ? annotate_segment(base, segsize) : nothing
    b = @lock a.lock add_segment!(arena!(a, ctx), base, segsize, sz; payload)
    Base.@atomic a.segments_allocated += 1
    return b
  catch
    @lock a.lock push!(a.orphans, (ctx, base, segsize))
    rethrow()
  end
end


## freeing

function release(a::CachingAllocator, mem::DeviceMemory, stream::CuStream,
                 stream_ctx::CuContext, generation::Int)
  addr = UInt(mem.ptr)
  annotating() && annotate_free(addr)
  if !isvalid(mem.ctx)
    # its segment was freed along with the context
    drop_destroyed!(a)
    return
  end

  # an event that is recorded after the last use of the memory, unless that use is known to
  # have completed: memory last used on the per-thread default stream is only released
  # after the context has been synchronized (see `release_now`), and recycled streams have
  # finished their work.
  fence = nothing
  if !on_per_thread_stream(stream) && generation == CUDACore.generation(stream)
    fence = @lock stream_disposal_lock begin
      released = release_stream(stream, stream_ctx, generation)
      # (the work of a destroyed context has ended)
      if released === nothing
        nothing
      else
        handle, ctx = released
        event = take_event!(a, ctx)
        try
          context!(() -> cuEventRecord(event, handle), ctx)
        catch
          @lock a.events_lock push!(get!(Vector{CuEvent}, a.events, ctx), event)
          rethrow()
        end
        recorder = Recorder(handle == stream.handle ? stream : nothing, handle, ctx)
        (event, recorder, Threads.atomic_add!(fence_seq, UInt64(1)))
      end
    end
  end

  @lock a.lock begin
    arena = a.arenas[mem.ctx]
    f = if fence === nothing
      nothing
    else
      event, recorder, seq = fence
      new_fence!(arena, event, Lane(stream, stream_ctx, generation), recorder, seq)
    end
    free_block!(arena, addr, f)
    recycle_events!(a, arena)
  end
  return
end

# forget about the memory of destroyed contexts, which was freed along with them
function drop_destroyed!(a::CachingAllocator)
  dropped = @lock a.lock begin
    dropped = 0
    for (ctx, arena) in collect(a.arenas)
      isvalid(ctx) && continue
      dropped += arena.resident + arena.queued + arena.failed
      delete!(a.arenas, ctx)
    end
    for (ctx, _, size) in a.orphans
      isvalid(ctx) || (dropped += size)
    end
    filter!(((ctx, _, _),) -> isvalid(ctx), a.orphans)
    dropped
  end
  dropped > 0 && return_credit!(a.credit, dropped)
  return
end


## returning memory to the driver

# keep the cache from growing beyond `RETENTION_FACTOR` times what's in use when allocating
# `segsize` more for a request of `cap` bytes, by evicting segments that haven't been used
# for a while
function limit_retention!(a::CachingAllocator, cap::Int, segsize::Int)
  target = max(Int(a.limits.soft), round(Int, RETENTION_FACTOR * (used(a) + cap)))
  excess = reserved(a) - queued_bytes(a) + segsize - target
  excess > 0 && evict!(a; min_age=MIN_FREE_AGE, bytes=excess)
  return
end

queued_bytes(a::CachingAllocator) =
  @lock a.lock sum(arena -> arena.queued, values(a.arenas); init=0)

# release wholly free segments of at least `bytes`, in stream order after their fences.
# doesn't wait for the GPU. returns whether segments were released.
function evict!(a::CachingAllocator; min_age::Integer, bytes::Integer)
  drop_destroyed!(a)
  # segments are only detached from the cache once captures have been excluded, so that
  # they aren't lost if releasing isn't possible right now
  remaining = Ref{Int}(bytes)
  evicted = Ref(false)
  releasing() do
    for ctx in @lock(a.lock, collect(keys(a.arenas)))
      remaining[] > 0 || break
      isvalid(ctx) || continue
      rel = SegmentRelease(a, ctx, disposal_stream(ctx))
      # (detached and registered together, so that the segments are owned by the release)
      registered = @lock a.lock begin
        arena = get(a.arenas, ctx, nothing)
        if arena === nothing
          false
        else
          sizehint!(a.releases, length(a.releases) + 1)
          rel.segments, rel.fences = evict_segments!(arena, remaining[]; min_age)
          isempty(rel.segments) || push!(a.releases, rel)
          !isempty(rel.segments)
        end
      end
      registered || continue
      evicted[] = true
      remaining[] -= sum(seg -> seg.size, rel.segments)
      try
        release_segments!(a, rel)
      finally
        Base.@atomic rel.ready = true
      end
      # memory freed in stream order is only returned once that stream is synchronized, so
      # do that once the work has finished
      rel.event === nothing ||
        hold!(ReleaseAction(finish_release!, rel, nothing, false), rel.event)
    end
  end
  return evicted[]
end

function release_segments!(a::CachingAllocator, rel::SegmentRelease)
  ctx, stream, segments, fences = rel.ctx, rel.stream, rel.segments, rel.fences
  if annotating()
    foreach(seg -> annotate_segment_release(seg.payload::Ptr{Cvoid}), segments)
  end
  a.release_barrier === nothing || Base.wait(a.release_barrier)
  # without support for freeing memory in stream order, `cuMemFree` waits for all running
  # kernels, so it's only called when reclaiming memory (after waiting for the fences)
  async_free_supported(a.dev) || return

  freed = Segment[]
  attempted = Ref(false)
  try
    context!(ctx) do
      event = CuEvent(EVENT_DISABLE_TIMING)
      for f in fences
        isvalid(f.event.ctx) && cuStreamWaitEvent(stream, f.event, 0)
      end
      for seg in segments
        attempted[] = true
        take_fault!(a, :release) && error("Injected failure")
        cuMemFreeAsync(reinterpret(CuPtr{Cvoid}, seg.base), stream)
        push!(freed, seg)
      end
      take_fault!(a, :release_event) && error("Injected failure")
      cuEventRecord(event, stream)
      rel.event = event
    end
  catch err
    @error "Failed to return GPU memory to the driver" exception=(err, catch_backtrace())
  finally
    # once freeing has been attempted, segments that weren't freed are kept as they are,
    # as that may have partially succeeded. if nothing was attempted, they are freed with
    # `cuMemFree` when reclaiming memory instead.
    attempted[] && @lock a.lock begin
      arena = a.arenas[ctx]
      foreach(seg -> seg in freed || segment_release_failed!(arena, seg), segments)
      rel.segments = freed
      rel.freed = true
      unhold_fences!(arena, fences)
      # (if no event was recorded, the frees are returned by synchronizing the stream)
      isempty(freed) && filter!(!=(rel), a.releases)
    end
  end
  return
end

# return the memory of releases that have finished, or with `blocking`, of all of them
function finish_releases!(a::CachingAllocator; blocking::Bool=false)
  rels = @lock a.lock copy(a.releases)
  isempty(rels) && return
  releasing(; blocking) do
    for rel in rels
      # (a release that's still being submitted is finished once it has been)
      Base.@atomic(rel.ready) || continue
      if blocking || (rel.event !== nothing && fence_completed(rel.event))
        attempt_release(ReleaseAction(finish_release!, rel, nothing, true))
      end
    end
  end
  return
end

function finish_release!(rel::SegmentRelease)
  @assert Base.@atomic(rel.ready)
  (Base.@atomicreplace rel.done false => true).success || return
  a = rel.allocator::CachingAllocator
  if isvalid(rel.ctx)
    context!(rel.ctx) do
      if !rel.freed
        for f in rel.fences
          isvalid(f.event.ctx) && synchronize(f.event)
        end
        for seg in rel.segments
          cuMemFree_v2(reinterpret(CuPtr{Cvoid}, seg.base))
        end
      elseif rel.event !== nothing
        # (`synchronize(event)` may find it done without synchronizing)
        cuEventSynchronize(rel.event)
      else
        cuStreamSynchronize(rel.stream)
      end
    end
  end
  # the arena's counters and the credit are updated together, so that dropping the arena
  # of a destroyed context (which returns the credit of its segments) can't interleave
  bytes = sum(seg -> seg.size, rel.segments; init=0)
  @lock a.lock begin
    arena = get(a.arenas, rel.ctx, nothing)
    if arena !== nothing
      foreach(seg -> segment_released!(arena, seg), rel.segments)
      rel.freed || unhold_fences!(arena, rel.fences)
      return_credit!(a.credit, bytes)
    end
    filter!(!=(rel), a.releases)
  end
  Base.@atomic a.segments_released += length(rel.segments)
  return
end

# when the device has been idle for a while, return what's cached beyond the retention
# target (the soft limit, by default)
function trim!(a::CachingAllocator)
  finish_releases!(a)
  excess = reserved(a) - queued_bytes(a) - a.retention
  excess > 0 && evict!(a; min_age=0, bytes=excess)
  return
end

# when reclaiming memory, return everything that isn't in use, waiting for it
function purge!(a::CachingAllocator)
  drop_destroyed!(a)
  evict!(a; min_age=0, bytes=typemax(Int))
  finish_releases!(a; blocking=true)
  # segments that couldn't be added to the cache
  orphans = @lock a.lock begin
    orphans = copy(a.orphans)
    empty!(a.orphans)
    orphans
  end
  for (ctx, base, size) in orphans
    attempt_release(ReleaseAction((ctx, base, size), nothing, true) do (ctx, base, size)
      isvalid(ctx) && context!(() -> cuMemFree_v2(reinterpret(CuPtr{Cvoid}, base)), ctx)
      return_credit!(a.credit, size)
    end)
  end
  return
end

grant_access!(a::CachingAllocator, peers::Vector{CuDevice}) = grant_peer_access(a.dev, peers)

function tree_capacity(tree::BlockTree)
  bytes = Ref(0)
  foreach(b -> bytes[] += b.size, tree)
  return bytes[]
end

function status(io::IO, a::CachingAllocator)
  stats = @lock a.lock begin
    used = requested = resident = queued = failed = pending = 0
    for arena in values(a.arenas)
      used += arena.used
      requested += arena.requested
      resident += arena.resident
      queued += arena.queued
      failed += arena.failed
      for trees in values(arena.lanes), tree in trees
        pending += tree_capacity(tree)
      end
    end
    (; used, requested, resident, queued, failed, pending)
  end
  @printf(io, "Cached memory: %s reusable, %s waiting for the GPU, %s being released\n",
          Base.format_bytes(stats.resident - stats.used - stats.pending),
          Base.format_bytes(stats.pending), Base.format_bytes(stats.queued))
  return
end
