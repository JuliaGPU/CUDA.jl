# high-level memory management


## allocation statistics

mutable struct AllocStats
  Base.@atomic alloc_count::Int
  Base.@atomic alloc_bytes::Int

  Base.@atomic free_count::Int
  Base.@atomic free_bytes::Int

  Base.@atomic total_time::Float64
end

AllocStats() = AllocStats(0, 0, 0, 0, 0.0)

Base.copy(alloc_stats::AllocStats) =
  AllocStats(alloc_stats.alloc_count, alloc_stats.alloc_bytes,
             alloc_stats.free_count, alloc_stats.free_bytes,
             alloc_stats.total_time)

Base.:(-)(a::AllocStats, b::AllocStats) = (;
    alloc_count = a.alloc_count - b.alloc_count,
    alloc_bytes = a.alloc_bytes - b.alloc_bytes,
    free_count  = a.free_count  - b.free_count,
    free_bytes  = a.free_bytes  - b.free_bytes,
    total_time  = a.total_time  - b.total_time)

const alloc_stats = AllocStats()


## memory accounting

mutable struct MemoryStats
  # maximum size of the memory heap
  Base.@atomic size::Int
  Base.@atomic size_updated::Float64

  # the amount of live bytes
  Base.@atomic live::Int

  Base.@atomic last_time::Float64
  Base.@atomic last_gc_time::Float64
  Base.@atomic last_freed::Int
end
MemoryStats() = MemoryStats(0, 0.0, 0, 0.0, 0.0, 0)

function account!(stats::MemoryStats, bytes::Integer)
  Base.@atomic stats.live += bytes
  if bytes > 0
    Base.@atomic stats.live += 1
  end
end

function memory_stats(dev::CuDevice=device())
  @memoize index=deviceid(dev)+1 begin
    MemoryStats()
  end::MemoryStats
end

# stats for memory that isn't part of any device pool (unified, host). these
# allocations are not tied to a specific device and can be migrated by the driver,
# so we track them globally and size them against system RAM rather than GPU memory.
const _host_stats = MemoryStats()

# lazily-initialized mutable flag controlling whether `maybe_collect` runs.
# `nothing` means uninitialized; once set it's read directly. mutable so we can
# disable it at runtime if `cuMemGetInfo` starts failing (see below).
const _early_gc = Ref{Union{Nothing,Bool}}(nothing)
function maybe_collect(will_block::Bool=false)
  enabled = _early_gc[]
  if enabled === nothing
    enabled = parse(Bool, get(ENV, "JULIA_CUDA_GC_EARLY", "true"))
    _early_gc[] = enabled
  end
  enabled || return
  stats = memory_stats()
  current_time = time()

  # periodically re-estimate the amount of memory available to this process.
  if current_time - stats.size_updated > 10
    limits = memory_limits()
    new_size = if limits.hard > 0
      limits.hard
    elseif limits.soft > 0
      limits.soft
    else
      # `cuMemGetInfo` can fail in unusual contexts — e.g. when a prior kernel
      # left the context in a sticky error state. propagating that error here
      # would obscure the real cause (see issue #2795), so on failure we
      # disable early GC for the rest of the session and bail out, letting
      # the actual allocation surface a more meaningful error.
      free = try
        free_memory()
      catch err
        isa(err, CuError) || rethrow()
        @warn "Failed to query free GPU memory; disabling early GC. \
               This usually indicates a prior CUDA error left the context in \
               a bad state." exception=(err, catch_backtrace())
        _early_gc[] = false
        return
      end
      size = free + stats.live
      # NOTE: we use stats.live so that we only count memory allocated here, ensuring
      #       the pressure calculation below reflects the heap we have control over.

      # also include reserved bytes
      dev = device()
      if stream_ordered(dev)
        size += (cached_memory() - used_memory())::Int
      end

      size
    end
    Base.@atomic stats.size = new_size
    Base.@atomic stats.size_updated = current_time
  end

  # similarly re-estimate the host memory budget for unified/host allocations.
  # `Sys.total_memory()` is cgroup-aware (it wraps `uv_get_constrained_memory`),
  # so this does the right thing in containers without us reinventing it.
  if current_time - _host_stats.size_updated > 10
    Base.@atomic _host_stats.size = Sys.total_memory()
    Base.@atomic _host_stats.size_updated = current_time
  end

  # compute pressure for both pools and operate on whichever is dominant.
  device_pressure = stats.size > 0 ? stats.live / stats.size : 0.0
  host_pressure = _host_stats.size > 0 ?
    _host_stats.live / _host_stats.size : 0.0
  pressure, dominant = device_pressure >= host_pressure ?
    (device_pressure, stats) : (host_pressure, _host_stats)

  min_pressure = 0.75
  ## if we're about to block anyway, now may be a good time for a GC pause
  if will_block
    min_pressure = 0.50
  end
  if pressure < min_pressure
    return
  end

  # ensure we don't collect too often by checking the GC rate
  last_time = dominant.last_time
  gc_rate = dominant.last_gc_time / (current_time - last_time)
  ## we tolerate 5% GC time
  max_gc_rate = 0.05
  ## if we freed a lot last time, bump that up
  if dominant.last_freed > 0.1*dominant.size
    max_gc_rate *= 2
  end
  ## if we're about to block, we can be more aggressive
  if will_block
    max_gc_rate *= 2
  end
  ## if we're under a lot of pressure, be even more aggressive
  if pressure > 0.90
    max_gc_rate *= 2
  end
  if pressure > 0.95
    max_gc_rate *= 2
  end
  if gc_rate > max_gc_rate
    return
  end
  Base.@atomic stats.last_time = current_time
  Base.@atomic _host_stats.last_time = current_time

  # finally, call the GC. snapshot live for both pools before/after, since
  # finalizers running during GC may free memory in either.
  pre_device_live = stats.live
  pre_host_live = _host_stats.live
  gc_time = Base.@elapsed begin
    GC.gc(false)
    # finalizers only retire memory, so release it before measuring what was freed
    drain_retired()
  end
  Base.@atomic stats.last_freed = pre_device_live - stats.live
  Base.@atomic _host_stats.last_freed = pre_host_live - _host_stats.live
  ## GC times can vary, so smooth them out
  Base.@atomic stats.last_gc_time = 0.75*stats.last_gc_time + 0.25*gc_time
  Base.@atomic _host_stats.last_gc_time = 0.75*_host_stats.last_gc_time + 0.25*gc_time

  return
end


## memory limits

# parse a memory limit, e.g. "1.5GiB" or "50%, to the number of bytes
function parse_limit(str::AbstractString)
    if endswith(str, "%")
        str = str[1:end-1]
        return round(UInt, parse(Float64, str) / 100 * total_memory())
    end

    si_units = [("k", "kB", "K", "KB"), ("M", "MB"), ("G", "GB")]
    for (i, units) in enumerate(si_units), unit in units
        if endswith(str, unit)
            multiplier = 1000^i
            str = str[1:end-length(unit)]
            return round(UInt, parse(Float64, str) * multiplier)
        end
    end

    iec_units = ["KiB", "MiB", "GiB"]
    for (i, unit) in enumerate(iec_units)
        if endswith(str, unit)
            multiplier = 1024^i
            str = str[1:end-length(unit)]
            return round(UInt, parse(Float64, str) * multiplier)
        end
    end

    return parse(UInt, str)
end

function memory_limits()
  @memoize begin
    soft = if haskey(ENV, "JULIA_CUDA_SOFT_MEMORY_LIMIT")
      parse_limit(ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"])
    else
      UInt(0)
    end

    hard = if haskey(ENV, "JULIA_CUDA_HARD_MEMORY_LIMIT")
      parse_limit(ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"])
    else
      UInt(0)
    end

    (; soft, hard)
  end::NamedTuple{(:soft, :hard), Tuple{UInt,UInt}}
end

function memory_limit_exceeded(bytes::Integer)
  limit = memory_limits()
  limit.hard > 0 || return false

  dev = device()
  used_bytes = if stream_ordered(dev) && driver_version() >= v"12.2"
    # we configured the memory pool to do this for us
    return false
  elseif stream_ordered(dev)
    pool = pool_create(dev)
    Int(attribute(UInt64, pool, MEMPOOL_ATTR_RESERVED_MEM_CURRENT))
  else
    # NOTE: cannot use `memory_info()`, because it only reports total & free memory.
    #       computing `total - free` would include memory allocated by other processes.
    #       NVML does report used memory, but is slow, and not available on all platforms.
    memory_stats().live
  end

  return used_bytes + bytes > limit.hard
end


## stream-ordered memory pool

function stream_ordered(dev::CuDevice)
  @memoize index=deviceid(dev)+1 begin
    CUDACore.driver_version() >= v"11.3" && memory_pools_supported(dev) && pools_enabled()
  end::Bool
end

# whether memory can be freed in stream order, which is also supported for memory that wasn't
# allocated from a pool
function async_free_supported(dev::CuDevice)
  @memoize index=deviceid(dev)+1 begin
    CUDACore.driver_version() >= v"11.3" && memory_pools_supported(dev)
  end::Bool
end

function pool_create(dev::CuDevice)
  @memoize index=deviceid(dev)+1 begin
    limits = memory_limits()

    # create a custom memory pool and assign it to the device
    # so that other libraries and applications will use it.
    pool = if limits.hard > 0 && CUDACore.driver_version() >= v"12.2"
      CuMemoryPool(dev; maxSize=limits.hard)
    else
      CuMemoryPool(dev)
    end
    memory_pool!(dev, pool)

    # allow the pool to use up all memory of this device
    attribute!(pool, MEMPOOL_ATTR_RELEASE_THRESHOLD,
               limits.soft == 0 ? typemax(UInt64) : limits.soft)

    # launch a task to periodically trim the pool
    if isinteractive() && !isassigned(__pool_cleanup)
      __pool_cleanup[] = errormonitor(Threads.@spawn pool_cleanup())
    end

    pool
  end::CuMemoryPool
end

# per-device flag indicating the status of the memory pool.
function pool_mark_ref(dev::CuDevice)
  @memoize index=deviceid(dev)+1 begin
    Ref{Union{Nothing,Bool}}(nothing)
  end::Base.RefValue{Union{Nothing,Bool}}
end
function pool_mark(dev::CuDevice)
  pool_mark_ref(dev)[]
end
function pool_mark!(dev::CuDevice, val)
  pool_mark_ref(dev)[] = val
  return
end

# reclaim unused pool memory after a certain time
const __pool_cleanup = Ref{Task}()
function pool_cleanup()
  idle_counters = Base.fill(0, ndevices())
  while true
    try
      sleep(60)
    catch ex
      if ex isa EOFError
        # If we get EOF here, it's because Julia is shutting down, so we should just exit the loop
        break
      else
        rethrow()
      end
    end

    for (i, dev) in enumerate(devices())
      stream_ordered(dev) || continue

      status = pool_mark(dev)
      status === nothing && continue

      if status
        idle_counters[i] = 0
      else
        idle_counters[i] += 1
      end
      pool_mark!(dev, false)

      if idle_counters[i] == 5
        # the pool hasn't been used for a while, so reclaim unused buffers
        device!(dev) do
          reclaim()
        end
      end
    end
  end
end


## OOM handling

export OutOfGPUMemoryError

struct MemoryInfo
  free_bytes::Int
  total_bytes::Int
  pool_reserved_bytes::Union{Int,Nothing}
  pool_used_bytes::Union{Int,Nothing}

  function MemoryInfo()
    free_bytes, total_bytes = memory_info()

    pool_reserved_bytes, pool_used_bytes = if stream_ordered(device())
      cached_memory(), used_memory()
    else
      nothing, nothing
    end

    new(free_bytes, total_bytes, pool_reserved_bytes, pool_used_bytes)
  end
end

"""
    pool_status([io=stdout])

Report to `io` on the memory status of the current GPU and the active memory pool.
"""
# (release memory that has been garbage collected, so that it isn't reported as being used)
function pool_status(io::IO=stdout, info::MemoryInfo=(drain_retired(); MemoryInfo()))
  state = active_state()
  ctx = context()

  used_bytes = info.total_bytes - info.free_bytes
  used_ratio = used_bytes / info.total_bytes
  @printf(io, "Effective GPU memory usage: %.2f%% (%s/%s)\n",
              100*used_ratio, Base.format_bytes(used_bytes),
              Base.format_bytes(info.total_bytes))

  if info.pool_reserved_bytes === nothing
    @printf(io, "No memory pool is in use.")
  else
    @printf(io, "Memory pool usage: %s (%s reserved)\n",
                Base.format_bytes(info.pool_used_bytes),
                Base.format_bytes(info.pool_reserved_bytes))

  end

  limits = memory_limits()
  if limits.soft > 0 || limits.hard > 0
    print(io, "Memory limit: ")
    if limits.soft > 0
      print(io, "soft = $(Base.format_bytes(limits.soft))")
    end
    if limits.hard > 0
      if limits.soft > 0
        print(io, ", ")
      end
      print(io, "hard = $(Base.format_bytes(limits.hard))")
    end
    println(io)
  end
end

"""
    OutOfGPUMemoryError()

An operation allocated too much GPU memory for either the system or the memory pool to
handle properly.
"""
struct OutOfGPUMemoryError <: Exception
  sz::Int
  info::Union{Nothing,MemoryInfo}

  function OutOfGPUMemoryError(sz::Integer=0)
    info = if task_local_state() === nothing
      # if this error was triggered before the TLS was initialized, we should not try to
      # fetch memory info as those API calls will just trigger TLS initialization again.
      nothing
    elseif in_oom_ctor[]
      # if we triggered an OOM while trying to construct an OOM object, break the cycle
      nothing
    else
      in_oom_ctor[] = true
      try
        MemoryInfo()
      catch err
        # when extremely close to OOM, just inspecting `memory_info()` may trigger an OOM again
        isa(err, OutOfGPUMemoryError) || rethrow()
        nothing
      finally
        in_oom_ctor[] = false
      end
    end
    new(sz, info)
  end
end
const in_oom_ctor = Ref{Bool}(false)

function Base.showerror(io::IO, err::OutOfGPUMemoryError)
    print(io, "Out of GPU memory")
    if err.sz > 0
      print(io, " trying to allocate $(Base.format_bytes(err.sz))")
    end
    if err.info !== nothing
      println(io)
      pool_status(io, err.info)
    end
end

## reclaim escalation
#
# `Reclaimable`/`register_reclaimable!`/`TaskLocalCache`/`drop!`/`purge!`
# are defined in utils/reclaim.jl. Here we add the ladder that drives them
# along with the allocator-specific sync/trim steps.
#
# Registered `drop!`/`purge!` callbacks must not switch the active device:
# the device is captured once per `reclaim` / `retry_reclaim` call.

"""
    ReclaimLevel

Escalation levels shared by `reclaim(level)` and `retry_reclaim`, from
cheapest to most aggressive:

| Level           | Action                                              |
| :---            | :---                                                |
| `RECLAIM_PURGE` | empty `HandleCache`s (no GC, no sync)               |
| `RECLAIM_SYNC`  | synchronize the device (lets async deallocs finish) |
| `RECLAIM_GC`    | run a full Julia GC, then sync + purge + trim       |
| `RECLAIM_DROP`  | also drop task-local library state, then GC + …     |

`RECLAIM_DROP` clears the calling task's library state — see
[`register_reclaimable!`](@ref). It assumes user-held descriptors/plans
are tied to a context, not to a specific library-handle instance (true
for all libraries CUDA.jl wraps). Live handles the user holds a
reference to are unaffected: the wrapper keeps the raw handle alive.

Steps that don't apply to the current allocator (e.g. trim on a
non-stream-ordered device) are silently skipped.
"""
@enum ReclaimLevel::Int begin
    RECLAIM_PURGE = 0
    RECLAIM_SYNC  = 1
    RECLAIM_GC    = 2
    RECLAIM_DROP  = 3
end


"""
    retry_reclaim(retry_if) do
        # code that may fail due to insufficient GPU memory
    end

Run a block of code repeatedly while `retry_if(ret)` holds for its return
value, escalating one `ReclaimLevel` between attempts. Returns the final
(or most recent) return value of the block.

This is intended for CUDA APIs that allocate outside the pool and report
failure via a status code. It's like `Base.retry`, but works on return
values instead of exceptions for performance reasons.
"""
@inline function retry_reclaim(f, retry_if)
    ret = f()
    retry_if(ret) || return ret
    return retry_reclaim_slow(f, retry_if, ret)
end

@noinline function retry_reclaim_slow(f, retry_if, ret)
    dev = active_state().device
    so = stream_ordered(dev)
    for level in instances(ReclaimLevel)
        reclaim_step(level, dev, so)
        ret = f()
        retry_if(ret) || return ret
    end
    return ret
end

# Each level is a complete reclaim at that aggressiveness — `reclaim(level)`
# just runs the matching step. `retry_reclaim` walks the levels in order to
# bisect on alloc failure. GC.gc(true) drains pending finalizers before
# returning, so the post-GC purge sees caches populated by wrapper finalizers.
function reclaim_step(level::ReclaimLevel, dev::CuDevice, stream_ordered::Bool)
    # memory is also freed asynchronously when not using a pool
    async = async_free_supported(dev)
    drain_retired()
    if level == RECLAIM_PURGE
        foreach_reclaimable(purge!)
    elseif level == RECLAIM_SYNC
        async && device_synchronize()
    elseif level == RECLAIM_GC
        GC.gc(true)
        drain_retired()
        async && device_synchronize()
        foreach_reclaimable(purge!)
        trim_pools(dev, stream_ordered)
    elseif level == RECLAIM_DROP
        foreach_reclaimable(drop!)
        GC.gc(true)
        drain_retired()
        async && device_synchronize()
        foreach_reclaimable(purge!)
        trim_pools(dev, stream_ordered)
    end
    return
end

function trim_pools(dev::CuDevice, stream_ordered::Bool)
    stream_ordered && trim(pool_create(dev))
    for pool in (host_pool(), unified_pool())
        pool === nothing || trim(pool)
    end
end


## managed memory

# to safely use allocated memory across tasks and devices, we don't simply return raw
# memory objects, but wrap them in a manager that ensures synchronization and ownership.

# XXX: immutable with atomic refs?
mutable struct Managed{M}
  const mem::M
  const lock::ReentrantLock

  # which stream is currently using the memory.
  stream::CuStream

  # whether accessing this memory can cause implicit synchronization
  synchronizing::Bool

  # whether there are outstanding operations that haven't been synchronized
  dirty::Bool

  # whether the memory has been captured in a way that would make the dirty bit unreliable
  captured::Bool

  function Managed(mem::AbstractMemory; stream = CUDACore.stream(), synchronizing = true,
                   dirty = true, captured = false)
    # NOTE: memory starts as dirty, because stream-ordered allocations are only
    #       guaranteed to be physically allocated at a synchronization event.
    new{typeof(mem)}(mem, ReentrantLock(), stream, synchronizing, dirty, captured)
  end
end

Base.sizeof(managed::Managed) = sizeof(managed.mem)

# wait for the current owner of memory to finish processing
function synchronize(managed::Managed)
  Base.@lock managed.lock begin
    synchronize(managed.stream)
    managed.dirty = false
  end
end
function maybe_synchronize(managed::Managed)
  Base.@lock managed.lock begin
    if managed.synchronizing && (managed.dirty || managed.captured)
      synchronize(managed)
    end
  end
end

# Transfer stream ownership of an allocation and mark it dirty in anticipation of a
# device-side operation. The caller must hold `managed.lock` until that operation has been
# submitted to `stream`, so the recorded owner cannot become visible before its submission.
function take_ownership!(managed::Managed{M}; state=active_state(),
                         stream::CuStream=state.stream,
                         capturing::Bool=is_capturing(stream)) where {M}
  sizeof(managed) == 0 && return managed

  # accessing memory during stream capture: taint the memory so that we always synchronize
  if capturing
    managed.captured = true
  end

  # accessing memory on another device: ensure the data is ready and accessible
  if M == DeviceMemory && state.context != managed.mem.ctx
    maybe_synchronize(managed)
    source_device = managed.mem.dev

    # enable peer-to-peer access
    if maybe_enable_peer_access(state.device, source_device) != 1
        throw(ArgumentError(
            """cannot take the GPU address of inaccessible device memory.

               You are trying to use memory from GPU $(deviceid(source_device)) on GPU $(deviceid(state.device)).
               P2P access between these devices is not possible; either switch to GPU $(deviceid(source_device))
               by calling `CUDA.device!($(deviceid(source_device)))`, or copy the data to an array allocated on device $(deviceid(state.device))."""))
    end

    # set pool visibility
    # XXX: disabled because of NVIDIA bug #6098762
    #if stream_ordered(source_device)
    #  pool = pool_create(source_device)
    #  access!(pool, state.device, ACCESS_FLAGS_PROT_READWRITE)
    #end
  end

  # accessing memory on another stream: ensure the data is ready and take ownership
  if managed.stream != stream
    maybe_synchronize(managed)
    managed.stream = stream
  end

  # prefetch unified memory as we're likely to use it on the GPU
  if M == UnifiedMemory
    can_prefetch = !capturing
    can_prefetch &= !__pinned(convert(Ptr{Cvoid}, managed.mem), managed.mem.ctx)
    can_prefetch &= attribute(state.device,
                              DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS) == 1
    can_prefetch &= ndevices() == 1
    can_prefetch && prefetch(managed.mem; device=state.device, stream)
  end

  managed.dirty = true
  return managed
end

function Base.convert(::Type{CuPtr{T}}, managed::Managed{M}) where {T,M}
  Base.@lock managed.lock begin
    # let null pointers pass through as-is
    ptr = convert(CuPtr{T}, managed.mem)
    ptr == CU_NULL && return ptr

    state = active_state()
    take_ownership!(managed; state, stream=state.stream)
    return ptr
  end
end

function Base.convert(::Type{Ptr{T}}, managed::Managed{M}) where {T,M}
  Base.@lock managed.lock begin
    # let null pointers pass through as-is
    ptr = convert(Ptr{T}, managed.mem)
    ptr == C_NULL && return ptr

    # accessing memory on the CPU: only allowed for host or unified allocations
    if M == DeviceMemory
      throw(ArgumentError(
          """cannot take the CPU address of GPU memory.

             You are probably falling back to or otherwise calling CPU functionality
             with GPU array inputs. This is not supported by regular device memory;
             ensure this operation is supported by CUDA.jl, and if it isn't, try to
             avoid it or rephrase it in terms of supported operations. Alternatively,
             you can consider using GPU arrays backed by unified memory by
             allocating using `cu(...; unified=true)`."""))
    end

    # make sure any work on the memory has finished.
    maybe_synchronize(managed)
    return ptr
  end
end


## retirement of memory freed by finalizers
#
# finalizers run on whatever thread triggers a collection, or releases a lock while
# finalizers are pending, and cannot switch tasks. releasing memory there could block that
# thread, as some driver calls wait for unrelated GPU work to finish (also blocking kernel
# launches from other threads in the meantime). if that work depends on a task on this
# thread, that would deadlock. so finalizers only retire resources, pushing them onto a
# lock-free list, and regular tasks dispose of them.

mutable struct Retired
  const resource::Any
  next::Union{Nothing,Retired}
end

mutable struct RetiredList
  Base.@atomic head::Union{Nothing,Retired}
end
const retired_memory = RetiredList(nothing)

function push_retired!(first::Retired, last::Retired)
  head = Base.@atomic :monotonic retired_memory.head
  while true
    last.next = head
    head, ok = Base.@atomicreplace :release :monotonic retired_memory.head head => first
    ok && return
  end
end

# can be called from a finalizer: doesn't switch tasks, take locks or call into CUDA.
# `dispose(resource)` is called later on, by a regular task.
function retire!(resource)
  node = Retired(resource, nothing)
  push_retired!(node, node)
  return
end

# drains detach the list, and process it holding a lock, so that a bounded drain can leave
# the rest for a later one (in `retired_backlog`) without having to put it back.
const drain_lock = ReentrantLock()
const retired_backlog = RetiredList(nothing)

"""
    drain_retired([limit])

Dispose of (at most `limit`) resources that have been retired by finalizers, e.g., making
memory available to future allocations. Returns the number of resources disposed of.
"""
function drain_retired(limit::Int=typemax(Int))
  GC.in_finalizer() && return 0
  if (Base.@atomic :monotonic retired_memory.head) === nothing &&
     (Base.@atomic :monotonic retired_backlog.head) === nothing
    pending_owner_count[] == 0 && return 0
  end

  # exhaustive drains wait for others to finish, bounded ones leave the work to them
  if limit == typemax(Int)
    lock(drain_lock)
  else
    trylock(drain_lock) || return 0
  end
  n = 0
  try
    while n < limit
      node = Base.@atomic :monotonic retired_backlog.head
      if node === nothing
        node = Base.@atomicswap :acquire retired_memory.head = nothing
        node === nothing && break
      end
      # (update the backlog first, as disposing may drain recursively)
      Base.@atomic :monotonic retired_backlog.head = node.next
      try
        dispose(node.resource)
      catch err
        @error "Failed to dispose of a $(typeof(node.resource))" exception=(err, catch_backtrace())
      end
      n += 1
    end
    release_owners!()
  finally
    unlock(drain_lock)
  end
  return n
end

# drain periodically, so that memory gets released when the application stops using CUDA
const retired_drainer = Threads.Atomic{Bool}(false)
function start_retired_drainer()
  Threads.atomic_cas!(retired_drainer, false, true) && return
  Timer(1; interval=1) do _
    try
      drain_retired()
    catch err
      @error "Failed to release retired GPU resources" exception=(err, catch_backtrace())
    end
  end
  return
end

# how many retired blocks to release when allocating, to bound the latency of an allocation
const ALLOC_DRAIN_LIMIT = 256

dispose(event::CuEvent) = unsafe_destroy!(event)

# memory that was last used on a stream may be retired after that stream, and released
# after the stream has been destroyed. it then needs to wait for the work that was submitted
# to the stream, which is captured by an event recorded before destroying it.
const stream_disposal_lock = ReentrantLock()
function dispose(stream::CuStream)
  @lock stream_disposal_lock begin
    isvalid(stream) || return
    context!(stream.ctx) do
      event = CuEvent(EVENT_DISABLE_TIMING)
      record(event, stream)
      stream.final_event = event
      cuStreamDestroy_v2(stream)
    end
    Base.@atomic stream.valid = false
  end
  return
end

# memory released after its stream was destroyed is released on a non-blocking stream that
# waits for the final event of that stream. (the legacy default stream would make other
# streams wait as well, which could deadlock.) it is created along with the first stream of
# each context, so that releasing memory doesn't create streams (which can block).
const disposal_streams = Dict{CuContext,CUstream}()
const disposal_streams_lock = ReentrantLock()
function disposal_stream(ctx::CuContext)
  @lock disposal_streams_lock get!(disposal_streams, ctx) do
    context!(ctx) do
      handle = Ref{CUstream}()
      cuStreamCreate(handle, STREAM_NON_BLOCKING)
      handle[]
    end
  end
end

# a stream to release memory on that was last used on `stream`, along with its context.
# needs to be called with `stream_disposal_lock` held.
function release_stream(stream::CuStream, ctx::CuContext)
  if isvalid(stream)
    return stream.handle, something(stream.ctx, ctx)
  end
  event = stream.final_event
  if event === nothing
    # explicitly destroyed, so the user should have made sure the work has finished
    return disposal_stream(ctx), ctx
  end
  handle = disposal_stream(event.ctx)
  context!(event.ctx) do
    cuStreamWaitEvent(handle, event, 0)
  end
  return handle, event.ctx
end

# make the disposal stream wait for the work on `stream`
function after_on_disposal_stream(stream::CUstream, ctx::CuContext)
  disposal = disposal_stream(ctx)
  stream == disposal && return disposal
  event = CuEvent(EVENT_DISABLE_TIMING)
  cuEventRecord(event, stream)
  cuStreamWaitEvent(disposal, event, 0)
  return disposal
end

# the per-thread default stream is specific to the thread that used it, which isn't known
# when releasing memory, so memory last used on it is only released when memory is
# reclaimed, after synchronizing the device.
on_per_thread_stream(stream::CuStream) = stream.handle == CU_STREAM_PER_THREAD
function synchronize_and(f, ctx::CuContext)
  return x -> begin
    context!(device_synchronize, ctx)
    f(x)
  end
end


## objects that need to be kept alive while the GPU uses them

struct RetiredOwner
  owner::Any
  managed::Managed
end

# owners whose memory might still be in use, along with an event signalling it isn't
const pending_owners = Tuple{CuEvent,Any}[]
const pending_owners_lock = ReentrantLock()
const pending_owner_count = Ref(0)

release_owner(owner, managed::Managed) =
  GC.in_finalizer() ? retire!(RetiredOwner(owner, managed)) :
                      dispose(RetiredOwner(owner, managed))

function dispose(retired::RetiredOwner)
  stream = retired.managed.stream
  ctx = retired.managed.mem.ctx
  if on_per_thread_stream(stream)
    destroy_later(synchronize_and(identity, ctx), retired.owner)
    return
  end
  event = @lock stream_disposal_lock begin
    handle, ctx = release_stream(stream, ctx)
    context!(ctx) do
      event = CuEvent(EVENT_DISABLE_TIMING)
      cuEventRecord(event, handle)
      event
    end
  end
  @lock pending_owners_lock begin
    push!(pending_owners, (event, retired.owner))
    pending_owner_count[] = length(pending_owners)
  end
  return
end

function release_owners!()
  pending_owner_count[] == 0 && return
  @lock pending_owners_lock begin
    filter!(((event, owner),) -> !isdone(event), pending_owners)
    pending_owner_count[] = length(pending_owners)
  end
  return
end


## public interface

"""
    pool_alloc([DeviceMemory], sz)::Managed{<:AbstractMemory}

Allocate a number of bytes `sz` from the memory pool on the current stream. Returns a
managed memory object; may throw an [`OutOfGPUMemoryError`](@ref) if the allocation request
cannot be satisfied.
"""
@inline pool_alloc(sz::Integer) = pool_alloc(DeviceMemory, sz)
@inline function pool_alloc(::Type{B}, sz) where {B<:AbstractMemory}
  # 0-byte allocations shouldn't hit the pool
  sz == 0 && return Managed(B())

  drain_retired(ALLOC_DRAIN_LIMIT)
  maybe_collect()
  time = Base.@elapsed begin
    mem = _pool_alloc(B, sz)
  end

  Base.@atomic alloc_stats.alloc_count += 1
  Base.@atomic alloc_stats.alloc_bytes += sz
  Base.@atomic alloc_stats.total_time += time
  # NOTE: total_time might be an over-estimation if we trigger GC somewhere else

  return Managed(mem)
end
@inline function _pool_alloc(::Type{DeviceMemory}, sz)
    state = active_state()

    mem = if stream_ordered(state.device)
      pool_mark!(state.device, true)
      pool = pool_create(state.device)

      retry_reclaim(isnothing) do
        memory_limit_exceeded(sz) && return nothing

        # try the actual allocation
        try
          alloc(DeviceMemory, sz; async=true, state.stream, pool)
        catch err
          isa(err, OutOfGPUMemoryError) || rethrow()
          return nothing
        end
      end
    else
      retry_reclaim(isnothing) do
        memory_limit_exceeded(sz) && return nothing

        # try the actual allocation
        try
          alloc(DeviceMemory, sz; async=false)
        catch err
          isa(err, OutOfGPUMemoryError) || rethrow()
          return nothing
        end
      end
    end
    # NOTE: the `retry_reclaim` body is duplicated to work around
    #       closure capture issues with the `pool` variable
    mem === nothing && throw(OutOfGPUMemoryError(sz))

    account!(memory_stats(state.device), sz)

    mem
end
@inline function _pool_alloc(::Type{UnifiedMemory}, sz)
  # NOTE: no `retry_reclaim` here. `cuMemAllocManaged` allocates lazily and
  # essentially never returns `ERROR_OUT_OF_MEMORY` — when host RAM is actually
  # exhausted, the OS kills the process on the page fault before the driver
  # call can fail. The only thing that can prevent OOM is the proactive
  # `maybe_collect` call in `pool_alloc`, which uses `_host_stats`.
  mem = alloc_unified(sz)
  account!(_host_stats, sizeof(mem))
  mem
end
@inline function _pool_alloc(::Type{HostMemory}, sz)
  mem = alloc_host(sz)
  account!(_host_stats, sizeof(mem))
  mem
end

"""
    pool_free(mem::Managed{<:AbstractMemory})

Releases memory to the pool. If possible, this operation will not block but will be ordered
against the stream that last used the memory.

When called from a finalizer, the memory is only retired, and released later by a regular
task (see [`drain_retired`](@ref)).
"""
@inline function pool_free(managed::Managed{<:AbstractMemory})
  # 0-byte allocations shouldn't hit the pool
  sizeof(managed.mem) == 0 && return

  if GC.in_finalizer()
    retire!(managed)
  else
    dispose(managed)
  end
  return
end

function dispose(managed::Managed)
  if on_per_thread_stream(managed.stream)
    destroy_later(synchronize_and(dispose_now, managed.mem.ctx), managed)
    return
  end
  dispose_now(managed)
end

function dispose_now(managed::Managed)
  Base.@lock managed.lock begin
    mem = managed.mem
    sz = sizeof(mem)

    # this function is called when draining retired memory, e.g. before allocating, where a
    # failure to free memory shouldn't be fatal, so perform our own error handling.
    try
      time = Base.@elapsed _pool_free(mem, managed.stream)

      Base.@atomic alloc_stats.free_count += 1
      Base.@atomic alloc_stats.free_bytes += sz
      Base.@atomic alloc_stats.total_time += time
    catch ex
      # NOTE: avoid `show`ing `mem` here since the buffer may be in a bad state
      # (often the reason free is failing); printing the byte count is safer.
      Base.showerror_nostdio(ex,
          "WARNING: Error while freeing $(Base.format_bytes(sz)) of GPU memory")
      Base.show_backtrace(Core.stdout, catch_backtrace())
      Core.println()
    end
  end

  return
end
@inline function _pool_free(mem::DeviceMemory, stream::CuStream)
    if mem.async || async_free_supported(mem.dev)
      # free in stream order. `cuMemFree` would wait for all work on the device to finish,
      # blocking kernel launches from other threads in the meantime. that also works for
      # memory that wasn't allocated from a pool.
      @lock stream_disposal_lock begin
        stream, ctx = release_stream(stream, mem.ctx)
        context!(ctx) do
          if !mem.async
            # when memory that wasn't allocated from a pool is freed in stream order,
            # destroying that stream waits for the free, i.e., for all work on the stream.
            # so free it on the disposal stream instead, after the work on this stream.
            stream = after_on_disposal_stream(stream, ctx)
          end
          cuMemFreeAsync(mem, stream)
        end
      end
    else
      # without support for freeing memory in stream order, `cuMemFree` is the only option,
      # which waits for all running kernels. defer that until memory is reclaimed.
      destroy_later(mem) do mem
        context!(mem.ctx) do
          free(mem)
        end
      end
    end
    account!(memory_stats(mem.dev), -sizeof(mem))
end
@inline function _pool_free(mem::Union{UnifiedMemory,HostMemory}, stream::CuStream)
  if mem.pooled
    @lock stream_disposal_lock begin
      stream, ctx = release_stream(stream, mem.ctx)
      context!(ctx) do
        cuMemFreeAsync(convert(CuPtr{Cvoid}, mem), stream)
      end
    end
  else
    cache_put!(mem isa HostMemory ? host_cache : unified_cache, mem, stream)
  end
  account!(_host_stats, -sizeof(mem))
end


## pinned host and unified memory
#
# freeing such memory with `cuMemFreeHost` or `cuMemFree` waits for all running kernels to
# finish, also blocking kernel launches from other threads in the meantime. where supported,
# it is allocated from memory pools instead, which can be freed in stream order without
# waiting. otherwise, freed allocations are cached for reuse, and only released when
# reclaiming memory, or when the cache grows too large.

function try_create_pool(f)
  try
    f()
  catch err
    isa(err, CuError) || rethrow()
    @debug "Could not create a memory pool" exception=(err, catch_backtrace())
    nothing
  end
end

pools_enabled() = get(ENV, "JULIA_CUDA_MEMORY_POOL", "cuda") == "cuda"

# a pool of pinned host memory, accessible from all devices
function host_pool()
  @memoize begin
    supported = driver_version() >= v"13.0" && pools_enabled() &&
                all(devices()) do dev
                  attribute(dev, DEVICE_ATTRIBUTE_HOST_MEMORY_POOLS_SUPPORTED) == 1
                end
    supported ? try_create_pool() do
      pool = CuMemoryPool(device(); location_type=CU_MEM_LOCATION_TYPE_HOST, location_id=0)
      access!(pool, collect(devices()), ACCESS_FLAGS_PROT_READWRITE)
      pool
    end : nothing
  end::Union{Nothing,CuMemoryPool}
end

# a pool of unified memory, without a preferred location
function unified_pool()
  @memoize begin
    supported = driver_version() >= v"13.0" && pools_enabled() &&
                all(devices()) do dev
                  memory_pools_supported(dev) &&
                    attribute(dev, DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS) == 1
                end
    supported ? try_create_pool() do
      CuMemoryPool(device(); alloc_type=CU_MEM_ALLOCATION_TYPE_MANAGED,
                   location_type=CU_MEM_LOCATION_TYPE_NONE, location_id=0)
    end : nothing
  end::Union{Nothing,CuMemoryPool}
end

function alloc_from_pool(pool::CuMemoryPool, sz, stream::CuStream)
  ptr = retry_reclaim(isnothing) do
    ref = Ref{CUdeviceptr}()
    res = unchecked_cuMemAllocFromPoolAsync(ref, sz, pool, stream)
    res == ERROR_OUT_OF_MEMORY && return nothing
    res == SUCCESS || throw_api_error(res)
    ref[]
  end
  ptr === nothing && throw(OutOfGPUMemoryError(sz))
  return ptr
end

# allocations that cannot be freed without blocking, kept for reuse by later allocations
struct CachedBlock{M}
  mem::M
  # recorded after the last use of the memory
  idle::CuEvent
end

mutable struct BlockCache{M} <: Reclaimable
  const lock::ReentrantLock
  const blocks::Dict{Tuple{CuContext,Int},Vector{CachedBlock{M}}}
  Base.@atomic bytes::Int
end
BlockCache{M}() where {M} =
  BlockCache{M}(ReentrantLock(), Dict{Tuple{CuContext,Int},Vector{CachedBlock{M}}}(), 0)

const host_cache = BlockCache{HostMemory}()
const unified_cache = BlockCache{UnifiedMemory}()

# allocations are rounded up, so that cached ones can be reused for similar sizes
cached_size(sz) = sz <= 1<<20 ? nextpow(2, max(sz, 512)) : cld(sz, 1<<20) << 20

# how many bytes a cache may hold before allocating releases cached memory. on devices that
# share memory with the CPU, cached memory competes with everything else.
cache_limit() = Sys.total_memory() ÷ 20

function cache_put!(cache::BlockCache{M}, mem::M, stream::CuStream) where {M}
  idle = @lock stream_disposal_lock begin
    stream, ctx = release_stream(stream, mem.ctx)
    # (the event needs to be created in the context of the stream it is recorded on)
    context!(ctx) do
      event = CuEvent(EVENT_DISABLE_TIMING)
      cuEventRecord(event, stream)
      event
    end
  end
  @lock cache.lock begin
    blocks = get!(Vector{CachedBlock{M}}, cache.blocks, (mem.ctx, sizeof(mem)))
    push!(blocks, CachedBlock(mem, idle))
    Base.@atomic cache.bytes += sizeof(mem)
  end
  return
end

# take a cached allocation that is not in use anymore
function cache_take!(cache::BlockCache{M}, ctx::CuContext, sz::Int) where {M}
  block = @lock cache.lock begin
    blocks = get(cache.blocks, (ctx, sz), nothing)
    blocks === nothing && return nothing
    i = findlast(block -> isdone(block.idle), blocks)
    i === nothing && return nothing
    Base.@atomic cache.bytes -= sz
    popat!(blocks, i)
  end
  block === nothing && return nothing
  return block.mem
end

# release cached allocations. this blocks, also kernel launches from other threads, until
# all running kernels have finished, so only do so when reclaiming memory.
function cache_release!(cache::BlockCache, bytes::Int=typemax(Int))
  released = 0
  while released < bytes
    block = @lock cache.lock begin
      key = findfirst(!isempty, cache.blocks)
      if key === nothing
        nothing
      else
        block = pop!(cache.blocks[key])
        Base.@atomic cache.bytes -= sizeof(block.mem)
        block
      end
    end
    block === nothing && break
    synchronize(block.idle)
    context!(block.mem.ctx) do
      free(block.mem)
    end
    released += sizeof(block.mem)
  end
  return released
end
purge!(cache::BlockCache) = (cache_release!(cache); nothing)

# before allocating more memory, release cached memory if the cache has grown too large
function maybe_release!(cache::BlockCache)
  limit = cache_limit()
  bytes = Base.@atomic cache.bytes
  bytes > limit && cache_release!(cache, bytes - limit ÷ 2)
  return
end

function alloc_host(sz)
  state = active_state()
  pool = host_pool()
  if pool !== nothing
    ptr = alloc_from_pool(pool, sz, state.stream)
    return HostMemory(state.context, reinterpret(Ptr{Cvoid}, ptr), sz, true)
  end

  sz = cached_size(sz)
  mem = cache_take!(host_cache, state.context, sz)
  mem === nothing || return mem
  maybe_release!(host_cache)
  return alloc(HostMemory, sz)
end

function alloc_unified(sz)
  state = active_state()
  pool = unified_pool()
  if pool !== nothing
    ptr = alloc_from_pool(pool, sz, state.stream)
    return UnifiedMemory(state.context, reinterpret(CuPtr{Cvoid}, ptr), sz, true)
  end

  sz = cached_size(sz)
  mem = cache_take!(unified_cache, state.context, sz)
  if mem !== nothing
    reset_advice!(mem)
    return mem
  end
  maybe_release!(unified_cache)
  return alloc(UnifiedMemory, sz)
end

# a reused allocation shouldn't keep advice given for its previous use. not all advice is
# supported everywhere, so ignore errors.
function reset_advice!(mem::UnifiedMemory)
  unchecked_cuMemAdvise(mem, sizeof(mem), MEM_ADVISE_UNSET_READ_MOSTLY, CU_DEVICE_CPU)
  unchecked_cuMemAdvise(mem, sizeof(mem), MEM_ADVISE_UNSET_PREFERRED_LOCATION, CU_DEVICE_CPU)
  unchecked_cuMemAdvise(mem, sizeof(mem), MEM_ADVISE_UNSET_ACCESSED_BY, CU_DEVICE_CPU)
  for dev in devices()
    unchecked_cuMemAdvise(mem, sizeof(mem), MEM_ADVISE_UNSET_ACCESSED_BY, dev.handle)
  end
end

## registered host memory
#
# unregistering host memory with `cuMemHostUnregister` waits for all running kernels to
# finish, also blocking kernel launches from other threads in the meantime. that is deferred
# until reclaiming memory, or until many registrations are pending, keeping the owner of the
# memory alive in the meantime so that it cannot be reused while still registered.

struct RetiredRegistration
  mem::HostMemory
  # registered by `__pin`, which counts registrations
  counted::Bool
  # what keeps the memory alive (if anything)
  owner::Any
end

mutable struct PendingRegistrations <: Reclaimable
  const lock::ReentrantLock
  const registrations::Vector{RetiredRegistration}
  Base.@atomic bytes::Int
end
const pending_registrations =
  PendingRegistrations(ReentrantLock(), RetiredRegistration[], 0)

function dispose(reg::RetiredRegistration)
  @lock pending_registrations.lock push!(pending_registrations.registrations, reg)
  Base.@atomic pending_registrations.bytes += sizeof(reg.mem)
  return
end

release_registration(reg::RetiredRegistration) =
  GC.in_finalizer() ? retire!(reg) : dispose(reg)

# unregister pending registrations. this blocks, also kernel launches from other threads,
# until all running kernels have finished, so only do so when reclaiming memory.
function unregister_pending!()
  drain_retired()
  regs = @lock pending_registrations.lock begin
    regs = copy(pending_registrations.registrations)
    empty!(pending_registrations.registrations)
    regs
  end
  for reg in regs
    mem = reg.mem
    try
      if reg.counted
        __unpin(pointer(mem), mem.ctx)
      else
        context!(mem.ctx) do
          unregister(mem)
        end
      end
      Base.@atomic pending_registrations.bytes -= sizeof(mem)
    catch err
      # the memory may still be registered, so keep its owner alive
      @error "Failed to unregister host memory" exception=(err, catch_backtrace())
      @lock pending_registrations.lock push!(pending_registrations.registrations, reg)
    end
  end
  return
end
purge!(::PendingRegistrations) = unregister_pending!()

function register_host_memory(ptr::Ptr, sz::Integer, flags=0)
  # before registering more memory, unregister pending registrations if there are many
  drain_retired(ALLOC_DRAIN_LIMIT)
  if (Base.@atomic pending_registrations.bytes) > cache_limit()
    unregister_pending!()
  end

  try
    register(HostMemory, ptr, sz, flags)
  catch err
    # the memory may still be registered, e.g., when the memory backing an array that was
    # wrapped by pointer has been freed and reused.
    (err isa CuError && err.code == ERROR_HOST_MEMORY_ALREADY_REGISTERED) || rethrow()
    unregister_pending!()
    register(HostMemory, ptr, sz, flags)
  end
end


## deferred destruction
#
# some objects (e.g., modules, texture arrays, and library handles or plans) can only be
# destroyed with calls that wait for all running kernels to finish, sometimes also blocking
# kernel launches from other threads. their destruction is deferred until memory is
# reclaimed, e.g., when running out of memory or when calling `reclaim()`.

struct DeferredDestruction
  f::Any
  obj::Any
end

mutable struct PendingDestructions <: Reclaimable
  const lock::ReentrantLock
  const items::Vector{DeferredDestruction}
end
const pending_destructions = PendingDestructions(ReentrantLock(), DeferredDestruction[])

"""
    CUDACore.destroy_later(f, obj)

Destroy `obj` by calling `f(obj)` when memory is reclaimed, instead of right away. This is
meant for objects whose destruction may wait for running kernels to finish, and can be used
from a finalizer, e.g., `finalizer(obj -> destroy_later(unsafe_destroy!, obj), obj)`.
"""
function destroy_later(f, obj)
  destruction = DeferredDestruction(f, obj)
  GC.in_finalizer() ? retire!(destruction) : dispose(destruction)
  return
end

function dispose(destruction::DeferredDestruction)
  @lock pending_destructions.lock push!(pending_destructions.items, destruction)
  return
end

function purge!(pending::PendingDestructions)
  drain_retired()
  items = @lock pending.lock begin
    items = copy(pending.items)
    empty!(pending.items)
    items
  end
  for destruction in items
    try
      # the destructor may have been defined after this code was compiled
      Base.invokelatest(destruction.f, destruction.obj)
    catch err
      @error "Failed to destroy $(typeof(destruction.obj))" exception=(err, catch_backtrace())
    end
  end
  return
end


"""
    reclaim([level::ReclaimLevel = RECLAIM_DROP])

Free GPU memory at the given [`ReclaimLevel`](@ref). The default drops
task-local library state, runs a full GC so handle wrappers finalize and
return their raw handles to caches, then destroys those caches and trims
the pool. Returns `nothing`.
"""
function reclaim(level::ReclaimLevel = RECLAIM_DROP)
    dev = active_state().device
    reclaim_step(level, dev, stream_ordered(dev))
    return
end


## utilities

"""
    @allocated

A macro to evaluate an expression, discarding the resulting value, instead returning the
total number of bytes allocated during evaluation of the expression.
"""
macro allocated(ex)
    quote
        let
            local f
            function f()
                b0 = alloc_stats.alloc_bytes
                $(esc(ex))
                alloc_stats.alloc_bytes - b0
            end
            f()
        end
    end
end

"""
    @time ex

Run expression `ex` and report on execution time and GPU/CPU memory behavior. The GPU is
synchronized right before and after executing `ex` to exclude any external effects.
"""
macro time(ex)
    quote
        local val, cpu_time,
            cpu_alloc_size, cpu_gc_time, cpu_mem_stats,
            gpu_alloc_size, gpu_mem_time, gpu_mem_stats = @timed $(esc(ex))

        local cpu_alloc_count = Base.gc_alloc_count(cpu_mem_stats)
        local gpu_alloc_count = gpu_mem_stats.alloc_count

        Printf.@printf("%10.6f seconds", cpu_time)
        for (typ, gctime, memtime, bytes, allocs) in
            (("CPU", cpu_gc_time, 0, cpu_alloc_size, cpu_alloc_count),
             ("GPU", 0, gpu_mem_time, gpu_alloc_size, gpu_alloc_count))
          if bytes != 0 || allocs != 0
              allocs, ma = Base.prettyprint_getunits(allocs, length(Base._cnt_units), Int64(1000))
              if ma == 1
                  Printf.@printf(" (%d%s %s allocation%s: ", allocs, Base._cnt_units[ma], typ, allocs==1 ? "" : "s")
              else
                  Printf.@printf(" (%.2f%s %s allocations: ", allocs, Base._cnt_units[ma], typ)
              end
              print(Base.format_bytes(bytes))
              if gctime > 0
                  Printf.@printf(", %.2f%% gc time", 100*gctime/cpu_time)
              end
              if memtime > 0
                  Printf.@printf(", %.2f%% memmgmt time", 100*memtime/cpu_time)
              end
              print(")")
          else
              if gctime > 0
                  Printf.@printf(", %.2f%% %s gc time", 100*gctime/cpu_time, typ)
              end
              if memtime > 0
                  Printf.@printf(", %.2f%% %s memmgmt time", 100*memtime/cpu_time, typ)
              end
          end
        end
        println()

        val
    end
end

macro timed(ex)
    quote
        Base.Experimental.@force_compile

        # coars-graned synchronization to exclude effects from previously-executed code
        device_synchronize()

        local gpu_mem_stats0 = copy(alloc_stats)
        local cpu_mem_stats0 = Base.gc_num()
        local cpu_time0 = time_ns()

        # fine-grained synchronization of the code under analysis
        local val = @sync $(esc(ex))

        local cpu_time1 = time_ns()
        local cpu_mem_stats1 = Base.gc_num()
        local gpu_mem_stats1 = copy(alloc_stats)

        local cpu_time = (cpu_time1 - cpu_time0) / 1e9

        local cpu_mem_stats = Base.GC_Diff(cpu_mem_stats1, cpu_mem_stats0)
        local gpu_mem_stats = gpu_mem_stats1 - gpu_mem_stats0

        (value=val, time=cpu_time,
         cpu_bytes=cpu_mem_stats.allocd, cpu_gctime=cpu_mem_stats.total_time / 1e9, cpu_gcstats=cpu_mem_stats,
         gpu_bytes=gpu_mem_stats.alloc_bytes, gpu_memtime=gpu_mem_stats.total_time, gpu_memstats=gpu_mem_stats)
    end
end
@public @allocated, @time, @timed, used_memory, cached_memory, pool_status, reclaim,
        RECLAIM_PURGE, RECLAIM_SYNC, RECLAIM_GC, RECLAIM_DROP

"""
    used_memory()

Returns the amount of memory from the CUDA memory pool that is currently in use by the
application.
"""
function used_memory()
  # not using `active_state()`, which creates the task's stream. that can block until the
  # GPU is idle, while this is called right before synchronizing (by `maybe_collect`).
  dev = device()
  if stream_ordered(dev)
    pool = pool_create(dev)
    Int(attribute(UInt64, pool, MEMPOOL_ATTR_USED_MEM_CURRENT))
  else
    missing
  end
end

"""
    cached_memory()

Returns the amount of backing memory currently allocated for the CUDA memory pool.
"""
function cached_memory()
  # not using `active_state()`, which creates the task's stream. that can block until the
  # GPU is idle, while this is called right before synchronizing (by `maybe_collect`).
  dev = device()
  if stream_ordered(dev)
    pool = pool_create(dev)
    Int(attribute(UInt64, pool, MEMPOOL_ATTR_RESERVED_MEM_CURRENT))
  else
    missing
  end
end
