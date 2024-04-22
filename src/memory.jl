# high-level memory management

using Printf
using Logging


## allocation statistics

mutable struct AllocStats
  alloc_count::Threads.Atomic{Int}
  alloc_bytes::Threads.Atomic{Int}

  free_count::Threads.Atomic{Int}
  free_bytes::Threads.Atomic{Int}

  total_time::Threads.Atomic{Float64}

  function AllocStats()
    new(Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
        Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
        Threads.Atomic{Float64}(0.0))
  end

  function AllocStats(alloc_count::Integer, alloc_bytes::Integer,
                      free_count::Integer, free_bytes::Integer,
                      total_time::Float64)
    new(Threads.Atomic{Int}(alloc_count), Threads.Atomic{Int}(alloc_bytes),
        Threads.Atomic{Int}(free_count), Threads.Atomic{Int}(free_bytes),
        Threads.Atomic{Float64}(total_time))
  end
end

Base.copy(alloc_stats::AllocStats) =
  AllocStats(alloc_stats.alloc_count[], alloc_stats.alloc_bytes[],
             alloc_stats.free_count[], alloc_stats.free_bytes[],
             alloc_stats.total_time[])

Base.:(-)(a::AllocStats, b::AllocStats) = (;
    alloc_count = a.alloc_count[] - b.alloc_count[],
    alloc_bytes = a.alloc_bytes[] - b.alloc_bytes[],
    free_count  = a.free_count[]  - b.free_count[],
    free_bytes  = a.free_bytes[]  - b.free_bytes[],
    total_time  = a.total_time[]  - b.total_time[])

const alloc_stats = AllocStats()


## memory accounting

struct MemoryStats
  # maximum size of the memory heap
  size::Threads.Atomic{Int}
  size_updated::Threads.Atomic{Float64}

  # the amount of live bytes
  live::Threads.Atomic{Int}

  last_time::Threads.Atomic{Float64}
  last_gc_time::Threads.Atomic{Float64}
  last_freed::Threads.Atomic{Int}
end

function account!(stats::MemoryStats, bytes::Integer)
  Threads.atomic_add!(stats.live, bytes)
  if bytes > 0
    Threads.atomic_add!(stats.live, 1)
  end
end

const _memory_stats = PerDevice{MemoryStats}()
function memory_stats(dev::CuDevice=device())
  get!(_memory_stats, dev) do
      MemoryStats(Threads.Atomic{Int}(0), Threads.Atomic{Float64}(0.0),
                  Threads.Atomic{Int}(0), Threads.Atomic{Float64}(0.0),
                  Threads.Atomic{Float64}(0.0), Threads.Atomic{Int}(0),)
  end
end

const _early_gc = LazyInitialized{Bool}()
function maybe_collect(will_block::Bool=false)
  enabled = get!(_early_gc) do
    parse(Bool, get(ENV, "JULIA_CUDA_GC_EARLY", "true"))
  end
  enabled || return
  stats = memory_stats()
  current_time = time()

  # periodically re-estimate the amount of memory available to this process.
  if current_time - stats.size_updated[] > 10
    limits = memory_limits()
    stats.size[] = if limits.hard > 0
      limits.hard
    elseif limits.soft > 0
      limits.soft
    else
      size = free_memory() + stats.live[]
      # NOTE: we use stats.live[] so that we only count memory allocated here, ensuring
      #       the pressure calculation below reflects the heap we have control over.

      # also include reserved bytes
      dev = device()
      if stream_ordered(dev)
        size += cached_memory() - used_memory()
      end

      size
    end
    stats.size_updated[] = current_time
  end

  # check that we're under memory pressure
  pressure = stats.live[] / stats.size[]
  min_pressure = 0.75
  ## if we're about to block anyway, now may be a good time for a GC pause
  if will_block
    min_pressure = 0.50
  end
  if pressure < min_pressure
    return
  end

  # ensure we don't collect too often by checking the GC rate
  last_time = stats.last_time[]
  gc_rate = stats.last_gc_time[] / (current_time - last_time)
  ## we tolerate 5% GC time
  max_gc_rate = 0.05
  ## if we freed a lot last time, bump that up
  if stats.last_freed[] > 0.1*stats.size[]
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
  stats.last_time[] = current_time

  # finally, call the GC
  pre_gc_live = stats.live[]
  gc_time = @elapsed GC.gc(false)
  post_gc_live = stats.live[]
  memory_freed = pre_gc_live - post_gc_live
  stats.last_freed[] = memory_freed
  ## GC times can vary, so smooth them out
  stats.last_gc_time[] = 0.75*stats.last_gc_time[] + 0.25*gc_time

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
    pool = memory_pool(dev)
    Int(attribute(UInt64, pool, MEMPOOL_ATTR_RESERVED_MEM_CURRENT))
  else
    # NOTE: cannot use `memory_info()`, because it only reports total & free memory.
    #       computing `total - free` would include memory allocated by other processes.
    #       NVML does report used memory, but is slow, and not available on all platforms.
    memory_stats().live[]
  end

  return used_bytes + bytes > limit.hard
end


## stream-ordered memory pool

# TODO: extract this into a @device_memoize macro, or teach @memoize about CuDevice?
#       this is a common pattern that could be applied to many more functions.
function stream_ordered(dev::CuDevice)
  devidx = deviceid(dev) + 1
  @memoize devidx::Int maxlen=ndevices() begin
    CUDA.driver_version() >= v"11.3" && memory_pools_supported(dev) &&
    get(ENV, "JULIA_CUDA_MEMORY_POOL", "cuda") == "cuda"
  end::Bool
end

# per-device flag indicating the status of a pool.
# `nothing` indicates the pool hasn't been initialized,
# `false` indicates the pool is idle, and `true` indicates it is active.
const _pool_status = PerDevice{Base.RefValue{Bool}}()
function pool_status(dev::CuDevice)
  status = get(_pool_status, dev, nothing)
  status === nothing && return nothing
  return status[]
end
function pool_status!(dev::CuDevice, val)
  box = get!(_pool_status, dev) do
    # nothing=uninitialized, false=idle, true=active
    Ref{Bool}()
  end
  box[] = val
  return
end
function pool_mark(dev::CuDevice)
  if pool_status(dev) === nothing
      limits = memory_limits()

      # create a custom memory pool and assign it to the device
      # so that other libraries and applications will use it.
      pool = if limits.hard > 0 && CUDA.driver_version() >= v"12.2"
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
  else
      pool = memory_pool(dev)
  end
  pool_status!(dev, true)
  return pool
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

      status = pool_status(dev)
      status === nothing && continue

      if status
        idle_counters[i] = 0
      else
        idle_counters[i] += 1
      end
      pool_status!(dev, false)

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
function pool_status(io::IO=stdout, info::MemoryInfo=MemoryInfo())
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

"""
    retry_reclaim(retry_if) do
        # code that may fail due to insufficient GPU memory
    end

Run a block of code repeatedly until it successfully allocates the memory it needs.
Retries are only attempted when calling `retry_if` with the current return value is true.
At each try, more and more memory is freed from the CUDA memory pool. When that is not
possible anymore, the latest returned value will be returned.

This function is intended for use with CUDA APIs, which sometimes allocate (outside of the
CUDA memory pool) and return a specific error code when failing to. It is similar to
`Base.retry`, but deals with return values instead of exceptions for performance reasons.
"""
@inline function retry_reclaim(f, retry_if)
  ret = f()
  if retry_if(ret)
    return retry_reclaim_slow(f, retry_if, ret)
  else
    return ret
  end
end
## slow path, incrementally reclaiming more memory until we succeed
@noinline function retry_reclaim_slow(f, retry_if, orig_ret)
  state = active_state()
  is_stream_ordered = stream_ordered(state.device)

  phase = 1
  while true
    if is_stream_ordered
      if phase == 1
        synchronize(state.stream)
      elseif phase == 2
        device_synchronize()
      elseif phase == 3
        GC.gc(false)
        device_synchronize()
      elseif phase == 4
        GC.gc(true)
        device_synchronize()
      elseif phase == 5
        # in case we had a release threshold configured
        trim(memory_pool(state.device))
      else
        break
      end
    else
      if phase == 1
        GC.gc(false)
      elseif phase == 2
        GC.gc(true)
      else
        break
      end
    end
    phase += 1

    ret = f()
    if !retry_if(ret)
      return ret
    end
  end

  return orig_ret
end


## managed memory

# to safely use allocated memory across tasks and devices, we don't simply return raw
# memory objects, but wrap them in a manager that ensures synchronization and ownership.

# XXX: immutable with atomic refs?
mutable struct Managed{M}
  const mem::M

  dirty::Bool         # whether the memory has been modified since the last sync
  stream::CuStream    # which stream is currently using the memory

  Managed(mem::AbstractMemory; dirty=false, stream=CUDA.stream()) =
    new{typeof(mem)}(mem, dirty, stream)
end

# wait for the current owner of memory to finish processing
function maybe_synchronize(managed::Managed)
  if managed.dirty
    maybe_synchronize(managed.stream)
    managed.dirty = false
  end
end
function synchronize(managed::Managed)
  # XXX: we can't always rely on the dirty flag to be correct, as memory can be modified
  #      after it was cleared (e.g. if a pointer to memory was stored somewhere). so when
  #      certain APIs ask to synchronize, _always_ synchronize.
  synchronize(managed.stream)
  managed.dirty = false
end

# take over memory for processing
function take_ownership(managed::Managed)
  current_stream = stream()

  if managed.stream != current_stream
    maybe_synchronize(managed)
    managed.stream = current_stream
  end
end

function Base.convert(::Type{CuPtr{T}}, managed::Managed{M}) where {T,M}
  if M == DeviceMemory && context() != managed.mem.ctx
    origin_device = device(managed.mem.ctx)
    throw(ArgumentError("cannot take the GPU address for device $(device()) of GPU memory allocated on device $origin_device"))
    # TODO: check and enable P2P if possible
  end

  # make sure any asynchronous operations that we weren't submitted by the current stream
  # have finished.
  take_ownership(managed)
  convert(CuPtr{T}, managed.mem)
end

function Base.convert(::Type{Ptr{T}}, managed::Managed{M}) where {T,M}
  if M == DeviceMemory
    throw(ArgumentError("cannot take the CPU address of GPU memory"))
  end
  # make sure _any_ work on the memory has finished.
  maybe_synchronize(managed)
  convert(Ptr{T}, managed.mem)
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

  # _alloc reports its own time measurement, since it may spend time in garbage collection
  # (and using Base.@timed/gc_num to exclude that time is too expensive)
  mem, time = _pool_alloc(B, sz)

  Threads.atomic_add!(alloc_stats.alloc_count, 1)
  Threads.atomic_add!(alloc_stats.alloc_bytes, sz)
  Threads.atomic_add!(alloc_stats.total_time, time)
  # NOTE: total_time might be an over-estimation if we trigger GC somewhere else

  return Managed(mem)
end
@inline function _pool_alloc(::Type{DeviceMemory}, sz)
    state = active_state()

    maybe_collect()

    actual_alloc = if stream_ordered(state.device)
      pool = pool_mark(state.device) # mark the pool as active
      (bytes) -> alloc(DeviceMemory, bytes; async=true, state.stream, pool)
    else
      (bytes) -> alloc(DeviceMemory, bytes; async=false)
    end

    time = Base.@elapsed begin
      mem = retry_reclaim(isnothing) do
        memory_limit_exceeded(sz) && return nothing

        # try the actual allocation
        mem = try
          actual_alloc(sz)
        catch err
          isa(err, OutOfGPUMemoryError) || rethrow()
          return nothing
        end

        return mem
      end
      mem === nothing && throw(OutOfGPUMemoryError(sz))
    end

    account!(memory_stats(state.device), sz)

    mem, time
end
@inline function _pool_alloc(::Type{UnifiedMemory}, sz)
  time = Base.@elapsed begin
    mem = alloc(UnifiedMemory, sz)
  end
  mem, time
end
@inline function _pool_alloc(::Type{HostMemory}, sz)
  time = Base.@elapsed begin
    mem = alloc(HostMemory, sz)
  end
  mem, time
end

"""
    pool_free(mem::Managed{<:AbstractMemory})

Releases memory to the pool. If possible, this operation will not block but will be ordered
against the stream that last used the memory.
"""
@inline function pool_free(managed::Managed{<:AbstractMemory})
  # XXX: have @timeit use the root timer, since we may be called from a finalizer

  # ensure this allocation is still alive
  isvalid(managed.mem.ctx) || return
  isvalid(managed.stream) || return
  mem = managed.mem

  # 0-byte allocations shouldn't hit the pool
  sz = sizeof(mem)
  sz == 0 && return

  # this function is typically called from a finalizer, where we can't switch tasks,
  # so perform our own error handling.
  try
    time = Base.@elapsed context!(mem.ctx) do
      _pool_free(mem, managed.stream)
    end

    Threads.atomic_add!(alloc_stats.free_count, 1)
    Threads.atomic_add!(alloc_stats.free_bytes, sz)
    Threads.atomic_add!(alloc_stats.total_time, time)
  catch ex
    Base.showerror_nostdio(ex, "WARNING: Error while freeing $mem")
    Base.show_backtrace(Core.stdout, catch_backtrace())
    Core.println()
  end

  return
end
@inline function _pool_free(mem::DeviceMemory, stream::CuStream)
    # verify that the caller has switched contexts
    if mem.ctx != context()
      error("Trying to free $mem from an unrelated context")
    end

    dev = device()
    if stream_ordered(dev)
      # mark the pool as active
      pool_mark(dev)

      # for safety, we default to the default stream and force this operation to be ordered
      # against all other streams. to opt out of this, pass a specific stream instead.
      free(mem; stream)
    else
      free(mem)
    end
    account!(memory_stats(dev), -sizeof(mem))
end
@inline _pool_free(mem::UnifiedMemory, stream::CuStream) = free(mem)
@inline _pool_free(mem::HostMemory, stream::CuStream) = nothing # XXX

"""
    reclaim([sz=typemax(Int)])

Reclaims `sz` bytes of cached memory. Use this to free GPU memory before calling into
functionality that does not use the CUDA memory pool. Returns the number of bytes
actually reclaimed.
"""
function reclaim(sz::Int=typemax(Int))
  dev = device()
  if stream_ordered(dev)
      device_synchronize()
      synchronize(context())
      trim(memory_pool(dev))
  else
    0
  end
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
                b0 = alloc_stats.alloc_bytes[]
                $(esc(ex))
                alloc_stats.alloc_bytes[] - b0
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
        while false; end # compiler heuristic: compile this block (alter this if the heuristic changes)

        # @time(d) might surround an application, so be sure to initialize CUDA before that
        CUDA.prepare_cuda_state()

        # coarse synchronization to exclude effects from previously-executed code
        synchronize()

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

"""
    used_memory()

Returns the amount of memory from the CUDA memory pool that is currently in use by the
application.
"""
function used_memory()
  state = active_state()
  if stream_ordered(state.device)
    pool = memory_pool(state.device)
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
  state = active_state()
  if stream_ordered(state.device)
    pool = memory_pool(state.device)
    Int(attribute(UInt64, pool, MEMPOOL_ATTR_RESERVED_MEM_CURRENT))
  else
    missing
  end
end
