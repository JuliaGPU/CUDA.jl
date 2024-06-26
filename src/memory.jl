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

const _memory_stats = PerDevice{MemoryStats}()
function memory_stats(dev::CuDevice=device())
  get!(_memory_stats, dev) do
      MemoryStats()
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
  if current_time - stats.size_updated > 10
    limits = memory_limits()
    Base.@atomic stats.size = if limits.hard > 0
      limits.hard
    elseif limits.soft > 0
      limits.soft
    else
      size = free_memory() + stats.live
      # NOTE: we use stats.live so that we only count memory allocated here, ensuring
      #       the pressure calculation below reflects the heap we have control over.

      # also include reserved bytes
      dev = device()
      if stream_ordered(dev)
        size += (cached_memory() - used_memory())::Int
      end

      size
    end
    Base.@atomic stats.size_updated = current_time
  end

  # check that we're under memory pressure
  pressure = stats.live / stats.size
  min_pressure = 0.75
  ## if we're about to block anyway, now may be a good time for a GC pause
  if will_block
    min_pressure = 0.50
  end
  if pressure < min_pressure
    return
  end

  # ensure we don't collect too often by checking the GC rate
  last_time = stats.last_time
  gc_rate = stats.last_gc_time / (current_time - last_time)
  ## we tolerate 5% GC time
  max_gc_rate = 0.05
  ## if we freed a lot last time, bump that up
  if stats.last_freed > 0.1*stats.size
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

  # finally, call the GC
  pre_gc_live = stats.live
  gc_time = Base.@elapsed GC.gc(pressure > 0.9 ? true : false)
  post_gc_live = stats.live
  memory_freed = pre_gc_live - post_gc_live
  Base.@atomic stats.last_freed = memory_freed
  ## GC times can vary, so smooth them out
  Base.@atomic stats.last_gc_time = 0.75*stats.last_gc_time + 0.25*gc_time

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

# TODO: extract this into a @device_memoize macro, or teach @memoize about CuDevice?
#       this is a common pattern that could be applied to many more functions.
function stream_ordered(dev::CuDevice)
  devidx = deviceid(dev) + 1
  @memoize devidx::Int maxlen=ndevices() begin
    CUDA.driver_version() >= v"11.3" && memory_pools_supported(dev) &&
    get(ENV, "JULIA_CUDA_MEMORY_POOL", "cuda") == "cuda"
  end::Bool
end

const _memory_pools = PerDevice{CuMemoryPool}()
function pool_create(dev::CuDevice)
  get!(_memory_pools, dev) do
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

      pool
  end
end

# per-device flag indicating the status of the memory pool.
const _pool_status = PerDevice{Base.RefValue{Bool}}()
function pool_mark(dev::CuDevice)
  status = get(_pool_status, dev, nothing)
  status === nothing && return nothing
  return status[]
end
function pool_mark!(dev::CuDevice, val)
  box = get!(_pool_status, dev) do
    Ref{Bool}()
  end
  box[] = val
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

const reclaim_hooks = Any[]

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
        trim(pool_create(state.device))
      elseif phase == 6
        for hook in reclaim_hooks
          hook()
        end
      else
        break
      end
    else
      if phase == 1
        GC.gc(false)
      elseif phase == 2
        GC.gc(true)
      elseif phase == 3
        for hook in reclaim_hooks
          hook()
        end
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

  # which stream is currently using the memory.
  stream::CuStream

  # whether there are outstanding operations that haven't been synchronized
  dirty::Bool

  # whether the memory has been captured in a way that would make the dirty bit unreliable
  captured::Bool

  function Managed(mem::AbstractMemory; stream=CUDA.stream(), dirty=true, captured=false)
    # NOTE: memory starts as dirty, because stream-ordered allocations are only
    #       guaranteed to be physically allocated at a synchronization event.
    new{typeof(mem)}(mem, stream, dirty, captured)
  end
end

# wait for the current owner of memory to finish processing
function synchronize(managed::Managed)
  synchronize(managed.stream)
  managed.dirty = false
end
function maybe_synchronize(managed::Managed)
  if managed.dirty || managed.captured
    synchronize(managed)
  end
end

function Base.convert(::Type{CuPtr{T}}, managed::Managed{M}) where {T,M}
  # let null pointers pass through as-is
  ptr = convert(CuPtr{T}, managed.mem)
  if ptr == CU_NULL
    return ptr
  end

  # accessing memory during stream capture: taint the memory so that we always synchronize
  state = active_state()
  if is_capturing(state.stream)
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
    if stream_ordered(source_device)
      pool = pool_create(source_device)
      access!(pool, state.device, ACCESS_FLAGS_PROT_READWRITE)
    end
  end

  # accessing memory on another stream: ensure the data is ready and take ownership
  if managed.stream != state.stream
    maybe_synchronize(managed)
    managed.stream = state.stream
  end

  managed.dirty = true
  return ptr
end

function Base.convert(::Type{Ptr{T}}, managed::Managed{M}) where {T,M}
  # let null pointers pass through as-is
  ptr = convert(Ptr{T}, managed.mem)
  if ptr == C_NULL
    return ptr
  end

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
  alloc(UnifiedMemory, sz)
end
@inline function _pool_alloc(::Type{HostMemory}, sz)
  alloc(HostMemory, sz)
end

"""
    pool_free(mem::Managed{<:AbstractMemory})

Releases memory to the pool. If possible, this operation will not block but will be ordered
against the stream that last used the memory.
"""
@inline function pool_free(managed::Managed{<:AbstractMemory})
  mem = managed.mem

  # 0-byte allocations shouldn't hit the pool
  sz = sizeof(mem)
  sz == 0 && return

  # this function is typically called from a finalizer, where we can't switch tasks,
  # so perform our own error handling.
  try
    time = Base.@elapsed _pool_free(mem, managed.stream)

    Base.@atomic alloc_stats.free_count += 1
    Base.@atomic alloc_stats.free_bytes += sz
    Base.@atomic alloc_stats.total_time += time
  catch ex
    Base.showerror_nostdio(ex, "WARNING: Error while freeing $mem")
    Base.show_backtrace(Core.stdout, catch_backtrace())
    Core.println()
  end

  return
end
@inline function _pool_free(mem::DeviceMemory, stream::CuStream)
    if mem.async
      # stream-ordered allocations are not tied to a context. we always need to free them,
      # and if the owning context (or stream) was destroyed, use a new (or default) one.
      if isvalid(mem.ctx) && isvalid(stream)
        context!(mem.ctx) do
          free(mem; stream)
        end
      else
        free(mem; stream=default_stream())
      end
    else
      # regular allocations are tied to a context, so ignore if the context was destroyed
      context!(mem.ctx; skip_destroyed=true) do
        free(mem)
      end
    end
    account!(memory_stats(mem.dev), -sizeof(mem))
end
@inline _pool_free(mem::UnifiedMemory, stream::CuStream) = free(mem)
@inline _pool_free(mem::HostMemory, stream::CuStream) = free(mem)

"""
    reclaim([sz=typemax(Int)])

Reclaims `sz` bytes of cached memory. Use this to free GPU memory before calling into
functionality that does not use the CUDA memory pool. Returns the number of bytes
actually reclaimed.
"""
function reclaim(sz::Int=typemax(Int))
  dev = device()
  for hook in reclaim_hooks
    hook()
  end
  if stream_ordered(dev)
      device_synchronize()
      synchronize(context())
      trim(pool_create(dev))
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

"""
    used_memory()

Returns the amount of memory from the CUDA memory pool that is currently in use by the
application.
"""
function used_memory()
  state = active_state()
  if stream_ordered(state.device)
    pool = pool_create(state.device)
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
    pool = pool_create(state.device)
    Int(attribute(UInt64, pool, MEMPOOL_ATTR_RESERVED_MEM_CURRENT))
  else
    missing
  end
end
