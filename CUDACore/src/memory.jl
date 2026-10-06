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

Report to `io` on the memory status of the current GPU and the active memory pool. Memory
that has been garbage collected, but not released yet, is released first.
"""
function pool_status(io::IO=stdout, info::MemoryInfo=(drain_retired(); MemoryInfo()))
  state = active_state()
  ctx = context()

  used_bytes = info.total_bytes - info.free_bytes
  used_ratio = used_bytes / info.total_bytes
  @printf(io, "Effective GPU memory usage: %.2f%% (%s/%s)\n",
              100*used_ratio, Base.format_bytes(used_bytes),
              Base.format_bytes(info.total_bytes))

  if info.pool_reserved_bytes === nothing
    @printf(io, "No memory pool is in use.\n")
  else
    @printf(io, "Memory pool usage: %s (%s reserved)\n",
                Base.format_bytes(info.pool_used_bytes),
                Base.format_bytes(info.pool_reserved_bytes))

  end

  active_handles = idle_handles = 0
  foreach_reclaimable() do resource
    if resource isa HandleCache
      counts = handle_cache_counts(resource)
      active_handles += counts.active
      idle_handles += counts.idle
    end
  end
  println(io, "Library handles: $active_handles active, $idle_handles reusable (released at reclaim)")

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
