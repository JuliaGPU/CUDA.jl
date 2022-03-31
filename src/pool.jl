# GPU memory management and pooling

using Printf
using Logging


## allocation statistics

mutable struct AllocStats
  alloc_count::Int
  alloc_bytes::Int

  free_count::Int
  free_bytes::Int

  total_time::Float64
end

const alloc_stats = AllocStats(0, 0, 0, 0, 0.0)

Base.copy(alloc_stats::AllocStats) =
  AllocStats((getfield(alloc_stats, field) for field in fieldnames(AllocStats))...)

AllocStats(b::AllocStats, a::AllocStats) =
  AllocStats(
    b.alloc_count - a.alloc_count,
    b.alloc_bytes - a.alloc_bytes,
    b.free_count - a.free_count,
    b.free_bytes - a.free_bytes,
    b.total_time - a.total_time)


## CUDA allocator

@timeit_ci function actual_alloc(bytes::Integer; async::Bool=false,
                                 stream::Union{CuStream,Nothing}=nothing)
  # try the actual allocation
  buf = try
    time = Base.@elapsed begin
      buf = @timeit_ci "Mem.alloc" begin
        Mem.alloc(Mem.Device, bytes; async, stream)
      end
    end

    buf
  catch err
    isa(err, OutOfGPUMemoryError) || rethrow()
    return nothing
  end

  return buf
end

@timeit_ci function actual_free(buf::Mem.DeviceBuffer;
                                stream::Union{CuStream,Nothing}=nothing)
  # free the memory
  time = Base.@elapsed begin
    @timeit_ci "Mem.free" Mem.free(buf; stream)
  end

  return
end


## stream-ordered memory pool

# TODO: extract this into a @device_memoize macro, or teach @memoize about CuDevice?
#       this is a common pattern that could be applied to many more functions.
function stream_ordered(dev::CuDevice)
  devidx = deviceid(dev) + 1
  @memoize devidx::Int maxlen=ndevices() begin
    memory_pools_supported(dev) && get(ENV, "JULIA_CUDA_MEMORY_POOL", "cuda") == "cuda"
  end::Bool
end

# per-device flag indicating the status of a pool
const _pool_status = PerDevice{Base.RefValue{Union{Nothing,Bool}}}()
pool_status(dev::CuDevice) = get!(_pool_status, dev) do
  # nothing=uninitialized, false=idle, true=active
  Ref{Union{Nothing,Bool}}(nothing)
end
function pool_mark(dev::CuDevice)
  status = pool_status(dev)
  if status[] === nothing
      pool = memory_pool(dev)

      # allow the pool to use up all memory of this device
      attribute!(memory_pool(dev), MEMPOOL_ATTR_RELEASE_THRESHOLD, typemax(UInt64))

      # launch a task to periodically trim the pool
      if isinteractive() && !isassigned(__pool_cleanup)
        __pool_cleanup[] = if VERSION < v"1.7"
          Threads.@spawn pool_cleanup()
        else
          errormonitor(Threads.@spawn pool_cleanup())
        end
      end
  end
  status[] = true
  return
end

# reclaim unused pool memory after a certain time
const __pool_cleanup = Ref{Task}()
function pool_cleanup()
  idle_counters = Base.fill(0, ndevices())
  while true
    for (i, dev) in enumerate(devices())
      stream_ordered(dev) || continue

      status = pool_status(dev)
      status[] === nothing && continue

      if status[]
        idle_counters[i] = 0
      else
        idle_counters[i] += 1
      end
      status[] = 0

      if idle_counters[i] == 5
        # the pool hasn't been used for a while, so reclaim unused buffers
        device!(dev) do
          reclaim()
        end
      end
    end

    sleep(60)
  end
end


## interface

export OutOfGPUMemoryError

struct MemoryInfo
  free_bytes::Int
  total_bytes::Int
  pool_reserved_bytes::Union{Int,Missing,Nothing}
  pool_used_bytes::Union{Int,Missing,Nothing}

  function MemoryInfo()
    free_bytes, total_bytes = Mem.info()

    pool_reserved_bytes, pool_used_bytes = if stream_ordered(device())
      if version() >= v"11.3"
        cached_memory(), used_memory()
      else
        missing, missing
      end
    else
      nothing, nothing
    end

    new(free_bytes, total_bytes, pool_reserved_bytes, pool_used_bytes)
  end
end

"""
    memory_status([io=stdout])

Report to `io` on the memory status of the current GPU and the active memory pool.
"""
function memory_status(io::IO=stdout, info::MemoryInfo=MemoryInfo())
  state = active_state()
  ctx = context()

  used_bytes = info.total_bytes - info.free_bytes
  used_ratio = used_bytes / info.total_bytes
  @printf(io, "Effective GPU memory usage: %.2f%% (%s/%s)\n",
              100*used_ratio, Base.format_bytes(used_bytes),
              Base.format_bytes(info.total_bytes))

  if info.pool_reserved_bytes === nothing
    @printf(io, "No memory pool is in use.")
  elseif info.pool_reserved_bytes === missing
    @printf(io, "Memory pool statistics require CUDA 11.3.")
  else
    @printf(io, "Memory pool usage: %s (%s reserved)",
                Base.format_bytes(info.pool_used_bytes),
                Base.format_bytes(info.pool_reserved_bytes))

  end
end

"""
    OutOfGPUMemoryError()

An operation allocated too much GPU memory for either the system or the memory pool to
handle properly.
"""
struct OutOfGPUMemoryError <: Exception
  sz::Int
  info::MemoryInfo

  OutOfGPUMemoryError(sz::Integer=0) = new(sz, MemoryInfo())
end

function Base.showerror(io::IO, err::OutOfGPUMemoryError)
    print(io, "Out of GPU memory")
    if err.sz > 0
      print(io, " trying to allocate $(Base.format_bytes(err.sz))")
    end
    println(io)
    memory_status(io, err.info)
end

"""
    @retry_reclaim isfailed(ret) ex

Run a block of code `ex` repeatedly until it successfully allocates the memory it needs.
Retries are only attempted when calling `isfailed` with the current return value is true.
At each try, more and more memory is freed from the CUDA memory pool. When that is not
possible anymore, the latest returned value will be returned.

This macro is intended for use with CUDA APIs, which sometimes allocate (outside of the
CUDA memory pool) and return a specific error code when failing to.
"""
macro retry_reclaim(isfailed, ex)
  quote
    ret = $(esc(ex))

    # slow path, incrementally reclaiming more memory until we succeed
    if $(esc(isfailed))(ret)
      state = active_state()
      is_stream_ordered = stream_ordered(state.device)

      phase = 1
      while true
        if is_stream_ordered
          # NOTE: the stream-ordered allocator only releases memory on actual API calls,
          #       and not when our synchronization routines query the relevant streams.
          #       we do still call our routines to minimize the time we block in libcuda.
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

        ret = $(esc(ex))
        $(esc(isfailed))(ret) || break
      end
    end

    ret
  end
end

# XXX: function version for use in CUDAdrv where we haven't loaded pool.jl yet
function retry_reclaim(f, check)
  @retry_reclaim check f()
end

"""
    alloc([::BufferType], sz; [stream::CuStream])

Allocate a number of bytes `sz` from the memory pool. Returns a buffer object; may throw
an [`OutOfGPUMemoryError`](@ref) if the allocation request cannot be satisfied.
"""
@inline @timeit_ci alloc(sz::Integer; kwargs...) = alloc(Mem.DeviceBuffer, sz; kwargs...)
@inline @timeit_ci function alloc(::Type{B}, sz; stream::Union{Nothing,CuStream}=nothing) where {B<:Mem.AbstractBuffer}
  # 0-byte allocations shouldn't hit the pool
  sz == 0 && return B()

  # _alloc reports its own time measurement, since it may spend time in garbage collection
  # (and using Base.@timed/gc_num to exclude that time is too expensive)
  buf, time = _alloc(B, sz; stream)

  alloc_stats.alloc_count += 1
  alloc_stats.alloc_bytes += sz
  alloc_stats.total_time += time
  # NOTE: total_time might be an over-estimation if we trigger GC somewhere else

  return buf
end
@inline function _alloc(::Type{Mem.DeviceBuffer}, sz; stream::Union{Nothing,CuStream})
    state = active_state()
    stream = something(stream, state.stream)

    gctime = 0.0
    time = Base.@elapsed begin
      buf = if stream_ordered(state.device)
        pool_mark(state.device) # mark the pool as active
        @retry_reclaim isnothing actual_alloc(sz; async=true, stream)
      else
        @retry_reclaim isnothing actual_alloc(sz; async=false, stream)
      end
      buf === nothing && throw(OutOfGPUMemoryError(sz))
    end

    buf, time - gctime
end
@inline function _alloc(::Type{Mem.UnifiedBuffer}, sz; stream::Union{Nothing,CuStream})
  time = Base.@elapsed begin
    buf = Mem.alloc(Mem.Unified, sz)
  end
  buf, time
end

"""
    free(buf)

Releases a buffer `buf` to the memory pool.
"""
@inline @timeit_ci function free(buf::Mem.AbstractBuffer;
                                 stream::Union{Nothing,CuStream}=nothing)
  # XXX: have @timeit use the root timer, since we may be called from a finalizer

  # 0-byte allocations shouldn't hit the pool
  sizeof(buf) == 0 && return

  # this function is typically called from a finalizer, where we can't switch tasks,
  # so perform our own error handling.
  try
    time = Base.@elapsed begin
      _free(buf; stream)
    end

    alloc_stats.free_count += 1
    alloc_stats.free_bytes += sizeof(buf)
    alloc_stats.total_time += time
  catch ex
    Base.showerror_nostdio(ex, "WARNING: Error while freeing $buf")
    Base.show_backtrace(Core.stdout, catch_backtrace())
    Core.println()
  end

  return
end
@inline function _free(buf::Mem.DeviceBuffer; stream::Union{Nothing,CuStream})
    # verify that the caller has switched contexts
    if buf.ctx != context()
      error("Trying to free $buf from an unrelated context")
    end

    dev = current_device()
    if stream_ordered(dev)
      # mark the pool as active
      pool_mark(dev)

      # for safety, we default to the default stream and force this operation to be ordered
      # against all other streams. to opt out of this, pass a specific stream instead.
      actual_free(buf; stream=something(stream, default_stream()))
    else
      actual_free(buf)
    end
end
@inline _free(buf::Mem.UnifiedBuffer; stream::Union{Nothing,CuStream}) = Mem.free(buf)
@inline _free(buf::Mem.HostBuffer; stream::Union{Nothing,CuStream}) = nothing

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
        local gpu_mem_stats = AllocStats(gpu_mem_stats1, gpu_mem_stats0)

        (value=val, time=cpu_time,
         cpu_bytes=cpu_mem_stats.allocd, cpu_gctime=cpu_mem_stats.total_time / 1e9, cpu_gcstats=cpu_mem_stats,
         gpu_bytes=gpu_mem_stats.alloc_bytes, gpu_memtime=gpu_mem_stats.total_time, gpu_memstats=gpu_mem_stats)
    end
end

"""
    used_memory()

Returns the amount of memory from the CUDA memory pool that is currently in use by the
application.

!!! warning

    This function is only available on CUDA driver 11.3 and later.
"""
function used_memory()
  state = active_state()
  if version() >= v"11.3" && stream_ordered(state.device)
    # we can only query the memory pool's reserved memory on CUDA 11.3 and later
    pool = memory_pool(state.device)
    Int(attribute(UInt64, pool, MEMPOOL_ATTR_USED_MEM_CURRENT))
  else
    missing
  end
end

"""
    cached_memory()

Returns the amount of backing memory currently allocated for the CUDA memory pool.

!!! warning

    This function is only available on CUDA driver 11.3 and later.
"""
function cached_memory()
  state = active_state()
  if version() >= v"11.3" && stream_ordered(state.device)
    # we can only query the memory pool's reserved memory on CUDA 11.3 and later
    pool = memory_pool(state.device)
    Int(attribute(UInt64, pool, MEMPOOL_ATTR_RESERVED_MEM_CURRENT))
  else
    missing
  end
end
