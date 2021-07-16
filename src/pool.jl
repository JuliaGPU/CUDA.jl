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

@timeit_ci function actual_alloc(bytes::Integer;
                                 stream_ordered::Bool=false,
                                 stream::Union{CuStream,Nothing}=nothing)
  # try the actual allocation
  buf = try
    time = Base.@elapsed begin
      buf = @timeit_ci "Mem.alloc" begin
        Mem.alloc(Mem.Device, bytes; async=true, stream_ordered, stream)
      end
    end

    buf
  catch err
    isa(err, OutOfGPUMemoryError) || rethrow()
    return nothing
  end

  return buf
end

@timeit_ci function actual_free(buf::Mem.DeviceBuffer; stream_ordered::Bool=false,
                                stream::Union{CuStream,Nothing}=nothing)
  # free the memory
  time = Base.@elapsed begin
    @timeit_ci "Mem.free" Mem.free(buf; async=true, stream_ordered, stream)
  end

  return
end


## stream-ordered memory pool

const __stream_ordered = LazyInitialized{Vector{Bool}}()
function stream_ordered(dev::CuDevice)
  flags = get!(__stream_ordered) do
    val = Vector{Bool}(undef, ndevices())
    if version() < v"11.2" || haskey(ENV, "CUDA_MEMCHECK")
      fill!(val, false)
    else
      for dev in devices()
        val[deviceid(dev)+1] = attribute(dev, DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED) == 1
      end
    end
    val
  end
  @inbounds flags[deviceid(dev)+1]
end

function allocatable_memory(dev::CuDevice)
  # NOTE: this function queries available memory, which obviously changes after we allocate.
  device!(dev) do
    available_memory()
  end
end

function reserved_memory(dev::CuDevice)
  # taken from TensorFlow's `MinSystemMemory`
  #
  # if the available memory is < 2GiB, we allocate 225MiB to system memory.
  # otherwise, depending on the capability version assign
  #  500MiB (for cuda_compute_capability <= 6.x) or
  # 1050MiB (for cuda_compute_capability <= 7.x) or
  # 1536MiB (for cuda_compute_capability >= 8.x)
  available = allocatable_memory(dev)
  if available <= 1<<31
    225 * 1024 * 1024
  else
    cap = capability(dev)
    if cap <= v"6"
      500 * 1024 * 1024
    elseif cap <= v"7"
      1050 * 1024 * 1024
    else
      1536 * 1024 * 1024
    end
  end
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

      # first time on this context, so configure the pool
      attribute!(memory_pool(dev), MEMPOOL_ATTR_RELEASE_THRESHOLD,
                 UInt64(reserved_memory(dev)))

      # also launch a task to periodically trim the pool
      if isinteractive() && !isassigned(__pool_cleanup)
        __pool_cleanup[] = @async pool_cleanup()
      end
  end
  status[] = true
  return
end

# reclaim unused pool memory after a certain time
const __pool_cleanup = Ref{Task}()
function pool_cleanup()
  idle_counters = fill(0, ndevices())
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

"""
    OutOfGPUMemoryError()

An operation allocated too much GPU memory for either the system or the memory pool to
handle properly.
"""
struct OutOfGPUMemoryError <: Exception
  sz::Int

  OutOfGPUMemoryError(sz::Integer=0) = new(sz)
end

function Base.showerror(io::IO, err::OutOfGPUMemoryError)
    print(io, "Out of GPU memory")
    if err.sz > 0
      print(io, " trying to allocate $(Base.format_bytes(err.sz))")
    end
    println(io)
    memory_status(io)
end

"""
    alloc(sz)

Allocate a number of bytes `sz` from the memory pool. Returns a buffer object; may throw
an [`OutOfGPUMemoryError`](@ref) if the allocation request cannot be satisfied.
"""
@inline @timeit_ci function alloc(sz; unified::Bool=false, stream::Union{Nothing,CuStream}=nothing)
  # 0-byte allocations shouldn't hit the pool
  sz == 0 && return Mem.DeviceBuffer(CU_NULL, 0)

  gctime = 0.0  # using Base.@timed/gc_num is too expensive
  time = Base.@elapsed if unified
    # TODO: integrate this with the non-unified code path (e.g. we want to retry & gc too)
    # TODO: add a memory type argument to `alloc`?
    buf = Mem.alloc(Mem.Unified, sz)
  else
    state = active_state()

    buf = nothing
    if stream_ordered(state.device)
      # mark the pool as active
      pool_mark(state.device)

      for phase in 1:4
          if phase == 2
              gctime += Base.@elapsed GC.gc(false)
          elseif phase == 3
              gctime += Base.@elapsed GC.gc(true)
          elseif phase == 4
              device_synchronize()
          end

          buf = actual_alloc(sz; stream_ordered=true, stream=something(stream, state.stream))
          buf === nothing || break
      end
    else
      for phase in 1:4
          if phase == 2
              gctime += Base.@elapsed GC.gc(false)
          elseif phase == 3
              gctime += Base.@elapsed GC.gc(true)
          end

          buf = actual_alloc(sz; stream_ordered=false)
          buf === nothing || break
      end
    end
    buf === nothing && throw(OutOfGPUMemoryError(sz))
  end

  alloc_stats.alloc_count += 1
  alloc_stats.alloc_bytes += sz
  alloc_stats.total_time += time - gctime
  # NOTE: total_time might be an over-estimation if we trigger GC somewhere else

  return buf
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
    time = Base.@elapsed if buf isa Mem.UnifiedBuffer
      Mem.free(buf)
    else
      state = active_state()
      if stream_ordered(state.device)
        # mark the pool as active
        pool_mark(state.device)

        actual_free(buf; stream_ordered=true, stream=something(stream, state.stream))
      else
        actual_free(buf; stream_ordered=false)
      end
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

"""
    reclaim([sz=typemax(Int)])

Reclaims `sz` bytes of cached memory. Use this to free GPU memory before calling into
functionality that does not use the CUDA memory pool. Returns the number of bytes
actually reclaimed.
"""
function reclaim(sz::Int=typemax(Int))
  dev = device()
  if stream_ordered(dev)
      # TODO: respect sz
      device_synchronize()
      trim(memory_pool(dev))
  else
    0
  end
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
    ret = nothing
    phase = 0
    while true
      phase += 1
      ret = $(esc(ex))
      $(esc(isfailed))(ret) || break

      # incrementally more costly reclaim of cached memory
      if stream_ordered(device())
        if phase == 1
          # synchronizing streams forces asynchronous free operations to finish.
          device_synchronize()
        elseif phase == 2
          GC.gc(false)
          device_synchronize()
        elseif phase == 3
          GC.gc(true)
          device_synchronize()
        elseif phase == 4
          # maybe this allocation doesn't use the pool, so trim it.
          reclaim()
          device_synchronize()
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
    end
    ret
  end
end

# XXX: function version for use in CUDAdrv where we haven't loaded pool.jl yet
function retry_reclaim(f, check)
  @retry_reclaim check f()
end


## utilities

used_memory(ctx=context()) = @lock allocated_lock begin
    mapreduce(sizeof∘first, +, values(allocated(ctx)); init=0)
end


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
        local val = @sync blocking=false $(esc(ex))

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
    memory_status([io=stdout])

Report to `io` on the memory status of the current GPU and the active memory pool.
"""
function memory_status(io::IO=stdout)
  state = active_state()
  ctx = context()

  free_bytes, total_bytes = Mem.info()
  used_bytes = total_bytes - free_bytes
  used_ratio = used_bytes / total_bytes
  @printf(io, "Effective GPU memory usage: %.2f%% (%s/%s)\n",
              100*used_ratio, Base.format_bytes(used_bytes),
              Base.format_bytes(total_bytes))

  if stream_ordered(state.device)
    pool = memory_pool(state.device)
    if version() >= v"11.3"
      pool_reserved_bytes = attribute(UInt64, pool, MEMPOOL_ATTR_RESERVED_MEM_CURRENT)
      pool_used_bytes = attribute(UInt64, pool, MEMPOOL_ATTR_USED_MEM_CURRENT)
      @printf(io, "Memory pool usage: %s (%s reserved)",
                  Base.format_bytes(pool_used_bytes),
                  Base.format_bytes(pool_reserved_bytes))
    else
      @printf(io, "Memory pool statistics require CUDA 11.3.")
    end
  else
    @printf(io, "No memory pool is in use.")
  end
end
