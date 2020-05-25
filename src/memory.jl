# GPU memory management and pooling

using Printf
using TimerOutputs

using Base: @lock
using Base.Threads: SpinLock

# global lock for shared object dicts (allocated, requested).
# stats are not covered by this and cannot be assumed to be exact.
# each allocator needs to lock its own resources separately too.
const memory_lock = SpinLock()

# the above spinlocks are taken around code that might gc, which might cause a deadlock
# if we try to acquire from the finalizer too. avoid that by temporarily disabling running finalizers,
# concurrently on this thread.
enable_finalizers(on::Bool) = ccall(:jl_gc_enable_finalizers, Cvoid, (Ptr{Cvoid}, Int32,), Core.getptls(), on)
macro safe_lock(l, ex)
  quote
    temp = $(esc(l))
    lock(temp)
    enable_finalizers(false)
    try
      $(esc(ex))
    finally
      unlock(temp)
      enable_finalizers(true)
    end
  end
end

const MEMDEBUG = ccall(:jl_is_memdebug, Bool, ())


## allocation statistics

mutable struct AllocStats
  # pool allocation requests
  pool_nalloc::Int
  pool_nfree::Int
  ## in bytes
  pool_alloc::Int

  # actual CUDA allocations
  actual_nalloc::Int
  actual_nfree::Int
  ## in bytes
  actual_alloc::Int
  actual_free::Int

  pool_time::Float64
  actual_time::Float64
end

const alloc_stats = AllocStats(0, 0, 0, 0, 0, 0, 0, 0, 0)

Base.copy(alloc_stats::AllocStats) =
  AllocStats((getfield(alloc_stats, field) for field in fieldnames(AllocStats))...)

AllocStats(b::AllocStats, a::AllocStats) =
  AllocStats(
    b.pool_nalloc - a.pool_nalloc,
    b.pool_nfree - a.pool_nfree,
    b.pool_alloc - a.pool_alloc,
    b.actual_nalloc - a.actual_nalloc,
    b.actual_nfree - a.actual_nfree,
    b.actual_alloc - a.actual_alloc,
    b.actual_free - a.actual_free,
    b.pool_time - a.pool_time,
    b.actual_time - a.actual_time)


## CUDA allocator

const alloc_to = TimerOutput()

"""
    alloc_timings()

Show the timings of the CUDA allocator. Assumes [`CUDA.enable_timings()`](@ref) has been
called.
"""
alloc_timings() = (show(alloc_to; allocations=false, sortby=:name); println())

const usage = Threads.Atomic{Int}(0)
const usage_limit = Ref{Union{Nothing,Int}}(nothing)

const allocated = Dict{CuPtr{Nothing},Mem.DeviceBuffer}()

function actual_alloc(bytes)
  # check the memory allocation limit
  if usage_limit[] !== nothing
    if usage[] + bytes > usage_limit[]
      return nothing
    end
  end

  # try the actual allocation
  time, buf = try
    time = Base.@elapsed begin
      @timeit_debug alloc_to "alloc" buf = Mem.alloc(Mem.Device, bytes)
    end
    Threads.atomic_add!(usage, bytes)
    time, buf
  catch err
    (isa(err, CuError) && err.code == ERROR_OUT_OF_MEMORY) || rethrow()
    return nothing
  end
  @assert sizeof(buf) == bytes
  ptr = convert(CuPtr{Nothing}, buf)

  # record the buffer
  @safe_lock memory_lock begin
    @assert !haskey(allocated, ptr)
    allocated[ptr] = buf
  end

  alloc_stats.actual_time += time
  alloc_stats.actual_nalloc += 1
  alloc_stats.actual_alloc += bytes

  return ptr
end

function actual_free(ptr::CuPtr{Nothing})
  # look up the buffer
  buf = @safe_lock memory_lock begin
    buf = allocated[ptr]
    delete!(allocated, ptr)
    buf
  end
  bytes = sizeof(buf)

  # free the memory
  @timeit_debug alloc_to "free" begin
    time = Base.@elapsed Mem.free(buf)
    Threads.atomic_sub!(usage, bytes)
  end

  alloc_stats.actual_time += time
  alloc_stats.actual_nfree += 1
  alloc_stats.actual_free += bytes

  return
end


## memory pools

const pool_to = TimerOutput()

macro pool_timeit(args...)
    TimerOutputs.timer_expr(CUDA, true, :($CUDA.pool_to), args...)
end

"""
    pool_timings()

Show the timings of the currently active memory pool. Assumes
[`CUDA.enable_timings()`](@ref) has been called.
"""
pool_timings() = (show(pool_to; allocations=false, sortby=:name); println())

# pool API:
# - init()
# - alloc(sz)::CuPtr{Nothing}
# - free(::CuPtr{Nothing})
# - reclaim(nb::Int=typemax(Int))::Int
# - used_memory()
# - cached_memory()

include("memory/binned.jl")
include("memory/simple.jl")
include("memory/split.jl")
include("memory/dummy.jl")

const pool = Ref{Module}(BinnedPool)


## interface

export OutOfGPUMemoryError

const requested = Dict{CuPtr{Nothing},Vector}()

"""
    OutOfGPUMemoryError()

An operation allocated too much GPU memory for either the system or the memory pool to
handle properly.
"""
struct OutOfGPUMemoryError <: Exception
  sz::Int
end

function Base.showerror(io::IO, err::OutOfGPUMemoryError)
    println(io, "Out of GPU memory trying to allocate $(Base.format_bytes(err.sz))")
    memory_status(io)
end

"""
    alloc(sz)

Allocate a number of bytes `sz` from the memory pool. Returns a `CuPtr{Nothing}`; may throw
a [`OutOfGPUMemoryError`](@ref) if the allocation request cannot be satisfied.
"""
@inline function alloc(sz)
  # 0-byte allocations shouldn't hit the pool
  sz == 0 && return CU_NULL

  time = Base.@elapsed begin
    @pool_timeit "pooled alloc" ptr = pool[].alloc(sz)::Union{Nothing,CuPtr{Nothing}}
  end
  ptr === nothing && throw(OutOfGPUMemoryError(sz))

  # record the allocation
  if Base.JLOptions().debug_level >= 2
    bt = backtrace()
    @lock memory_lock begin
      @assert !haskey(requested, ptr)
      requested[ptr] = bt
    end
  end

  alloc_stats.pool_time += time
  alloc_stats.pool_nalloc += 1
  alloc_stats.pool_alloc += sz

  if MEMDEBUG && ptr == CuPtr{Cvoid}(0xbbbbbbbbbbbbbbbb)
    error("Allocated a scrubbed pointer")
  end

  return ptr
end

"""
    free(sz)

Releases a buffer pointed to by `ptr` to the memory pool.
"""
@inline function free(ptr::CuPtr{Nothing})
  # 0-byte allocations shouldn't hit the pool
  ptr == CU_NULL && return

  if MEMDEBUG && ptr == CuPtr{Cvoid}(0xbbbbbbbbbbbbbbbb)
    Core.println("Freeing a scrubbed pointer!")
  end

  # this function is typically called from a finalizer, where we can't switch tasks,
  # so perform our own error handling.
  try
    # record the allocation
    if Base.JLOptions().debug_level >= 2
      @lock memory_lock begin
        @assert haskey(requested, ptr)
        delete!(requested, ptr)
      end
    end

    time = Base.@elapsed begin
      @pool_timeit "pooled free" pool[].free(ptr)
    end

    alloc_stats.pool_time += time
    alloc_stats.pool_nfree += 1
  catch ex
    Base.showerror_nostdio(ex, "WARNING: Error while freeing $ptr")
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
reclaim(sz::Int=typemax(Int)) = pool[].reclaim(sz)

"""
    @retry_reclaim fail ex

Run a block of code `ex` repeatedly until it successfully allocates the memory it needs;
Failure to do so indicated by returning `fail`. At each try, more and more memory is freed
from the CUDA memory pool. When that is not possible anymore, `fail` will be returned.

This macro is intended for use with CUDA APIs, which sometimes allocate (outside of the
CUDA memory pool) and return a specific error code when failing to.
"""
macro retry_reclaim(fail, ex)
  quote
    ret = nothing
    for phase in 1:3
      ret = $(esc(ex))
      ret == $(esc(fail)) || break

      # incrementally more costly reclaim of cached memory
      if phase == 1
        reclaim()
      elseif phase == 2
        GC.gc(false)
        reclaim()
      elseif phase == 3
        GC.gc(true)
        reclaim()
      end
    end
    ret
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
                b0 = alloc_stats.pool_alloc
                $(esc(ex))
                alloc_stats.pool_alloc - b0
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
        gpu_alloc_size, gpu_gc_time, gpu_mem_stats = @timed $(esc(ex))

        local cpu_alloc_count = Base.gc_alloc_count(cpu_mem_stats)
        local gpu_alloc_count = gpu_mem_stats.pool_nalloc

        local gpu_lib_time = gpu_mem_stats.actual_time

        Printf.@printf("%10.6f seconds", cpu_time)
        for (typ, gctime, libtime, bytes, allocs) in
            (("CPU", cpu_gc_time, 0, cpu_alloc_size, cpu_alloc_count),
             ("GPU", gpu_gc_time, gpu_lib_time, gpu_alloc_size, gpu_alloc_count))
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
                if libtime > 0
                    Printf.@printf(" of which %.2f%% spent allocating", 100*libtime/gctime)
                end
              end
              print(")")
          elseif gctime > 0
              Printf.@printf(", %.2f%% %s gc time", 100*gctime/cpu_time, typ)
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
        CUDA.prepare_cuda_call()

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
         gpu_bytes=gpu_mem_stats.pool_alloc, gpu_gctime=gpu_mem_stats.pool_time, gpu_gcstate=gpu_mem_stats)
    end
end

"""
    memory_status([io=stdout])

Report to `io` on the memory status of the current GPU and the active memory pool.
"""
function memory_status(io::IO=stdout)
  free_bytes, total_bytes = Mem.info()
  used_bytes = total_bytes - free_bytes
  used_ratio = used_bytes / total_bytes
  @printf(io, "Effective GPU memory usage: %.2f%% (%s/%s)\n",
              100*used_ratio, Base.format_bytes(used_bytes),
              Base.format_bytes(total_bytes))

  @printf(io, "CUDA allocator usage: %s", Base.format_bytes(usage[]))
  if usage_limit[] !== nothing
    @printf(io, " (capped at %s)", Base.format_bytes(usage_limit[]))
  end
  println(io)

  alloc_used_bytes = pool[].used_memory()
  alloc_cached_bytes = pool[].cached_memory()
  alloc_total_bytes = alloc_used_bytes + alloc_cached_bytes
  @printf(io, "%s usage: %s (%s allocated, %s cached)\n", nameof(pool[]),
              Base.format_bytes(alloc_total_bytes), Base.format_bytes(alloc_used_bytes),
              Base.format_bytes(alloc_cached_bytes))

  # check if the memory usage as counted by the CUDA allocator wrapper
  # matches what is reported by the pool implementation
  discrepancy = Base.abs(usage[] - alloc_total_bytes)
  if discrepancy != 0
    println(io, "Discrepancy of $(Base.format_bytes(discrepancy)) between memory pool and allocator!")
  end

  if Base.JLOptions().debug_level >= 2
    requested′, allocated′ = @lock memory_lock begin
      copy(requested), copy(allocated)
    end
    for (ptr, bt) in requested′
      buf = allocated′[ptr]
      @printf(io, "\nOutstanding memory allocation of %s at %p",
              Base.format_bytes(sizeof(buf)), Int(ptr))
      stack = stacktrace(bt, false)
      StackTraces.remove_frames!(stack, :alloc)
      Base.show_backtrace(io, stack)
      println(io)
    end
  end
end


## init

"""
    enable_timings()

Enable the recording of debug timings.
"""
enable_timings() = (TimerOutputs.enable_debug_timings(CUDA); return)
disable_timings() = (TimerOutputs.disable_debug_timings(CUDA); return)

function __init_memory__()
  # memory limit configuration
  memory_limit_str = if haskey(ENV, "JULIA_CUDA_MEMORY_LIMIT")
    ENV["JULIA_CUDA_MEMORY_LIMIT"]
  elseif haskey(ENV, "CUARRAYS_MEMORY_LIMIT")
    Base.depwarn("The CUARRAYS_MEMORY_LIMIT environment flag is deprecated, please use JULIA_CUDA_MEMORY_LIMIT instead.", :__init_memory__)
    ENV["CUARRAYS_MEMORY_LIMIT"]
  else
    nothing
  end
  if memory_limit_str !== nothing
    usage_limit[] = parse(Int, memory_limit_str)
  end

  # memory pool configuration
  memory_pool_str = if haskey(ENV, "JULIA_CUDA_MEMORY_POOL")
    ENV["JULIA_CUDA_MEMORY_POOL"]
  elseif haskey(ENV, "CUARRAYS_MEMORY_POOL")
    Base.depwarn("The CUARRAYS_MEMORY_POOL environment flag is deprecated, please use JULIA_CUDA_MEMORY_POOL instead.", :__init_memory__)
    ENV["CUARRAYS_MEMORY_POOL"]
  else
    nothing
  end
  if memory_pool_str !== nothing
    pool[] =
      if memory_pool_str == "binned"
        BinnedPool
      elseif memory_pool_str == "simple"
        SimplePool
      elseif memory_pool_str == "split"
        SplittingPool
      elseif memory_pool_str == "none"
        DummyPool
      else
        error("Invalid allocator selected")
      end

    # the user hand-picked an allocator, so be a little verbose
    atexit(()->begin
      Core.println("""
        CUDA.jl $(nameof(pool[])) statistics:
         - $(alloc_stats.pool_nalloc) pool allocations: $(Base.format_bytes(alloc_stats.pool_alloc)) in $(Base.round(alloc_stats.pool_time; digits=2))s
         - $(alloc_stats.actual_nalloc) CUDA allocations: $(Base.format_bytes(alloc_stats.actual_alloc)) in $(Base.round(alloc_stats.actual_time; digits=2))s""")
    end)
  end

  # initialization
  pool[].init()
  reset_timers!()
end

"""
    reset_timers!()

Reset all debug timers. This is automatically called at initialization time,
"""
function reset_timers!()
  TimerOutputs.reset_timer!(alloc_to)
  TimerOutputs.reset_timer!(pool_to)
end
