# GPU memory management and pooling

using Printf
using Logging
using TimerOutputs

include("pool/utils.jl")
using .PoolUtils


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

const usage = PerDevice{Threads.Atomic{Int}}() do dev
  Threads.Atomic{Int}(0)
end
const usage_limit = PerDevice{Int}() do dev
  if haskey(ENV, "JULIA_CUDA_MEMORY_LIMIT")
    parse(Int, ENV["JULIA_CUDA_MEMORY_LIMIT"])
  elseif haskey(ENV, "CUARRAYS_MEMORY_LIMIT")
    Base.depwarn("The CUARRAYS_MEMORY_LIMIT environment flag is deprecated, please use JULIA_CUDA_MEMORY_LIMIT instead.", :__init_pool__)
    parse(Int, ENV["CUARRAYS_MEMORY_LIMIT"])
  else
    typemax(Int)
  end
end

@memoize function allocatable_memory(dev::CuDevice)
  # NOTE: this function queries available memory, which obviously changes after we allocate,
  #       so we memoize it to ensure only the first value is ever returned.
  device!(dev) do
    Base.min(available_memory(), usage_limit[dev])
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

# This limit depends on the actually available memory, which might be an underestimation
# (e.g. when the GPU was in use initially). The hard limit does not rely on such heuristics.
@memoize function soft_limit(dev::CuDevice)
  available = allocatable_memory(dev)
  reserve = reserved_memory(dev)
  return available - reserve
end

function hard_limit(dev::CuDevice)
  # ignore the available memory heuristic, and even allow to eat in to the reserve
  usage_limit[dev]
end

function actual_alloc(dev::CuDevice, bytes::Integer, last_resort::Bool=false)
  buf = @device! dev begin
    # check the memory allocation limit
    if usage[dev][] + bytes > (last_resort ? hard_limit(dev) : soft_limit(dev))
      return nothing
    end

    # try the actual allocation
    try
      time = Base.@elapsed begin
        @timeit_debug alloc_to "alloc" begin
          buf = Mem.alloc(Mem.Device, bytes; async=true,
                          pool = (async_alloc[] ? pool() : nothing))
          # we only need a memory pool when we'll be using the async allocator.
          # this avoids a needless warning when running under cuda-memcheck,
          # which doesn't support the stream-ordered memory allocator.
        end
      end

      Threads.atomic_add!(usage[dev], bytes)
      alloc_stats.actual_time += time
      alloc_stats.actual_nalloc += 1
      alloc_stats.actual_alloc += bytes

      buf
    catch err
      (isa(err, CuError) && err.code == ERROR_OUT_OF_MEMORY) || rethrow()
      return nothing
    end
  end

  return Block(buf, bytes; state=AVAILABLE)
end

function actual_free(dev::CuDevice, block::Block)
  @assert iswhole(block) "Cannot free $block: block is not whole"
  @assert block.off == 0
  @assert block.state == AVAILABLE "Cannot free $block: block is not available"

  @device! dev begin
    # free the memory
    @timeit_debug alloc_to "free" begin
      time = Base.@elapsed begin
        Mem.free(block.buf; async=true)
      end
      block.state = INVALID

      Threads.atomic_sub!(usage[dev], sizeof(block.buf))
      alloc_stats.actual_time += time
      alloc_stats.actual_nfree += 1
      alloc_stats.actual_free += sizeof(block.buf)
    end
  end

  return
end


## memory pools

"""
    pool_timings()

Show the timings of the currently active memory pool. Assumes
[`CUDA.enable_timings()`](@ref) has been called.
"""
pool_timings() = (show(PoolUtils.to; allocations=false, sortby=:name); println())

# pool API:
# - init()
# - alloc(::CuDevice, sz)::Block
# - free(::CuDevice, ::Block)
# - reclaim(::CuDevice, nb::Int=typemax(Int))::Int
# - cached_memory()

module Pool
@enum MemoryPool None Simple Binned Split
end
const active_pool = Ref{Pool.MemoryPool}()
const async_alloc = Ref{Bool}()

macro pooled(ex)
    @assert Meta.isexpr(ex, :call)
    f, args... = ex.args
    quote
      if active_pool[] == Pool.None
        NoPool.$(f)($(map(esc, args)...))
      elseif active_pool[] == Pool.Simple
        SimplePool.$(f)($(map(esc, args)...))
      elseif active_pool[] == Pool.Binned
        BinnedPool.$(f)($(map(esc, args)...))
      elseif active_pool[] == Pool.Split
        SplitPool.$(f)($(map(esc, args)...))
      else
        error("unreachable")
      end
    end
end

include("pool/none.jl")
include("pool/simple.jl")
include("pool/binned.jl")
include("pool/split.jl")


## interface

export OutOfGPUMemoryError

const allocated_lock = NonReentrantLock()
const allocated = PerDevice{Dict{CuPtr,Tuple{Block,Int}}}() do dev
    Dict{CuPtr,Tuple{Block,Int}}()
end

const requested_lock = NonReentrantLock()
const requested = PerDevice{Dict{CuPtr{Nothing},Vector}}() do dev
  Dict{CuPtr{Nothing},Vector}()
end

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

Allocate a number of bytes `sz` from the memory pool. Returns a `CuPtr{Nothing}`; may throw
a [`OutOfGPUMemoryError`](@ref) if the allocation request cannot be satisfied.
"""
@inline function alloc(sz)
  # 0-byte allocations shouldn't hit the pool
  sz == 0 && return CU_NULL

  dev = device()

  time = Base.@elapsed begin
    @pool_timeit "pooled alloc" block = @pooled alloc(dev, sz)
  end
  block::Union{Nothing,Block}
  block === nothing && throw(OutOfGPUMemoryError(sz))

  # record the memory block
  ptr = pointer(block)
  @lock allocated_lock begin
      @assert !haskey(allocated[dev], ptr)
      allocated[dev][ptr] = block, 1
  end

  # record the allocation site
  if Base.JLOptions().debug_level >= 2
    bt = backtrace()
    @lock requested_lock begin
      @assert !haskey(requested[dev], ptr)
      requested[dev][ptr] = bt
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
    alias(ptr)

Increase the reference count of a buffer pointed to by `ptr`. As a result, it will require
multiple calls to `free` before this buffer is put back into the memory pool.
"""
@inline function alias(ptr::CuPtr{Nothing})
  # 0-byte allocations shouldn't hit the pool
  ptr == CU_NULL && return

  dev = device()

  # look up the memory block
  @spinlock allocated_lock begin
    block, refcount = allocated[dev][ptr]
    allocated[dev][ptr] = block, refcount+1
  end

  return
end

"""
    free(ptr)

Releases a buffer pointed to by `ptr` to the memory pool.
"""
@inline function free(ptr::CuPtr{Nothing})
  # 0-byte allocations shouldn't hit the pool
  ptr == CU_NULL && return

  dev = device()
  last_use[dev] = time()

  if MEMDEBUG && ptr == CuPtr{Cvoid}(0xbbbbbbbbbbbbbbbb)
    Core.println("Freeing a scrubbed pointer!")
  end

  # this function is typically called from a finalizer, where we can't switch tasks,
  # so perform our own error handling.
  try
    # look up the memory block, and bail out if its refcount isn't 1
    block = @spinlock allocated_lock begin
        block, refcount = allocated[dev][ptr]
        if refcount == 1
          delete!(allocated[dev], ptr)
          block
        else
          # we can't actually free this block yet, so decrease its refcount and return
          allocated[dev][ptr] = block, refcount-1
          return
        end
    end

    # look up the allocation site
    if Base.JLOptions().debug_level >= 2
      @lock requested_lock begin
        @assert haskey(requested[dev], ptr)
        delete!(requested[dev], ptr)
      end
    end

    time = Base.@elapsed begin
      @pool_timeit "pooled free" @pooled free(dev, block)
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
function reclaim(sz::Int=typemax(Int))
  dev = device()
  @pooled reclaim(dev, sz)
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
    for phase in 1:4
      ret = $(esc(ex))
      $(esc(isfailed))(ret) || break

      # incrementally more costly reclaim of cached memory
      if phase == 1
        reclaim()
      elseif phase == 2
        GC.gc(false)
        reclaim()
      elseif phase == 3
        GC.gc(true)
        reclaim()
      elseif phase == 4 && async_alloc[]
        # this phase is unique to retry_reclaim, as regular allocations come from the pool
        # so are assumed to never need to trim its contents.
        trim(pool())
      end
    end
    ret
  end
end

# XXX: function version for use in CUDAdrv where we haven't loaded pool.jl yet
function retry_reclaim(f, check)
  @retry_reclaim check f()
end


## management

const last_use = PerDevice{Union{Nothing,Float64}}() do dev
  nothing
end

# reclaim unused pool memory after a certain time
function pool_cleanup()
  while true
    t1 = time()
    @pool_timeit "cleanup" for dev in devices()
      t0 = last_use[dev]
      t0 === nothing && continue

      if t1-t0 > 300
        # the pool hasn't been used for a while, so reclaim unused buffers
        @pooled reclaim(dev)
      end
    end

    sleep(60)
  end
end


## utilities

used_memory(dev=device()) = @lock allocated_lock begin
    mapreduce(sizeof∘first, +, values(allocated[dev]); init=0)
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
        CUDA.initialize_cuda_context()

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
         gpu_bytes=gpu_mem_stats.pool_alloc, gpu_gctime=gpu_mem_stats.pool_time, gpu_gcstate=gpu_mem_stats)
    end
end

cached_memory() = @pooled cached_memory()

"""
    memory_status([io=stdout])

Report to `io` on the memory status of the current GPU and the active memory pool.
"""
function memory_status(io::IO=stdout)
  dev = device()

  free_bytes, total_bytes = Mem.info()
  used_bytes = total_bytes - free_bytes
  used_ratio = used_bytes / total_bytes
  @printf(io, "Effective GPU memory usage: %.2f%% (%s/%s)\n",
              100*used_ratio, Base.format_bytes(used_bytes),
              Base.format_bytes(total_bytes))

  @printf(io, "CUDA allocator usage: %s", Base.format_bytes(usage[dev][]))
  if usage_limit[dev] !== typemax(Int)
    @printf(io, " (capped at %s)", Base.format_bytes(usage_limit[dev]))
  end
  println(io)

  alloc_used_bytes = used_memory()
  alloc_cached_bytes = cached_memory()
  alloc_total_bytes = alloc_used_bytes + alloc_cached_bytes
  @printf(io, "%s usage: %s (%s allocated, %s cached)\n", pool_name,
              Base.format_bytes(alloc_total_bytes), Base.format_bytes(alloc_used_bytes),
              Base.format_bytes(alloc_cached_bytes))

  # check if the memory usage as counted by the CUDA allocator wrapper
  # matches what is reported by the pool implementation
  discrepancy = Base.abs(usage[dev][] - alloc_total_bytes)
  if discrepancy != 0
    println(io, "Discrepancy of $(Base.format_bytes(discrepancy)) between memory pool and allocator!")
  end

  if Base.JLOptions().debug_level >= 2
    requested′, allocated′ = @lock requested_lock begin
      copy(requested[dev]), copy(allocated[dev])
    end
    for (ptr, bt) in requested′
      block = allocated′[ptr]
      @printf(io, "\nOutstanding memory allocation of %s at %p",
              Base.format_bytes(sizeof(block)), Int(ptr))
      stack = stacktrace(bt, false)
      StackTraces.remove_frames!(stack, :alloc)
      Base.show_backtrace(io, stack)
      println(io)
    end
  end
end


## init

function __init_pool__()
  # usage
  initialize!(usage, ndevices())
  initialize!(last_use, ndevices())
  initialize!(usage_limit, ndevices())

  # allocation tracking
  initialize!(allocated, ndevices())
  initialize!(requested, ndevices())

  # memory pool configuration
  default_pool = version() >= v"11.2" ? "cuda" : "binned"
  pool_name = get(ENV, "JULIA_CUDA_MEMORY_POOL", default_pool)
  active_pool[], async_alloc[] = if pool_name == "none"
      Pool.None, false
  elseif pool_name == "simple"
      Pool.Simple, false
  elseif pool_name == "binned"
      Pool.Binned, false
  elseif pool_name == "split"
      Pool.Split, false
  elseif pool_name == "cuda"
      @assert version() >= v"11.2" "The CUDA memory pool is only supported on CUDA 11.2+"
      Pool.None, true
  else
      error("Invalid memory pool '$pool_name'")
  end
  @pooled init()

  TimerOutputs.reset_timer!(alloc_to)
  TimerOutputs.reset_timer!(PoolUtils.to)

  if isinteractive()
    @async @pooled pool_cleanup()
  end
end
