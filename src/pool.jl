# GPU memory management and pooling

using Printf
using Logging
using TimerOutputs

using Base: @lock

# a simple non-reentrant lock that errors when trying to reenter on the same task
struct NonReentrantLock <: Threads.AbstractLock
  rl::ReentrantLock
  NonReentrantLock() = new(ReentrantLock())
end

function Base.lock(nrl::NonReentrantLock)
  @assert !islocked(nrl.rl) || nrl.rl.locked_by !== current_task()
  lock(nrl.rl)
end

function Base.trylock(nrl::NonReentrantLock)
  @assert !islocked(nrl.rl) || nrl.rl.locked_by !== current_task()
  trylock(nrl.rl)
end

Base.unlock(nrl::NonReentrantLock) = unlock(nrl.rl)

# the above lock is taken around code that might gc, which might reenter through finalizers.
# avoid that by temporarily disabling finalizers running concurrently on this thread.
enable_finalizers(on::Bool) = ccall(:jl_gc_enable_finalizers, Cvoid,
                                    (Ptr{Cvoid}, Int32,), Core.getptls(), on)
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

# if we actually want to acquire these locks from a finalizer, we can't just wait on them
# (which might cause a task switch). as the lock can only be taken by another thread that
# should be running, and not a concurrent task we'd need to switch to, we can safely spin.
macro safe_lock_spin(l, ex)
  quote
    temp = $(esc(l))
    while !trylock(temp)
      # we can't yield here
    end
    enable_finalizers(false) # retains compatibility with non-finalizer callers
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


## block of memory

@enum BlockState begin
    INVALID
    AVAILABLE
    ALLOCATED
    FREED
end

mutable struct Block
    buf::Mem.DeviceBuffer # base allocation
    sz::Int               # size into it
    off::Int              # offset into it

    state::BlockState
    prev::Union{Nothing,Block}
    next::Union{Nothing,Block}

    Block(buf, sz; off=0, state=INVALID, prev=nothing, next=nothing) =
        new(buf, sz, off, state, prev, next)
end

Base.sizeof(block::Block) = block.sz
Base.pointer(block::Block) = pointer(block.buf) + block.off

iswhole(block::Block) = block.prev === nothing && block.next === nothing

function Base.show(io::IO, block::Block)
    fields = [@sprintf("%p", Int(pointer(block)))]
    push!(fields, Base.format_bytes(sizeof(block)))
    push!(fields, "$(block.state)")
    block.off != 0 && push!(fields, "offset=$(block.off)")
    block.prev !== nothing && push!(fields, "prev=Block(offset=$(block.prev.off))")
    block.next !== nothing && push!(fields, "next=Block(offset=$(block.next.off))")

    print(io, "Block(", join(fields, ", "), ")")
end


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

function actual_alloc(dev::CuDevice, bytes::Integer)
  buf = @device! dev begin
    # check the memory allocation limit
    if usage[dev][] + bytes > usage_limit[dev]
      return nothing
    end

    # try the actual allocation
    try
      time = Base.@elapsed begin
        @timeit_debug alloc_to "alloc" begin
          buf = Mem.alloc(Mem.Device, bytes)
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
        Mem.free(block.buf)
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
# - pool_init()
# - pool_alloc(::CuDevice, sz)::Block
# - pool_free(::CuDevice, ::Block)
# - pool_reclaim(::CuDevice, nb::Int=typemax(Int))::Int
# - cached_memory()

const pool_name = get(ENV, "JULIA_CUDA_MEMORY_POOL", "binned")
let pool_path = joinpath(@__DIR__, "pool", "$(pool_name).jl")
  isfile(pool_path) || error("Unknown memory pool $pool_name")
  include(pool_path)
end


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

  dev = device()

  time = Base.@elapsed begin
    @pool_timeit "pooled alloc" block = pool_alloc(dev, sz)::Union{Nothing,Block}
  end
  block === nothing && throw(OutOfGPUMemoryError(sz))

  # record the memory block
  ptr = pointer(block)
  @safe_lock allocated_lock begin
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
  @safe_lock_spin allocated_lock begin
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
    block = @safe_lock_spin allocated_lock begin
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
      @pool_timeit "pooled free" pool_free(dev, block)
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
  pool_reclaim(dev, sz)
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
    for phase in 1:3
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
      end
    end
    ret
  end
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
        pool_reclaim(dev)
      end
    end

    sleep(60)
  end
end


## utilities

used_memory(dev=device()) = @safe_lock allocated_lock begin
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
        CUDA.prepare_cuda_call()

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
  runtime_pool_name = get(ENV, "JULIA_CUDA_MEMORY_POOL", "binned")
  if runtime_pool_name != pool_name
      error("Cannot use memory pool '$runtime_pool_name' when CUDA.jl was precompiled for memory pool '$pool_name'.")
  end
  pool_init()

  TimerOutputs.reset_timer!(alloc_to)
  TimerOutputs.reset_timer!(pool_to)

  if isinteractive()
    @async pool_cleanup()
  end
end
