import Base.GC: gc

# dynamic memory pool allocator
#
# this allocator sits between CuArray constructors
# and the actual memory allocation in CUDAdrv.Mem
#
# the core design is a pretty simple:
# - bin allocations into multiple pools according to their size (see `poolidx`)
# - when requested memory, check the pool for unused memory, or allocate dynamically
# - conversely, when released memory, put it in the appropriate pool for future use
#
# to avoid memory hogging and/or trashing the Julia GC:
# - keep track of used and available memory, in order to determine the usage of each pool
# - keep track of each pool's usage, as well as a window of previous usages
# - regularly release memory from underused pools (see `reclaim(false)`)
#
# possible improvements:
# - context management: either switch contexts when performing memory operations,
#                       or just use unified memory for all allocations.
# - per-device pools

const pool_lock = ReentrantLock()

function __init_pool_()
  pool_timings!()
end


## infrastructure

const pools_used = Vector{Set{Mem.Buffer}}()
const pools_avail = Vector{Vector{Mem.Buffer}}()

poolidx(n) = ceil(Int, log2(n))+1
poolsize(idx) = 2^(idx-1)

function create_pools(idx)
  if length(pool_usage) >= idx
    # fast-path without taking a lock
    return
  end

  lock(pool_lock) do
    while length(pool_usage) < idx
      push!(pool_usage, 1)
      push!(pool_history, initial_usage)
      push!(pools_used, Set{Mem.Buffer}())
      push!(pools_avail, Vector{Mem.Buffer}())
    end
  end
end


## timings

using TimerOutputs
const to = Ref{TimerOutput}()

function pool_timings!(new=TimerOutput())
  global to
  to[] = new
  return
end

pool_timings() = (TimerOutputs.print_timer(to[]; allocations=false); println())


## management

const USAGE_WINDOW = 5
const initial_usage = Tuple(1 for _ in 1:USAGE_WINDOW)

const pool_usage = Vector{Float64}()
const pool_history = Vector{NTuple{USAGE_WINDOW,Float64}}()

# min and max time between successive background task iterations.
# when the pool usages don't change, scan less regularly.
#
# together with USAGE_WINDOW, this determines how long it takes for objects to get reclaimed
const MIN_DELAY = 1.0
const MAX_DELAY = 5.0

# debug stats
mutable struct PoolStats
  # allocation requests
  req_nalloc::Int
  req_nfree::Int
  ## in bytes
  req_alloc::Int
  user_free::Int

  # actual allocations
  actual_nalloc::Int
  actual_nfree::Int
  ## in bytes
  actual_alloc::Int
  actual_free::Int

  cuda_time::Float64
  total_time::Float64
end
const stats = PoolStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
Base.copy(stats::PoolStats) =
  PoolStats((getfield(stats, field) for field in fieldnames(PoolStats))...)

# allocation traces
const tracing = parse(Bool, get(ENV, "CUARRAYS_TRACE_POOL", "false"))
const BackTrace = Vector{Union{Ptr{Nothing}, Base.InterpreterIP}}
const alloc_sites = Dict{Mem.Buffer, Tuple{Int, BackTrace}}()
const alloc_collectables = Dict{BackTrace, Tuple{Int, Int, Int}}()

function __init_memory__()
  create_pools(30) # up to 512 MiB

  managed = parse(Bool, get(ENV, "CUARRAYS_MANAGED_POOL", "true"))
  if managed
    delay = MIN_DELAY
    @async begin
      while true
        @timeit to[] "background task" lock(pool_lock) do
          if scan()
            delay = MIN_DELAY
          else
            delay = min(delay*2, MAX_DELAY)
          end

          reclaim()
        end

        sleep(delay)
      end
    end
  end

  verbose = haskey(ENV, "CUARRAYS_MANAGED_POOL")
  if verbose
    atexit(()->begin
      Core.println("""
        Pool statistics (managed: $(managed ? "yes" : "no")):
         - requested alloc/free: $(stats.req_nalloc)/$(stats.req_nfree) ($(Base.format_bytes(stats.req_nalloc))/$(Base.format_bytes(stats.req_free)))
         - actual alloc/free: $(stats.actual_nalloc)/$(stats.actual_nfree) ($(Base.format_bytes(stats.actual_alloc))/$(Base.format_bytes(stats.actual_free)))""")
    end)
  end
end

# scan every pool and manage the usage history
#
# returns a boolean indicating whether any pool is active (this can be a false negative)
function scan()
  gc(false) # quick, incremental collection

  active = false

  @inbounds for pid in 1:length(pool_history)
    nused = length(pools_used[pid])
    navail = length(pools_avail[pid])
    history = pool_history[pid]

    if nused+navail > 0
      usage = pool_usage[pid]
      current_usage = nused / (nused + navail)

      # shift the history window with the recorded usage
      history = pool_history[pid]
      pool_history[pid] = (Base.tail(pool_history[pid])..., usage)

      # reset the usage with the current one
      pool_usage[pid] = current_usage

      if usage != current_usage
        active = true
      end
    else
      pool_usage[pid] = 1
      pool_history[pid] = initial_usage
    end
  end

  active
end

# reclaim unused buffers
function reclaim(full::Bool=false, target_bytes::Int=typemax(Int))
  stats.total_time += Base.@elapsed begin
    # find inactive buffers
    @timeit to[] "scan" begin
      pools_inactive = Vector{Int}(undef, length(pools_avail)) # pid => buffers that can be freed
      if full
        # consider all currently unused buffers
        for (pid, avail) in enumerate(pools_avail)
          pools_inactive[pid] = length(avail)
        end
      else
        # only consider inactive buffers
        @inbounds for pid in 1:length(pool_usage)
          nused = length(pools_used[pid])
          navail = length(pools_avail[pid])
          recent_usage = (pool_history[pid]..., pool_usage[pid])

          if navail > 0
            # reclaim as much as the usage allows
            reclaimable = floor(Int, (1-maximum(recent_usage))*(nused+navail))
            pools_inactive[pid] = reclaimable
          else
            pools_inactive[pid] = 0
          end
        end
      end
    end

    # reclaim buffers (in reverse, to discard largest buffers first)
    @timeit to[] "reclaim" begin
      for pid in reverse(eachindex(pools_inactive))
        bytes = poolsize(pid)
        avail = pools_avail[pid]

        bufcount = pools_inactive[pid]
        @assert bufcount <= length(avail)
        for i in 1:bufcount
          buf = pop!(avail)

          stats.actual_nfree += 1
          stats.cuda_time += Base.@elapsed Mem.free(buf)
          stats.actual_free += bytes

          target_bytes -= bytes
          target_bytes <= 0 && return true
        end
      end
    end
  end

  return false
end


## allocator state machine

function try_cuda_alloc(bytes)
  buf = nothing
  try
    stats.cuda_time += Base.@elapsed begin
      buf = Mem.alloc(bytes)
    end
    stats.actual_nalloc += 1
    stats.actual_alloc += bytes
  catch ex
    ex == CUDAdrv.ERROR_OUT_OF_MEMORY || rethrow()
  end

  return buf
end

function try_alloc(bytes, pid=-1)
  # NOTE: checking the pool is really fast, and not included in the timings
  if pid != -1 && !isempty(pools_avail[pid])
    return pop!(pools_avail[pid])
  end

  @timeit to[] "1 try alloc" begin
    let buf = try_cuda_alloc(bytes)
      buf !== nothing && return buf
    end
  end

  # trace buffers that are ready to be collected by the Julia GC.
  # such objects hinder efficient memory management, and maybe should be `unsafe_free!`d
  if tracing
    alloc_sites_old = copy(alloc_sites)

    gc(true)

    for (buf, (bytes, bt)) in sort(collect(alloc_sites_old), by=x->x[2][1])
      if !haskey(alloc_sites, buf)
        if !haskey(alloc_collectables, bt)
          alloc_collectables[bt] = (1, bytes, bytes)
        else
          nalloc, _, total_bytes = alloc_collectables[bt]
          alloc_collectables[bt] = (nalloc+1, bytes, bytes+total_bytes)
        end
      end
    end
  end

  @timeit to[] "2 gc(false)" begin
    gc(false) # incremental collection
  end

  if pid != -1 && !isempty(pools_avail[pid])
    return pop!(pools_avail[pid])
  end

  # TODO: we could return a larger allocation here, but that increases memory pressure and
  #       would require proper block splitting + compaction to be any efficient.

  @timeit to[] "3 reclaim unused" begin
    reclaim(true, bytes)
  end

  @timeit to[] "4 try alloc" begin
    let buf = try_cuda_alloc(bytes)
      buf !== nothing && return buf
    end
  end

  @timeit to[] "5 gc(true)" begin
    gc(true) # full collection
  end

  if pid != -1 && !isempty(pools_avail[pid])
    return pop!(pools_avail[pid])
  end

  @timeit to[] "6 reclaim unused" begin
    reclaim(true, bytes)
  end

  @timeit to[] "7 try alloc" begin
    let buf = try_cuda_alloc(bytes)
      buf !== nothing && return buf
    end
  end

  @timeit to[] "8 reclaim everything" begin
    reclaim(true)
  end

  @timeit to[] "9 try alloc" begin
    let buf = try_cuda_alloc(bytes)
      buf !== nothing && return buf
    end
  end

  if tracing
    for (buf, (bytes, bt)) in alloc_sites
      st = stacktrace(bt, false)
      Core.print(Core.stderr, "WARNING: outstanding a GPU allocation of $(Base.format_bytes(bytes))")
      Base.show_backtrace(Core.stderr, st)
      Core.println(Core.stderr)
    end
  end

  throw(OutOfMemoryError())
end


## interface

const MAX_POOL = 100*1024^2 # 100 MiB

function alloc(bytes)
  # 0-byte allocations shouldn't hit the pool
  bytes == 0 && return Mem.alloc(0)

  stats.req_nalloc += 1
  stats.req_alloc += bytes
  stats.total_time += Base.@elapsed begin
    # only manage small allocations in the pool
    if bytes <= MAX_POOL
      pid = poolidx(bytes)
      create_pools(pid)
      alloc_bytes = poolsize(pid)

      @inbounds used = pools_used[pid]
      @inbounds avail = pools_avail[pid]

      lock(pool_lock) do
        buf = @timeit to[] "pooled alloc" try_alloc(alloc_bytes, pid)

        # mark the buffer as used
        push!(used, buf)

        # update pool usage
        current_usage = length(used) / (length(avail) + length(used))
        pool_usage[pid] = max(pool_usage[pid], current_usage)
      end
    else
      buf = @timeit to[] "large alloc" try_alloc(bytes)
    end
  end

  if tracing
    alloc_sites[buf] = (bytes, backtrace())
  end

  buf
end

function dealloc(buf, bytes)
  # 0-byte allocations shouldn't hit the pool
  bytes == 0 && return

  stats.req_nfree += 1
  stats.user_free += bytes
  stats.total_time += Base.@elapsed begin
    # was this a pooled buffer?
    if bytes <= MAX_POOL
      pid = poolidx(bytes)
      @assert pid <= length(pools_used)

      @inbounds used = pools_used[pid]
      @inbounds avail = pools_avail[pid]

      lock(pool_lock) do
        # mark the buffer as available
        delete!(used, buf)
        push!(avail, buf)

        # update pool usage
        current_usage = length(used) / (length(used) + length(avail))
        pool_usage[pid] = max(pool_usage[pid], current_usage)
      end
    else
      @timeit to[] "large dealloc" Mem.free(buf)
    end
  end

  if tracing
    delete!(alloc_sites, buf)
  end

  return
end


## utilities

using Printf

macro allocated(ex)
    quote
        let
            local f
            function f()
                b0 = stats.req_alloc
                $(esc(ex))
                stats.req_alloc - b0
            end
            f()
        end
    end
end

macro time(ex)
    quote
        local gpu_mem_stats0 = copy(stats)
        local cpu_mem_stats0 = Base.gc_num()
        local cpu_time0 = time_ns()

        local val = $(esc(ex))

        local cpu_time1 = time_ns()
        local cpu_mem_stats1 = Base.gc_num()
        local gpu_mem_stats1 = copy(stats)

        local cpu_time = (cpu_time1 - cpu_time0) / 1e9
        local gpu_gc_time = gpu_mem_stats1.total_time - gpu_mem_stats0.total_time
        local gpu_lib_time = gpu_mem_stats1.cuda_time - gpu_mem_stats0.cuda_time
        local gpu_alloc_count = gpu_mem_stats1.req_nalloc - gpu_mem_stats0.req_nalloc
        local gpu_alloc_size = gpu_mem_stats1.req_alloc - gpu_mem_stats0.req_alloc
        local cpu_mem_stats = Base.GC_Diff(cpu_mem_stats1, cpu_mem_stats0)
        local cpu_gc_time = cpu_mem_stats.total_time / 1e9
        local cpu_alloc_count = Base.gc_alloc_count(cpu_mem_stats)
        local cpu_alloc_size = cpu_mem_stats.allocd

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

function pool_status()
  used_pool_buffers = 0
  used_pool_bytes = 0
  for (pid, pl) in enumerate(pools_used)
    bytes = poolsize(pid)
    used_pool_buffers += length(pl)
    used_pool_bytes += bytes * length(pl)
  end

  avail_pool_buffers = 0
  avail_pool_bytes = 0
  for (pid, pl) in enumerate(pools_avail)
    bytes = poolsize(pid)
    avail_pool_buffers += length(pl)
    avail_pool_bytes += bytes * length(pl)
  end

  free_bytes, total_bytes = CUDAdrv.Mem.info()
  used_bytes = total_bytes - free_bytes
  used_ratio = used_bytes / total_bytes

  pool_ratio = (used_pool_bytes + avail_pool_bytes) / used_bytes

  println("Total GPU memory usage: $(100*round(used_ratio; digits=2))% ($(Base.format_bytes(used_bytes))/$(Base.format_bytes(total_bytes)))")
  println("CuArrays.jl pool usage: $(100*round(pool_ratio; digits=2))% ($(Base.format_bytes(used_pool_bytes)) in use by $used_pool_buffers buffer(s), $(Base.format_bytes(avail_pool_bytes)) idle)")

  return
end

function pool_collectables()
  if !tracing
    error("Allocation tracing disabled, please start Julia and precompile CuArrays.jl with CUARRAYS_TRACE_POOL=1")
  end

  for (bt, (nalloc, bytes, total_bytes)) in sort(collect(alloc_collectables), by=x->x[2][3])
    st = stacktrace(bt, false)
    print("Eagerly collecting the following $nalloc GPU allocations of each $(Base.format_bytes(bytes)) would unblock $(Base.format_bytes(total_bytes)):")
    Base.show_backtrace(stdout, st)
    println()
  end
end
