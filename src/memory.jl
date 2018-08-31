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
# - pressure: have the `reclaim` background task reclaim more aggressively,
#             and call it from the failure cascade in `alloc`
# - context management: either switch contexts when performing memory operations,
#                       or just use unified memory for all allocations.
# - per-device pools

const pool_lock = ReentrantLock()


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

  # internal stats
  alloc_1::Int
  alloc_2::Int
  alloc_3::Int
  alloc_4::Int
end
const pool_stats = PoolStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

function __init_memory__()
  create_pools(30) # up to 512 MiB

  managed = parse(Bool, get(ENV, "CUARRAYS_MANAGED_POOL", "true"))
  if managed
    delay = MIN_DELAY
    @async begin
      while true
        if scan()
          delay = MIN_DELAY
        else
          delay = min(delay*2, MAX_DELAY)
        end

        reclaim()

        sleep(delay)
      end
    end
  end

  verbose = haskey(ENV, "CUARRAYS_MANAGED_POOL")
  if verbose
    atexit(()->begin
      Core.println("""
        Pool statistics (managed: $(managed ? "yes" : "no")):
         - requested alloc/free: $(pool_stats.req_nalloc)/$(pool_stats.req_nfree) ($(pool_stats.req_nlloc)/$(pool_stats.req_free) bytes)
         - actual alloc/free: $(pool_stats.actual_nalloc)/$(pool_stats.actual_nfree) ($(pool_stats.actual_alloc)/$(pool_stats.actual_free) bytes)
         - alloc types: $(pool_stats.alloc_1) $(pool_stats.alloc_2) $(pool_stats.alloc_3) $(pool_stats.alloc_4)""")
    end)
  end

end

# scan every pool and manage the usage history
#
# returns a boolean indicating whether any pool is active (this can be a false negative)
function scan()
  gc(false) # quick, incremental collection

  lock(pool_lock) do
    active = false

    @inbounds for pid in 1:length(pool_history)
      nused = length(pools_used[pid])
      navail = length(pools_avail[pid])
      history = pool_history[pid]

      if nused+navail > 0
        usage = pool_usage[pid]
        current_usage = nused / (nused + navail)

        if any(usage->usage != current_usage, history)
          # shift the history window with the recorded usage
          history = pool_history[pid]
          pool_history[pid] = (Base.tail(pool_history[pid])..., usage)

          # reset the usage with the current one
          pool_usage[pid] = current_usage
        end

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
end

# reclaim free objects
function reclaim(full::Bool=false)
  lock(pool_lock) do
    if full
      # reclaim all currently unused buffers
      for (pid, pl) in enumerate(pools_avail)
        for buf in pl
          pool_stats.actual_nfree += 1
          Mem.free(buf)
          pool_stats.actual_free += poolsize(pid)
        end
        empty!(pl)
      end
    else
      # only reclaim really unused buffers
      @inbounds for pid in 1:length(pool_usage)
        nused = length(pools_used[pid])
        navail = length(pools_avail[pid])
        recent_usage = (pool_history[pid]..., pool_usage[pid])

        if navail > 0
          # reclaim as much as the usage allows
          reclaimable = floor(Int, (1-maximum(recent_usage))*(nused+navail))
          @assert reclaimable <= navail

          while reclaimable > 0
            buf = pop!(pools_avail[pid])
            pool_stats.actual_nfree += 1
            Mem.free(buf)
            pool_stats.actual_free += poolsize(pid)
            reclaimable -= 1
          end
        end
      end
    end
  end
end


## interface

function alloc(bytes)
  # 0-byte allocations shouldn't hit the pool
  bytes == 0 && return Mem.alloc(0)
  pool_stats.req_nalloc += 1
  pool_stats.req_alloc += bytes

  pid = poolidx(bytes)
  create_pools(pid)

  @inbounds used = pools_used[pid]
  @inbounds avail = pools_avail[pid]

  lock(pool_lock) do
    # 1. find an unused buffer in our pool
    buf = if !isempty(avail)
      pool_stats.alloc_1 += 1
      pop!(avail)
    else
      try
        # 2. didn't have one, so allocate a new buffer
        buf = Mem.alloc(poolsize(pid))
        pool_stats.alloc_2 += 1
        pool_stats.actual_nalloc += 1
        pool_stats.actual_alloc += poolsize(pid)
        buf
      catch e
        e == CUDAdrv.CuError(2) || rethrow()
        # 3. that failed; make Julia collect objects and check do 1. again
        gc(true) # full collection
        if !isempty(avail)
          pool_stats.alloc_3 += 1
          buf = pop!(avail)
        else
          # 4. didn't have one, so reclaim all other unused buffers and do 2. again
          reclaim(true)
          buf = Mem.alloc(poolsize(pid))
          pool_stats.alloc_4 += 1
          pool_stats.actual_nalloc += 1
          pool_stats.actual_alloc += poolsize(pid)
          buf
        end
      end
    end

    push!(used, buf)

    current_usage = length(used) / (length(avail) + length(used))
    pool_usage[pid] = max(pool_usage[pid], current_usage)

    buf
  end
end

function dealloc(buf, bytes)
  pool_stats.req_nfree += 1

  # 0-byte allocations shouldn't hit the pool
  bytes == 0 && return Mem.alloc(0)
  pool_stats.user_free += bytes

  pid = poolidx(bytes)

  @inbounds used = pools_used[pid]
  @inbounds avail = pools_avail[pid]

  lock(pool_lock) do
    delete!(used, buf)

    push!(avail, buf)

    current_usage = length(used) / (length(used) + length(avail))
    pool_usage[pid] = max(pool_usage[pid], current_usage)
  end

  return
end


## utility macros

macro allocated(ex)
    quote
        let
            local f
            function f()
                b0 = pool_stats.req_alloc
                $(esc(ex))
                pool_stats.req_alloc - b0
            end
            f()
        end
    end
end
