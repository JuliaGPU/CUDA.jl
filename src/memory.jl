# pooled gpu memory allocator
#
# this allocator sits between CuArray constructors
# and the actual memory allocation in CUDAdrv.Mem
#
# the core design is pretty simple:
# - bin allocations according to their size (see `poolidx`)
# - when requested memory, check for previously allocated memory that has been released.
# - conversely, when released memory, put it aside for future use.
#
# to avoid consuming all available memory, and/or trashing the Julia GPU when running out:
# - keep track of free and used memory, in order to determine the usage of each pool
# - keep track of each pool's usage, as well as a window of previous usages.
# - regularly release memory from underused pools (see `reclaim(false)`).
#
# improvements:
# - pressure: have the `reclaim` background task reclaim more aggressively,
#             and call it from the failure cascade in `alloc`
# - context management: either switch contexts when performing memory operations,
#                       or just use unified memory for all allocations.
# - per-device pools

const lock_pools = ReentrantLock()


## core pool management

const pools_used = Vector{Set{Mem.Buffer}}()
const pools_free = Vector{Vector{Mem.Buffer}}()

poolidx(n) = ceil(Int, log2(n))+1
poolsize(idx) = 2^(idx-1)

function create_pools(idx)
  if length(pool_usage) >= idx
    return
  end

  lock(lock_pools) do
    while length(pool_usage) < idx
      push!(pool_usage, 1)
      push!(pool_history, initial_usage)
      push!(pools_used, Set{Mem.Buffer}())
      push!(pools_free, Vector{Mem.Buffer}())
    end
  end
end


## memory management

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

function __init_memory__()
  create_pools(30) # up to 512 MiB

  delay = MIN_DELAY
  @schedule begin
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

  atexit(()->begin
    Core.println("call_reclaim = $(call_reclaim)")
    Core.println("call_alloc = $(call_alloc)")
    Core.println("call_dealloc = $(call_dealloc)")
    Core.println("stat_free = $(stat_free)")
    Core.println("stat_alloc = $(stat_alloc)")
    Core.println("mem_free = $(mem_free)")
    Core.println("mem_alloc = $(mem_alloc)")
  end)
end

# scan every pool and manage the usage history
#
# returns a boolean indicating whether any pool is active, i.e. allocs or deallocs.
# this can be a false negative.
function scan()
  gc(false) # quick, incremental scan

  lock(lock_pools) do
    active = false

    @inbounds for pid in 1:length(pool_history)
      nfree = length(pools_free[pid])
      nused = length(pools_used[pid])
      history = pool_history[pid]

      if nfree+nused > 0
        usage = pool_usage[pid]
        current_usage = nused / (nfree + nused)

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
  global call_reclaim, stat_free, mem_free
  call_reclaim += 1

  lock(lock_pools) do
    if full
      # reclaim all currently unused buffers
      for (pid, pl) in enumerate(pools_free)
        for buf in pl
          stat_free += 1
          Mem.free(buf)
          mem_free += poolsize(pid)
        end
        empty!(pl)
      end
    else
      # only reclaim really unused buffers
      @inbounds for pid in 1:length(pool_usage)
        nfree = length(pools_free[pid])
        nused = length(pools_used[pid])
        recent_usage = (pool_history[pid]..., pool_usage[pid])

        if nfree > 0
          # reclaim as much as the usage allows
          reclaimable = floor(Int, (1-maximum(recent_usage))*(nfree+nused))
          @assert reclaimable <= nfree

          while reclaimable > 0
            buf = pop!(pools_free[pid])
            stat_free += 1
            Mem.free(buf)
            mem_free += poolsize(pid)
            reclaimable -= 1
          end
        end
      end
    end
  end
end

# debug stats
call_reclaim = 0
call_alloc = 0
call_dealloc = 0
stat_free = 0
stat_alloc = 0
mem_alloc = 0
mem_free = 0


## interface

function alloc(bytes)
  global call_alloc, stat_alloc, mem_alloc
  call_alloc += 1

  pid = poolidx(bytes)
  create_pools(pid)

  @inbounds free = pools_free[pid]
  @inbounds used = pools_used[pid]

  lock(lock_pools) do
    # 1. find an unused buffer in our pool
    buf = if !isempty(free)
      pop!(free)
    else
      try
        # 2. didn't have one, so allocate a new buffer
        stat_alloc += 1
        buf = Mem.alloc(poolsize(pid))
        mem_alloc += poolsize(pid)
        buf
      catch e
        e == CUDAdrv.CuError(2) || rethrow()
        # 3. that failed; make Julia collect objects and check do 1. again
        gc(true)
        if !isempty(free)
          buf = pop!(free)
        else
          # 4. didn't have one, so reclaim all other unused buffers and do 2. again
          reclaim(true)
          buf = Mem.alloc(poolsize(pid))
          mem_alloc += poolsize(pid)
          buf
        end
      end
    end

    push!(used, buf)

    current_usage = length(used) / (length(free) + length(used))
    pool_usage[pid] = max(pool_usage[pid], current_usage)

    buf
  end
end

function dealloc(buf, bytes)
  global call_dealloc
  call_dealloc += 1

  pid = poolidx(bytes)

  @inbounds used = pools_used[pid]
  @inbounds free = pools_free[pid]

  lock(lock_pools) do
    delete!(used, buf)

    push!(free, buf)

    current_usage = length(used) / (length(free) + length(used))
    pool_usage[pid] = max(pool_usage[pid], current_usage)
  end

  return
end
