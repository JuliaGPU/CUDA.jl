# pooled gpu memory allocator
#
# this allocator sits between CuArray constructors
# and the actual memory allocation in CUDAdrv.Mem
#
# the core design is pretty simple:
# - bin allocations according to their size (see `poolidx`)
# - when requested memory, check for previously allocated memory that has been released
# - conversely, when released memory, put it aside for future use
#
# to avoid consuming all available memory, and/or trashing the Julia GC when running out:
# - keep track of free and used memory, in order to determine the usage of each pool
# - keep track of each pool's usage, as well as a window of previous usages
# - regularly release memory from underused pools (see `reclaim(false)`)
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
const pools_avail = Vector{Vector{Mem.Buffer}}()

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
      push!(pools_avail, Vector{Mem.Buffer}())
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

# debug stats
req_alloc = 0
req_free = 0
actual_alloc = 0
actual_free = 0
amount_alloc = 0
amount_free = 0
alloc_1 = 0
alloc_2 = 0
alloc_3 = 0
alloc_4 = 0

function __init_memory__()
  create_pools(30) # up to 512 MiB

  if parse(Bool, get(ENV, "CUARRAYS_MANAGED_POOL", "true"))
    Core.println("Managing pool")
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
  end

  atexit(()->begin
    Core.println("req_alloc: $req_alloc")
    Core.println("req_free: $req_free")
    Core.println("actual_alloc: $actual_alloc")
    Core.println("actual_free: $actual_free")
    Core.println("amount_alloc: $amount_alloc")
    Core.println("amount_free: $amount_free")
    Core.println("alloc types: $alloc_1 $alloc_2 $alloc_3 $alloc_4")
  end)
end

# scan every pool and manage the usage history
#
# returns a boolean indicating whether any pool is active, i.e. allocs or deallocs.
# this can be a false negative.
function scan()
  gc(false) # quick, incremental collection

  lock(lock_pools) do
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
  global actual_free, amount_free

  lock(lock_pools) do
    if full
      # reclaim all currently unused buffers
      for (pid, pl) in enumerate(pools_avail)
        for buf in pl
          actual_free += 1
          Mem.free(buf)
          amount_free += poolsize(pid)
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
            actual_free += 1
            Mem.free(buf)
            amount_free += poolsize(pid)
            reclaimable -= 1
          end
        end
      end
    end
  end
end


## interface

function alloc(bytes)
  global req_alloc, alloc_1, alloc_2, alloc_3, alloc_4, actual_alloc, amount_alloc
  req_alloc += 1

  pid = poolidx(bytes)
  create_pools(pid)

  @inbounds used = pools_used[pid]
  @inbounds avail = pools_avail[pid]

  lock(lock_pools) do
    # 1. find an unused buffer in our pool
    buf = if !isempty(avail)
      alloc_1 += 1
      pop!(avail)
    else
      try
        # 2. didn't have one, so allocate a new buffer
        buf = Mem.alloc(poolsize(pid))
        alloc_2 += 1
        actual_alloc += 1
        amount_alloc += poolsize(pid)
        buf
      catch e
        e == CUDAdrv.CuError(2) || rethrow()
        # 3. that failed; make Julia collect objects and check do 1. again
        gc(true) # full collection
        if !isempty(avail)
          alloc_3 += 1
          buf = pop!(avail)
        else
          # 4. didn't have one, so reclaim all other unused buffers and do 2. again
          reclaim(true)
          buf = Mem.alloc(poolsize(pid))
          alloc_4 += 1
          actual_alloc += 1
          amount_alloc += poolsize(pid)
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
  global req_free
  req_free += 1

  pid = poolidx(bytes)

  @inbounds used = pools_used[pid]
  @inbounds avail = pools_avail[pid]

  lock(lock_pools) do
    delete!(used, buf)

    push!(avail, buf)

    current_usage = length(used) / (length(used) + length(avail))
    pool_usage[pid] = max(pool_usage[pid], current_usage)
  end

  return
end
