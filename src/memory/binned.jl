module BinnedPool

# binned memory pool allocator
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

using ..CuArrays
using ..CuArrays: @pool_timeit

using CUDAdrv

const pool_lock = ReentrantLock()


## tunables

const MAX_POOL = 2^27 # 128 MiB

const USAGE_WINDOW = 5

# min and max time between successive background task iterations.
# when the pool usages don't change, scan less regularly.
#
# together with USAGE_WINDOW, this determines how long it takes for objects to get reclaimed
const MIN_DELAY = 1.0
const MAX_DELAY = 5.0


## block of memory

struct Block
    ptr::CuPtr{Nothing}
    sz::Int
end

Base.pointer(block::Block) = block.ptr
Base.sizeof(block::Block) = block.sz

@inline function actual_alloc(sz)
    ptr = CuArrays.actual_alloc(sz)
    block = ptr === nothing ? nothing : Block(ptr, sz)
end

function actual_free(block::Block)
    CuArrays.actual_free(pointer(block))
    return
end


## infrastructure

const pools_used = Vector{Set{Block}}()
const pools_avail = Vector{Vector{Block}}()
const allocated = Dict{CuPtr{Nothing},Block}()

poolidx(n) = ceil(Int, log2(n))+1
poolsize(idx) = 2^(idx-1)

@assert poolsize(poolidx(MAX_POOL)) <= MAX_POOL "MAX_POOL cutoff should close a pool"

function create_pools(idx)
  if length(pool_usage) >= idx
    # fast-path without taking a lock
    return
  end

  lock(pool_lock) do
    while length(pool_usage) < idx
      push!(pool_usage, 1)
      push!(pool_history, initial_usage)
      push!(pools_used, Set{Block}())
      push!(pools_avail, Vector{Block}())
    end
  end
end


## management

const initial_usage = Tuple(1 for _ in 1:USAGE_WINDOW)

const pool_usage = Vector{Float64}()
const pool_history = Vector{NTuple{USAGE_WINDOW,Float64}}()

# scan every pool and manage the usage history
#
# returns a boolean indicating whether any pool is active (this can be a false negative)
function scan()
  GC.gc(false) # quick, incremental collection

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
  # find inactive buffers
  @pool_timeit "scan" begin
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
  @pool_timeit "reclaim" begin
    for pid in reverse(eachindex(pools_inactive))
      bytes = poolsize(pid)
      avail = pools_avail[pid]

      bufcount = pools_inactive[pid]
      @assert bufcount <= length(avail)
      for i in 1:bufcount
        block = pop!(avail)

        actual_free(block)

        target_bytes -= bytes
        target_bytes <= 0 && return true
      end
    end
  end

  return false
end


## allocator state machine

function pool_alloc(bytes, pid=-1)
  # NOTE: checking the pool is really fast, and not included in the timings
  if pid != -1 && !isempty(pools_avail[pid])
    return pop!(pools_avail[pid])
  end

  @pool_timeit "1. try alloc" begin
    let block = actual_alloc(bytes)
      block !== nothing && return block
    end
  end

  @pool_timeit "2. gc (incremental)" begin
    GC.gc(VERSION >= v"1.4.0-DEV.257" ? GC.Incremental : false)
  end

  if pid != -1 && !isempty(pools_avail[pid])
    return pop!(pools_avail[pid])
  end

  # TODO: we could return a larger allocation here, but that increases memory pressure and
  #       would require proper block splitting + compaction to be any efficient.

  @pool_timeit "3. reclaim unused" begin
    reclaim(true, bytes)
  end

  @pool_timeit "4. try alloc" begin
    let block = actual_alloc(bytes)
      block !== nothing && return block
    end
  end

  @pool_timeit "5. gc (full)" begin
    GC.gc(VERSION >= v"1.4.0-DEV.257" ? GC.Incremental : false)
  end

  if pid != -1 && !isempty(pools_avail[pid])
    return pop!(pools_avail[pid])
  end

  @pool_timeit "6. reclaim unused" begin
    reclaim(true, bytes)
  end

  @pool_timeit "7. try alloc" begin
    let block = actual_alloc(bytes)
      block !== nothing && return block
    end
  end

  @pool_timeit "8. reclaim everything" begin
    reclaim(true)
  end

  @pool_timeit "9. try alloc" begin
    let block = actual_alloc(bytes)
      block !== nothing && return block
    end
  end

  return nothing
end


## interface

function init()
  create_pools(30) # up to 512 MiB

  managed = parse(Bool, get(ENV, "CUARRAYS_MANAGED_POOL", "true"))
  if managed
    delay = MIN_DELAY
    @async begin
      while true
        @pool_timeit "background task" lock(pool_lock) do
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
end

deinit() = error("Not implemented")

function alloc(bytes)
  # only manage small allocations in the pool
  if bytes <= MAX_POOL
    pid = poolidx(bytes)
    create_pools(pid)
    alloc_bytes = poolsize(pid)

    @inbounds used = pools_used[pid]
    @inbounds avail = pools_avail[pid]

    lock(pool_lock) do
      block = pool_alloc(alloc_bytes, pid)

      if block !== nothing
        # mark the buffer as used
        push!(used, block)

        # update pool usage
        current_usage = length(used) / (length(avail) + length(used))
        pool_usage[pid] = max(pool_usage[pid], current_usage)
      end
    end
  else
    block = pool_alloc(bytes)
  end

  if block !== nothing
    ptr = pointer(block)
    allocated[ptr] = block
    return ptr
  else
    return nothing
  end
end

function free(ptr)
  block = allocated[ptr]
  delete!(allocated, ptr)
  bytes = sizeof(block)

  # was this a pooled buffer?
  if bytes <= MAX_POOL
    pid = poolidx(bytes)
    @assert pid <= length(pools_used)

    @inbounds used = pools_used[pid]
    @inbounds avail = pools_avail[pid]

    lock(pool_lock) do
      # mark the buffer as available
      delete!(used, block)
      push!(avail, block)

      # update pool usage
      current_usage = length(used) / (length(used) + length(avail))
      pool_usage[pid] = max(pool_usage[pid], current_usage)
    end
  else
    actual_free(block)
  end

  return
end

function used_memory()
  sz = 0
  for (pid, pl) in enumerate(pools_used)
    bytes = poolsize(pid)
    sz += bytes * length(pl)
  end

  return sz
end

function cached_memory()
  sz = 0
  for (pid, pl) in enumerate(pools_avail)
    bytes = poolsize(pid)
    sz += bytes * length(pl)
  end

  return sz
end

dump() = return

end
