# binned memory pool allocator
#
# the core design is a pretty simple:
# - bin allocations into multiple pools according to their size (see `poolidx`)
# - when requested memory, check the pool for unused memory, or allocate dynamically
# - conversely, when released memory, put it in the appropriate pool for future use


## tunables

const MAX_POOL = 2^27 # 128 MiB

const USAGE_WINDOW = 5

# min and max time between successive background task iterations.
# when the pool usages don't change, scan less regularly.
#
# together with USAGE_WINDOW, this determines how long it takes for objects to get reclaimed
const MIN_DELAY = 1.0
const MAX_DELAY = 5.0


## infrastructure

Base.@kwdef struct BinnedPool <: AbstractPool
  dev::CuDevice
  stream_ordered::Bool

  # TODO: npools
  lock::ReentrantLock = ReentrantLock()
  active::Vector{Set{Block}} = Vector{Set{Block}}()
  cache::Vector{Vector{Block}} = Vector{Vector{Block}}()

  freed_lock::NonReentrantLock = NonReentrantLock()
  freed::Vector{Block} = Vector{Block}()
end

poolidx(n) = Base.ceil(Int, Base.log2(n))+1
poolsize(idx) = 2^(idx-1)

@assert poolsize(poolidx(MAX_POOL)) <= MAX_POOL "MAX_POOL cutoff should close a pool"

function create_pools(pool::BinnedPool, idx)
  if length(pool.cache) >= idx
    # fast-path without taking a lock
    return
  end

  @lock pool.lock begin
    while length(pool.cache) < idx
      push!(pool.active, Set{Block}())
      push!(pool.cache, Vector{Block}())
    end
  end
end


## pooling

# reclaim unused buffers
function reclaim(pool::BinnedPool, target_bytes::Int=typemax(Int))
  pool_repopulate(pool)

  @lock pool.lock begin
    @pool_timeit "reclaim" begin
      freed_bytes = 0

      # process pools in reverse, to discard largest buffers first
      for pid in reverse(1:length(pool.cache))
        bytes = poolsize(pid)
        cache = pool.cache[pid]

        bufcount = length(cache)
        for i in 1:bufcount
          block = pop!(cache)

          actual_free(pool.dev, block; pool.stream_ordered)

          freed_bytes += bytes
          if freed_bytes >= target_bytes
            return freed_bytes
          end
        end
      end

      return freed_bytes
    end
  end
end

# repopulate the "available" pools from the list of freed blocks
function pool_repopulate(pool::BinnedPool)
  blocks = @lock pool.freed_lock begin
    isempty(pool.freed) && return
    blocks = Set(pool.freed)
    empty!(pool.freed)
    blocks
  end

  @lock pool.lock begin
    for block in blocks
      pid = poolidx(sizeof(block))

      @inbounds active = pool.active[pid]
      @inbounds cache = pool.cache[pid]

      # mark the buffer as available
      delete!(active, block)
      push!(cache, block)
    end
  end

  return
end

function alloc(pool::BinnedPool, bytes)
  if bytes <= MAX_POOL
    pid = poolidx(bytes)
    create_pools(pool, pid)
    bytes = poolsize(pid)
  else
    pid = -1
  end

  block = nothing

  # NOTE: checking the pool is really fast, and not included in the timings
  @lock pool.lock begin
    if pid != -1 && !isempty(pool.cache[pid])
      block = pop!(pool.cache[pid])
    end
  end

  if block === nothing
    @pool_timeit "0. repopulate" begin
      pool_repopulate(pool)
    end

    @lock pool.lock begin
      if pid != -1 && !isempty(pool.cache[pid])
        block = pop!(pool.cache[pid])
      end
    end
  end

  if block === nothing
    @pool_timeit "1. try alloc" begin
      block = actual_alloc(pool.dev, bytes; pool.stream_ordered)
    end
  end

  if block === nothing
    @pool_timeit "2a. gc (incremental)" begin
      GC.gc(false)
    end

    @pool_timeit "2b. repopulate" begin
      pool_repopulate(pool)
    end

    @lock pool.lock begin
      if pid != -1 && !isempty(pool.cache[pid])
        block = pop!(pool.cache[pid])
      end
    end
  end

  # TODO: we could return a larger allocation here, but that increases memory pressure and
  #       would require proper block splitting + compaction to be any efficient.

  if block === nothing
    @pool_timeit "3. reclaim" begin
      reclaim(pool, bytes)
    end

    @pool_timeit "4. try alloc" begin
      block = actual_alloc(pool.dev, bytes; pool.stream_ordered)
    end
  end

  if block === nothing
    @pool_timeit "5a. gc (full)" begin
      GC.gc(true)
    end

    @pool_timeit "5b. repopulate" begin
      pool_repopulate(pool)
    end

    @lock pool.lock begin
      if pid != -1 && !isempty(pool.cache[pid])
        block = pop!(pool.cache[pid])
      end
    end
  end

  if block === nothing
    @pool_timeit "6. reclaim" begin
      reclaim(pool, bytes)
    end

    @pool_timeit "7. try alloc" begin
      block = actual_alloc(pool.dev, bytes; pool.stream_ordered)
    end
  end

  if block === nothing
    @pool_timeit "8. reclaim everything" begin
      reclaim(pool, typemax(Int))
    end

    @pool_timeit "9. try alloc" begin
      block = actual_alloc(pool.dev, bytes, true; pool.stream_ordered)
    end
  end

  if block !== nothing && pid != -1
    @lock pool.lock begin
      @inbounds active = pool.active[pid]
      @inbounds cache = pool.cache[pid]

      # mark the buffer as active
      push!(active, block)
    end
  end

  return block
end

function free(pool::BinnedPool, block)
  # was this a pooled buffer?
  bytes = sizeof(block)
  if bytes > MAX_POOL
    actual_free(pool.dev, block; pool.stream_ordered)
    return
  end

  # we don't do any work here to reduce pressure on the GC (spending time in finalizers)
  # and to simplify locking (preventing concurrent access during GC interventions)
  @spinlock pool.freed_lock begin
    push!(pool.freed, block)
  end
end

function cached_memory(pool::BinnedPool)
  sz = @lock pool.freed_lock mapreduce(sizeof, +, pool.freed; init=0)
  @lock pool.lock for (pid, pl) in enumerate(pool.cache)
    bytes = poolsize(pid)
    sz += bytes * length(pl)
  end
  return sz
end
