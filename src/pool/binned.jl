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

# TODO: npools
const pool_lock = ReentrantLock()
const pools_used = PerDevice{Vector{Set{Block}}}((dev)->Vector{Set{Block}}())
const pools_avail = PerDevice{Vector{Vector{Block}}}((dev)->Vector{Vector{Block}}())

poolidx(n) = Base.ceil(Int, Base.log2(n))+1
poolsize(idx) = 2^(idx-1)

@assert poolsize(poolidx(MAX_POOL)) <= MAX_POOL "MAX_POOL cutoff should close a pool"

function create_pools(dev, idx)
  if length(pools_avail[dev]) >= idx
    # fast-path without taking a lock
    return
  end

  @lock pool_lock begin
    while length(pools_avail[dev]) < idx
      push!(pools_used[dev], Set{Block}())
      push!(pools_avail[dev], Vector{Block}())
    end
  end
end


## pooling

const freed_lock = NonReentrantLock()
const freed = PerDevice{Vector{Block}}((dev)->Vector{Block}())

# reclaim unused buffers
function pool_reclaim(dev, target_bytes::Int=typemax(Int))
  pool_repopulate(dev)

  @lock pool_lock begin
    @pool_timeit "reclaim" begin
      freed_bytes = 0

      # process pools in reverse, to discard largest buffers first
      for pid in reverse(1:length(pools_avail[dev]))
        bytes = poolsize(pid)
        avail = pools_avail[dev][pid]

        bufcount = length(avail)
        for i in 1:bufcount
          block = pop!(avail)

          actual_free(dev, block)

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
function pool_repopulate(dev)
  blocks = @lock freed_lock begin
    isempty(freed[dev]) && return
    blocks = Set(freed[dev])
    empty!(freed[dev])
    blocks
  end

  @lock pool_lock begin
    for block in blocks
      pid = poolidx(sizeof(block))

      @inbounds used = pools_used[dev][pid]
      @inbounds avail = pools_avail[dev][pid]

      # mark the buffer as available
      delete!(used, block)
      push!(avail, block)
    end
  end

  return
end

function pool_alloc(dev, bytes)
  if bytes <= MAX_POOL
    pid = poolidx(bytes)
    create_pools(dev, pid)
    bytes = poolsize(pid)
  else
    pid = -1
  end

  block = nothing

  # NOTE: checking the pool is really fast, and not included in the timings
  @lock pool_lock begin
    if pid != -1 && !isempty(pools_avail[dev][pid])
      block = pop!(pools_avail[dev][pid])
    end
  end

  if block === nothing
    @pool_timeit "0. repopulate" begin
      pool_repopulate(dev)
    end

    @lock pool_lock begin
      if pid != -1 && !isempty(pools_avail[dev][pid])
        block = pop!(pools_avail[dev][pid])
      end
    end
  end

  if block === nothing
    @pool_timeit "1. try alloc" begin
      block = actual_alloc(dev, bytes)
    end
  end

  if block === nothing
    @pool_timeit "2a. gc (incremental)" begin
      GC.gc(false)
    end

    @pool_timeit "2b. repopulate" begin
      pool_repopulate(dev)
    end

    @lock pool_lock begin
      if pid != -1 && !isempty(pools_avail[dev][pid])
        block = pop!(pools_avail[dev][pid])
      end
    end
  end

  # TODO: we could return a larger allocation here, but that increases memory pressure and
  #       would require proper block splitting + compaction to be any efficient.

  if block === nothing
    @pool_timeit "3. reclaim" begin
      pool_reclaim(dev, bytes)
    end

    @pool_timeit "4. try alloc" begin
      block = actual_alloc(dev, bytes)
    end
  end

  if block === nothing
    @pool_timeit "5a. gc (full)" begin
      GC.gc(true)
    end

    @pool_timeit "5b. repopulate" begin
      pool_repopulate(dev)
    end

    @lock pool_lock begin
      if pid != -1 && !isempty(pools_avail[dev][pid])
        block = pop!(pools_avail[dev][pid])
      end
    end
  end

  if block === nothing
    @pool_timeit "6. reclaim" begin
      pool_reclaim(dev, bytes)
    end

    @pool_timeit "7. try alloc" begin
      block = actual_alloc(dev, bytes)
    end
  end

  if block === nothing
    @pool_timeit "8. reclaim everything" begin
      pool_reclaim(dev, typemax(Int))
    end

    @pool_timeit "9. try alloc" begin
      block = actual_alloc(dev, bytes)
    end
  end

  if block !== nothing && pid != -1
    @lock pool_lock begin
      @inbounds used = pools_used[dev][pid]
      @inbounds avail = pools_avail[dev][pid]

      # mark the buffer as used
      push!(used, block)
    end
  end

  return block
end

function pool_free(dev, block)
  # was this a pooled buffer?
  bytes = sizeof(block)
  if bytes > MAX_POOL
    actual_free(dev, block)
    return
  end

  # we don't do any work here to reduce pressure on the GC (spending time in finalizers)
  # and to simplify locking (preventing concurrent access during GC interventions)
  @spinlock freed_lock begin
    push!(freed[dev], block)
  end
end

function pool_init()
  initialize!(freed, ndevices())

  initialize!(pools_used, ndevices())
  initialize!(pools_avail, ndevices())
end

function cached_memory(dev=device())
  sz = @lock freed_lock mapreduce(sizeof, +, freed[dev]; init=0)
  @lock pool_lock for (pid, pl) in enumerate(pools_avail[dev])
    bytes = poolsize(pid)
    sz += bytes * length(pl)
  end
  return sz
end
