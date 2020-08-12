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

# TODO: move the management thread one level up, to be shared by all allocators


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

const pool_lock = ReentrantLock()
const pools_used = PerDevice{Vector{Set{Block}}}((dev)->Vector{Set{Block}}())
const pools_avail = PerDevice{Vector{Vector{Block}}}((dev)->Vector{Vector{Block}}())

poolidx(n) = Base.ceil(Int, Base.log2(n))+1
poolsize(idx) = 2^(idx-1)

@assert poolsize(poolidx(MAX_POOL)) <= MAX_POOL "MAX_POOL cutoff should close a pool"

function create_pools(dev, idx)
  if length(pool_usage[dev]) >= idx
    # fast-path without taking a lock
    return
  end

  @lock pool_lock begin
    while length(pool_usage[dev]) < idx
      push!(pool_usage[dev], 1)
      push!(pool_history[dev], initial_usage)
      push!(pools_used[dev], Set{Block}())
      push!(pools_avail[dev], Vector{Block}())
    end
  end
end


## pooling

const initial_usage = Tuple(1 for _ in 1:USAGE_WINDOW)

const pool_usage = PerDevice{Vector{Float64}}((dev)->Vector{Float64}())
const pool_history = PerDevice{Vector{NTuple{USAGE_WINDOW,Float64}}}((dev)->Vector{NTuple{USAGE_WINDOW,Float64}}())

const freed_lock = NonReentrantLock()
const freed = PerDevice{Vector{Block}}((dev)->Vector{Block}())

# scan every pool and manage the usage history
#
# returns a boolean indicating whether any pool is active (this can be a false negative)
function pool_scan(dev)
  GC.gc(false) # quick, incremental collection

  active = false

  @lock pool_lock begin
    @inbounds for pid in 1:length(pool_history[dev])
      nused = length(pools_used[dev][pid])
      navail = length(pools_avail[dev][pid])
      history = pool_history[dev][pid]

      if nused+navail > 0
        usage = pool_usage[dev][pid]
        current_usage = nused / (nused + navail)

        # shift the history window with the recorded usage
        history = pool_history[dev][pid]
        pool_history[dev][pid] = (Base.tail(pool_history[dev][pid])..., usage)

        # reset the usage with the current one
        pool_usage[dev][pid] = current_usage

        if usage != current_usage
          active = true
        end
      else
        pool_usage[dev][pid] = 1
        pool_history[dev][pid] = initial_usage
      end
    end
  end

  active
end

# reclaim unused buffers
function pool_reclaim(dev, target_bytes::Int=typemax(Int); full::Bool=true)
  pool_repopulate(dev)

  @lock pool_lock begin
    # find inactive buffers
    @pool_timeit "scan" begin
      pools_inactive = Vector{Int}(undef, length(pools_avail[dev])) # pid => buffers that can be freed
      if full
        # consider all currently unused buffers
        for (pid, avail) in enumerate(pools_avail[dev])
          pools_inactive[pid] = length(avail)
        end
      else
        # only consider inactive buffers
        @inbounds for pid in 1:length(pool_usage[dev])
          nused = length(pools_used[dev][pid])
          navail = length(pools_avail[dev][pid])
          recent_usage = (pool_history[dev][pid]..., pool_usage[dev][pid])

          if navail > 0
            # reclaim as much as the usage allows
            reclaimable = Base.floor(Int, (1-maximum(recent_usage))*(nused+navail))
            pools_inactive[pid] = reclaimable
          else
            pools_inactive[pid] = 0
          end
        end
      end
    end

    # reclaim buffers (in reverse, to discard largest buffers first)
    @pool_timeit "reclaim" begin
      freed_bytes = 0
      for pid in reverse(eachindex(pools_inactive))
        bytes = poolsize(pid)
        avail = pools_avail[dev][pid]

        bufcount = pools_inactive[pid]
        @assert bufcount <= length(avail)
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
  blocks = @safe_lock freed_lock begin
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

      # update pool usage
      current_usage = length(used) / (length(used) + length(avail))
      pool_usage[dev][pid] = Base.max(pool_usage[dev][pid], current_usage)
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
    @pool_timeit "3. reclaim unused" begin
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
    @pool_timeit "6. reclaim unused" begin
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

      # update pool usage
      current_usage = length(used) / (length(avail) + length(used))
      pool_usage[dev][pid] = Base.max(pool_usage[dev][pid], current_usage)
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
  @safe_lock_spin freed_lock begin
    push!(freed[dev], block)
  end
end

function pool_init()
  initialize!(freed, ndevices())

  initialize!(pool_usage, ndevices())
  initialize!(pool_history, ndevices())

  initialize!(pools_used, ndevices())
  initialize!(pools_avail, ndevices())

  managed_str = if haskey(ENV, "JULIA_CUDA_MEMORY_POOL_MANAGED")
    ENV["JULIA_CUDA_MEMORY_POOL_MANAGED"]
  elseif haskey(ENV, "CUARRAYS_MANAGED_POOL")
    Base.depwarn("The CUARRAYS_MANAGED_POOL environment flag is deprecated, please use JULIA_CUDA_MEMORY_POOL_MANAGED instead.", :__init_pool__)
    ENV["CUARRAYS_MANAGED_POOL"]
  else
    nothing
  end

  managed = parse(Bool, something(managed_str, "true"))
  if managed
    delay = MIN_DELAY
    @async begin
      # HACK: wait until the initialization tests have executed (but this also makes sense
      #       for users, which are unlikely to need this management that early)
      sleep(60)

      while true
        dev = CuCurrentdevice()
        if dev !== nothing
          @pool_timeit "background task" begin
            if pool_scan(dev)
              delay = MIN_DELAY
            else
              delay = Base.min(delay*2, MAX_DELAY)
            end

            reclaim(full=false)
          end
        end

        sleep(delay)
      end
    end
  end
end

function cached_memory(dev=device())
  sz = @safe_lock freed_lock mapreduce(sizeof, +, freed[dev]; init=0)
  @lock pool_lock for (pid, pl) in enumerate(pools_avail[dev])
    bytes = poolsize(pid)
    sz += bytes * length(pl)
  end
  return sz
end
