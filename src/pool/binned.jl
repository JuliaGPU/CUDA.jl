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

# TODO: move the management thread one level up, to be shared by all allocators

using ..CUDA
using ..CUDA: @pool_timeit, @safe_lock, @safe_lock_spin, NonReentrantLock

using Base: @lock


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
    ptr = CUDA.actual_alloc(sz)
    block = ptr === nothing ? nothing : Block(ptr, sz)
end

function actual_free(block::Block)
    CUDA.actual_free(pointer(block))
    return
end


## infrastructure

const pool_lock = ReentrantLock()
const pools_used = Vector{Set{Block}}()
const pools_avail = Vector{Vector{Block}}()

poolidx(n) = Base.ceil(Int, Base.log2(n))+1
poolsize(idx) = 2^(idx-1)

@assert poolsize(poolidx(MAX_POOL)) <= MAX_POOL "MAX_POOL cutoff should close a pool"

function create_pools(idx)
  if length(pool_usage) >= idx
    # fast-path without taking a lock
    return
  end

  @lock pool_lock begin
    while length(pool_usage) < idx
      push!(pool_usage, 1)
      push!(pool_history, initial_usage)
      push!(pools_used, Set{Block}())
      push!(pools_avail, Vector{Block}())
    end
  end
end


## pooling

const initial_usage = Tuple(1 for _ in 1:USAGE_WINDOW)

const pool_usage = Vector{Float64}()
const pool_history = Vector{NTuple{USAGE_WINDOW,Float64}}()

const freed_lock = NonReentrantLock()
const freed = Vector{Block}()

# scan every pool and manage the usage history
#
# returns a boolean indicating whether any pool is active (this can be a false negative)
function scan()
  GC.gc(false) # quick, incremental collection

  active = false

  @lock pool_lock begin
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
  end

  active
end

# reclaim unused buffers
function reclaim(target_bytes::Int=typemax(Int); full::Bool=true)
  repopulate()

  @lock pool_lock begin
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
        avail = pools_avail[pid]

        bufcount = pools_inactive[pid]
        @assert bufcount <= length(avail)
        for i in 1:bufcount
          block = pop!(avail)

          actual_free(block)

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
function repopulate()
  blocks = @safe_lock freed_lock begin
    isempty(freed) && return
    blocks = Set(freed)
    empty!(freed)
    blocks
  end

  @lock pool_lock begin
    for block in blocks
      pid = poolidx(sizeof(block))

      @inbounds used = pools_used[pid]
      @inbounds avail = pools_avail[pid]

      # mark the buffer as available
      delete!(used, block)
      push!(avail, block)

      # update pool usage
      current_usage = length(used) / (length(used) + length(avail))
      pool_usage[pid] = Base.max(pool_usage[pid], current_usage)
    end
  end

  return
end

function pool_alloc(bytes, pid=-1)
  block = nothing

  # NOTE: checking the pool is really fast, and not included in the timings
  @lock pool_lock begin
    if pid != -1 && !isempty(pools_avail[pid])
      block = pop!(pools_avail[pid])
    end
  end

  if block === nothing
    @pool_timeit "0. repopulate" begin
      repopulate()
    end

    @lock pool_lock begin
      if pid != -1 && !isempty(pools_avail[pid])
        block = pop!(pools_avail[pid])
      end
    end
  end

  if block === nothing
    @pool_timeit "1. try alloc" begin
      block = actual_alloc(bytes)
    end
  end

  if block === nothing
    @pool_timeit "2a. gc (incremental)" begin
      GC.gc(false)
    end

    @pool_timeit "2b. repopulate" begin
      repopulate()
    end

    @lock pool_lock begin
      if pid != -1 && !isempty(pools_avail[pid])
        block = pop!(pools_avail[pid])
      end
    end
  end

  # TODO: we could return a larger allocation here, but that increases memory pressure and
  #       would require proper block splitting + compaction to be any efficient.

  if block === nothing
    @pool_timeit "3. reclaim unused" begin
      reclaim(bytes)
    end

    @pool_timeit "4. try alloc" begin
      block = actual_alloc(bytes)
    end
  end

  if block === nothing
    @pool_timeit "5a. gc (full)" begin
      GC.gc(true)
    end

    @pool_timeit "5b. repopulate" begin
      repopulate()
    end

    @lock pool_lock begin
      if pid != -1 && !isempty(pools_avail[pid])
        block = pop!(pools_avail[pid])
      end
    end
  end

  if block === nothing
    @pool_timeit "6. reclaim unused" begin
      reclaim(bytes)
    end

    @pool_timeit "7. try alloc" begin
      block = actual_alloc(bytes)
    end
  end

  if block === nothing
    @pool_timeit "8. reclaim everything" begin
      reclaim()
    end

    @pool_timeit "9. try alloc" begin
      block = actual_alloc(bytes)
    end
  end

  if block !== nothing && pid != -1
    @lock pool_lock begin
      @inbounds used = pools_used[pid]
      @inbounds avail = pools_avail[pid]

      # mark the buffer as used
      push!(used, block)

      # update pool usage
      current_usage = length(used) / (length(avail) + length(used))
      pool_usage[pid] = Base.max(pool_usage[pid], current_usage)
    end
  end

  return block
end

function pool_free(block)
    # we don't do any work here to reduce pressure on the GC (spending time in finalizers)
    # and to simplify locking (preventing concurrent access during GC interventions)
  @safe_lock_spin freed_lock begin
    push!(freed, block)
  end
end


## interface

const allocated_lock = NonReentrantLock()
const allocated = Dict{CuPtr{Nothing},Block}()

function init()
  create_pools(30) # up to 512 MiB

  managed_str = if haskey(ENV, "JULIA_CUDA_MEMORY_POOL_MANAGED")
    ENV["JULIA_CUDA_MEMORY_POOL_MANAGED"]
  elseif haskey(ENV, "CUARRAYS_MANAGED_POOL")
    Base.depwarn("The CUARRAYS_MANAGED_POOL environment flag is deprecated, please use JULIA_CUDA_MEMORY_POOL_MANAGED instead.", :__init_memory__)
    ENV["CUARRAYS_MANAGED_POOL"]
  else
    nothing
  end

  managed = parse(Bool, something(managed_str, "true"))
  if managed
    delay = MIN_DELAY
    @async begin
      while true
        @pool_timeit "background task" begin
          if scan()
            delay = MIN_DELAY
          else
            delay = Base.min(delay*2, MAX_DELAY)
          end

          reclaim(full=false)
        end

        sleep(delay)
      end
    end
  end
end

function alloc(bytes)
  @assert bytes > 0

  # only manage small allocations in the pool
  block = if bytes <= MAX_POOL
    pid = poolidx(bytes)
    create_pools(pid)
    alloc_bytes = poolsize(pid)
    pool_alloc(alloc_bytes, pid)
  else
    pool_alloc(bytes)
  end

  if block !== nothing
    ptr = pointer(block)
    @safe_lock allocated_lock begin
      allocated[ptr] = block
    end
    return ptr
  else
    return nothing
  end
end

function free(ptr)
  block = @safe_lock_spin allocated_lock begin
    block = allocated[ptr]
    delete!(allocated, ptr)
    block
  end
  bytes = sizeof(block)
  @assert bytes > 0

  # was this a pooled buffer?
  if bytes <= MAX_POOL
    pid = poolidx(bytes)
    @assert pid <= length(pools_used)
    @assert pid == poolidx(sizeof(block))
    pool_free(block)
  else
    actual_free(block)
  end

  return
end

used_memory() = @safe_lock allocated_lock mapreduce(sizeof, +, values(allocated); init=0)

function cached_memory()
  sz = @safe_lock freed_lock mapreduce(sizeof, +, freed; init=0)
  @lock pool_lock for (pid, pl) in enumerate(pools_avail)
    bytes = poolsize(pid)
    sz += bytes * length(pl)
  end
  return sz
end

dump() = return

end
