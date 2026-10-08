## public interface

"""
    pool_alloc([DeviceMemory], sz)::Managed{<:AbstractMemory}

Allocate a number of bytes `sz` from the memory pool on the current stream. Returns a
managed memory object; may throw an [`OutOfGPUMemoryError`](@ref) if the allocation request
cannot be satisfied.
"""
@inline pool_alloc(sz::Integer) = pool_alloc(DeviceMemory, sz)
@inline function pool_alloc(::Type{B}, sz) where {B<:AbstractMemory}
  # 0-byte allocations shouldn't hit the pool
  sz == 0 && return Managed(B())

  # LLVM implements 8- and 16-bit atomics on the containing 32-bit word, which may extend
  # past the end of the array. That never faults (allocations are 256-byte aligned), but
  # compute-sanitizer flags it, so make the allocation cover that word.
  sz = cld(sz, 4) * 4

  state = active_state()
  # (deciding to allocate for a capture is part of the submission, see `CaptureScope`)
  managed = capture_submission(state.stream) do
    in_capture(state.stream) ? capture_alloc(B, sz, state) : nothing
  end
  managed === nothing || return managed

  drain_retired(ALLOC_DRAIN_LIMIT)
  maybe_collect()
  time = Base.@elapsed begin
    mem = _pool_alloc(B, sz, state)
  end

  Base.@atomic alloc_stats.alloc_count += 1
  Base.@atomic alloc_stats.alloc_bytes += sz
  Base.@atomic alloc_stats.total_time += time
  # NOTE: total_time might be an over-estimation if we trigger GC somewhere else

  return Managed(mem; owned_allocation=true)
end
@inline function _pool_alloc(::Type{DeviceMemory}, sz, state)

    mem = if stream_ordered(state.device)
      pool_mark!(state.device, true)
      pool = pool_create(state.device)

      retry_reclaim(isnothing) do
        memory_limit_exceeded(sz) && return nothing

        # try the actual allocation
        try
          alloc(DeviceMemory, sz; async=true, state.stream, pool)
        catch err
          isa(err, OutOfGPUMemoryError) || rethrow()
          return nothing
        end
      end
    else
      retry_reclaim(isnothing) do
        memory_limit_exceeded(sz) && return nothing

        # try the actual allocation
        try
          alloc(DeviceMemory, sz; async=false)
        catch err
          isa(err, OutOfGPUMemoryError) || rethrow()
          return nothing
        end
      end
    end
    # NOTE: the `retry_reclaim` body is duplicated to work around
    #       closure capture issues with the `pool` variable
    mem === nothing && throw(OutOfGPUMemoryError(sz))

    account!(memory_stats(state.device), sz)

    mem
end
@inline function _pool_alloc(::Type{UnifiedMemory}, sz, state)
  # NOTE: allocating unified memory rarely fails, as it is only backed by physical memory
  #       when used. when host memory runs out, the OS kills the process on a page fault
  #       instead, so the proactive `maybe_collect` in `pool_alloc` is what prevents that.
  mem = alloc_unified(sz, state)
  account!(_host_stats, sizeof(mem))
  mem
end
@inline function _pool_alloc(::Type{HostMemory}, sz, state)
  mem = alloc_host(sz, state)
  account!(_host_stats, sizeof(mem))
  mem
end

"""
    pool_free(mem::Managed{<:AbstractMemory})

Releases memory to the pool. If possible, this operation will not block but will be ordered
against the stream that last used the memory.

When called from a finalizer, the memory is only retired, and released later by a regular
task (see [`drain_retired`](@ref)). Memory leased by a graph is released after its last
lease ends.
"""
@inline function pool_free(managed::Managed{<:AbstractMemory})
  # 0-byte allocations shouldn't hit the pool
  sizeof(managed.mem) == 0 && return

  release(discard, managed)
  return
end

function release_now(managed::Managed)
  if on_per_thread_stream(managed.stream)
    destroy_later(synchronize_and(free_now, managed.stream_ctx), managed)
    return
  end
  free_now(managed)
end

function free_now(managed::Managed)
  # nothing references this memory anymore, so its stream can't change anymore either
  mem = managed.mem
  sz = sizeof(mem)

  try
    time = Base.@elapsed _pool_free(mem, managed.stream, managed.stream_ctx,
                                    managed.generation, managed.owned_allocation,
                                    managed.attachment)
    Base.@atomic alloc_stats.free_count += 1
    Base.@atomic alloc_stats.free_bytes += sz
    Base.@atomic alloc_stats.total_time += time
  catch err
    release_failed!(ReleaseAction(identity, managed, nothing, false), err, catch_backtrace())
  end

  return
end
@inline function _pool_free(mem::DeviceMemory, stream::CuStream, stream_ctx::CuContext,
                            generation::Int, owned_allocation::Bool, ::Attachment)
    if mem.async || async_free_supported(mem.dev)
      # free in stream order. `cuMemFree` would wait for all work on the device to finish,
      # blocking kernel launches from other threads in the meantime. that also works for
      # memory that wasn't allocated from a pool.
      @lock stream_disposal_lock begin
        handle, ctx = release_stream(stream, stream_ctx, generation)
        context!(ctx) do
          free_stream = if !mem.async
            # when memory that wasn't allocated from a pool is freed in stream order,
            # destroying that stream waits for the free, i.e., for all work on the stream.
            # so free it on the disposal stream instead, after the work on this stream.
            after_on_disposal_stream(handle, ctx)
          else
            handle
          end
          cuMemFreeAsync(mem, free_stream)
        end
      end
    else
      # without support for freeing memory in stream order, `cuMemFree` is the only option,
      # which waits for all running kernels. defer that until memory is reclaimed.
      destroy_later(mem) do mem
        context!(mem.ctx) do
          free(mem)
        end
      end
    end
    owned_allocation && account!(memory_stats(mem.dev), -sizeof(mem))
end
@inline function _pool_free(mem::Union{UnifiedMemory,HostMemory}, stream::CuStream,
                            stream_ctx::CuContext, generation::Int, owned_allocation::Bool,
                            attachment::Attachment)
  if mem.pooled
    @lock stream_disposal_lock begin
      stream, ctx = release_stream(stream, stream_ctx, generation)
      context!(ctx) do
        cuMemFreeAsync(convert(CuPtr{Cvoid}, mem), stream)
      end
    end
  elseif owned_allocation
    # cached unified memory is attached to the host once it's idle, like a new allocation
    attach_host = attachment in (GLOBALLY_ATTACHED, ALWAYS_GLOBAL)
    cache_put!(mem isa HostMemory ? host_cache : unified_cache, mem, stream, stream_ctx,
               generation; attach_host)
  else
    # imported memory may not have been allocated like we do, so isn't reused. freeing it
    # waits for all running kernels to finish, so defer that until memory is reclaimed.
    defer_release(free, mem; ctx=mem.ctx, blocking=true)
  end
  owned_allocation && account!(_host_stats, -sizeof(mem))
end

# freed memory may only be released when reclaiming memory, so do so when running out
function alloc_or_reclaim(::Type{M}, sz, args...) where {M}
  mem = retry_reclaim(isnothing) do
    try
      alloc(M, sz, args...)
    catch err
      err isa OutOfGPUMemoryError || rethrow()
      nothing
    end
  end
  mem === nothing && throw(OutOfGPUMemoryError(sz))
  return mem
end


## pinned host and unified memory
#
# freeing such memory with `cuMemFreeHost` or `cuMemFree` waits for all running kernels to
# finish, also blocking kernel launches from other threads in the meantime. where supported,
# it is allocated from memory pools instead, which can be freed in stream order without
# waiting. otherwise, freed allocations are cached for reuse, and only released when
# reclaiming memory.

function try_create_pool(f)
  try
    f()
  catch err
    isa(err, CuError) || rethrow()
    @debug "Could not create a memory pool" exception=(err, catch_backtrace())
    nothing
  end
end

pools_enabled() = get(ENV, "JULIA_CUDA_MEMORY_POOL", "cuda") == "cuda"

# a pool of pinned host memory, accessible from all devices
function host_pool()
  @memoize begin
    supported = driver_version() >= v"13.0" && pools_enabled() &&
                all(devices()) do dev
                  attribute(dev, DEVICE_ATTRIBUTE_HOST_MEMORY_POOLS_SUPPORTED) == 1
                end
    supported ? try_create_pool() do
      pool = CuMemoryPool(device(); location_type=CU_MEM_LOCATION_TYPE_HOST, location_id=0)
      access!(pool, collect(devices()), ACCESS_FLAGS_PROT_READWRITE)
      pool
    end : nothing
  end::Union{Nothing,CuMemoryPool}
end

# a pool of unified memory, without a preferred location
function unified_pool()
  @memoize begin
    supported = driver_version() >= v"13.0" && pools_enabled() &&
                all(devices()) do dev
                  memory_pools_supported(dev) && concurrent_managed_access(dev)
                end
    supported ? try_create_pool() do
      CuMemoryPool(device(); alloc_type=CU_MEM_ALLOCATION_TYPE_MANAGED,
                   location_type=CU_MEM_LOCATION_TYPE_NONE, location_id=0)
    end : nothing
  end::Union{Nothing,CuMemoryPool}
end

function alloc_from_pool(pool::CuMemoryPool, sz, stream::CuStream)
  ptr = retry_reclaim(isnothing) do
    ref = Ref{CUdeviceptr}()
    res = unchecked_cuMemAllocFromPoolAsync(ref, sz, pool, stream)
    res == ERROR_OUT_OF_MEMORY && return nothing
    res == SUCCESS || throw_api_error(res)
    ref[]
  end
  ptr === nothing && throw(OutOfGPUMemoryError(sz))
  return ptr
end

# allocations that cannot be freed without blocking, kept for reuse by later allocations
struct CachedBlock{M}
  mem::M
  # recorded after the last use of the memory
  idle::CuEvent
end

mutable struct BlockCache{M} <: Reclaimable
  const lock::ReentrantLock
  const blocks::Dict{Tuple{CuContext,Int},Vector{CachedBlock{M}}}
  Base.@atomic bytes::Int
end
BlockCache{M}() where {M} =
  BlockCache{M}(ReentrantLock(), Dict{Tuple{CuContext,Int},Vector{CachedBlock{M}}}(), 0)

const host_cache = BlockCache{HostMemory}()
const unified_cache = BlockCache{UnifiedMemory}()

# allocations are rounded up, so that cached ones can be reused for similar sizes
cached_size(sz) = sz <= 1<<20 ? nextpow(2, max(sz, 512)) : cld(sz, 1<<20) << 20

function cache_put!(cache::BlockCache{M}, mem::M, stream::CuStream,
                    stream_ctx::CuContext, generation::Int; attach_host::Bool=false) where {M}
  idle = @lock stream_disposal_lock begin
    stream, ctx = release_stream(stream, stream_ctx, generation)
    # (the event needs to be created in the context of the stream it is recorded on)
    context!(ctx) do
      if attach_host
        relaxed_capture_mode() do
          cuStreamAttachMemAsync(stream, mem, 0, MEM_ATTACH_HOST)
        end
      end
      event = CuEvent(EVENT_DISABLE_TIMING)
      cuEventRecord(event, stream)
      event
    end
  end
  @lock cache.lock begin
    blocks = get!(Vector{CachedBlock{M}}, cache.blocks, (mem.ctx, sizeof(mem)))
    push!(blocks, CachedBlock(mem, idle))
    Base.@atomic cache.bytes += sizeof(mem)
  end
  return
end

# take a cached allocation that is not in use anymore
function cache_take!(cache::BlockCache{M}, ctx::CuContext, sz::Int) where {M}
  block = @lock cache.lock begin
    blocks = get(cache.blocks, (ctx, sz), nothing)
    blocks === nothing && return nothing
    i = findlast(block -> isdone(block.idle), blocks)
    i === nothing && return nothing
    Base.@atomic cache.bytes -= sz
    popat!(blocks, i)
  end
  block === nothing && return nothing
  return block.mem
end

# release cached allocations. this blocks, also kernel launches from other threads, until
# all running kernels have finished, so only do so when reclaiming memory.
function cache_release!(cache::BlockCache)
  while true
    block = @lock cache.lock begin
      key = findfirst(!isempty, cache.blocks)
      if key === nothing
        nothing
      else
        block = pop!(cache.blocks[key])
        Base.@atomic cache.bytes -= sizeof(block.mem)
        block
      end
    end
    block === nothing && break
    attempt_release(ReleaseAction(block, nothing, true) do block
      synchronize(block.idle)
      context!(block.mem.ctx) do
        free(block.mem)
      end
    end)
  end
  return
end
purge!(cache::BlockCache) = cache_release!(cache)

function alloc_host(sz, state=active_state())
  pool = host_pool()
  if pool !== nothing
    ptr = alloc_from_pool(pool, sz, state.stream)
    return HostMemory(state.context, reinterpret(Ptr{Cvoid}, ptr), sz, true)
  end

  sz = cached_size(sz)
  mem = cache_take!(host_cache, state.context, sz)
  mem === nothing || return mem
  return alloc_or_reclaim(HostMemory, sz)
end

function alloc_unified(sz, state=active_state())
  pool = unified_pool()
  if pool !== nothing
    ptr = alloc_from_pool(pool, sz, state.stream)
    return UnifiedMemory(state.context, reinterpret(CuPtr{Cvoid}, ptr), sz, true)
  end

  sz = cached_size(sz)
  mem = cache_take!(unified_cache, state.context, sz)
  if mem !== nothing
    reset_advice!(mem)
    return mem
  end
  # see `Attachment` for why memory is attached to the host on some devices
  flags = concurrent_managed_access(state.device) ? MEM_ATTACH_GLOBAL : MEM_ATTACH_HOST
  return alloc_or_reclaim(UnifiedMemory, sz, flags)
end

# a reused allocation shouldn't keep advice given for its previous use. not all advice is
# supported everywhere, so ignore errors.
function reset_advice!(mem::UnifiedMemory)
  unchecked_cuMemAdvise(mem, sizeof(mem), MEM_ADVISE_UNSET_READ_MOSTLY, CU_DEVICE_CPU)
  unchecked_cuMemAdvise(mem, sizeof(mem), MEM_ADVISE_UNSET_PREFERRED_LOCATION, CU_DEVICE_CPU)
  unchecked_cuMemAdvise(mem, sizeof(mem), MEM_ADVISE_UNSET_ACCESSED_BY, CU_DEVICE_CPU)
  for dev in devices()
    unchecked_cuMemAdvise(mem, sizeof(mem), MEM_ADVISE_UNSET_ACCESSED_BY, dev.handle)
  end
end
