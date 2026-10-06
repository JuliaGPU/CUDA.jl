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

  drain_retired(ALLOC_DRAIN_LIMIT)
  maybe_collect()
  time = Base.@elapsed begin
    mem = _pool_alloc(B, sz)
  end

  Base.@atomic alloc_stats.alloc_count += 1
  Base.@atomic alloc_stats.alloc_bytes += sz
  Base.@atomic alloc_stats.total_time += time
  # NOTE: total_time might be an over-estimation if we trigger GC somewhere else

  return Managed(mem; owned_allocation=true)
end
@inline function _pool_alloc(::Type{DeviceMemory}, sz)
    state = active_state()

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
@inline function _pool_alloc(::Type{UnifiedMemory}, sz)
  # NOTE: allocating unified memory rarely fails, as it is only backed by physical memory
  #       when used. when host memory runs out, the OS kills the process on a page fault
  #       instead, so the proactive `maybe_collect` in `pool_alloc` is what prevents that.
  mem = alloc_or_reclaim(UnifiedMemory, sz)
  account!(_host_stats, sizeof(mem))
  mem
end
@inline function _pool_alloc(::Type{HostMemory}, sz)
  mem = alloc_or_reclaim(HostMemory, sz)
  account!(_host_stats, sizeof(mem))
  mem
end

"""
    pool_free(mem::Managed{<:AbstractMemory})

Releases memory to the pool. If possible, this operation will not block but will be ordered
against the stream that last used the memory.

When called from a finalizer, the memory is only retired, and released later by a regular
task (see [`drain_retired`](@ref)).
"""
@inline function pool_free(managed::Managed{<:AbstractMemory})
  # 0-byte allocations shouldn't hit the pool
  sizeof(managed.mem) == 0 && return

  discard(managed)
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
                                    managed.owned_allocation)
    Base.@atomic alloc_stats.free_count += 1
    Base.@atomic alloc_stats.free_bytes += sz
    Base.@atomic alloc_stats.total_time += time
  catch err
    release_failed!(ReleaseAction(identity, managed, nothing, false), err, catch_backtrace())
  end

  return
end
@inline function _pool_free(mem::DeviceMemory, stream::CuStream, stream_ctx::CuContext,
                            owned_allocation::Bool)
    if mem.async || async_free_supported(mem.dev)
      # free in stream order. `cuMemFree` would wait for all work on the device to finish,
      # blocking kernel launches from other threads in the meantime. that also works for
      # memory that wasn't allocated from a pool.
      @lock stream_disposal_lock begin
        handle, ctx = release_stream(stream, stream_ctx)
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
                            stream_ctx::CuContext, owned_allocation::Bool)
  # freeing such memory with `cuMemFreeHost` or `cuMemFree` waits for all running kernels to
  # finish (including those using the memory), also blocking kernel launches from other
  # threads in the meantime. so defer that until memory is reclaimed.
  defer_release(free, mem; ctx=mem.ctx, blocking=true)
  owned_allocation && account!(_host_stats, -sizeof(mem))
end

# freed memory may only be released when reclaiming memory, so do so when running out
function alloc_or_reclaim(::Type{M}, sz) where {M}
  mem = retry_reclaim(isnothing) do
    try
      alloc(M, sz)
    catch err
      err isa OutOfGPUMemoryError || rethrow()
      nothing
    end
  end
  mem === nothing && throw(OutOfGPUMemoryError(sz))
  return mem
end
