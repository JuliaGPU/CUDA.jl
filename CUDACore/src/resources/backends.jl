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

  maybe_collect()
  time = Base.@elapsed begin
    mem = _pool_alloc(B, sz)
  end

  Base.@atomic alloc_stats.alloc_count += 1
  Base.@atomic alloc_stats.alloc_bytes += sz
  Base.@atomic alloc_stats.total_time += time
  # NOTE: total_time might be an over-estimation if we trigger GC somewhere else

  return Managed(mem)
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
  # NOTE: no `retry_reclaim` here. `cuMemAllocManaged` allocates lazily and
  # essentially never returns `ERROR_OUT_OF_MEMORY` — when host RAM is actually
  # exhausted, the OS kills the process on the page fault before the driver
  # call can fail. The only thing that can prevent OOM is the proactive
  # `maybe_collect` call in `pool_alloc`, which uses `_host_stats`.
  mem = alloc(UnifiedMemory, sz)
  account!(_host_stats, sz)
  mem
end
@inline function _pool_alloc(::Type{HostMemory}, sz)
  mem = alloc(HostMemory, sz)
  account!(_host_stats, sz)
  mem
end

"""
    pool_free(mem::Managed{<:AbstractMemory})

Releases memory to the pool. If possible, this operation will not block but will be ordered
against the stream that last used the memory.
"""
@inline function pool_free(managed::Managed{<:AbstractMemory})
  Base.@lock managed.lock begin
    mem = managed.mem

    # 0-byte allocations shouldn't hit the pool
    sz = sizeof(mem)
    sz == 0 && return

    # this function is typically called from a finalizer, where we can't switch tasks,
    # so perform our own error handling.
    try
      time = Base.@elapsed _pool_free(mem, managed.stream)

      Base.@atomic alloc_stats.free_count += 1
      Base.@atomic alloc_stats.free_bytes += sz
      Base.@atomic alloc_stats.total_time += time
    catch ex
      # NOTE: avoid `show`ing `mem` here since the buffer may be in a bad state
      # (often the reason free is failing); printing the byte count is safer.
      Base.showerror_nostdio(ex,
          "WARNING: Error while freeing $(Base.format_bytes(sz)) of GPU memory")
      Base.show_backtrace(Core.stdout, catch_backtrace())
      Core.println()
    end
  end

  return
end
@inline function _pool_free(mem::DeviceMemory, stream::CuStream)
    if mem.async
      # stream-ordered allocations are not tied to a context. we always need to free them,
      # and if the owning stream was destroyed, use a default one.
      if isvalid(stream)
        context!(mem.ctx) do
          free(mem; stream)
        end
      else
        free(mem; stream=default_stream())
      end
    else
      # regular allocations are tied to a context, so free them in their owning context.
      context!(mem.ctx) do
        free(mem)
      end
    end
    account!(memory_stats(mem.dev), -sizeof(mem))
end
@inline function _pool_free(mem::UnifiedMemory, stream::CuStream)
  free(mem)
  account!(_host_stats, -sizeof(mem))
end
@inline function _pool_free(mem::HostMemory, stream::CuStream)
  free(mem)
  account!(_host_stats, -sizeof(mem))
end
