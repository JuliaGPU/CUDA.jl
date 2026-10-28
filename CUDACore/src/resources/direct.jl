## direct allocation with `cuMemAlloc`

struct DirectAllocator <: DeviceAllocator
  dev::CuDevice
  # bytes that have been freed, but not returned to the driver yet
  pending::Threads.Atomic{Int}
end
DirectAllocator(dev::CuDevice) = DirectAllocator(dev, Threads.Atomic{Int}(0))

name(::DirectAllocator) = "none (`cuMemAlloc` for every allocation)"

function try_allocate(a::DirectAllocator, sz, state)
  # (memory that has been freed but not returned yet isn't counted)
  limit = memory_limits().hard
  limit > 0 && used(a) + sz > limit && return nothing
  try
    alloc(DeviceMemory, sz; async=false)
  catch err
    isa(err, OutOfGPUMemoryError) || rethrow()
    nothing
  end
end

function release(a::DirectAllocator, mem::DeviceMemory, stream::CuStream,
                 stream_ctx::CuContext, generation::Int)
  # (freed along with its context)
  isvalid(mem.ctx) || return
  if async_free_supported(mem.dev)
    # free in stream order. `cuMemFree` would wait for all work on the device to finish,
    # blocking kernel launches from other threads in the meantime.
    @lock stream_disposal_lock begin
      handle, ctx = @something(release_stream(stream, stream_ctx, generation),
                               (disposal_stream(mem.ctx), mem.ctx))
      context!(ctx) do
        # destroying a stream that memory was freed on waits for the free, i.e., for all
        # work on the stream. so free it on the disposal stream instead, after the work on
        # this stream.
        free_stream = after_on_disposal_stream(handle, ctx)
        cuMemFreeAsync(mem, free_stream)
        # (counted until it has been returned, see `record_free_events`)
        key = (ctx, mem.dev)
        unsynced_frees[key] = get(unsynced_frees, key, 0) + sizeof(mem)
        Threads.atomic_add!(a.pending, sizeof(mem))
      end
    end
  else
    # without support for freeing memory in stream order, `cuMemFree` is the only option,
    # which waits for all running kernels. defer that until memory is reclaimed.
    Threads.atomic_add!(a.pending, sizeof(mem))
    defer_release(mem; ctx=mem.ctx, blocking=true) do mem
      free(mem)
      Threads.atomic_sub!(direct_allocator(mem.dev).pending, sizeof(mem))
    end
  end
  return
end

used(a::DirectAllocator) = memory_stats(a.dev).live
reserved(a::DirectAllocator) = used(a) + a.pending[]

trim!(::DirectAllocator) = nothing
set_retention!(::DirectAllocator, bytes) = nothing
grant_access!(::DirectAllocator, peers::Vector{CuDevice}) = nothing
