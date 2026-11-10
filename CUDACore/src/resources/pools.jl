## CUDA's stream-ordered memory pool

struct DriverPool <: DeviceAllocator
  dev::CuDevice
  pool::CuMemoryPool
  limits::@NamedTuple{soft::UInt, hard::UInt}
end

# with `install`, the pool is also used by other libraries and applications
# (`cuDeviceSetMemPool`), and is trimmed when idle
function DriverPool(dev::CuDevice; limits=memory_limits(), install::Bool=true)
  pool = if limits.hard > 0 && CUDACore.driver_version() >= v"12.2"
    CuMemoryPool(dev; maxSize=limits.hard)
  else
    CuMemoryPool(dev)
  end
  install && memory_pool!(dev, pool)

  # allow the pool to use up all memory of this device
  attribute!(pool, MEMPOOL_ATTR_RELEASE_THRESHOLD,
             limits.soft == 0 ? typemax(UInt64) : limits.soft)

  install && start_pool_cleanup()
  return DriverPool(dev, pool, limits)
end

name(::DriverPool) = "cuda (CUDA's stream-ordered memory pool)"

function try_allocate(a::DriverPool, sz, state)
  pool_mark!(a.dev, true)
  exceeds_limit(a, sz) && return nothing
  try
    alloc(DeviceMemory, sz; async=true, state.stream, a.pool)
  catch err
    isa(err, OutOfGPUMemoryError) || rethrow()
    nothing
  end
end

function exceeds_limit(a::DriverPool, sz)
  limit = a.limits.hard
  limit > 0 || return false
  # (since CUDA 12.2, the pool is configured to enforce the limit itself)
  driver_version() >= v"12.2" && return false
  return reserved(a) + sz > limit
end

# memory allocated from a pool is freed in stream order, which doesn't wait for anything
release(::DriverPool, mem::DeviceMemory, stream::CuStream, stream_ctx::CuContext,
        generation::Int) = release(DriverPool, mem, stream, stream_ctx, generation)
function release(::Type{DriverPool}, mem::DeviceMemory, stream::CuStream,
                 stream_ctx::CuContext, generation::Int)
  free_ctx = free_context(mem)
  free_ctx === nothing && return
  @lock stream_disposal_lock begin
    handle, ctx = @something(release_stream(stream, stream_ctx, generation),
                             (disposal_stream(free_ctx), free_ctx))
    context!(ctx) do
      cuMemFreeAsync(mem, handle)
    end
  end
  return
end

used(a::DriverPool) = Int(attribute(UInt64, a.pool, MEMPOOL_ATTR_USED_MEM_CURRENT))
reserved(a::DriverPool) = Int(attribute(UInt64, a.pool, MEMPOOL_ATTR_RESERVED_MEM_CURRENT))

trim!(a::DriverPool) = trim(a.pool)
set_retention!(a::DriverPool, bytes) =
  attribute!(a.pool, MEMPOOL_ATTR_RELEASE_THRESHOLD, UInt64(bytes))
grant_access!(a::DriverPool, peers::Vector{CuDevice}) =
  isempty(peers) || access!(a.pool, peers, ACCESS_FLAGS_PROT_READWRITE)
