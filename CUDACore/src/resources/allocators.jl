## device memory allocators
#
# device memory is allocated by one of several allocators, chosen per device with
# `JULIA_CUDA_MEMORY_POOL` (see `allocator`):
# - `cuda` (`DriverPool`): CUDA's stream-ordered memory pool, the default;
# - `none` (`DirectAllocator`): `cuMemAlloc` for every allocation, without caching.
#
# every allocator implements:
# - `try_allocate(a, sz, state)`: allocate `sz` bytes for use on `state.stream`, or return
#   `nothing` when out of memory, after which the caller reclaims memory and retries;
# - `release(a, mem, stream, ctx, generation)`: free memory it allocated, in stream order
#   after its last use on `stream` (during `generation`, in context `ctx`);
# - `used(a)` and `reserved(a)`: the bytes of live allocations, and of the backing memory
#   (including memory that has been freed but not returned to the driver yet);
# - `trim!(a)`: return unused backing memory to the driver, without waiting for the GPU;
# - `set_retention!(a, bytes)`: how much unused memory to keep around;
# - `grant_access!(a, peers)`: make its memory accessible from other devices;
# - `name(a)`, for reporting.
#
# memory is freed by the allocator that made it, regardless of the current setting (see
# `release(::DeviceMemory, ...)`).

abstract type DeviceAllocator end

# whether memory can be freed in stream order, which is also supported for memory that wasn't
# allocated from a pool
function async_free_supported(dev::CuDevice)
  @memoize index=deviceid(dev)+1 begin
    CUDACore.driver_version() >= v"11.3" && memory_pools_supported(dev)
  end::Bool
end

# allocate `sz` bytes of device memory with `a`, reclaiming memory when running out
@inline function allocate(a::DeviceAllocator, sz, state)
  mem = retry_reclaim(isnothing) do
    try_allocate(a, sz, state)
  end
  mem === nothing && throw(OutOfGPUMemoryError(sz))
  account!(memory_stats(state.device), sz)
  return mem
end

## idle cleanup

# devices whose allocator has allocated memory since the cleanup task last looked, or
# `false` if it hasn't, or `nothing` if it never did
function pool_mark_ref(dev::CuDevice)
  @memoize index=deviceid(dev)+1 begin
    Ref{Union{Nothing,Bool}}(nothing)
  end::Base.RefValue{Union{Nothing,Bool}}
end
pool_mark(dev::CuDevice) = pool_mark_ref(dev)[]
pool_mark!(dev::CuDevice, val) = (pool_mark_ref(dev)[] = val; return)

# return unused memory to the driver once an allocator hasn't been used for a while
const __pool_cleanup = Ref{Task}()
function start_pool_cleanup()
  if isinteractive() && !isassigned(__pool_cleanup)
    __pool_cleanup[] = errormonitor(Threads.@spawn pool_cleanup())
  end
end
function pool_cleanup()
  idle_counters = Base.fill(0, ndevices())
  while true
    try
      sleep(60)
    catch ex
      if ex isa EOFError
        # If we get EOF here, it's because Julia is shutting down, so we should just exit the loop
        break
      else
        rethrow()
      end
    end

    for (i, dev) in enumerate(devices())
      status = pool_mark(dev)
      status === nothing && continue

      if status
        idle_counters[i] = 0
      else
        idle_counters[i] += 1
      end
      pool_mark!(dev, false)

      if idle_counters[i] == 5
        # the allocator hasn't been used for a while, so release unused memory. (not
        # using `reclaim`, which would release resources whose release may wait for the GPU)
        device!(dev) do
          trim!(allocator(dev))
        end
      end
    end
  end
end
