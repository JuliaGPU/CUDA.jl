## stream-ordered memory pool

function stream_ordered(dev::CuDevice)
  @memoize index=deviceid(dev)+1 begin
    CUDACore.driver_version() >= v"11.3" && memory_pools_supported(dev) && pools_enabled()
  end::Bool
end

# whether memory can be freed in stream order, which is also supported for memory that wasn't
# allocated from a pool
function async_free_supported(dev::CuDevice)
  @memoize index=deviceid(dev)+1 begin
    CUDACore.driver_version() >= v"11.3" && memory_pools_supported(dev)
  end::Bool
end

function pool_create(dev::CuDevice)
  @memoize index=deviceid(dev)+1 begin
    limits = memory_limits()

    # create a custom memory pool and assign it to the device
    # so that other libraries and applications will use it.
    pool = if limits.hard > 0 && CUDACore.driver_version() >= v"12.2"
      CuMemoryPool(dev; maxSize=limits.hard)
    else
      CuMemoryPool(dev)
    end
    memory_pool!(dev, pool)

    # allow the pool to use up all memory of this device
    attribute!(pool, MEMPOOL_ATTR_RELEASE_THRESHOLD,
               limits.soft == 0 ? typemax(UInt64) : limits.soft)

    # launch a task to periodically trim the pool
    if isinteractive() && !isassigned(__pool_cleanup)
      __pool_cleanup[] = errormonitor(Threads.@spawn pool_cleanup())
    end

    pool
  end::CuMemoryPool
end

# per-device flag indicating the status of the memory pool.
function pool_mark_ref(dev::CuDevice)
  @memoize index=deviceid(dev)+1 begin
    Ref{Union{Nothing,Bool}}(nothing)
  end::Base.RefValue{Union{Nothing,Bool}}
end
function pool_mark(dev::CuDevice)
  pool_mark_ref(dev)[]
end
function pool_mark!(dev::CuDevice, val)
  pool_mark_ref(dev)[] = val
  return
end

# reclaim unused pool memory after a certain time
const __pool_cleanup = Ref{Task}()
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
      stream_ordered(dev) || continue

      status = pool_mark(dev)
      status === nothing && continue

      if status
        idle_counters[i] = 0
      else
        idle_counters[i] += 1
      end
      pool_mark!(dev, false)

      if idle_counters[i] == 5
        # the pool hasn't been used for a while, so release unused memory. (not using
        # `reclaim`, which would release resources whose release may wait for the GPU)
        device!(dev) do
          trim(pool_create(dev))
        end
      end
    end
  end
end
