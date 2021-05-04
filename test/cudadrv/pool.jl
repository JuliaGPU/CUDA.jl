dev = device()
if CUDA.version() >= v"11.2" && attribute(dev, CUDA.DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED) == 1

pool = memory_pool(dev)

pool2 = CuMemoryPool(dev)
@test pool2 != pool
memory_pool!(dev, pool2)
@test pool2 == memory_pool(dev)
@test pool2 != default_memory_pool(dev)

memory_pool!(dev, pool)
@test pool == memory_pool(dev)

@test attribute(UInt64, pool2, CUDA.MEMPOOL_ATTR_RELEASE_THRESHOLD) == 0
attribute!(pool2, CUDA.MEMPOOL_ATTR_RELEASE_THRESHOLD, UInt64(2^30))
@test attribute(UInt64, pool2, CUDA.MEMPOOL_ATTR_RELEASE_THRESHOLD) == 2^30

end
