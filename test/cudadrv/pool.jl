dev = device()

pool = memory_pool(dev)

pool2 = CuMemoryPool(dev)
@test pool2 != pool
memory_pool!(dev, pool2)
@test pool2 == memory_pool(dev)
@test pool2 != default_memory_pool(dev)

memory_pool!(dev, pool)
@test pool == memory_pool(dev)
