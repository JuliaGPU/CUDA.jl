const CuCacheAllocator = GPUArrays.AllocCache.PerDeviceCacheAllocator(CuArray; free_immediately=true)

GPUArrays.AllocCache.cache_allocator(::Type{<: CuArray}) = CuCacheAllocator

GPUArrays.AllocCache.device(::Type{<: CuArray}) = CUDA.device()
