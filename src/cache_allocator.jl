const CuCacheAllocator = GPUArrays.AllocCache.PerDeviceCacheAllocator(CuArray)

GPUArrays.AllocCache.cache_allocator(::Type{<: CuArray}) = CuCacheAllocator

GPUArrays.AllocCache.device(::Type{<: CuArray}) = CUDA.device()
