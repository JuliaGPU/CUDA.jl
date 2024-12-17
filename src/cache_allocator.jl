const CuCacheAllocator = GPUArrays.PerDeviceCacheAllocator(CuArray; free_immediately=true)

GPUArrays.cache_allocator(::CUDABackend) = CuCacheAllocator

GPUArrays.device(::CUDABackend) = CUDA.device()
