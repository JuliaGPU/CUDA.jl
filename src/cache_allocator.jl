const CacheAllocatorName = ScopedValue(:none)

const CuCacheAllocator = GPUArrays.PerDeviceCacheAllocator(CuArray)

GPUArrays.cache_alloc_scope(::CUDABackend) = CacheAllocatorName

GPUArrays.cache_allocator(::CUDABackend) = CuCacheAllocator

GPUArrays.free_busy_cache_alloc!(pdcache::GPUArrays.PerDeviceCacheAllocator{CuArray}, name::Symbol) =
    GPUArrays.free_busy!(GPUArrays.named_cache_allocator!(pdcache, CUDA.device(), name))

GPUArrays.invalidate_cache_allocator!(pdcache::GPUArrays.PerDeviceCacheAllocator{CuArray}, name::Symbol) =
    GPUArrays.invalidate_cache_allocator!(pdcache, CUDA.device(), name)
