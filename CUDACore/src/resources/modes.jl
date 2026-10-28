## selecting an allocator

"""
    allocator(dev::CuDevice)

The allocator for device memory on `dev`, as selected by `JULIA_CUDA_MEMORY_POOL`: `cuda`
(the default) for CUDA's stream-ordered memory pool, or `none` to allocate every array
with `cuMemAlloc`. Devices or drivers that don't support memory pools fall back to `none`.
"""
function allocator(dev::CuDevice)
  @memoize index=deviceid(dev)+1 begin
    mode = get(ENV, "JULIA_CUDA_MEMORY_POOL", "cuda")
    if mode == "cuda"
      if driver_version() >= v"11.3" && memory_pools_supported(dev)
        DriverPool(dev)
      else
        DirectAllocator(dev)
      end
    elseif mode == "none"
      DirectAllocator(dev)
    else
      throw(ArgumentError("Unsupported value JULIA_CUDA_MEMORY_POOL=$mode; " *
                          "use `cuda` or `none`"))
    end
  end::Union{DriverPool,DirectAllocator}
end

# (memory may have been allocated directly when another allocator is in use, e.g., when
# wrapped with `unsafe_wrap(...; own=true)`)
function direct_allocator(dev::CuDevice)
  a = allocator(dev)
  a isa DirectAllocator && return a
  @memoize index=deviceid(dev)+1 begin
    DirectAllocator(dev)
  end::DirectAllocator
end

# free memory with the allocator that made it
function release(mem::DeviceMemory, stream::CuStream, stream_ctx::CuContext,
                 generation::Int)
  if mem.async
    release(DriverPool, mem, stream, stream_ctx, generation)
  else
    release(direct_allocator(mem.dev), mem, stream, stream_ctx, generation)
  end
end
