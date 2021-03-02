# Stream-orderdered memory allocator

export CuMemoryPool, default_memory_pool, memory_pool, memory_pool!, trim

mutable struct CuMemoryPool
    handle::CUmemoryPool
    ctx::CuContext
end

function CuMemoryPool(dev::CuDevice)
    props = Ref(CUmemPoolProps(
        CU_MEM_ALLOCATION_TYPE_PINNED,
        CU_MEM_HANDLE_TYPE_NONE,
        CUmemLocation(
            CU_MEM_LOCATION_TYPE_DEVICE,
            deviceid(dev)
        ),
        C_NULL,
        ntuple(i->Cuchar(0), 64)
    ))
    handle_ref = Ref{CUmemoryPool}()
    cuMemPoolCreate(handle_ref, props)

    ctx = CuCurrentContext()
    obj = CuMemoryPool(handle_ref[], ctx)
    finalizer(unsafe_destroy!, obj)
    return obj
end

function unsafe_destroy!(pool::CuMemoryPool)
    if isvalid(pool.ctx)
        cuMemPoolDestroy(pool)
    end
end

Base.unsafe_convert(::Type{CUmemoryPool}, pool::CuMemoryPool) = pool.handle

Base.:(==)(a::CuMemoryPool, b::CuMemoryPool) = a.handle == b.handle
Base.hash(pool::CuMemoryPool, h::UInt) = hash(pool.handle, h)

function default_memory_pool(dev::CuDevice)
    handle_ref = Ref{CUmemoryPool}()
    cuDeviceGetDefaultMemPool(handle_ref, dev)

    ctx = CuCurrentContext()
    CuMemoryPool(handle_ref[], ctx)
end

function memory_pool(dev::CuDevice)
    handle_ref = Ref{CUmemoryPool}()
    cuDeviceGetMemPool(handle_ref, dev)

    ctx = CuCurrentContext()
    CuMemoryPool(handle_ref[], ctx)
end

memory_pool!(dev::CuDevice, pool::CuMemoryPool) = cuDeviceSetMemPool(dev, pool)

trim(pool::CuMemoryPool, bytes_to_keep::Integer=0) = cuMemPoolTrimTo(pool, bytes_to_keep)
