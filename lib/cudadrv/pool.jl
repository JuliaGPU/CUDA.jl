# Stream-orderdered memory allocator

export CuMemoryPool, default_memory_pool, memory_pool, memory_pool!, trim, attribute, attribute!

mutable struct CuMemoryPool
    handle::CUmemoryPool
    ctx::CuContext

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
        obj = new(handle_ref[], ctx)
        finalizer(unsafe_destroy!, obj)
        return obj
    end

    global function default_memory_pool(dev::CuDevice)
        handle_ref = Ref{CUmemoryPool}()
        cuDeviceGetDefaultMemPool(handle_ref, dev)

        ctx = CuCurrentContext()
        new(handle_ref[], ctx)
    end

    global function memory_pool(dev::CuDevice)
        handle_ref = Ref{CUmemoryPool}()
        cuDeviceGetMemPool(handle_ref, dev)

        ctx = CuCurrentContext()::CuContext
        new(handle_ref[], ctx)
    end
end

function unsafe_destroy!(pool::CuMemoryPool)
    @finalize_in_ctx pool.ctx cuMemPoolDestroy(pool)
end

Base.unsafe_convert(::Type{CUmemoryPool}, pool::CuMemoryPool) = pool.handle

Base.:(==)(a::CuMemoryPool, b::CuMemoryPool) = a.handle == b.handle
Base.hash(pool::CuMemoryPool, h::UInt) = hash(pool.handle, h)

memory_pool!(dev::CuDevice, pool::CuMemoryPool) = cuDeviceSetMemPool(dev, pool)

trim(pool::CuMemoryPool, bytes_to_keep::Integer=0) = cuMemPoolTrimTo(pool, bytes_to_keep)


## pool attributes

@enum_without_prefix CUmemPool_attribute CU_

"""
    attribute(X, pool::CuMemoryPool, attr)

Returns attribute `attr` about `pool`. The type of the returned value depends on the
attribute, and as such must be passed as the `X` parameter.
"""
function attribute(X::Type, pool::CuMemoryPool, attr::CUmemPool_attribute) where {T}
    value = Ref{X}()
    cuMemPoolGetAttribute(pool, attr, value)
    return value[]
end

"""
    attribute!(ptr::Union{Ptr,CuPtr}, attr, val)

Sets attribute` attr` on a pointer `ptr` to `val`.
"""
function attribute!(pool::CuMemoryPool, attr::CUmemPool_attribute, value) where {T}
    cuMemPoolSetAttribute(pool, attr, Ref(value))
    return
end
