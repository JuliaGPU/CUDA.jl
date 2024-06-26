# Stream-orderdered memory allocator

export CuMemoryPool, default_memory_pool, memory_pool, memory_pool!, trim,
       attribute, attribute!, access!

@enum_without_prefix CUmemAllocationType CU_MEM_
@enum_without_prefix CUmemAllocationHandleType CU_MEM_
@enum_without_prefix CUmemAccess_flags_enum CU_MEM_

mutable struct CuMemoryPool
    handle::CUmemoryPool
    ctx::CuContext

    function CuMemoryPool(dev::CuDevice;
                          alloc_type::CUmemAllocationType=ALLOCATION_TYPE_PINNED,
                          handle_type::CUmemAllocationHandleType=HANDLE_TYPE_NONE,
                          maxSize::Integer=0)
        props = Ref(CUmemPoolProps(
            alloc_type,
            handle_type,
            CUmemLocation(
                CU_MEM_LOCATION_TYPE_DEVICE,
                deviceid(dev)
            ),
            C_NULL,
            maxSize,
            ntuple(i->Cuchar(0), 56)
        ))
        handle_ref = Ref{CUmemoryPool}()
        cuMemPoolCreate(handle_ref, props)

        ctx = current_context()
        new(handle_ref[], ctx)
        # NOTE: we cannot attach a finalizer to this object, as the pool can be active
        #       without any references to it (similar to how contexts work).
    end

    global function default_memory_pool(dev::CuDevice)
        handle_ref = Ref{CUmemoryPool}()
        cuDeviceGetDefaultMemPool(handle_ref, dev)

        ctx = current_context()
        new(handle_ref[], ctx)
    end

    global function memory_pool(dev::CuDevice)
        handle_ref = Ref{CUmemoryPool}()
        cuDeviceGetMemPool(handle_ref, dev)

        ctx = current_context()
        new(handle_ref[], ctx)
    end
end

function unsafe_destroy!(pool::CuMemoryPool)
    context!(pool.ctx; skip_destroyed=true) do
        cuMemPoolDestroy(pool)
    end
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
function attribute(::Type{T}, pool::CuMemoryPool, attr::CUmemPool_attribute) where T
    value = Ref{T}()
    cuMemPoolGetAttribute(pool, attr, value)
    return value[]
end

"""
    attribute!(ptr::Union{Ptr,CuPtr}, attr, val)

Sets attribute` attr` on a pointer `ptr` to `val`.
"""
function attribute!(pool::CuMemoryPool, attr::CUmemPool_attribute, value)
    cuMemPoolSetAttribute(pool, attr, Ref(value))
    return
end


## pool access

@enum_without_prefix CUmemAccess_flags_enum CU_MEM_

"""
    access!(pool::CuMemoryPool, dev::CuDevice, flags::CUmemAccess_flags)
    access!(pool::CuMemoryPool, devs::Vector{CuDevice}, flags::CUmemAccess_flags)

Control the visibility of memory pool `pool` on device `dev` or a list of devices `devs`.
"""
function access!(pool::CuMemoryPool, devs::Vector{CuDevice}, flags::CUmemAccess_flags)
    map = Vector{CUmemAccessDesc}(undef, length(devs))
    for (i, dev) in enumerate(devs)
        location = CUmemLocation(CU_MEM_LOCATION_TYPE_DEVICE, dev.handle)
        access = CUmemAccessDesc(location, flags)
        map[i] = access
    end
    cuMemPoolSetAccess(pool, map, length(map))
end
access!(pool::CuMemoryPool, dev::CuDevice, flags::CUmemAccess_flags) =
    access!(pool, [dev], flags)
