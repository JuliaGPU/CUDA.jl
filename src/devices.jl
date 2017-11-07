# Device type and auxiliary functions

export
    CuDevice, name, totalmem, attribute


const CuDevice_t = Cint

"""
    CuDevice(i::Integer)

Get a handle to a compute device.
"""
struct CuDevice
    ordinal::Cint
    handle::CuDevice_t

    function CuDevice(i::Integer)
        ordinal = convert(Cint, i)
        handle_ref = Ref{CuDevice_t}()
        @apicall(:cuDeviceGet, (Ptr{CuDevice_t}, Cint), handle_ref, ordinal)
        new(ordinal, handle_ref[])
    end
end

Base.convert(::Type{CuDevice_t}, dev::CuDevice) = dev.handle

Base.:(==)(a::CuDevice, b::CuDevice) = a.handle == b.handle
Base.hash(dev::CuDevice, h::UInt) = hash(dev.handle, h)

function Base.show(io::IO, ::MIME"text/plain", dev::CuDevice)
    print(io, "CuDevice($(dev.ordinal)): $(name(dev))")
end

"""
    name(dev::CuDevice)

Returns an identifier string for the device.
"""
function name(dev::CuDevice)
    buflen = 256
    buf = Array{Cchar}(buflen)
    @apicall(:cuDeviceGetName, (Ptr{Cchar}, Cint, CuDevice_t),
                               buf, buflen, dev)
    buf[end] = 0
    return unsafe_string(pointer(buf))
end

"""
    totalmem(dev::CuDevice)

Returns the total amount of memory (in bytes) on the device.
"""
function totalmem(dev::CuDevice)
    mem_ref = Ref{Csize_t}()
    @apicall(:cuDeviceTotalMem, (Ptr{Csize_t}, CuDevice_t), mem_ref, dev)
    return mem_ref[]
end

@enum(CUdevice_attribute, MAX_THREADS_PER_BLOCK = Cint(1),
                          MAX_BLOCK_DIM_X,
                          MAX_BLOCK_DIM_Y,
                          MAX_BLOCK_DIM_Z,
                          MAX_GRID_DIM_X,
                          MAX_GRID_DIM_Y,
                          MAX_GRID_DIM_Z,
                          MAX_SHARED_MEMORY_PER_BLOCK,
                          TOTAL_CONSTANT_MEMORY,
                          WARP_SIZE,
                          MAX_PITCH,
                          MAX_REGISTERS_PER_BLOCK,
                          CLOCK_RATE,
                          TEXTURE_ALIGNMENT,
                          GPU_OVERLAP,
                          MULTIPROCESSOR_COUNT,
                          KERNEL_EXEC_TIMEOUT,
                          INTEGRATED,
                          CAN_MAP_HOST_MEMORY,
                          COMPUTE_MODE,
                          MAXIMUM_TEXTURE1D_WIDTH,
                          MAXIMUM_TEXTURE2D_WIDTH,
                          MAXIMUM_TEXTURE2D_HEIGHT,
                          MAXIMUM_TEXTURE3D_WIDTH,
                          MAXIMUM_TEXTURE3D_HEIGHT,
                          MAXIMUM_TEXTURE3D_DEPTH,
                          MAXIMUM_TEXTURE2D_LAYERED_WIDTH,
                          MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,
                          MAXIMUM_TEXTURE2D_LAYERED_LAYERS,
                          SURFACE_ALIGNMENT,
                          CONCURRENT_KERNELS,
                          ECC_ENABLED,
                          PCI_BUS_ID,
                          PCI_DEVICE_ID,
                          TCC_DRIVER,
                          MEMORY_CLOCK_RATE,
                          GLOBAL_MEMORY_BUS_WIDTH,
                          L2_CACHE_SIZE,
                          MAX_THREADS_PER_MULTIPROCESSOR,
                          ASYNC_ENGINE_COUNT,
                          UNIFIED_ADDRESSING,
                          MAXIMUM_TEXTURE1D_LAYERED_WIDTH,
                          MAXIMUM_TEXTURE1D_LAYERED_LAYERS,
                          CAN_TEX2D_GATHER,
                          MAXIMUM_TEXTURE2D_GATHER_WIDTH,
                          MAXIMUM_TEXTURE2D_GATHER_HEIGHT,
                          MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE,
                          MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE,
                          MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE,
                          PCI_DOMAIN_ID,
                          TEXTURE_PITCH_ALIGNMENT,
                          MAXIMUM_TEXTURECUBEMAP_WIDTH,
                          MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH,
                          MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS,
                          MAXIMUM_SURFACE1D_WIDTH,
                          MAXIMUM_SURFACE2D_WIDTH,
                          MAXIMUM_SURFACE2D_HEIGHT,
                          MAXIMUM_SURFACE3D_WIDTH,
                          MAXIMUM_SURFACE3D_HEIGHT,
                          MAXIMUM_SURFACE3D_DEPTH,
                          MAXIMUM_SURFACE1D_LAYERED_WIDTH,
                          MAXIMUM_SURFACE1D_LAYERED_LAYERS,
                          MAXIMUM_SURFACE2D_LAYERED_WIDTH,
                          MAXIMUM_SURFACE2D_LAYERED_HEIGHT,
                          MAXIMUM_SURFACE2D_LAYERED_LAYERS,
                          MAXIMUM_SURFACECUBEMAP_WIDTH,
                          MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH,
                          MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS,
                          MAXIMUM_TEXTURE1D_LINEAR_WIDTH,
                          MAXIMUM_TEXTURE2D_LINEAR_WIDTH,
                          MAXIMUM_TEXTURE2D_LINEAR_HEIGHT,
                          MAXIMUM_TEXTURE2D_LINEAR_PITCH,
                          MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH,
                          MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT,
                          COMPUTE_CAPABILITY_MAJOR,
                          COMPUTE_CAPABILITY_MINOR,
                          MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH,
                          STREAM_PRIORITIES_SUPPORTED,
                          GLOBAL_L1_CACHE_SUPPORTED,
                          LOCAL_L1_CACHE_SUPPORTED,
                          MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                          MAX_REGISTERS_PER_MULTIPROCESSOR,
                          MANAGED_MEMORY,
                          MULTI_GPU_BOARD,
                          MULTI_GPU_BOARD_GROUP_ID)
@assert Cint(MULTI_GPU_BOARD_GROUP_ID) == 85

"""
    attribute(dev::CuDevice, code)

Returns information about the device.
"""
function attribute(dev::CuDevice, code::CUdevice_attribute)
    value_ref = Ref{Cint}()
    @apicall(:cuDeviceGetAttribute, (Ptr{Cint}, Cint, CuDevice_t),
                                    value_ref, code, dev)
    return value_ref[]
end


## device iteration

export devices

struct DeviceSet end

"""
    devices()

Get an iterator for the compute devices.
"""
devices() = DeviceSet()

Base.eltype(::DeviceSet) = CuDevice

Base.start(::DeviceSet) = 0

Base.next(::DeviceSet, state) =
    (CuDevice(state), state+1)

Base.done(iter::DeviceSet, state) = state == Base.length(iter)

function Base.length(::DeviceSet)
    count_ref = Ref{Cint}()
    @apicall(:cuDeviceGetCount, (Ptr{Cint},), count_ref)
    return count_ref[]
end

Base.iteratorsize(::DeviceSet) = Base.HasLength()


## convenience attribute getters

export warpsize, capability

"""
    warpsize(dev::CuDevice)

Returns the warp size (in threads) of the device.
"""
warpsize(dev::CuDevice) = attribute(dev, WARP_SIZE)

"""
    capability(dev::CuDevice)

Returns the compute capability of the device.
"""
function capability(dev::CuDevice)
    return VersionNumber(attribute(dev, COMPUTE_CAPABILITY_MAJOR),
                         attribute(dev, COMPUTE_CAPABILITY_MINOR))
end
