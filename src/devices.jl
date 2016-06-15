# Device type and auxiliary functions

import Base: unsafe_convert, @deprecate_binding

export
    devcount,
    CuDevice, name, totalmem, attribute, capability, architecture,
    list_devices


typealias CuDevice_t Cint

"Return the number of available CUDA devices"
function devcount()
    count_ref = Ref{Cint}()
    @apicall(:cuDeviceGetCount, (Ptr{Cint},), count_ref)
    return count_ref[]
end

immutable CuDevice
    ordinal::Cint
    handle::CuDevice_t

    function CuDevice(i::Integer)
        ordinal = convert(Cint, i)
        handle_ref = Ref{CuDevice_t}()
        @apicall(:cuDeviceGet, (Ptr{CuDevice_t}, Cint), handle_ref, ordinal)
        new(ordinal, handle_ref[])
    end
end

unsafe_convert(::Type{CuDevice_t}, dev::CuDevice) = dev.handle

"Get the name of a CUDA device"
function name(dev::CuDevice)
    const buflen = 256
    buf = Array(Cchar, buflen)
    @apicall(:cuDeviceGetName, (Ptr{Cchar}, Cint, CuDevice_t),
                              buf, buflen, dev.handle)
    buf[end] = 0
    return unsafe_string(pointer(buf))
end

"Get the amount of GPU memory (in bytes) of a CUDA device"
function totalmem(dev::CuDevice)
    mem_ref = Ref{Csize_t}()
    @apicall(:cuDeviceTotalMem, (Ptr{Csize_t}, CuDevice_t), mem_ref, dev.handle)
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
@deprecate_binding SHARED_MEMORY_PER_BLOCK MAX_SHARED_MEMORY_PER_BLOCK
@deprecate_binding REGISTERS_PER_BLOCK MAX_REGISTERS_PER_BLOCK
Base.deprecate(:GPU_OVERLAP)
@deprecate_binding MAXIMUM_TEXTURE2D_ARRAY_WIDTH MAXIMUM_TEXTURE2D_LAYERED_WIDTH
@deprecate_binding MAXIMUM_TEXTURE2D_ARRAY_HEIGHT MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
@deprecate_binding MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES MAXIMUM_TEXTURE2D_LAYERED_LAYERS
Base.deprecate(:CAN_TEX2D_GATHER)


function attribute(dev::CuDevice, attrcode::CUdevice_attribute)
    value_ref = Ref{Cint}()
    @apicall(:cuDeviceGetAttribute, (Ptr{Cint}, Cint, CuDevice_t),
                                   value_ref, attrcode, dev.handle)
    return value_ref[]
end

"Return the compute capabilities of a CUDA device"
function capability(dev::CuDevice)
    major_ref = Ref{Cint}()
    minor_ref = Ref{Cint}()
    @apicall(:cuDeviceComputeCapability, (Ptr{Cint}, Ptr{Cint}, CuDevice_t),
                                        major_ref, minor_ref, dev.handle)
    return VersionNumber(major_ref[], minor_ref[])
end

"Database of compute capabilities and matching shader model (boolean flag indicates which version of LLVM supports the architecture)"
const architectures = [
    (v"1.0", "sm_10", nothing),
    (v"1.1", "sm_11", nothing),
    (v"1.2", "sm_12", nothing),
    (v"1.3", "sm_13", nothing),
    (v"2.0", "sm_20", v"3.2"),
    (v"2.1", "sm_21", v"3.2"),
    (v"3.0", "sm_30", v"3.2"),
    (v"3.2", "sm_32", v"3.7"),
    (v"3.5", "sm_35", v"3.2"),
    (v"3.7", "sm_37", v"3.7"),
    (v"5.0", "sm_50", v"3.5"),
    (v"5.2", "sm_52", v"3.7"),
    (v"5.3", "sm_53", v"3.7"),
    (v"6.0", "sm_60", nothing),
    (v"6.1", "sm_61", nothing) ]

"Return the most recent supported architecture for a CUDA device"
function architecture(dev::CuDevice)
    cap = capability(dev)
    if cap < architectures[1][1]
        error("No support for SM < $(architectures[1][1])")
    end

    for i = 2:length(architectures)
        if cap < architectures[i][1]
            return architectures[i-1][2]
            break
        end
    end
    return architectures[length(architectures)][2]
end

"List all CUDA devices with their capabilities and attributes"
function list_devices()
    cnt = devcount()
    if cnt == 0
        println("No CUDA-capable device found.")
        return
    end

    for i = 0:cnt-1
        dev = CuDevice(i)
        nam = name(dev)
        tmem = round(Integer, totalmem(dev) / (1024^2))
        cap = capability(dev)

        println("device[$i]: $(nam), capability $(cap.major).$(cap.minor), total mem = $tmem MB")
    end
end
