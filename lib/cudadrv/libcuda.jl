using CEnum

# CUstream is defined as Ptr{CUstream_st}, but CUDA's headers contain aliases like
# CUstream(0x01) which cannot be directly converted to a Julia Ptr; so add a method:
mutable struct CUstream_st end
const CUstream = Ptr{CUstream_st}
CUstream(x::UInt8) = CUstream(Int(x))

# use our pointer types where possible
const CUdeviceptr = CuPtr{Cvoid}
const CUarray = CuArrayPtr{Cvoid}

# provide aliases for OpenGL-interop
const GLuint = Cuint
const GLenum = Cuint

@inline function initialize_context()
    prepare_cuda_state()
    return
end

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == ERROR_OUT_OF_MEMORY
        throw(OutOfGPUMemoryError())
    else
        throw(CuError(res))
    end
end

macro check(ex)
    quote
        res = $(esc(ex))
        if res != SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end


const CUdevice_v1 = Cint

const CUdevice = CUdevice_v1

@cenum cudaError_enum::UInt32 begin
    CUDA_SUCCESS = 0
    CUDA_ERROR_INVALID_VALUE = 1
    CUDA_ERROR_OUT_OF_MEMORY = 2
    CUDA_ERROR_NOT_INITIALIZED = 3
    CUDA_ERROR_DEINITIALIZED = 4
    CUDA_ERROR_PROFILER_DISABLED = 5
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8
    CUDA_ERROR_STUB_LIBRARY = 34
    CUDA_ERROR_DEVICE_UNAVAILABLE = 46
    CUDA_ERROR_NO_DEVICE = 100
    CUDA_ERROR_INVALID_DEVICE = 101
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102
    CUDA_ERROR_INVALID_IMAGE = 200
    CUDA_ERROR_INVALID_CONTEXT = 201
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202
    CUDA_ERROR_MAP_FAILED = 205
    CUDA_ERROR_UNMAP_FAILED = 206
    CUDA_ERROR_ARRAY_IS_MAPPED = 207
    CUDA_ERROR_ALREADY_MAPPED = 208
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209
    CUDA_ERROR_ALREADY_ACQUIRED = 210
    CUDA_ERROR_NOT_MAPPED = 211
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213
    CUDA_ERROR_ECC_UNCORRECTABLE = 214
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217
    CUDA_ERROR_INVALID_PTX = 218
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222
    CUDA_ERROR_JIT_COMPILATION_DISABLED = 223
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224
    CUDA_ERROR_INVALID_SOURCE = 300
    CUDA_ERROR_FILE_NOT_FOUND = 301
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303
    CUDA_ERROR_OPERATING_SYSTEM = 304
    CUDA_ERROR_INVALID_HANDLE = 400
    CUDA_ERROR_ILLEGAL_STATE = 401
    CUDA_ERROR_NOT_FOUND = 500
    CUDA_ERROR_NOT_READY = 600
    CUDA_ERROR_ILLEGAL_ADDRESS = 700
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701
    CUDA_ERROR_LAUNCH_TIMEOUT = 702
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709
    CUDA_ERROR_ASSERT = 710
    CUDA_ERROR_TOO_MANY_PEERS = 711
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715
    CUDA_ERROR_MISALIGNED_ADDRESS = 716
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717
    CUDA_ERROR_INVALID_PC = 718
    CUDA_ERROR_LAUNCH_FAILED = 719
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720
    CUDA_ERROR_NOT_PERMITTED = 800
    CUDA_ERROR_NOT_SUPPORTED = 801
    CUDA_ERROR_SYSTEM_NOT_READY = 802
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804
    CUDA_ERROR_MPS_CONNECTION_FAILED = 805
    CUDA_ERROR_MPS_RPC_FAILURE = 806
    CUDA_ERROR_MPS_SERVER_NOT_READY = 807
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809
    CUDA_ERROR_MPS_CLIENT_TERMINATED = 810
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906
    CUDA_ERROR_CAPTURED_EVENT = 907
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908
    CUDA_ERROR_TIMEOUT = 909
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910
    CUDA_ERROR_EXTERNAL_DEVICE = 911
    CUDA_ERROR_INVALID_CLUSTER_SIZE = 912
    CUDA_ERROR_UNKNOWN = 999
end

const CUresult = cudaError_enum

@checked function cuDeviceTotalMem_v2(bytes, dev)
        ccall((:cuDeviceTotalMem_v2, libcuda), CUresult, (Ptr{Csize_t}, CUdevice), bytes, dev)
    end

mutable struct CUctx_st end

const CUcontext = Ptr{CUctx_st}

@checked function cuCtxCreate_v2(pctx, flags, dev)
        ccall((:cuCtxCreate_v2, libcuda), CUresult, (Ptr{CUcontext}, Cuint, CUdevice), pctx, flags, dev)
    end

mutable struct CUmod_st end

const CUmodule = Ptr{CUmod_st}

@checked function cuModuleGetGlobal_v2(dptr, bytes, hmod, name)
        initialize_context()
        ccall((:cuModuleGetGlobal_v2, libcuda), CUresult, (Ptr{CUdeviceptr}, Ptr{Csize_t}, CUmodule, Cstring), dptr, bytes, hmod, name)
    end

@checked function cuMemGetInfo_v2(free, total)
        initialize_context()
        ccall((:cuMemGetInfo_v2, libcuda), CUresult, (Ptr{Csize_t}, Ptr{Csize_t}), free, total)
    end

@checked function cuMemAlloc_v2(dptr, bytesize)
        initialize_context()
        ccall((:cuMemAlloc_v2, libcuda), CUresult, (Ptr{CUdeviceptr}, Csize_t), dptr, bytesize)
    end

@checked function cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes)
        initialize_context()
        ccall((:cuMemAllocPitch_v2, libcuda), CUresult, (Ptr{CUdeviceptr}, Ptr{Csize_t}, Csize_t, Csize_t, Cuint), dptr, pPitch, WidthInBytes, Height, ElementSizeBytes)
    end

@checked function cuMemFree_v2(dptr)
        initialize_context()
        ccall((:cuMemFree_v2, libcuda), CUresult, (CUdeviceptr,), dptr)
    end

@checked function cuMemGetAddressRange_v2(pbase, psize, dptr)
        initialize_context()
        ccall((:cuMemGetAddressRange_v2, libcuda), CUresult, (Ptr{CUdeviceptr}, Ptr{Csize_t}, CUdeviceptr), pbase, psize, dptr)
    end

@checked function cuMemAllocHost_v2(pp, bytesize)
        initialize_context()
        ccall((:cuMemAllocHost_v2, libcuda), CUresult, (Ptr{Ptr{Cvoid}}, Csize_t), pp, bytesize)
    end

@checked function cuMemHostGetDevicePointer_v2(pdptr, p, Flags)
        initialize_context()
        ccall((:cuMemHostGetDevicePointer_v2, libcuda), CUresult, (Ptr{CUdeviceptr}, Ptr{Cvoid}, Cuint), pdptr, p, Flags)
    end

@checked function cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount)
        initialize_context()
        ccall((:cuMemcpyHtoD_v2, libcuda), CUresult, (CUdeviceptr, Ptr{Cvoid}, Csize_t), dstDevice, srcHost, ByteCount)
    end

@checked function cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount)
        initialize_context()
        ccall((:cuMemcpyDtoH_v2, libcuda), CUresult, (Ptr{Cvoid}, CUdeviceptr, Csize_t), dstHost, srcDevice, ByteCount)
    end

@checked function cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount)
        initialize_context()
        ccall((:cuMemcpyDtoD_v2, libcuda), CUresult, (CUdeviceptr, CUdeviceptr, Csize_t), dstDevice, srcDevice, ByteCount)
    end

mutable struct CUarray_st end

@checked function cuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount)
        initialize_context()
        ccall((:cuMemcpyDtoA_v2, libcuda), CUresult, (CUarray, Csize_t, CUdeviceptr, Csize_t), dstArray, dstOffset, srcDevice, ByteCount)
    end

@checked function cuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount)
        initialize_context()
        ccall((:cuMemcpyAtoD_v2, libcuda), CUresult, (CUdeviceptr, CUarray, Csize_t, Csize_t), dstDevice, srcArray, srcOffset, ByteCount)
    end

@checked function cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount)
        initialize_context()
        ccall((:cuMemcpyHtoA_v2, libcuda), CUresult, (CUarray, Csize_t, Ptr{Cvoid}, Csize_t), dstArray, dstOffset, srcHost, ByteCount)
    end

@checked function cuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount)
        initialize_context()
        ccall((:cuMemcpyAtoH_v2, libcuda), CUresult, (Ptr{Cvoid}, CUarray, Csize_t, Csize_t), dstHost, srcArray, srcOffset, ByteCount)
    end

@checked function cuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount)
        initialize_context()
        ccall((:cuMemcpyAtoA_v2, libcuda), CUresult, (CUarray, Csize_t, CUarray, Csize_t, Csize_t), dstArray, dstOffset, srcArray, srcOffset, ByteCount)
    end

@checked function cuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost, ByteCount, hStream)
        initialize_context()
        ccall((:cuMemcpyHtoAAsync_v2, libcuda), CUresult, (CUarray, Csize_t, Ptr{Cvoid}, Csize_t, CUstream), dstArray, dstOffset, srcHost, ByteCount, hStream)
    end

@checked function cuMemcpyAtoHAsync_v2(dstHost, srcArray, srcOffset, ByteCount, hStream)
        initialize_context()
        ccall((:cuMemcpyAtoHAsync_v2, libcuda), CUresult, (Ptr{Cvoid}, CUarray, Csize_t, Csize_t, CUstream), dstHost, srcArray, srcOffset, ByteCount, hStream)
    end

@cenum CUmemorytype_enum::UInt32 begin
    CU_MEMORYTYPE_HOST = 1
    CU_MEMORYTYPE_DEVICE = 2
    CU_MEMORYTYPE_ARRAY = 3
    CU_MEMORYTYPE_UNIFIED = 4
end

const CUmemorytype = CUmemorytype_enum

struct CUDA_MEMCPY2D_st
    srcXInBytes::Csize_t
    srcY::Csize_t
    srcMemoryType::CUmemorytype
    srcHost::Ptr{Cvoid}
    srcDevice::CUdeviceptr
    srcArray::CUarray
    srcPitch::Csize_t
    dstXInBytes::Csize_t
    dstY::Csize_t
    dstMemoryType::CUmemorytype
    dstHost::Ptr{Cvoid}
    dstDevice::CUdeviceptr
    dstArray::CUarray
    dstPitch::Csize_t
    WidthInBytes::Csize_t
    Height::Csize_t
end

const CUDA_MEMCPY2D_v2 = CUDA_MEMCPY2D_st

const CUDA_MEMCPY2D = CUDA_MEMCPY2D_v2

@checked function cuMemcpy2D_v2(pCopy)
        initialize_context()
        ccall((:cuMemcpy2D_v2, libcuda), CUresult, (Ptr{CUDA_MEMCPY2D},), pCopy)
    end

@checked function cuMemcpy2DUnaligned_v2(pCopy)
        initialize_context()
        ccall((:cuMemcpy2DUnaligned_v2, libcuda), CUresult, (Ptr{CUDA_MEMCPY2D},), pCopy)
    end

struct CUDA_MEMCPY3D_st
    srcXInBytes::Csize_t
    srcY::Csize_t
    srcZ::Csize_t
    srcLOD::Csize_t
    srcMemoryType::CUmemorytype
    srcHost::Ptr{Cvoid}
    srcDevice::CUdeviceptr
    srcArray::CUarray
    reserved0::Ptr{Cvoid}
    srcPitch::Csize_t
    srcHeight::Csize_t
    dstXInBytes::Csize_t
    dstY::Csize_t
    dstZ::Csize_t
    dstLOD::Csize_t
    dstMemoryType::CUmemorytype
    dstHost::Ptr{Cvoid}
    dstDevice::CUdeviceptr
    dstArray::CUarray
    reserved1::Ptr{Cvoid}
    dstPitch::Csize_t
    dstHeight::Csize_t
    WidthInBytes::Csize_t
    Height::Csize_t
    Depth::Csize_t
end

const CUDA_MEMCPY3D_v2 = CUDA_MEMCPY3D_st

const CUDA_MEMCPY3D = CUDA_MEMCPY3D_v2

@checked function cuMemcpy3D_v2(pCopy)
        initialize_context()
        ccall((:cuMemcpy3D_v2, libcuda), CUresult, (Ptr{CUDA_MEMCPY3D},), pCopy)
    end

@checked function cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream)
        initialize_context()
        ccall((:cuMemcpyHtoDAsync_v2, libcuda), CUresult, (CUdeviceptr, Ptr{Cvoid}, Csize_t, CUstream), dstDevice, srcHost, ByteCount, hStream)
    end

@checked function cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream)
        initialize_context()
        ccall((:cuMemcpyDtoHAsync_v2, libcuda), CUresult, (Ptr{Cvoid}, CUdeviceptr, Csize_t, CUstream), dstHost, srcDevice, ByteCount, hStream)
    end

@checked function cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream)
        initialize_context()
        ccall((:cuMemcpyDtoDAsync_v2, libcuda), CUresult, (CUdeviceptr, CUdeviceptr, Csize_t, CUstream), dstDevice, srcDevice, ByteCount, hStream)
    end

@checked function cuMemcpy2DAsync_v2(pCopy, hStream)
        initialize_context()
        ccall((:cuMemcpy2DAsync_v2, libcuda), CUresult, (Ptr{CUDA_MEMCPY2D}, CUstream), pCopy, hStream)
    end

@checked function cuMemcpy3DAsync_v2(pCopy, hStream)
        initialize_context()
        ccall((:cuMemcpy3DAsync_v2, libcuda), CUresult, (Ptr{CUDA_MEMCPY3D}, CUstream), pCopy, hStream)
    end

@checked function cuMemsetD8_v2(dstDevice, uc, N)
        initialize_context()
        ccall((:cuMemsetD8_v2, libcuda), CUresult, (CUdeviceptr, Cuchar, Csize_t), dstDevice, uc, N)
    end

@checked function cuMemsetD16_v2(dstDevice, us, N)
        initialize_context()
        ccall((:cuMemsetD16_v2, libcuda), CUresult, (CUdeviceptr, Cushort, Csize_t), dstDevice, us, N)
    end

@checked function cuMemsetD32_v2(dstDevice, ui, N)
        initialize_context()
        ccall((:cuMemsetD32_v2, libcuda), CUresult, (CUdeviceptr, Cuint, Csize_t), dstDevice, ui, N)
    end

@checked function cuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height)
        initialize_context()
        ccall((:cuMemsetD2D8_v2, libcuda), CUresult, (CUdeviceptr, Csize_t, Cuchar, Csize_t, Csize_t), dstDevice, dstPitch, uc, Width, Height)
    end

@checked function cuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height)
        initialize_context()
        ccall((:cuMemsetD2D16_v2, libcuda), CUresult, (CUdeviceptr, Csize_t, Cushort, Csize_t, Csize_t), dstDevice, dstPitch, us, Width, Height)
    end

@checked function cuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height)
        initialize_context()
        ccall((:cuMemsetD2D32_v2, libcuda), CUresult, (CUdeviceptr, Csize_t, Cuint, Csize_t, Csize_t), dstDevice, dstPitch, ui, Width, Height)
    end

@cenum CUarray_format_enum::UInt32 begin
    CU_AD_FORMAT_UNSIGNED_INT8 = 1
    CU_AD_FORMAT_UNSIGNED_INT16 = 2
    CU_AD_FORMAT_UNSIGNED_INT32 = 3
    CU_AD_FORMAT_SIGNED_INT8 = 8
    CU_AD_FORMAT_SIGNED_INT16 = 9
    CU_AD_FORMAT_SIGNED_INT32 = 10
    CU_AD_FORMAT_HALF = 16
    CU_AD_FORMAT_FLOAT = 32
    CU_AD_FORMAT_NV12 = 176
    CU_AD_FORMAT_UNORM_INT8X1 = 192
    CU_AD_FORMAT_UNORM_INT8X2 = 193
    CU_AD_FORMAT_UNORM_INT8X4 = 194
    CU_AD_FORMAT_UNORM_INT16X1 = 195
    CU_AD_FORMAT_UNORM_INT16X2 = 196
    CU_AD_FORMAT_UNORM_INT16X4 = 197
    CU_AD_FORMAT_SNORM_INT8X1 = 198
    CU_AD_FORMAT_SNORM_INT8X2 = 199
    CU_AD_FORMAT_SNORM_INT8X4 = 200
    CU_AD_FORMAT_SNORM_INT16X1 = 201
    CU_AD_FORMAT_SNORM_INT16X2 = 202
    CU_AD_FORMAT_SNORM_INT16X4 = 203
    CU_AD_FORMAT_BC1_UNORM = 145
    CU_AD_FORMAT_BC1_UNORM_SRGB = 146
    CU_AD_FORMAT_BC2_UNORM = 147
    CU_AD_FORMAT_BC2_UNORM_SRGB = 148
    CU_AD_FORMAT_BC3_UNORM = 149
    CU_AD_FORMAT_BC3_UNORM_SRGB = 150
    CU_AD_FORMAT_BC4_UNORM = 151
    CU_AD_FORMAT_BC4_SNORM = 152
    CU_AD_FORMAT_BC5_UNORM = 153
    CU_AD_FORMAT_BC5_SNORM = 154
    CU_AD_FORMAT_BC6H_UF16 = 155
    CU_AD_FORMAT_BC6H_SF16 = 156
    CU_AD_FORMAT_BC7_UNORM = 157
    CU_AD_FORMAT_BC7_UNORM_SRGB = 158
end

const CUarray_format = CUarray_format_enum

struct CUDA_ARRAY_DESCRIPTOR_st
    Width::Csize_t
    Height::Csize_t
    Format::CUarray_format
    NumChannels::Cuint
end

const CUDA_ARRAY_DESCRIPTOR_v2 = CUDA_ARRAY_DESCRIPTOR_st

const CUDA_ARRAY_DESCRIPTOR = CUDA_ARRAY_DESCRIPTOR_v2

@checked function cuArrayCreate_v2(pHandle, pAllocateArray)
        initialize_context()
        ccall((:cuArrayCreate_v2, libcuda), CUresult, (Ptr{CUarray}, Ptr{CUDA_ARRAY_DESCRIPTOR}), pHandle, pAllocateArray)
    end

@checked function cuArrayGetDescriptor_v2(pArrayDescriptor, hArray)
        initialize_context()
        ccall((:cuArrayGetDescriptor_v2, libcuda), CUresult, (Ptr{CUDA_ARRAY_DESCRIPTOR}, CUarray), pArrayDescriptor, hArray)
    end

struct CUDA_ARRAY3D_DESCRIPTOR_st
    Width::Csize_t
    Height::Csize_t
    Depth::Csize_t
    Format::CUarray_format
    NumChannels::Cuint
    Flags::Cuint
end

const CUDA_ARRAY3D_DESCRIPTOR_v2 = CUDA_ARRAY3D_DESCRIPTOR_st

const CUDA_ARRAY3D_DESCRIPTOR = CUDA_ARRAY3D_DESCRIPTOR_v2

@checked function cuArray3DCreate_v2(pHandle, pAllocateArray)
        initialize_context()
        ccall((:cuArray3DCreate_v2, libcuda), CUresult, (Ptr{CUarray}, Ptr{CUDA_ARRAY3D_DESCRIPTOR}), pHandle, pAllocateArray)
    end

@checked function cuArray3DGetDescriptor_v2(pArrayDescriptor, hArray)
        initialize_context()
        ccall((:cuArray3DGetDescriptor_v2, libcuda), CUresult, (Ptr{CUDA_ARRAY3D_DESCRIPTOR}, CUarray), pArrayDescriptor, hArray)
    end

mutable struct CUtexref_st end

const CUtexref = Ptr{CUtexref_st}

@checked function cuTexRefSetAddress_v2(ByteOffset, hTexRef, dptr, bytes)
        initialize_context()
        ccall((:cuTexRefSetAddress_v2, libcuda), CUresult, (Ptr{Csize_t}, CUtexref, CUdeviceptr, Csize_t), ByteOffset, hTexRef, dptr, bytes)
    end

@checked function cuTexRefGetAddress_v2(pdptr, hTexRef)
        initialize_context()
        ccall((:cuTexRefGetAddress_v2, libcuda), CUresult, (Ptr{CUdeviceptr}, CUtexref), pdptr, hTexRef)
    end

mutable struct CUgraphicsResource_st end

const CUgraphicsResource = Ptr{CUgraphicsResource_st}

@checked function cuGraphicsResourceGetMappedPointer_v2(pDevPtr, pSize, resource)
        initialize_context()
        ccall((:cuGraphicsResourceGetMappedPointer_v2, libcuda), CUresult, (Ptr{CUdeviceptr}, Ptr{Csize_t}, CUgraphicsResource), pDevPtr, pSize, resource)
    end

@checked function cuCtxDestroy_v2(ctx)
        ccall((:cuCtxDestroy_v2, libcuda), CUresult, (CUcontext,), ctx)
    end

@checked function cuCtxPopCurrent_v2(pctx)
        ccall((:cuCtxPopCurrent_v2, libcuda), CUresult, (Ptr{CUcontext},), pctx)
    end

@checked function cuCtxPushCurrent_v2(ctx)
        ccall((:cuCtxPushCurrent_v2, libcuda), CUresult, (CUcontext,), ctx)
    end

@checked function cuStreamDestroy_v2(hStream)
        initialize_context()
        ccall((:cuStreamDestroy_v2, libcuda), CUresult, (CUstream,), hStream)
    end

mutable struct CUevent_st end

const CUevent = Ptr{CUevent_st}

@checked function cuEventDestroy_v2(hEvent)
        initialize_context()
        ccall((:cuEventDestroy_v2, libcuda), CUresult, (CUevent,), hEvent)
    end

@checked function cuTexRefSetAddress2D_v3(hTexRef, desc, dptr, Pitch)
        initialize_context()
        ccall((:cuTexRefSetAddress2D_v3, libcuda), CUresult, (CUtexref, Ptr{CUDA_ARRAY_DESCRIPTOR}, CUdeviceptr, Csize_t), hTexRef, desc, dptr, Pitch)
    end

@cenum CUjit_option_enum::UInt32 begin
    CU_JIT_MAX_REGISTERS = 0
    CU_JIT_THREADS_PER_BLOCK = 1
    CU_JIT_WALL_TIME = 2
    CU_JIT_INFO_LOG_BUFFER = 3
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4
    CU_JIT_ERROR_LOG_BUFFER = 5
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
    CU_JIT_OPTIMIZATION_LEVEL = 7
    CU_JIT_TARGET_FROM_CUCONTEXT = 8
    CU_JIT_TARGET = 9
    CU_JIT_FALLBACK_STRATEGY = 10
    CU_JIT_GENERATE_DEBUG_INFO = 11
    CU_JIT_LOG_VERBOSE = 12
    CU_JIT_GENERATE_LINE_INFO = 13
    CU_JIT_CACHE_MODE = 14
    CU_JIT_NEW_SM3X_OPT = 15
    CU_JIT_FAST_COMPILE = 16
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19
    CU_JIT_LTO = 20
    CU_JIT_FTZ = 21
    CU_JIT_PREC_DIV = 22
    CU_JIT_PREC_SQRT = 23
    CU_JIT_FMA = 24
    CU_JIT_REFERENCED_KERNEL_NAMES = 25
    CU_JIT_REFERENCED_KERNEL_COUNT = 26
    CU_JIT_REFERENCED_VARIABLE_NAMES = 27
    CU_JIT_REFERENCED_VARIABLE_COUNT = 28
    CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES = 29
    CU_JIT_NUM_OPTIONS = 30
end

const CUjit_option = CUjit_option_enum

mutable struct CUlinkState_st end

const CUlinkState = Ptr{CUlinkState_st}

@checked function cuLinkCreate_v2(numOptions, options, optionValues, stateOut)
        initialize_context()
        ccall((:cuLinkCreate_v2, libcuda), CUresult, (Cuint, Ptr{CUjit_option}, Ptr{Ptr{Cvoid}}, Ptr{CUlinkState}), numOptions, options, optionValues, stateOut)
    end

@cenum CUjitInputType_enum::UInt32 begin
    CU_JIT_INPUT_CUBIN = 0
    CU_JIT_INPUT_PTX = 1
    CU_JIT_INPUT_FATBINARY = 2
    CU_JIT_INPUT_OBJECT = 3
    CU_JIT_INPUT_LIBRARY = 4
    CU_JIT_INPUT_NVVM = 5
    CU_JIT_NUM_INPUT_TYPES = 6
end

const CUjitInputType = CUjitInputType_enum

@checked function cuLinkAddData_v2(state, type, data, size, name, numOptions, options, optionValues)
        initialize_context()
        ccall((:cuLinkAddData_v2, libcuda), CUresult, (CUlinkState, CUjitInputType, Ptr{Cvoid}, Csize_t, Cstring, Cuint, Ptr{CUjit_option}, Ptr{Ptr{Cvoid}}), state, type, data, size, name, numOptions, options, optionValues)
    end

@checked function cuLinkAddFile_v2(state, type, path, numOptions, options, optionValues)
        initialize_context()
        ccall((:cuLinkAddFile_v2, libcuda), CUresult, (CUlinkState, CUjitInputType, Cstring, Cuint, Ptr{CUjit_option}, Ptr{Ptr{Cvoid}}), state, type, path, numOptions, options, optionValues)
    end

@checked function cuMemHostRegister_v2(p, bytesize, Flags)
        initialize_context()
        ccall((:cuMemHostRegister_v2, libcuda), CUresult, (Ptr{Cvoid}, Csize_t, Cuint), p, bytesize, Flags)
    end

@checked function cuGraphicsResourceSetMapFlags_v2(resource, flags)
        initialize_context()
        ccall((:cuGraphicsResourceSetMapFlags_v2, libcuda), CUresult, (CUgraphicsResource, Cuint), resource, flags)
    end

@cenum CUstreamCaptureMode_enum::UInt32 begin
    CU_STREAM_CAPTURE_MODE_GLOBAL = 0
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1
    CU_STREAM_CAPTURE_MODE_RELAXED = 2
end

const CUstreamCaptureMode = CUstreamCaptureMode_enum

@checked function cuStreamBeginCapture_v2(hStream, mode)
        initialize_context()
        ccall((:cuStreamBeginCapture_v2, libcuda), CUresult, (CUstream, CUstreamCaptureMode), hStream, mode)
    end

@checked function cuDevicePrimaryCtxRelease_v2(dev)
        ccall((:cuDevicePrimaryCtxRelease_v2, libcuda), CUresult, (CUdevice,), dev)
    end

@checked function cuDevicePrimaryCtxReset_v2(dev)
        ccall((:cuDevicePrimaryCtxReset_v2, libcuda), CUresult, (CUdevice,), dev)
    end

@checked function cuDevicePrimaryCtxSetFlags_v2(dev, flags)
        ccall((:cuDevicePrimaryCtxSetFlags_v2, libcuda), CUresult, (CUdevice, Cuint), dev, flags)
    end

struct CUipcMemHandle_st
    reserved::NTuple{64, Cchar}
end

const CUipcMemHandle_v1 = CUipcMemHandle_st

const CUipcMemHandle = CUipcMemHandle_v1

@checked function cuIpcOpenMemHandle_v2(pdptr, handle, Flags)
        initialize_context()
        ccall((:cuIpcOpenMemHandle_v2, libcuda), CUresult, (Ptr{CUdeviceptr}, CUipcMemHandle, Cuint), pdptr, handle, Flags)
    end

mutable struct CUgraphExec_st end

const CUgraphExec = Ptr{CUgraphExec_st}

mutable struct CUgraph_st end

const CUgraph = Ptr{CUgraph_st}

mutable struct CUgraphNode_st end

const CUgraphNode = Ptr{CUgraphNode_st}

@checked function cuGraphInstantiate_v2(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize)
        initialize_context()
        ccall((:cuGraphInstantiate_v2, libcuda), CUresult, (Ptr{CUgraphExec}, CUgraph, Ptr{CUgraphNode}, Cstring, Csize_t), phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize)
    end

const cuuint32_t = UInt32

const cuuint64_t = UInt64

mutable struct CUfunc_st end

const CUfunction = Ptr{CUfunc_st}

mutable struct CUmipmappedArray_st end

const CUmipmappedArray = Ptr{CUmipmappedArray_st}

mutable struct CUsurfref_st end

const CUsurfref = Ptr{CUsurfref_st}

const CUtexObject_v1 = Culonglong

const CUtexObject = CUtexObject_v1

const CUsurfObject_v1 = Culonglong

const CUsurfObject = CUsurfObject_v1

mutable struct CUextMemory_st end

const CUexternalMemory = Ptr{CUextMemory_st}

mutable struct CUextSemaphore_st end

const CUexternalSemaphore = Ptr{CUextSemaphore_st}

mutable struct CUmemPoolHandle_st end

const CUmemoryPool = Ptr{CUmemPoolHandle_st}

mutable struct CUuserObject_st end

const CUuserObject = Ptr{CUuserObject_st}

struct CUuuid_st
    bytes::NTuple{16, Cchar}
end

const CUuuid = CUuuid_st

struct CUipcEventHandle_st
    reserved::NTuple{64, Cchar}
end

const CUipcEventHandle_v1 = CUipcEventHandle_st

const CUipcEventHandle = CUipcEventHandle_v1

@cenum CUipcMem_flags_enum::UInt32 begin
    CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1
end

const CUipcMem_flags = CUipcMem_flags_enum

@cenum CUmemAttach_flags_enum::UInt32 begin
    CU_MEM_ATTACH_GLOBAL = 1
    CU_MEM_ATTACH_HOST = 2
    CU_MEM_ATTACH_SINGLE = 4
end

const CUmemAttach_flags = CUmemAttach_flags_enum

@cenum CUctx_flags_enum::UInt32 begin
    CU_CTX_SCHED_AUTO = 0
    CU_CTX_SCHED_SPIN = 1
    CU_CTX_SCHED_YIELD = 2
    CU_CTX_SCHED_BLOCKING_SYNC = 4
    CU_CTX_BLOCKING_SYNC = 4
    CU_CTX_SCHED_MASK = 7
    CU_CTX_MAP_HOST = 8
    CU_CTX_LMEM_RESIZE_TO_MAX = 16
    CU_CTX_FLAGS_MASK = 31
end

const CUctx_flags = CUctx_flags_enum

@cenum CUevent_sched_flags_enum::UInt32 begin
    CU_EVENT_SCHED_AUTO = 0
    CU_EVENT_SCHED_SPIN = 1
    CU_EVENT_SCHED_YIELD = 2
    CU_EVENT_SCHED_BLOCKING_SYNC = 4
end

const CUevent_sched_flags = CUevent_sched_flags_enum

@cenum cl_event_flags_enum::UInt32 begin
    NVCL_EVENT_SCHED_AUTO = 0
    NVCL_EVENT_SCHED_SPIN = 1
    NVCL_EVENT_SCHED_YIELD = 2
    NVCL_EVENT_SCHED_BLOCKING_SYNC = 4
end

const cl_event_flags = cl_event_flags_enum

@cenum cl_context_flags_enum::UInt32 begin
    NVCL_CTX_SCHED_AUTO = 0
    NVCL_CTX_SCHED_SPIN = 1
    NVCL_CTX_SCHED_YIELD = 2
    NVCL_CTX_SCHED_BLOCKING_SYNC = 4
end

const cl_context_flags = cl_context_flags_enum

@cenum CUstream_flags_enum::UInt32 begin
    CU_STREAM_DEFAULT = 0
    CU_STREAM_NON_BLOCKING = 1
end

const CUstream_flags = CUstream_flags_enum

@cenum CUevent_flags_enum::UInt32 begin
    CU_EVENT_DEFAULT = 0
    CU_EVENT_BLOCKING_SYNC = 1
    CU_EVENT_DISABLE_TIMING = 2
    CU_EVENT_INTERPROCESS = 4
end

const CUevent_flags = CUevent_flags_enum

@cenum CUevent_record_flags_enum::UInt32 begin
    CU_EVENT_RECORD_DEFAULT = 0
    CU_EVENT_RECORD_EXTERNAL = 1
end

const CUevent_record_flags = CUevent_record_flags_enum

@cenum CUevent_wait_flags_enum::UInt32 begin
    CU_EVENT_WAIT_DEFAULT = 0
    CU_EVENT_WAIT_EXTERNAL = 1
end

const CUevent_wait_flags = CUevent_wait_flags_enum

@cenum CUstreamWaitValue_flags_enum::UInt32 begin
    CU_STREAM_WAIT_VALUE_GEQ = 0
    CU_STREAM_WAIT_VALUE_EQ = 1
    CU_STREAM_WAIT_VALUE_AND = 2
    CU_STREAM_WAIT_VALUE_NOR = 3
    CU_STREAM_WAIT_VALUE_FLUSH = 1073741824
end

const CUstreamWaitValue_flags = CUstreamWaitValue_flags_enum

@cenum CUstreamWriteValue_flags_enum::UInt32 begin
    CU_STREAM_WRITE_VALUE_DEFAULT = 0
    CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 1
end

const CUstreamWriteValue_flags = CUstreamWriteValue_flags_enum

@cenum CUstreamBatchMemOpType_enum::UInt32 begin
    CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1
    CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2
    CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4
    CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5
    CU_STREAM_MEM_OP_BARRIER = 6
    CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3
end

const CUstreamBatchMemOpType = CUstreamBatchMemOpType_enum

@cenum CUstreamMemoryBarrier_flags_enum::UInt32 begin
    CU_STREAM_MEMORY_BARRIER_TYPE_SYS = 0
    CU_STREAM_MEMORY_BARRIER_TYPE_GPU = 1
end

const CUstreamMemoryBarrier_flags = CUstreamMemoryBarrier_flags_enum

struct CUstreamBatchMemOpParams_union
    data::NTuple{48, UInt8}
end

function Base.getproperty(x::Ptr{CUstreamBatchMemOpParams_union}, f::Symbol)
    f === :operation && return Ptr{CUstreamBatchMemOpType}(x + 0)
    f === :waitValue && return Ptr{CUstreamMemOpWaitValueParams_st}(x + 0)
    f === :writeValue && return Ptr{CUstreamMemOpWriteValueParams_st}(x + 0)
    f === :flushRemoteWrites && return Ptr{CUstreamMemOpFlushRemoteWritesParams_st}(x + 0)
    f === :memoryBarrier && return Ptr{CUstreamMemOpMemoryBarrierParams_st}(x + 0)
    f === :pad && return Ptr{NTuple{6, cuuint64_t}}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::CUstreamBatchMemOpParams_union, f::Symbol)
    r = Ref{CUstreamBatchMemOpParams_union}(x)
    ptr = Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_union}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{CUstreamBatchMemOpParams_union}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

const CUstreamBatchMemOpParams_v1 = CUstreamBatchMemOpParams_union

const CUstreamBatchMemOpParams = CUstreamBatchMemOpParams_v1

struct CUDA_BATCH_MEM_OP_NODE_PARAMS_st
    ctx::CUcontext
    count::Cuint
    paramArray::Ptr{CUstreamBatchMemOpParams}
    flags::Cuint
end

const CUDA_BATCH_MEM_OP_NODE_PARAMS = CUDA_BATCH_MEM_OP_NODE_PARAMS_st

@cenum CUoccupancy_flags_enum::UInt32 begin
    CU_OCCUPANCY_DEFAULT = 0
    CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = 1
end

const CUoccupancy_flags = CUoccupancy_flags_enum

@cenum CUstreamUpdateCaptureDependencies_flags_enum::UInt32 begin
    CU_STREAM_ADD_CAPTURE_DEPENDENCIES = 0
    CU_STREAM_SET_CAPTURE_DEPENDENCIES = 1
end

const CUstreamUpdateCaptureDependencies_flags = CUstreamUpdateCaptureDependencies_flags_enum

@cenum CUaddress_mode_enum::UInt32 begin
    CU_TR_ADDRESS_MODE_WRAP = 0
    CU_TR_ADDRESS_MODE_CLAMP = 1
    CU_TR_ADDRESS_MODE_MIRROR = 2
    CU_TR_ADDRESS_MODE_BORDER = 3
end

const CUaddress_mode = CUaddress_mode_enum

@cenum CUfilter_mode_enum::UInt32 begin
    CU_TR_FILTER_MODE_POINT = 0
    CU_TR_FILTER_MODE_LINEAR = 1
end

const CUfilter_mode = CUfilter_mode_enum

@cenum CUdevice_attribute_enum::UInt32 begin
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
    CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12
    CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V2 = 122
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V2 = 123
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124
    CU_DEVICE_ATTRIBUTE_MAX = 125
end

const CUdevice_attribute = CUdevice_attribute_enum

struct CUdevprop_st
    maxThreadsPerBlock::Cint
    maxThreadsDim::NTuple{3, Cint}
    maxGridSize::NTuple{3, Cint}
    sharedMemPerBlock::Cint
    totalConstantMemory::Cint
    SIMDWidth::Cint
    memPitch::Cint
    regsPerBlock::Cint
    clockRate::Cint
    textureAlign::Cint
end

const CUdevprop_v1 = CUdevprop_st

const CUdevprop = CUdevprop_v1

@cenum CUpointer_attribute_enum::UInt32 begin
    CU_POINTER_ATTRIBUTE_CONTEXT = 1
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3
    CU_POINTER_ATTRIBUTE_HOST_POINTER = 4
    CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5
    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6
    CU_POINTER_ATTRIBUTE_BUFFER_ID = 7
    CU_POINTER_ATTRIBUTE_IS_MANAGED = 8
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9
    CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11
    CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12
    CU_POINTER_ATTRIBUTE_MAPPED = 13
    CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14
    CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15
    CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16
    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17
    CU_POINTER_ATTRIBUTE_MAPPING_SIZE = 18
    CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR = 19
    CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID = 20
end

const CUpointer_attribute = CUpointer_attribute_enum

@cenum CUfunction_attribute_enum::UInt32 begin
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4
    CU_FUNC_ATTRIBUTE_PTX_VERSION = 5
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9
    CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET = 10
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = 11
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT = 12
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH = 13
    CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14
    CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 15
    CU_FUNC_ATTRIBUTE_MAX = 16
end

const CUfunction_attribute = CUfunction_attribute_enum

@cenum CUfunc_cache_enum::UInt32 begin
    CU_FUNC_CACHE_PREFER_NONE = 0
    CU_FUNC_CACHE_PREFER_SHARED = 1
    CU_FUNC_CACHE_PREFER_L1 = 2
    CU_FUNC_CACHE_PREFER_EQUAL = 3
end

const CUfunc_cache = CUfunc_cache_enum

@cenum CUsharedconfig_enum::UInt32 begin
    CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0
    CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 1
    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 2
end

const CUsharedconfig = CUsharedconfig_enum

@cenum CUshared_carveout_enum::Int32 begin
    CU_SHAREDMEM_CARVEOUT_DEFAULT = -1
    CU_SHAREDMEM_CARVEOUT_MAX_SHARED = 100
    CU_SHAREDMEM_CARVEOUT_MAX_L1 = 0
end

const CUshared_carveout = CUshared_carveout_enum

@cenum CUcomputemode_enum::UInt32 begin
    CU_COMPUTEMODE_DEFAULT = 0
    CU_COMPUTEMODE_PROHIBITED = 2
    CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3
end

const CUcomputemode = CUcomputemode_enum

@cenum CUmem_advise_enum::UInt32 begin
    CU_MEM_ADVISE_SET_READ_MOSTLY = 1
    CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2
    CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3
    CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4
    CU_MEM_ADVISE_SET_ACCESSED_BY = 5
    CU_MEM_ADVISE_UNSET_ACCESSED_BY = 6
end

const CUmem_advise = CUmem_advise_enum

@cenum CUmem_range_attribute_enum::UInt32 begin
    CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1
    CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2
    CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3
    CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4
end

const CUmem_range_attribute = CUmem_range_attribute_enum

@cenum CUjit_target_enum::UInt32 begin
    CU_TARGET_COMPUTE_20 = 20
    CU_TARGET_COMPUTE_21 = 21
    CU_TARGET_COMPUTE_30 = 30
    CU_TARGET_COMPUTE_32 = 32
    CU_TARGET_COMPUTE_35 = 35
    CU_TARGET_COMPUTE_37 = 37
    CU_TARGET_COMPUTE_50 = 50
    CU_TARGET_COMPUTE_52 = 52
    CU_TARGET_COMPUTE_53 = 53
    CU_TARGET_COMPUTE_60 = 60
    CU_TARGET_COMPUTE_61 = 61
    CU_TARGET_COMPUTE_62 = 62
    CU_TARGET_COMPUTE_70 = 70
    CU_TARGET_COMPUTE_72 = 72
    CU_TARGET_COMPUTE_75 = 75
    CU_TARGET_COMPUTE_80 = 80
    CU_TARGET_COMPUTE_86 = 86
    CU_TARGET_COMPUTE_87 = 87
    CU_TARGET_COMPUTE_89 = 89
    CU_TARGET_COMPUTE_90 = 90
end

const CUjit_target = CUjit_target_enum

@cenum CUjit_fallback_enum::UInt32 begin
    CU_PREFER_PTX = 0
    CU_PREFER_BINARY = 1
end

const CUjit_fallback = CUjit_fallback_enum

@cenum CUjit_cacheMode_enum::UInt32 begin
    CU_JIT_CACHE_OPTION_NONE = 0
    CU_JIT_CACHE_OPTION_CG = 1
    CU_JIT_CACHE_OPTION_CA = 2
end

const CUjit_cacheMode = CUjit_cacheMode_enum

@cenum CUgraphicsRegisterFlags_enum::UInt32 begin
    CU_GRAPHICS_REGISTER_FLAGS_NONE = 0
    CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY = 1
    CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2
    CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 4
    CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 8
end

const CUgraphicsRegisterFlags = CUgraphicsRegisterFlags_enum

@cenum CUgraphicsMapResourceFlags_enum::UInt32 begin
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 1
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 2
end

const CUgraphicsMapResourceFlags = CUgraphicsMapResourceFlags_enum

@cenum CUarray_cubemap_face_enum::UInt32 begin
    CU_CUBEMAP_FACE_POSITIVE_X = 0
    CU_CUBEMAP_FACE_NEGATIVE_X = 1
    CU_CUBEMAP_FACE_POSITIVE_Y = 2
    CU_CUBEMAP_FACE_NEGATIVE_Y = 3
    CU_CUBEMAP_FACE_POSITIVE_Z = 4
    CU_CUBEMAP_FACE_NEGATIVE_Z = 5
end

const CUarray_cubemap_face = CUarray_cubemap_face_enum

@cenum CUlimit_enum::UInt32 begin
    CU_LIMIT_STACK_SIZE = 0
    CU_LIMIT_PRINTF_FIFO_SIZE = 1
    CU_LIMIT_MALLOC_HEAP_SIZE = 2
    CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 3
    CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 4
    CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 5
    CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 6
    CU_LIMIT_MAX = 7
end

const CUlimit = CUlimit_enum

@cenum CUresourcetype_enum::UInt32 begin
    CU_RESOURCE_TYPE_ARRAY = 0
    CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1
    CU_RESOURCE_TYPE_LINEAR = 2
    CU_RESOURCE_TYPE_PITCH2D = 3
end

const CUresourcetype = CUresourcetype_enum

# typedef void ( CUDA_CB * CUhostFn ) ( void * userData )
const CUhostFn = Ptr{Cvoid}

@cenum CUaccessProperty_enum::UInt32 begin
    CU_ACCESS_PROPERTY_NORMAL = 0
    CU_ACCESS_PROPERTY_STREAMING = 1
    CU_ACCESS_PROPERTY_PERSISTING = 2
end

const CUaccessProperty = CUaccessProperty_enum

struct CUaccessPolicyWindow_st
    base_ptr::Ptr{Cvoid}
    num_bytes::Csize_t
    hitRatio::Cfloat
    hitProp::CUaccessProperty
    missProp::CUaccessProperty
end

const CUaccessPolicyWindow_v1 = CUaccessPolicyWindow_st

const CUaccessPolicyWindow = CUaccessPolicyWindow_v1

struct CUDA_KERNEL_NODE_PARAMS_st
    func::CUfunction
    gridDimX::Cuint
    gridDimY::Cuint
    gridDimZ::Cuint
    blockDimX::Cuint
    blockDimY::Cuint
    blockDimZ::Cuint
    sharedMemBytes::Cuint
    kernelParams::Ptr{Ptr{Cvoid}}
    extra::Ptr{Ptr{Cvoid}}
end

const CUDA_KERNEL_NODE_PARAMS_v1 = CUDA_KERNEL_NODE_PARAMS_st

const CUDA_KERNEL_NODE_PARAMS = CUDA_KERNEL_NODE_PARAMS_v1

struct CUDA_MEMSET_NODE_PARAMS_st
    dst::CUdeviceptr
    pitch::Csize_t
    value::Cuint
    elementSize::Cuint
    width::Csize_t
    height::Csize_t
end

const CUDA_MEMSET_NODE_PARAMS_v1 = CUDA_MEMSET_NODE_PARAMS_st

const CUDA_MEMSET_NODE_PARAMS = CUDA_MEMSET_NODE_PARAMS_v1

struct CUDA_HOST_NODE_PARAMS_st
    fn::CUhostFn
    userData::Ptr{Cvoid}
end

const CUDA_HOST_NODE_PARAMS_v1 = CUDA_HOST_NODE_PARAMS_st

const CUDA_HOST_NODE_PARAMS = CUDA_HOST_NODE_PARAMS_v1

@cenum CUgraphNodeType_enum::UInt32 begin
    CU_GRAPH_NODE_TYPE_KERNEL = 0
    CU_GRAPH_NODE_TYPE_MEMCPY = 1
    CU_GRAPH_NODE_TYPE_MEMSET = 2
    CU_GRAPH_NODE_TYPE_HOST = 3
    CU_GRAPH_NODE_TYPE_GRAPH = 4
    CU_GRAPH_NODE_TYPE_EMPTY = 5
    CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6
    CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9
    CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10
    CU_GRAPH_NODE_TYPE_MEM_FREE = 11
    CU_GRAPH_NODE_TYPE_BATCH_MEM_OP = 12
end

const CUgraphNodeType = CUgraphNodeType_enum

@cenum CUsynchronizationPolicy_enum::UInt32 begin
    CU_SYNC_POLICY_AUTO = 1
    CU_SYNC_POLICY_SPIN = 2
    CU_SYNC_POLICY_YIELD = 3
    CU_SYNC_POLICY_BLOCKING_SYNC = 4
end

const CUsynchronizationPolicy = CUsynchronizationPolicy_enum

@cenum CUclusterSchedulingPolicy_enum::UInt32 begin
    CU_CLUSTER_SCHEDULING_POLICY_DEFAULT = 0
    CU_CLUSTER_SCHEDULING_POLICY_SPREAD = 1
    CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING = 2
end

const CUclusterSchedulingPolicy = CUclusterSchedulingPolicy_enum

@cenum CUlaunchAttributeID_enum::UInt32 begin
    CU_LAUNCH_ATTRIBUTE_IGNORE = 0
    CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1
    CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2
    CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4
    CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = 7
    CU_LAUNCH_ATTRIBUTE_PRIORITY = 8
end

const CUlaunchAttributeID = CUlaunchAttributeID_enum

struct CUlaunchAttributeValue_union
    data::NTuple{64, UInt8}
end

function Base.getproperty(x::Ptr{CUlaunchAttributeValue_union}, f::Symbol)
    f === :pad && return Ptr{NTuple{64, Cchar}}(x + 0)
    f === :accessPolicyWindow && return Ptr{CUaccessPolicyWindow}(x + 0)
    f === :cooperative && return Ptr{Cint}(x + 0)
    f === :syncPolicy && return Ptr{CUsynchronizationPolicy}(x + 0)
    f === :clusterDim && return Ptr{var"##Ctag#380"}(x + 0)
    f === :clusterSchedulingPolicyPreference && return Ptr{CUclusterSchedulingPolicy}(x + 0)
    f === :programmaticStreamSerializationAllowed && return Ptr{Cint}(x + 0)
    f === :programmaticEvent && return Ptr{var"##Ctag#381"}(x + 0)
    f === :priority && return Ptr{Cint}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::CUlaunchAttributeValue_union, f::Symbol)
    r = Ref{CUlaunchAttributeValue_union}(x)
    ptr = Base.unsafe_convert(Ptr{CUlaunchAttributeValue_union}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{CUlaunchAttributeValue_union}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

const CUlaunchAttributeValue = CUlaunchAttributeValue_union

struct CUlaunchAttribute_st
    id::CUlaunchAttributeID
    pad::NTuple{4, Cchar}
    value::CUlaunchAttributeValue
end

const CUlaunchAttribute = CUlaunchAttribute_st

struct CUlaunchConfig_st
    gridDimX::Cuint
    gridDimY::Cuint
    gridDimZ::Cuint
    blockDimX::Cuint
    blockDimY::Cuint
    blockDimZ::Cuint
    sharedMemBytes::Cuint
    hStream::CUstream
    attrs::Ptr{CUlaunchAttribute}
    numAttrs::Cuint
end

const CUlaunchConfig = CUlaunchConfig_st

const CUkernelNodeAttrID = CUlaunchAttributeID

const CUkernelNodeAttrValue_v1 = CUlaunchAttributeValue

const CUkernelNodeAttrValue = CUkernelNodeAttrValue_v1

@cenum CUstreamCaptureStatus_enum::UInt32 begin
    CU_STREAM_CAPTURE_STATUS_NONE = 0
    CU_STREAM_CAPTURE_STATUS_ACTIVE = 1
    CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2
end

const CUstreamCaptureStatus = CUstreamCaptureStatus_enum

const CUstreamAttrID = CUlaunchAttributeID

const CUstreamAttrValue_v1 = CUlaunchAttributeValue

const CUstreamAttrValue = CUstreamAttrValue_v1

@cenum CUdriverProcAddress_flags_enum::UInt32 begin
    CU_GET_PROC_ADDRESS_DEFAULT = 0
    CU_GET_PROC_ADDRESS_LEGACY_STREAM = 1
    CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM = 2
end

const CUdriverProcAddress_flags = CUdriverProcAddress_flags_enum

@cenum CUexecAffinityType_enum::UInt32 begin
    CU_EXEC_AFFINITY_TYPE_SM_COUNT = 0
    CU_EXEC_AFFINITY_TYPE_MAX = 1
end

const CUexecAffinityType = CUexecAffinityType_enum

struct CUexecAffinitySmCount_st
    val::Cuint
end

const CUexecAffinitySmCount_v1 = CUexecAffinitySmCount_st

const CUexecAffinitySmCount = CUexecAffinitySmCount_v1

struct var"##Ctag#384"
    data::NTuple{4, UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#384"}, f::Symbol)
    f === :smCount && return Ptr{CUexecAffinitySmCount}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#384", f::Symbol)
    r = Ref{var"##Ctag#384"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#384"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#384"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct CUexecAffinityParam_st
    data::NTuple{8, UInt8}
end

function Base.getproperty(x::Ptr{CUexecAffinityParam_st}, f::Symbol)
    f === :type && return Ptr{CUexecAffinityType}(x + 0)
    f === :param && return Ptr{var"##Ctag#384"}(x + 4)
    return getfield(x, f)
end

function Base.getproperty(x::CUexecAffinityParam_st, f::Symbol)
    r = Ref{CUexecAffinityParam_st}(x)
    ptr = Base.unsafe_convert(Ptr{CUexecAffinityParam_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{CUexecAffinityParam_st}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

const CUexecAffinityParam_v1 = CUexecAffinityParam_st

const CUexecAffinityParam = CUexecAffinityParam_v1

@cenum CUdevice_P2PAttribute_enum::UInt32 begin
    CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 1
    CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 2
    CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 3
    CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = 4
    CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = 4
end

const CUdevice_P2PAttribute = CUdevice_P2PAttribute_enum

# typedef void ( CUDA_CB * CUstreamCallback ) ( CUstream hStream , CUresult status , void * userData )
const CUstreamCallback = Ptr{Cvoid}

# typedef size_t ( CUDA_CB * CUoccupancyB2DSize ) ( int blockSize )
const CUoccupancyB2DSize = Ptr{Cvoid}

struct CUDA_MEMCPY3D_PEER_st
    srcXInBytes::Csize_t
    srcY::Csize_t
    srcZ::Csize_t
    srcLOD::Csize_t
    srcMemoryType::CUmemorytype
    srcHost::Ptr{Cvoid}
    srcDevice::CUdeviceptr
    srcArray::CUarray
    srcContext::CUcontext
    srcPitch::Csize_t
    srcHeight::Csize_t
    dstXInBytes::Csize_t
    dstY::Csize_t
    dstZ::Csize_t
    dstLOD::Csize_t
    dstMemoryType::CUmemorytype
    dstHost::Ptr{Cvoid}
    dstDevice::CUdeviceptr
    dstArray::CUarray
    dstContext::CUcontext
    dstPitch::Csize_t
    dstHeight::Csize_t
    WidthInBytes::Csize_t
    Height::Csize_t
    Depth::Csize_t
end

const CUDA_MEMCPY3D_PEER_v1 = CUDA_MEMCPY3D_PEER_st

const CUDA_MEMCPY3D_PEER = CUDA_MEMCPY3D_PEER_v1

struct var"##Ctag#370"
    width::Cuint
    height::Cuint
    depth::Cuint
end
function Base.getproperty(x::Ptr{var"##Ctag#370"}, f::Symbol)
    f === :width && return Ptr{Cuint}(x + 0)
    f === :height && return Ptr{Cuint}(x + 4)
    f === :depth && return Ptr{Cuint}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#370", f::Symbol)
    r = Ref{var"##Ctag#370"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#370"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#370"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct CUDA_ARRAY_SPARSE_PROPERTIES_st
    data::NTuple{48, UInt8}
end

function Base.getproperty(x::Ptr{CUDA_ARRAY_SPARSE_PROPERTIES_st}, f::Symbol)
    f === :tileExtent && return Ptr{var"##Ctag#370"}(x + 0)
    f === :miptailFirstLevel && return Ptr{Cuint}(x + 12)
    f === :miptailSize && return Ptr{Culonglong}(x + 16)
    f === :flags && return Ptr{Cuint}(x + 24)
    f === :reserved && return Ptr{NTuple{4, Cuint}}(x + 28)
    return getfield(x, f)
end

function Base.getproperty(x::CUDA_ARRAY_SPARSE_PROPERTIES_st, f::Symbol)
    r = Ref{CUDA_ARRAY_SPARSE_PROPERTIES_st}(x)
    ptr = Base.unsafe_convert(Ptr{CUDA_ARRAY_SPARSE_PROPERTIES_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{CUDA_ARRAY_SPARSE_PROPERTIES_st}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

const CUDA_ARRAY_SPARSE_PROPERTIES_v1 = CUDA_ARRAY_SPARSE_PROPERTIES_st

const CUDA_ARRAY_SPARSE_PROPERTIES = CUDA_ARRAY_SPARSE_PROPERTIES_v1

struct CUDA_ARRAY_MEMORY_REQUIREMENTS_st
    size::Csize_t
    alignment::Csize_t
    reserved::NTuple{4, Cuint}
end

const CUDA_ARRAY_MEMORY_REQUIREMENTS_v1 = CUDA_ARRAY_MEMORY_REQUIREMENTS_st

const CUDA_ARRAY_MEMORY_REQUIREMENTS = CUDA_ARRAY_MEMORY_REQUIREMENTS_v1

struct var"##Ctag#362"
    data::NTuple{128, UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#362"}, f::Symbol)
    f === :array && return Ptr{var"##Ctag#363"}(x + 0)
    f === :mipmap && return Ptr{var"##Ctag#364"}(x + 0)
    f === :linear && return Ptr{var"##Ctag#365"}(x + 0)
    f === :pitch2D && return Ptr{var"##Ctag#366"}(x + 0)
    f === :reserved && return Ptr{var"##Ctag#367"}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#362", f::Symbol)
    r = Ref{var"##Ctag#362"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#362"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#362"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct CUDA_RESOURCE_DESC_st
    data::NTuple{144, UInt8}
end

function Base.getproperty(x::Ptr{CUDA_RESOURCE_DESC_st}, f::Symbol)
    f === :resType && return Ptr{CUresourcetype}(x + 0)
    f === :res && return Ptr{var"##Ctag#362"}(x + 8)
    f === :flags && return Ptr{Cuint}(x + 136)
    return getfield(x, f)
end

function Base.getproperty(x::CUDA_RESOURCE_DESC_st, f::Symbol)
    r = Ref{CUDA_RESOURCE_DESC_st}(x)
    ptr = Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{CUDA_RESOURCE_DESC_st}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

const CUDA_RESOURCE_DESC_v1 = CUDA_RESOURCE_DESC_st

const CUDA_RESOURCE_DESC = CUDA_RESOURCE_DESC_v1

struct CUDA_TEXTURE_DESC_st
    addressMode::NTuple{3, CUaddress_mode}
    filterMode::CUfilter_mode
    flags::Cuint
    maxAnisotropy::Cuint
    mipmapFilterMode::CUfilter_mode
    mipmapLevelBias::Cfloat
    minMipmapLevelClamp::Cfloat
    maxMipmapLevelClamp::Cfloat
    borderColor::NTuple{4, Cfloat}
    reserved::NTuple{12, Cint}
end

const CUDA_TEXTURE_DESC_v1 = CUDA_TEXTURE_DESC_st

const CUDA_TEXTURE_DESC = CUDA_TEXTURE_DESC_v1

@cenum CUresourceViewFormat_enum::UInt32 begin
    CU_RES_VIEW_FORMAT_NONE = 0
    CU_RES_VIEW_FORMAT_UINT_1X8 = 1
    CU_RES_VIEW_FORMAT_UINT_2X8 = 2
    CU_RES_VIEW_FORMAT_UINT_4X8 = 3
    CU_RES_VIEW_FORMAT_SINT_1X8 = 4
    CU_RES_VIEW_FORMAT_SINT_2X8 = 5
    CU_RES_VIEW_FORMAT_SINT_4X8 = 6
    CU_RES_VIEW_FORMAT_UINT_1X16 = 7
    CU_RES_VIEW_FORMAT_UINT_2X16 = 8
    CU_RES_VIEW_FORMAT_UINT_4X16 = 9
    CU_RES_VIEW_FORMAT_SINT_1X16 = 10
    CU_RES_VIEW_FORMAT_SINT_2X16 = 11
    CU_RES_VIEW_FORMAT_SINT_4X16 = 12
    CU_RES_VIEW_FORMAT_UINT_1X32 = 13
    CU_RES_VIEW_FORMAT_UINT_2X32 = 14
    CU_RES_VIEW_FORMAT_UINT_4X32 = 15
    CU_RES_VIEW_FORMAT_SINT_1X32 = 16
    CU_RES_VIEW_FORMAT_SINT_2X32 = 17
    CU_RES_VIEW_FORMAT_SINT_4X32 = 18
    CU_RES_VIEW_FORMAT_FLOAT_1X16 = 19
    CU_RES_VIEW_FORMAT_FLOAT_2X16 = 20
    CU_RES_VIEW_FORMAT_FLOAT_4X16 = 21
    CU_RES_VIEW_FORMAT_FLOAT_1X32 = 22
    CU_RES_VIEW_FORMAT_FLOAT_2X32 = 23
    CU_RES_VIEW_FORMAT_FLOAT_4X32 = 24
    CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = 25
    CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = 26
    CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = 27
    CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = 28
    CU_RES_VIEW_FORMAT_SIGNED_BC4 = 29
    CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = 30
    CU_RES_VIEW_FORMAT_SIGNED_BC5 = 31
    CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = 32
    CU_RES_VIEW_FORMAT_SIGNED_BC6H = 33
    CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = 34
end

const CUresourceViewFormat = CUresourceViewFormat_enum

struct CUDA_RESOURCE_VIEW_DESC_st
    format::CUresourceViewFormat
    width::Csize_t
    height::Csize_t
    depth::Csize_t
    firstMipmapLevel::Cuint
    lastMipmapLevel::Cuint
    firstLayer::Cuint
    lastLayer::Cuint
    reserved::NTuple{16, Cuint}
end

const CUDA_RESOURCE_VIEW_DESC_v1 = CUDA_RESOURCE_VIEW_DESC_st

const CUDA_RESOURCE_VIEW_DESC = CUDA_RESOURCE_VIEW_DESC_v1

struct CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st
    p2pToken::Culonglong
    vaSpaceToken::Cuint
end

const CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1 = CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st

const CUDA_POINTER_ATTRIBUTE_P2P_TOKENS = CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1

@cenum CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum::UInt32 begin
    CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE = 0
    CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ = 1
    CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE = 3
end

const CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS = CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum

struct CUDA_LAUNCH_PARAMS_st
    _function::CUfunction
    gridDimX::Cuint
    gridDimY::Cuint
    gridDimZ::Cuint
    blockDimX::Cuint
    blockDimY::Cuint
    blockDimZ::Cuint
    sharedMemBytes::Cuint
    hStream::CUstream
    kernelParams::Ptr{Ptr{Cvoid}}
end

const CUDA_LAUNCH_PARAMS_v1 = CUDA_LAUNCH_PARAMS_st

const CUDA_LAUNCH_PARAMS = CUDA_LAUNCH_PARAMS_v1

@cenum CUexternalMemoryHandleType_enum::UInt32 begin
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = 4
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = 6
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8
end

const CUexternalMemoryHandleType = CUexternalMemoryHandleType_enum

struct var"##Ctag#382"
    data::NTuple{16, UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#382"}, f::Symbol)
    f === :fd && return Ptr{Cint}(x + 0)
    f === :win32 && return Ptr{var"##Ctag#383"}(x + 0)
    f === :nvSciBufObject && return Ptr{Ptr{Cvoid}}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#382", f::Symbol)
    r = Ref{var"##Ctag#382"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#382"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#382"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
    data::NTuple{104, UInt8}
end

function Base.getproperty(x::Ptr{CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st}, f::Symbol)
    f === :type && return Ptr{CUexternalMemoryHandleType}(x + 0)
    f === :handle && return Ptr{var"##Ctag#382"}(x + 8)
    f === :size && return Ptr{Culonglong}(x + 24)
    f === :flags && return Ptr{Cuint}(x + 32)
    f === :reserved && return Ptr{NTuple{16, Cuint}}(x + 36)
    return getfield(x, f)
end

function Base.getproperty(x::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st, f::Symbol)
    r = Ref{CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st}(x)
    ptr = Base.unsafe_convert(Ptr{CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

const CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 = CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st

const CUDA_EXTERNAL_MEMORY_HANDLE_DESC = CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1

struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
    offset::Culonglong
    size::Culonglong
    flags::Cuint
    reserved::NTuple{16, Cuint}
end

const CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 = CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st

const CUDA_EXTERNAL_MEMORY_BUFFER_DESC = CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1

struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st
    offset::Culonglong
    arrayDesc::CUDA_ARRAY3D_DESCRIPTOR
    numLevels::Cuint
    reserved::NTuple{16, Cuint}
end

const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1 = CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st

const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC = CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1

@cenum CUexternalSemaphoreHandleType_enum::UInt32 begin
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = 1
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = 2
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = 4
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE = 5
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC = 6
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX = 7
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT = 8
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD = 9
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32 = 10
end

const CUexternalSemaphoreHandleType = CUexternalSemaphoreHandleType_enum

struct var"##Ctag#368"
    data::NTuple{16, UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#368"}, f::Symbol)
    f === :fd && return Ptr{Cint}(x + 0)
    f === :win32 && return Ptr{var"##Ctag#369"}(x + 0)
    f === :nvSciSyncObj && return Ptr{Ptr{Cvoid}}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#368", f::Symbol)
    r = Ref{var"##Ctag#368"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#368"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#368"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
    data::NTuple{96, UInt8}
end

function Base.getproperty(x::Ptr{CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st}, f::Symbol)
    f === :type && return Ptr{CUexternalSemaphoreHandleType}(x + 0)
    f === :handle && return Ptr{var"##Ctag#368"}(x + 8)
    f === :flags && return Ptr{Cuint}(x + 24)
    f === :reserved && return Ptr{NTuple{16, Cuint}}(x + 28)
    return getfield(x, f)
end

function Base.getproperty(x::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st, f::Symbol)
    r = Ref{CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st}(x)
    ptr = Base.unsafe_convert(Ptr{CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1 = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st

const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1

struct var"##Ctag#386"
    value::Culonglong
end
function Base.getproperty(x::Ptr{var"##Ctag#386"}, f::Symbol)
    f === :value && return Ptr{Culonglong}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#386", f::Symbol)
    r = Ref{var"##Ctag#386"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#386"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#386"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct var"##Ctag#387"
    data::NTuple{8, UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#387"}, f::Symbol)
    f === :fence && return Ptr{Ptr{Cvoid}}(x + 0)
    f === :reserved && return Ptr{Culonglong}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#387", f::Symbol)
    r = Ref{var"##Ctag#387"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#387"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#387"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct var"##Ctag#388"
    key::Culonglong
end
function Base.getproperty(x::Ptr{var"##Ctag#388"}, f::Symbol)
    f === :key && return Ptr{Culonglong}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#388", f::Symbol)
    r = Ref{var"##Ctag#388"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#388"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#388"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct var"##Ctag#385"
    data::NTuple{72, UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#385"}, f::Symbol)
    f === :fence && return Ptr{var"##Ctag#386"}(x + 0)
    f === :nvSciSync && return Ptr{var"##Ctag#387"}(x + 8)
    f === :keyedMutex && return Ptr{var"##Ctag#388"}(x + 16)
    f === :reserved && return Ptr{NTuple{12, Cuint}}(x + 24)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#385", f::Symbol)
    r = Ref{var"##Ctag#385"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#385"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#385"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
    data::NTuple{144, UInt8}
end

function Base.getproperty(x::Ptr{CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st}, f::Symbol)
    f === :params && return Ptr{Cvoid}(x + 0)
    f === :flags && return Ptr{Cuint}(x + 72)
    f === :reserved && return Ptr{NTuple{16, Cuint}}(x + 76)
    return getfield(x, f)
end

function Base.getproperty(x::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st, f::Symbol)
    r = Ref{CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st}(x)
    ptr = Base.unsafe_convert(Ptr{CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1 = CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st

const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS = CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1

struct var"##Ctag#377"
    value::Culonglong
end
function Base.getproperty(x::Ptr{var"##Ctag#377"}, f::Symbol)
    f === :value && return Ptr{Culonglong}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#377", f::Symbol)
    r = Ref{var"##Ctag#377"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#377"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#377"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct var"##Ctag#378"
    data::NTuple{8, UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#378"}, f::Symbol)
    f === :fence && return Ptr{Ptr{Cvoid}}(x + 0)
    f === :reserved && return Ptr{Culonglong}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#378", f::Symbol)
    r = Ref{var"##Ctag#378"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#378"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#378"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct var"##Ctag#379"
    key::Culonglong
    timeoutMs::Cuint
end
function Base.getproperty(x::Ptr{var"##Ctag#379"}, f::Symbol)
    f === :key && return Ptr{Culonglong}(x + 0)
    f === :timeoutMs && return Ptr{Cuint}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#379", f::Symbol)
    r = Ref{var"##Ctag#379"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#379"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#379"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct var"##Ctag#376"
    data::NTuple{72, UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#376"}, f::Symbol)
    f === :fence && return Ptr{var"##Ctag#377"}(x + 0)
    f === :nvSciSync && return Ptr{var"##Ctag#378"}(x + 8)
    f === :keyedMutex && return Ptr{var"##Ctag#379"}(x + 16)
    f === :reserved && return Ptr{NTuple{10, Cuint}}(x + 32)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#376", f::Symbol)
    r = Ref{var"##Ctag#376"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#376"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#376"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
    data::NTuple{144, UInt8}
end

function Base.getproperty(x::Ptr{CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st}, f::Symbol)
    f === :params && return Ptr{Cvoid}(x + 0)
    f === :flags && return Ptr{Cuint}(x + 72)
    f === :reserved && return Ptr{NTuple{16, Cuint}}(x + 76)
    return getfield(x, f)
end

function Base.getproperty(x::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st, f::Symbol)
    r = Ref{CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st}(x)
    ptr = Base.unsafe_convert(Ptr{CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1 = CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st

const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1

struct CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st
    extSemArray::Ptr{CUexternalSemaphore}
    paramsArray::Ptr{CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS}
    numExtSems::Cuint
end

const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 = CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st

const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS = CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1

struct CUDA_EXT_SEM_WAIT_NODE_PARAMS_st
    extSemArray::Ptr{CUexternalSemaphore}
    paramsArray::Ptr{CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS}
    numExtSems::Cuint
end

const CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1 = CUDA_EXT_SEM_WAIT_NODE_PARAMS_st

const CUDA_EXT_SEM_WAIT_NODE_PARAMS = CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1

const CUmemGenericAllocationHandle_v1 = Culonglong

const CUmemGenericAllocationHandle = CUmemGenericAllocationHandle_v1

@cenum CUmemAllocationHandleType_enum::UInt32 begin
    CU_MEM_HANDLE_TYPE_NONE = 0
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
    CU_MEM_HANDLE_TYPE_WIN32 = 2
    CU_MEM_HANDLE_TYPE_WIN32_KMT = 4
    CU_MEM_HANDLE_TYPE_MAX = 2147483647
end

const CUmemAllocationHandleType = CUmemAllocationHandleType_enum

@cenum CUmemAccess_flags_enum::UInt32 begin
    CU_MEM_ACCESS_FLAGS_PROT_NONE = 0
    CU_MEM_ACCESS_FLAGS_PROT_READ = 1
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
    CU_MEM_ACCESS_FLAGS_PROT_MAX = 2147483647
end

const CUmemAccess_flags = CUmemAccess_flags_enum

@cenum CUmemLocationType_enum::UInt32 begin
    CU_MEM_LOCATION_TYPE_INVALID = 0
    CU_MEM_LOCATION_TYPE_DEVICE = 1
    CU_MEM_LOCATION_TYPE_MAX = 2147483647
end

const CUmemLocationType = CUmemLocationType_enum

@cenum CUmemAllocationType_enum::UInt32 begin
    CU_MEM_ALLOCATION_TYPE_INVALID = 0
    CU_MEM_ALLOCATION_TYPE_PINNED = 1
    CU_MEM_ALLOCATION_TYPE_MAX = 2147483647
end

const CUmemAllocationType = CUmemAllocationType_enum

@cenum CUmemAllocationGranularity_flags_enum::UInt32 begin
    CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1
end

const CUmemAllocationGranularity_flags = CUmemAllocationGranularity_flags_enum

@cenum CUmemRangeHandleType_enum::UInt32 begin
    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 1
    CU_MEM_RANGE_HANDLE_TYPE_MAX = 2147483647
end

const CUmemRangeHandleType = CUmemRangeHandleType_enum

@cenum CUarraySparseSubresourceType_enum::UInt32 begin
    CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL = 0
    CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL = 1
end

const CUarraySparseSubresourceType = CUarraySparseSubresourceType_enum

@cenum CUmemOperationType_enum::UInt32 begin
    CU_MEM_OPERATION_TYPE_MAP = 1
    CU_MEM_OPERATION_TYPE_UNMAP = 2
end

const CUmemOperationType = CUmemOperationType_enum

@cenum CUmemHandleType_enum::UInt32 begin
    CU_MEM_HANDLE_TYPE_GENERIC = 0
end

const CUmemHandleType = CUmemHandleType_enum

struct var"##Ctag#371"
    data::NTuple{8, UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#371"}, f::Symbol)
    f === :mipmap && return Ptr{CUmipmappedArray}(x + 0)
    f === :array && return Ptr{CUarray}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#371", f::Symbol)
    r = Ref{var"##Ctag#371"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#371"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#371"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct var"##Ctag#372"
    data::NTuple{32, UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#372"}, f::Symbol)
    f === :sparseLevel && return Ptr{var"##Ctag#373"}(x + 0)
    f === :miptail && return Ptr{var"##Ctag#374"}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#372", f::Symbol)
    r = Ref{var"##Ctag#372"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#372"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#372"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct var"##Ctag#375"
    data::NTuple{8, UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#375"}, f::Symbol)
    f === :memHandle && return Ptr{CUmemGenericAllocationHandle}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#375", f::Symbol)
    r = Ref{var"##Ctag#375"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#375"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#375"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct CUarrayMapInfo_st
    data::NTuple{96, UInt8}
end

function Base.getproperty(x::Ptr{CUarrayMapInfo_st}, f::Symbol)
    f === :resourceType && return Ptr{CUresourcetype}(x + 0)
    f === :resource && return Ptr{var"##Ctag#371"}(x + 8)
    f === :subresourceType && return Ptr{CUarraySparseSubresourceType}(x + 16)
    f === :subresource && return Ptr{var"##Ctag#372"}(x + 24)
    f === :memOperationType && return Ptr{CUmemOperationType}(x + 56)
    f === :memHandleType && return Ptr{CUmemHandleType}(x + 60)
    f === :memHandle && return Ptr{var"##Ctag#375"}(x + 64)
    f === :offset && return Ptr{Culonglong}(x + 72)
    f === :deviceBitMask && return Ptr{Cuint}(x + 80)
    f === :flags && return Ptr{Cuint}(x + 84)
    f === :reserved && return Ptr{NTuple{2, Cuint}}(x + 88)
    return getfield(x, f)
end

function Base.getproperty(x::CUarrayMapInfo_st, f::Symbol)
    r = Ref{CUarrayMapInfo_st}(x)
    ptr = Base.unsafe_convert(Ptr{CUarrayMapInfo_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{CUarrayMapInfo_st}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

const CUarrayMapInfo_v1 = CUarrayMapInfo_st

const CUarrayMapInfo = CUarrayMapInfo_v1

struct CUmemLocation_st
    type::CUmemLocationType
    id::Cint
end

const CUmemLocation_v1 = CUmemLocation_st

const CUmemLocation = CUmemLocation_v1

@cenum CUmemAllocationCompType_enum::UInt32 begin
    CU_MEM_ALLOCATION_COMP_NONE = 0
    CU_MEM_ALLOCATION_COMP_GENERIC = 1
end

const CUmemAllocationCompType = CUmemAllocationCompType_enum

struct var"##Ctag#361"
    compressionType::Cuchar
    gpuDirectRDMACapable::Cuchar
    usage::Cushort
    reserved::NTuple{4, Cuchar}
end
function Base.getproperty(x::Ptr{var"##Ctag#361"}, f::Symbol)
    f === :compressionType && return Ptr{Cuchar}(x + 0)
    f === :gpuDirectRDMACapable && return Ptr{Cuchar}(x + 1)
    f === :usage && return Ptr{Cushort}(x + 2)
    f === :reserved && return Ptr{NTuple{4, Cuchar}}(x + 4)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#361", f::Symbol)
    r = Ref{var"##Ctag#361"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#361"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#361"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct CUmemAllocationProp_st
    data::NTuple{32, UInt8}
end

function Base.getproperty(x::Ptr{CUmemAllocationProp_st}, f::Symbol)
    f === :type && return Ptr{CUmemAllocationType}(x + 0)
    f === :requestedHandleTypes && return Ptr{CUmemAllocationHandleType}(x + 4)
    f === :location && return Ptr{CUmemLocation}(x + 8)
    f === :win32HandleMetaData && return Ptr{Ptr{Cvoid}}(x + 16)
    f === :allocFlags && return Ptr{var"##Ctag#361"}(x + 24)
    return getfield(x, f)
end

function Base.getproperty(x::CUmemAllocationProp_st, f::Symbol)
    r = Ref{CUmemAllocationProp_st}(x)
    ptr = Base.unsafe_convert(Ptr{CUmemAllocationProp_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{CUmemAllocationProp_st}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

const CUmemAllocationProp_v1 = CUmemAllocationProp_st

const CUmemAllocationProp = CUmemAllocationProp_v1

struct CUmemAccessDesc_st
    location::CUmemLocation
    flags::CUmemAccess_flags
end

const CUmemAccessDesc_v1 = CUmemAccessDesc_st

const CUmemAccessDesc = CUmemAccessDesc_v1

@cenum CUgraphExecUpdateResult_enum::UInt32 begin
    CU_GRAPH_EXEC_UPDATE_SUCCESS = 0
    CU_GRAPH_EXEC_UPDATE_ERROR = 1
    CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = 2
    CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = 3
    CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = 4
    CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = 5
    CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = 6
    CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE = 7
    CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED = 8
end

const CUgraphExecUpdateResult = CUgraphExecUpdateResult_enum

@cenum CUmemPool_attribute_enum::UInt32 begin
    CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 1
    CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = 2
    CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 3
    CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4
    CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 5
    CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = 6
    CU_MEMPOOL_ATTR_USED_MEM_CURRENT = 7
    CU_MEMPOOL_ATTR_USED_MEM_HIGH = 8
end

const CUmemPool_attribute = CUmemPool_attribute_enum

struct CUmemPoolProps_st
    allocType::CUmemAllocationType
    handleTypes::CUmemAllocationHandleType
    location::CUmemLocation
    win32SecurityAttributes::Ptr{Cvoid}
    reserved::NTuple{64, Cuchar}
end

const CUmemPoolProps_v1 = CUmemPoolProps_st

const CUmemPoolProps = CUmemPoolProps_v1

struct CUmemPoolPtrExportData_st
    reserved::NTuple{64, Cuchar}
end

const CUmemPoolPtrExportData_v1 = CUmemPoolPtrExportData_st

const CUmemPoolPtrExportData = CUmemPoolPtrExportData_v1

struct CUDA_MEM_ALLOC_NODE_PARAMS_st
    poolProps::CUmemPoolProps
    accessDescs::Ptr{CUmemAccessDesc}
    accessDescCount::Csize_t
    bytesize::Csize_t
    dptr::CUdeviceptr
end

const CUDA_MEM_ALLOC_NODE_PARAMS = CUDA_MEM_ALLOC_NODE_PARAMS_st

@cenum CUgraphMem_attribute_enum::UInt32 begin
    CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT = 0
    CU_GRAPH_MEM_ATTR_USED_MEM_HIGH = 1
    CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT = 2
    CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH = 3
end

const CUgraphMem_attribute = CUgraphMem_attribute_enum

@cenum CUflushGPUDirectRDMAWritesOptions_enum::UInt32 begin
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST = 1
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS = 2
end

const CUflushGPUDirectRDMAWritesOptions = CUflushGPUDirectRDMAWritesOptions_enum

@cenum CUGPUDirectRDMAWritesOrdering_enum::UInt32 begin
    CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE = 0
    CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER = 100
    CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES = 200
end

const CUGPUDirectRDMAWritesOrdering = CUGPUDirectRDMAWritesOrdering_enum

@cenum CUflushGPUDirectRDMAWritesScope_enum::UInt32 begin
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER = 100
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES = 200
end

const CUflushGPUDirectRDMAWritesScope = CUflushGPUDirectRDMAWritesScope_enum

@cenum CUflushGPUDirectRDMAWritesTarget_enum::UInt32 begin
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX = 0
end

const CUflushGPUDirectRDMAWritesTarget = CUflushGPUDirectRDMAWritesTarget_enum

@cenum CUgraphDebugDot_flags_enum::UInt32 begin
    CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE = 1
    CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES = 2
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS = 4
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS = 8
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS = 16
    CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS = 32
    CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS = 64
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS = 128
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS = 256
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES = 512
    CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES = 1024
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS = 2048
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS = 4096
    CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS = 8192
end

const CUgraphDebugDot_flags = CUgraphDebugDot_flags_enum

@cenum CUuserObject_flags_enum::UInt32 begin
    CU_USER_OBJECT_NO_DESTRUCTOR_SYNC = 1
end

const CUuserObject_flags = CUuserObject_flags_enum

@cenum CUuserObjectRetain_flags_enum::UInt32 begin
    CU_GRAPH_USER_OBJECT_MOVE = 1
end

const CUuserObjectRetain_flags = CUuserObjectRetain_flags_enum

@cenum CUgraphInstantiate_flags_enum::UInt32 begin
    CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 1
    CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = 8
end

const CUgraphInstantiate_flags = CUgraphInstantiate_flags_enum

@checked function cuGetErrorString(error, pStr)
        ccall((:cuGetErrorString, libcuda), CUresult, (CUresult, Ptr{Cstring}), error, pStr)
    end

@checked function cuGetErrorName(error, pStr)
        ccall((:cuGetErrorName, libcuda), CUresult, (CUresult, Ptr{Cstring}), error, pStr)
    end

@checked function cuInit(Flags)
        ccall((:cuInit, libcuda), CUresult, (Cuint,), Flags)
    end

@checked function cuDriverGetVersion(driverVersion)
        ccall((:cuDriverGetVersion, libcuda), CUresult, (Ptr{Cint},), driverVersion)
    end

@checked function cuDeviceGet(device, ordinal)
        ccall((:cuDeviceGet, libcuda), CUresult, (Ptr{CUdevice}, Cint), device, ordinal)
    end

@checked function cuDeviceGetCount(count)
        ccall((:cuDeviceGetCount, libcuda), CUresult, (Ptr{Cint},), count)
    end

@checked function cuDeviceGetName(name, len, dev)
        ccall((:cuDeviceGetName, libcuda), CUresult, (Cstring, Cint, CUdevice), name, len, dev)
    end

@checked function cuDeviceGetUuid(uuid, dev)
        ccall((:cuDeviceGetUuid, libcuda), CUresult, (Ptr{CUuuid}, CUdevice), uuid, dev)
    end

@checked function cuDeviceGetUuid_v2(uuid, dev)
        ccall((:cuDeviceGetUuid_v2, libcuda), CUresult, (Ptr{CUuuid}, CUdevice), uuid, dev)
    end

@checked function cuDeviceGetLuid(luid, deviceNodeMask, dev)
        ccall((:cuDeviceGetLuid, libcuda), CUresult, (Cstring, Ptr{Cuint}, CUdevice), luid, deviceNodeMask, dev)
    end

@checked function cuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, format, numChannels, dev)
        initialize_context()
        ccall((:cuDeviceGetTexture1DLinearMaxWidth, libcuda), CUresult, (Ptr{Csize_t}, CUarray_format, Cuint, CUdevice), maxWidthInElements, format, numChannels, dev)
    end

@checked function cuDeviceGetAttribute(pi, attrib, dev)
        ccall((:cuDeviceGetAttribute, libcuda), CUresult, (Ptr{Cint}, CUdevice_attribute, CUdevice), pi, attrib, dev)
    end

@checked function cuDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, dev, flags)
        initialize_context()
        ccall((:cuDeviceGetNvSciSyncAttributes, libcuda), CUresult, (Ptr{Cvoid}, CUdevice, Cint), nvSciSyncAttrList, dev, flags)
    end

@checked function cuDeviceSetMemPool(dev, pool)
        initialize_context()
        ccall((:cuDeviceSetMemPool, libcuda), CUresult, (CUdevice, CUmemoryPool), dev, pool)
    end

@checked function cuDeviceGetMemPool(pool, dev)
        initialize_context()
        ccall((:cuDeviceGetMemPool, libcuda), CUresult, (Ptr{CUmemoryPool}, CUdevice), pool, dev)
    end

@checked function cuDeviceGetDefaultMemPool(pool_out, dev)
        initialize_context()
        ccall((:cuDeviceGetDefaultMemPool, libcuda), CUresult, (Ptr{CUmemoryPool}, CUdevice), pool_out, dev)
    end

@checked function cuFlushGPUDirectRDMAWrites(target, scope)
        initialize_context()
        ccall((:cuFlushGPUDirectRDMAWrites, libcuda), CUresult, (CUflushGPUDirectRDMAWritesTarget, CUflushGPUDirectRDMAWritesScope), target, scope)
    end

@checked function cuDeviceGetProperties(prop, dev)
        ccall((:cuDeviceGetProperties, libcuda), CUresult, (Ptr{CUdevprop}, CUdevice), prop, dev)
    end

@checked function cuDeviceComputeCapability(major, minor, dev)
        ccall((:cuDeviceComputeCapability, libcuda), CUresult, (Ptr{Cint}, Ptr{Cint}, CUdevice), major, minor, dev)
    end

@checked function cuDevicePrimaryCtxRetain(pctx, dev)
        ccall((:cuDevicePrimaryCtxRetain, libcuda), CUresult, (Ptr{CUcontext}, CUdevice), pctx, dev)
    end

@checked function cuDevicePrimaryCtxGetState(dev, flags, active)
        ccall((:cuDevicePrimaryCtxGetState, libcuda), CUresult, (CUdevice, Ptr{Cuint}, Ptr{Cint}), dev, flags, active)
    end

@checked function cuDeviceGetExecAffinitySupport(pi, type, dev)
        initialize_context()
        ccall((:cuDeviceGetExecAffinitySupport, libcuda), CUresult, (Ptr{Cint}, CUexecAffinityType, CUdevice), pi, type, dev)
    end

@checked function cuCtxCreate_v3(pctx, paramsArray, numParams, flags, dev)
        initialize_context()
        ccall((:cuCtxCreate_v3, libcuda), CUresult, (Ptr{CUcontext}, Ptr{CUexecAffinityParam}, Cint, Cuint, CUdevice), pctx, paramsArray, numParams, flags, dev)
    end

@checked function cuCtxSetCurrent(ctx)
        ccall((:cuCtxSetCurrent, libcuda), CUresult, (CUcontext,), ctx)
    end

@checked function cuCtxGetCurrent(pctx)
        ccall((:cuCtxGetCurrent, libcuda), CUresult, (Ptr{CUcontext},), pctx)
    end

@checked function cuCtxGetDevice(device)
        ccall((:cuCtxGetDevice, libcuda), CUresult, (Ptr{CUdevice},), device)
    end

@checked function cuCtxGetFlags(flags)
        initialize_context()
        ccall((:cuCtxGetFlags, libcuda), CUresult, (Ptr{Cuint},), flags)
    end

@checked function cuCtxSynchronize()
        initialize_context()
        ccall((:cuCtxSynchronize, libcuda), CUresult, ())
    end

@checked function cuCtxSetLimit(limit, value)
        initialize_context()
        ccall((:cuCtxSetLimit, libcuda), CUresult, (CUlimit, Csize_t), limit, value)
    end

@checked function cuCtxGetLimit(pvalue, limit)
        initialize_context()
        ccall((:cuCtxGetLimit, libcuda), CUresult, (Ptr{Csize_t}, CUlimit), pvalue, limit)
    end

@checked function cuCtxGetCacheConfig(pconfig)
        initialize_context()
        ccall((:cuCtxGetCacheConfig, libcuda), CUresult, (Ptr{CUfunc_cache},), pconfig)
    end

@checked function cuCtxSetCacheConfig(config)
        initialize_context()
        ccall((:cuCtxSetCacheConfig, libcuda), CUresult, (CUfunc_cache,), config)
    end

@checked function cuCtxGetSharedMemConfig(pConfig)
        initialize_context()
        ccall((:cuCtxGetSharedMemConfig, libcuda), CUresult, (Ptr{CUsharedconfig},), pConfig)
    end

@checked function cuCtxSetSharedMemConfig(config)
        initialize_context()
        ccall((:cuCtxSetSharedMemConfig, libcuda), CUresult, (CUsharedconfig,), config)
    end

@checked function cuCtxGetApiVersion(ctx, version)
        initialize_context()
        ccall((:cuCtxGetApiVersion, libcuda), CUresult, (CUcontext, Ptr{Cuint}), ctx, version)
    end

@checked function cuCtxGetStreamPriorityRange(leastPriority, greatestPriority)
        initialize_context()
        ccall((:cuCtxGetStreamPriorityRange, libcuda), CUresult, (Ptr{Cint}, Ptr{Cint}), leastPriority, greatestPriority)
    end

@checked function cuCtxResetPersistingL2Cache()
        initialize_context()
        ccall((:cuCtxResetPersistingL2Cache, libcuda), CUresult, ())
    end

@checked function cuCtxGetExecAffinity(pExecAffinity, type)
        initialize_context()
        ccall((:cuCtxGetExecAffinity, libcuda), CUresult, (Ptr{CUexecAffinityParam}, CUexecAffinityType), pExecAffinity, type)
    end

@checked function cuCtxAttach(pctx, flags)
        initialize_context()
        ccall((:cuCtxAttach, libcuda), CUresult, (Ptr{CUcontext}, Cuint), pctx, flags)
    end

@checked function cuCtxDetach(ctx)
        initialize_context()
        ccall((:cuCtxDetach, libcuda), CUresult, (CUcontext,), ctx)
    end

@checked function cuModuleLoad(_module, fname)
        initialize_context()
        ccall((:cuModuleLoad, libcuda), CUresult, (Ptr{CUmodule}, Cstring), _module, fname)
    end

@checked function cuModuleLoadData(_module, image)
        initialize_context()
        ccall((:cuModuleLoadData, libcuda), CUresult, (Ptr{CUmodule}, Ptr{Cvoid}), _module, image)
    end

@checked function cuModuleLoadDataEx(_module, image, numOptions, options, optionValues)
        initialize_context()
        ccall((:cuModuleLoadDataEx, libcuda), CUresult, (Ptr{CUmodule}, Ptr{Cvoid}, Cuint, Ptr{CUjit_option}, Ptr{Ptr{Cvoid}}), _module, image, numOptions, options, optionValues)
    end

@checked function cuModuleLoadFatBinary(_module, fatCubin)
        initialize_context()
        ccall((:cuModuleLoadFatBinary, libcuda), CUresult, (Ptr{CUmodule}, Ptr{Cvoid}), _module, fatCubin)
    end

@checked function cuModuleUnload(hmod)
        initialize_context()
        ccall((:cuModuleUnload, libcuda), CUresult, (CUmodule,), hmod)
    end

@cenum CUmoduleLoadingMode_enum::UInt32 begin
    CU_MODULE_EAGER_LOADING = 1
    CU_MODULE_LAZY_LOADING = 2
end

const CUmoduleLoadingMode = CUmoduleLoadingMode_enum

@checked function cuModuleGetLoadingMode(mode)
        initialize_context()
        ccall((:cuModuleGetLoadingMode, libcuda), CUresult, (Ptr{CUmoduleLoadingMode},), mode)
    end

@checked function cuModuleGetFunction(hfunc, hmod, name)
        initialize_context()
        ccall((:cuModuleGetFunction, libcuda), CUresult, (Ptr{CUfunction}, CUmodule, Cstring), hfunc, hmod, name)
    end

@checked function cuModuleGetTexRef(pTexRef, hmod, name)
        initialize_context()
        ccall((:cuModuleGetTexRef, libcuda), CUresult, (Ptr{CUtexref}, CUmodule, Cstring), pTexRef, hmod, name)
    end

@checked function cuModuleGetSurfRef(pSurfRef, hmod, name)
        initialize_context()
        ccall((:cuModuleGetSurfRef, libcuda), CUresult, (Ptr{CUsurfref}, CUmodule, Cstring), pSurfRef, hmod, name)
    end

@checked function cuLinkComplete(state, cubinOut, sizeOut)
        initialize_context()
        ccall((:cuLinkComplete, libcuda), CUresult, (CUlinkState, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}), state, cubinOut, sizeOut)
    end

@checked function cuLinkDestroy(state)
        initialize_context()
        ccall((:cuLinkDestroy, libcuda), CUresult, (CUlinkState,), state)
    end

@checked function cuMemFreeHost(p)
        initialize_context()
        ccall((:cuMemFreeHost, libcuda), CUresult, (Ptr{Cvoid},), p)
    end

@checked function cuMemHostAlloc(pp, bytesize, Flags)
        initialize_context()
        ccall((:cuMemHostAlloc, libcuda), CUresult, (Ptr{Ptr{Cvoid}}, Csize_t, Cuint), pp, bytesize, Flags)
    end

@checked function cuMemHostGetFlags(pFlags, p)
        initialize_context()
        ccall((:cuMemHostGetFlags, libcuda), CUresult, (Ptr{Cuint}, Ptr{Cvoid}), pFlags, p)
    end

@checked function cuMemAllocManaged(dptr, bytesize, flags)
        initialize_context()
        ccall((:cuMemAllocManaged, libcuda), CUresult, (Ptr{CUdeviceptr}, Csize_t, Cuint), dptr, bytesize, flags)
    end

@checked function cuDeviceGetByPCIBusId(dev, pciBusId)
        initialize_context()
        ccall((:cuDeviceGetByPCIBusId, libcuda), CUresult, (Ptr{CUdevice}, Cstring), dev, pciBusId)
    end

@checked function cuDeviceGetPCIBusId(pciBusId, len, dev)
        initialize_context()
        ccall((:cuDeviceGetPCIBusId, libcuda), CUresult, (Cstring, Cint, CUdevice), pciBusId, len, dev)
    end

@checked function cuIpcGetEventHandle(pHandle, event)
        initialize_context()
        ccall((:cuIpcGetEventHandle, libcuda), CUresult, (Ptr{CUipcEventHandle}, CUevent), pHandle, event)
    end

@checked function cuIpcOpenEventHandle(phEvent, handle)
        initialize_context()
        ccall((:cuIpcOpenEventHandle, libcuda), CUresult, (Ptr{CUevent}, CUipcEventHandle), phEvent, handle)
    end

@checked function cuIpcGetMemHandle(pHandle, dptr)
        initialize_context()
        ccall((:cuIpcGetMemHandle, libcuda), CUresult, (Ptr{CUipcMemHandle}, CUdeviceptr), pHandle, dptr)
    end

@checked function cuIpcCloseMemHandle(dptr)
        initialize_context()
        ccall((:cuIpcCloseMemHandle, libcuda), CUresult, (CUdeviceptr,), dptr)
    end

@checked function cuMemHostUnregister(p)
        initialize_context()
        ccall((:cuMemHostUnregister, libcuda), CUresult, (Ptr{Cvoid},), p)
    end

@checked function cuMemcpy(dst, src, ByteCount)
        initialize_context()
        ccall((:cuMemcpy, libcuda), CUresult, (CUdeviceptr, CUdeviceptr, Csize_t), dst, src, ByteCount)
    end

@checked function cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount)
        initialize_context()
        ccall((:cuMemcpyPeer, libcuda), CUresult, (CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, Csize_t), dstDevice, dstContext, srcDevice, srcContext, ByteCount)
    end

@checked function cuMemcpy3DPeer(pCopy)
        initialize_context()
        ccall((:cuMemcpy3DPeer, libcuda), CUresult, (Ptr{CUDA_MEMCPY3D_PEER},), pCopy)
    end

@checked function cuMemcpyAsync(dst, src, ByteCount, hStream)
        initialize_context()
        ccall((:cuMemcpyAsync, libcuda), CUresult, (CUdeviceptr, CUdeviceptr, Csize_t, CUstream), dst, src, ByteCount, hStream)
    end

@checked function cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream)
        initialize_context()
        ccall((:cuMemcpyPeerAsync, libcuda), CUresult, (CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, Csize_t, CUstream), dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream)
    end

@checked function cuMemcpy3DPeerAsync(pCopy, hStream)
        initialize_context()
        ccall((:cuMemcpy3DPeerAsync, libcuda), CUresult, (Ptr{CUDA_MEMCPY3D_PEER}, CUstream), pCopy, hStream)
    end

@checked function cuMemsetD8Async(dstDevice, uc, N, hStream)
        initialize_context()
        ccall((:cuMemsetD8Async, libcuda), CUresult, (CUdeviceptr, Cuchar, Csize_t, CUstream), dstDevice, uc, N, hStream)
    end

@checked function cuMemsetD16Async(dstDevice, us, N, hStream)
        initialize_context()
        ccall((:cuMemsetD16Async, libcuda), CUresult, (CUdeviceptr, Cushort, Csize_t, CUstream), dstDevice, us, N, hStream)
    end

@checked function cuMemsetD32Async(dstDevice, ui, N, hStream)
        initialize_context()
        ccall((:cuMemsetD32Async, libcuda), CUresult, (CUdeviceptr, Cuint, Csize_t, CUstream), dstDevice, ui, N, hStream)
    end

@checked function cuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream)
        initialize_context()
        ccall((:cuMemsetD2D8Async, libcuda), CUresult, (CUdeviceptr, Csize_t, Cuchar, Csize_t, Csize_t, CUstream), dstDevice, dstPitch, uc, Width, Height, hStream)
    end

@checked function cuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream)
        initialize_context()
        ccall((:cuMemsetD2D16Async, libcuda), CUresult, (CUdeviceptr, Csize_t, Cushort, Csize_t, Csize_t, CUstream), dstDevice, dstPitch, us, Width, Height, hStream)
    end

@checked function cuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream)
        initialize_context()
        ccall((:cuMemsetD2D32Async, libcuda), CUresult, (CUdeviceptr, Csize_t, Cuint, Csize_t, Csize_t, CUstream), dstDevice, dstPitch, ui, Width, Height, hStream)
    end

@checked function cuArrayGetSparseProperties(sparseProperties, array)
        initialize_context()
        ccall((:cuArrayGetSparseProperties, libcuda), CUresult, (Ptr{CUDA_ARRAY_SPARSE_PROPERTIES}, CUarray), sparseProperties, array)
    end

@checked function cuMipmappedArrayGetSparseProperties(sparseProperties, mipmap)
        initialize_context()
        ccall((:cuMipmappedArrayGetSparseProperties, libcuda), CUresult, (Ptr{CUDA_ARRAY_SPARSE_PROPERTIES}, CUmipmappedArray), sparseProperties, mipmap)
    end

@checked function cuArrayGetMemoryRequirements(memoryRequirements, array, device)
        initialize_context()
        ccall((:cuArrayGetMemoryRequirements, libcuda), CUresult, (Ptr{CUDA_ARRAY_MEMORY_REQUIREMENTS}, CUarray, CUdevice), memoryRequirements, array, device)
    end

@checked function cuMipmappedArrayGetMemoryRequirements(memoryRequirements, mipmap, device)
        initialize_context()
        ccall((:cuMipmappedArrayGetMemoryRequirements, libcuda), CUresult, (Ptr{CUDA_ARRAY_MEMORY_REQUIREMENTS}, CUmipmappedArray, CUdevice), memoryRequirements, mipmap, device)
    end

@checked function cuArrayGetPlane(pPlaneArray, hArray, planeIdx)
        initialize_context()
        ccall((:cuArrayGetPlane, libcuda), CUresult, (Ptr{CUarray}, CUarray, Cuint), pPlaneArray, hArray, planeIdx)
    end

@checked function cuArrayDestroy(hArray)
        initialize_context()
        ccall((:cuArrayDestroy, libcuda), CUresult, (CUarray,), hArray)
    end

@checked function cuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels)
        initialize_context()
        ccall((:cuMipmappedArrayCreate, libcuda), CUresult, (Ptr{CUmipmappedArray}, Ptr{CUDA_ARRAY3D_DESCRIPTOR}, Cuint), pHandle, pMipmappedArrayDesc, numMipmapLevels)
    end

@checked function cuMipmappedArrayGetLevel(pLevelArray, hMipmappedArray, level)
        initialize_context()
        ccall((:cuMipmappedArrayGetLevel, libcuda), CUresult, (Ptr{CUarray}, CUmipmappedArray, Cuint), pLevelArray, hMipmappedArray, level)
    end

@checked function cuMipmappedArrayDestroy(hMipmappedArray)
        initialize_context()
        ccall((:cuMipmappedArrayDestroy, libcuda), CUresult, (CUmipmappedArray,), hMipmappedArray)
    end

@checked function cuMemGetHandleForAddressRange(handle, dptr, size, handleType, flags)
        initialize_context()
        ccall((:cuMemGetHandleForAddressRange, libcuda), CUresult, (Ptr{Cvoid}, CUdeviceptr, Csize_t, CUmemRangeHandleType, Culonglong), handle, dptr, size, handleType, flags)
    end

@checked function cuMemAddressReserve(ptr, size, alignment, addr, flags)
        initialize_context()
        ccall((:cuMemAddressReserve, libcuda), CUresult, (Ptr{CUdeviceptr}, Csize_t, Csize_t, CUdeviceptr, Culonglong), ptr, size, alignment, addr, flags)
    end

@checked function cuMemAddressFree(ptr, size)
        initialize_context()
        ccall((:cuMemAddressFree, libcuda), CUresult, (CUdeviceptr, Csize_t), ptr, size)
    end

@checked function cuMemCreate(handle, size, prop, flags)
        initialize_context()
        ccall((:cuMemCreate, libcuda), CUresult, (Ptr{CUmemGenericAllocationHandle}, Csize_t, Ptr{CUmemAllocationProp}, Culonglong), handle, size, prop, flags)
    end

@checked function cuMemRelease(handle)
        initialize_context()
        ccall((:cuMemRelease, libcuda), CUresult, (CUmemGenericAllocationHandle,), handle)
    end

@checked function cuMemMap(ptr, size, offset, handle, flags)
        initialize_context()
        ccall((:cuMemMap, libcuda), CUresult, (CUdeviceptr, Csize_t, Csize_t, CUmemGenericAllocationHandle, Culonglong), ptr, size, offset, handle, flags)
    end

@checked function cuMemMapArrayAsync(mapInfoList, count, hStream)
        initialize_context()
        ccall((:cuMemMapArrayAsync, libcuda), CUresult, (Ptr{CUarrayMapInfo}, Cuint, CUstream), mapInfoList, count, hStream)
    end

@checked function cuMemUnmap(ptr, size)
        initialize_context()
        ccall((:cuMemUnmap, libcuda), CUresult, (CUdeviceptr, Csize_t), ptr, size)
    end

@checked function cuMemSetAccess(ptr, size, desc, count)
        initialize_context()
        ccall((:cuMemSetAccess, libcuda), CUresult, (CUdeviceptr, Csize_t, Ptr{CUmemAccessDesc}, Csize_t), ptr, size, desc, count)
    end

@checked function cuMemGetAccess(flags, location, ptr)
        initialize_context()
        ccall((:cuMemGetAccess, libcuda), CUresult, (Ptr{Culonglong}, Ptr{CUmemLocation}, CUdeviceptr), flags, location, ptr)
    end

@checked function cuMemExportToShareableHandle(shareableHandle, handle, handleType, flags)
        initialize_context()
        ccall((:cuMemExportToShareableHandle, libcuda), CUresult, (Ptr{Cvoid}, CUmemGenericAllocationHandle, CUmemAllocationHandleType, Culonglong), shareableHandle, handle, handleType, flags)
    end

@checked function cuMemImportFromShareableHandle(handle, osHandle, shHandleType)
        initialize_context()
        ccall((:cuMemImportFromShareableHandle, libcuda), CUresult, (Ptr{CUmemGenericAllocationHandle}, Ptr{Cvoid}, CUmemAllocationHandleType), handle, osHandle, shHandleType)
    end

@checked function cuMemGetAllocationGranularity(granularity, prop, option)
        initialize_context()
        ccall((:cuMemGetAllocationGranularity, libcuda), CUresult, (Ptr{Csize_t}, Ptr{CUmemAllocationProp}, CUmemAllocationGranularity_flags), granularity, prop, option)
    end

@checked function cuMemGetAllocationPropertiesFromHandle(prop, handle)
        initialize_context()
        ccall((:cuMemGetAllocationPropertiesFromHandle, libcuda), CUresult, (Ptr{CUmemAllocationProp}, CUmemGenericAllocationHandle), prop, handle)
    end

@checked function cuMemRetainAllocationHandle(handle, addr)
        initialize_context()
        ccall((:cuMemRetainAllocationHandle, libcuda), CUresult, (Ptr{CUmemGenericAllocationHandle}, Ptr{Cvoid}), handle, addr)
    end

@checked function cuMemFreeAsync(dptr, hStream)
        initialize_context()
        ccall((:cuMemFreeAsync, libcuda), CUresult, (CUdeviceptr, CUstream), dptr, hStream)
    end

@checked function cuMemAllocAsync(dptr, bytesize, hStream)
        initialize_context()
        ccall((:cuMemAllocAsync, libcuda), CUresult, (Ptr{CUdeviceptr}, Csize_t, CUstream), dptr, bytesize, hStream)
    end

@checked function cuMemPoolTrimTo(pool, minBytesToKeep)
        initialize_context()
        ccall((:cuMemPoolTrimTo, libcuda), CUresult, (CUmemoryPool, Csize_t), pool, minBytesToKeep)
    end

@checked function cuMemPoolSetAttribute(pool, attr, value)
        initialize_context()
        ccall((:cuMemPoolSetAttribute, libcuda), CUresult, (CUmemoryPool, CUmemPool_attribute, Ptr{Cvoid}), pool, attr, value)
    end

@checked function cuMemPoolGetAttribute(pool, attr, value)
        initialize_context()
        ccall((:cuMemPoolGetAttribute, libcuda), CUresult, (CUmemoryPool, CUmemPool_attribute, Ptr{Cvoid}), pool, attr, value)
    end

@checked function cuMemPoolSetAccess(pool, map, count)
        initialize_context()
        ccall((:cuMemPoolSetAccess, libcuda), CUresult, (CUmemoryPool, Ptr{CUmemAccessDesc}, Csize_t), pool, map, count)
    end

@checked function cuMemPoolGetAccess(flags, memPool, location)
        initialize_context()
        ccall((:cuMemPoolGetAccess, libcuda), CUresult, (Ptr{CUmemAccess_flags}, CUmemoryPool, Ptr{CUmemLocation}), flags, memPool, location)
    end

@checked function cuMemPoolCreate(pool, poolProps)
        initialize_context()
        ccall((:cuMemPoolCreate, libcuda), CUresult, (Ptr{CUmemoryPool}, Ptr{CUmemPoolProps}), pool, poolProps)
    end

@checked function cuMemPoolDestroy(pool)
        initialize_context()
        ccall((:cuMemPoolDestroy, libcuda), CUresult, (CUmemoryPool,), pool)
    end

@checked function cuMemAllocFromPoolAsync(dptr, bytesize, pool, hStream)
        initialize_context()
        ccall((:cuMemAllocFromPoolAsync, libcuda), CUresult, (Ptr{CUdeviceptr}, Csize_t, CUmemoryPool, CUstream), dptr, bytesize, pool, hStream)
    end

@checked function cuMemPoolExportToShareableHandle(handle_out, pool, handleType, flags)
        initialize_context()
        ccall((:cuMemPoolExportToShareableHandle, libcuda), CUresult, (Ptr{Cvoid}, CUmemoryPool, CUmemAllocationHandleType, Culonglong), handle_out, pool, handleType, flags)
    end

@checked function cuMemPoolImportFromShareableHandle(pool_out, handle, handleType, flags)
        initialize_context()
        ccall((:cuMemPoolImportFromShareableHandle, libcuda), CUresult, (Ptr{CUmemoryPool}, Ptr{Cvoid}, CUmemAllocationHandleType, Culonglong), pool_out, handle, handleType, flags)
    end

@checked function cuMemPoolExportPointer(shareData_out, ptr)
        initialize_context()
        ccall((:cuMemPoolExportPointer, libcuda), CUresult, (Ptr{CUmemPoolPtrExportData}, CUdeviceptr), shareData_out, ptr)
    end

@checked function cuMemPoolImportPointer(ptr_out, pool, shareData)
        initialize_context()
        ccall((:cuMemPoolImportPointer, libcuda), CUresult, (Ptr{CUdeviceptr}, CUmemoryPool, Ptr{CUmemPoolPtrExportData}), ptr_out, pool, shareData)
    end

@checked function cuPointerGetAttribute(data, attribute, ptr)
        initialize_context()
        ccall((:cuPointerGetAttribute, libcuda), CUresult, (Ptr{Cvoid}, CUpointer_attribute, CUdeviceptr), data, attribute, ptr)
    end

@checked function cuMemPrefetchAsync(devPtr, count, dstDevice, hStream)
        initialize_context()
        ccall((:cuMemPrefetchAsync, libcuda), CUresult, (CUdeviceptr, Csize_t, CUdevice, CUstream), devPtr, count, dstDevice, hStream)
    end

@checked function cuMemAdvise(devPtr, count, advice, device)
        initialize_context()
        ccall((:cuMemAdvise, libcuda), CUresult, (CUdeviceptr, Csize_t, CUmem_advise, CUdevice), devPtr, count, advice, device)
    end

@checked function cuMemRangeGetAttribute(data, dataSize, attribute, devPtr, count)
        initialize_context()
        ccall((:cuMemRangeGetAttribute, libcuda), CUresult, (Ptr{Cvoid}, Csize_t, CUmem_range_attribute, CUdeviceptr, Csize_t), data, dataSize, attribute, devPtr, count)
    end

@checked function cuMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count)
        initialize_context()
        ccall((:cuMemRangeGetAttributes, libcuda), CUresult, (Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{CUmem_range_attribute}, Csize_t, CUdeviceptr, Csize_t), data, dataSizes, attributes, numAttributes, devPtr, count)
    end

@checked function cuPointerSetAttribute(value, attribute, ptr)
        initialize_context()
        ccall((:cuPointerSetAttribute, libcuda), CUresult, (Ptr{Cvoid}, CUpointer_attribute, CUdeviceptr), value, attribute, ptr)
    end

@checked function cuPointerGetAttributes(numAttributes, attributes, data, ptr)
        initialize_context()
        ccall((:cuPointerGetAttributes, libcuda), CUresult, (Cuint, Ptr{CUpointer_attribute}, Ptr{Ptr{Cvoid}}, CUdeviceptr), numAttributes, attributes, data, ptr)
    end

@checked function cuStreamCreate(phStream, Flags)
        initialize_context()
        ccall((:cuStreamCreate, libcuda), CUresult, (Ptr{CUstream}, Cuint), phStream, Flags)
    end

@checked function cuStreamCreateWithPriority(phStream, flags, priority)
        initialize_context()
        ccall((:cuStreamCreateWithPriority, libcuda), CUresult, (Ptr{CUstream}, Cuint, Cint), phStream, flags, priority)
    end

@checked function cuStreamGetPriority(hStream, priority)
        initialize_context()
        ccall((:cuStreamGetPriority, libcuda), CUresult, (CUstream, Ptr{Cint}), hStream, priority)
    end

@checked function cuStreamGetFlags(hStream, flags)
        initialize_context()
        ccall((:cuStreamGetFlags, libcuda), CUresult, (CUstream, Ptr{Cuint}), hStream, flags)
    end

@checked function cuStreamGetCtx(hStream, pctx)
        initialize_context()
        ccall((:cuStreamGetCtx, libcuda), CUresult, (CUstream, Ptr{CUcontext}), hStream, pctx)
    end

@checked function cuStreamWaitEvent(hStream, hEvent, Flags)
        initialize_context()
        ccall((:cuStreamWaitEvent, libcuda), CUresult, (CUstream, CUevent, Cuint), hStream, hEvent, Flags)
    end

@checked function cuStreamAddCallback(hStream, callback, userData, flags)
        initialize_context()
        ccall((:cuStreamAddCallback, libcuda), CUresult, (CUstream, CUstreamCallback, Ptr{Cvoid}, Cuint), hStream, callback, userData, flags)
    end

@checked function cuThreadExchangeStreamCaptureMode(mode)
        initialize_context()
        ccall((:cuThreadExchangeStreamCaptureMode, libcuda), CUresult, (Ptr{CUstreamCaptureMode},), mode)
    end

@checked function cuStreamEndCapture(hStream, phGraph)
        initialize_context()
        ccall((:cuStreamEndCapture, libcuda), CUresult, (CUstream, Ptr{CUgraph}), hStream, phGraph)
    end

@checked function cuStreamIsCapturing(hStream, captureStatus)
        initialize_context()
        ccall((:cuStreamIsCapturing, libcuda), CUresult, (CUstream, Ptr{CUstreamCaptureStatus}), hStream, captureStatus)
    end

@checked function cuStreamGetCaptureInfo(hStream, captureStatus_out, id_out)
        initialize_context()
        ccall((:cuStreamGetCaptureInfo, libcuda), CUresult, (CUstream, Ptr{CUstreamCaptureStatus}, Ptr{cuuint64_t}), hStream, captureStatus_out, id_out)
    end

@checked function cuStreamGetCaptureInfo_v2(hStream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out)
        initialize_context()
        ccall((:cuStreamGetCaptureInfo_v2, libcuda), CUresult, (CUstream, Ptr{CUstreamCaptureStatus}, Ptr{cuuint64_t}, Ptr{CUgraph}, Ptr{Ptr{CUgraphNode}}, Ptr{Csize_t}), hStream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out)
    end

@checked function cuStreamUpdateCaptureDependencies(hStream, dependencies, numDependencies, flags)
        initialize_context()
        ccall((:cuStreamUpdateCaptureDependencies, libcuda), CUresult, (CUstream, Ptr{CUgraphNode}, Csize_t, Cuint), hStream, dependencies, numDependencies, flags)
    end

@checked function cuStreamAttachMemAsync(hStream, dptr, length, flags)
        initialize_context()
        ccall((:cuStreamAttachMemAsync, libcuda), CUresult, (CUstream, CUdeviceptr, Csize_t, Cuint), hStream, dptr, length, flags)
    end

@checked function cuStreamQuery(hStream)
        initialize_context()
        ccall((:cuStreamQuery, libcuda), CUresult, (CUstream,), hStream)
    end

@checked function cuStreamSynchronize(hStream)
        initialize_context()
        ccall((:cuStreamSynchronize, libcuda), CUresult, (CUstream,), hStream)
    end

@checked function cuStreamCopyAttributes(dst, src)
        initialize_context()
        ccall((:cuStreamCopyAttributes, libcuda), CUresult, (CUstream, CUstream), dst, src)
    end

@checked function cuStreamGetAttribute(hStream, attr, value_out)
        initialize_context()
        ccall((:cuStreamGetAttribute, libcuda), CUresult, (CUstream, CUstreamAttrID, Ptr{CUstreamAttrValue}), hStream, attr, value_out)
    end

@checked function cuStreamSetAttribute(hStream, attr, value)
        initialize_context()
        ccall((:cuStreamSetAttribute, libcuda), CUresult, (CUstream, CUstreamAttrID, Ptr{CUstreamAttrValue}), hStream, attr, value)
    end

@checked function cuEventCreate(phEvent, Flags)
        initialize_context()
        ccall((:cuEventCreate, libcuda), CUresult, (Ptr{CUevent}, Cuint), phEvent, Flags)
    end

@checked function cuEventRecord(hEvent, hStream)
        initialize_context()
        ccall((:cuEventRecord, libcuda), CUresult, (CUevent, CUstream), hEvent, hStream)
    end

@checked function cuEventRecordWithFlags(hEvent, hStream, flags)
        initialize_context()
        ccall((:cuEventRecordWithFlags, libcuda), CUresult, (CUevent, CUstream, Cuint), hEvent, hStream, flags)
    end

@checked function cuEventQuery(hEvent)
        initialize_context()
        ccall((:cuEventQuery, libcuda), CUresult, (CUevent,), hEvent)
    end

@checked function cuEventSynchronize(hEvent)
        initialize_context()
        ccall((:cuEventSynchronize, libcuda), CUresult, (CUevent,), hEvent)
    end

@checked function cuEventElapsedTime(pMilliseconds, hStart, hEnd)
        initialize_context()
        ccall((:cuEventElapsedTime, libcuda), CUresult, (Ptr{Cfloat}, CUevent, CUevent), pMilliseconds, hStart, hEnd)
    end

@checked function cuImportExternalMemory(extMem_out, memHandleDesc)
        initialize_context()
        ccall((:cuImportExternalMemory, libcuda), CUresult, (Ptr{CUexternalMemory}, Ptr{CUDA_EXTERNAL_MEMORY_HANDLE_DESC}), extMem_out, memHandleDesc)
    end

@checked function cuExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc)
        initialize_context()
        ccall((:cuExternalMemoryGetMappedBuffer, libcuda), CUresult, (Ptr{CUdeviceptr}, CUexternalMemory, Ptr{CUDA_EXTERNAL_MEMORY_BUFFER_DESC}), devPtr, extMem, bufferDesc)
    end

@checked function cuExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc)
        initialize_context()
        ccall((:cuExternalMemoryGetMappedMipmappedArray, libcuda), CUresult, (Ptr{CUmipmappedArray}, CUexternalMemory, Ptr{CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC}), mipmap, extMem, mipmapDesc)
    end

@checked function cuDestroyExternalMemory(extMem)
        initialize_context()
        ccall((:cuDestroyExternalMemory, libcuda), CUresult, (CUexternalMemory,), extMem)
    end

@checked function cuImportExternalSemaphore(extSem_out, semHandleDesc)
        initialize_context()
        ccall((:cuImportExternalSemaphore, libcuda), CUresult, (Ptr{CUexternalSemaphore}, Ptr{CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC}), extSem_out, semHandleDesc)
    end

@checked function cuSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)
        initialize_context()
        ccall((:cuSignalExternalSemaphoresAsync, libcuda), CUresult, (Ptr{CUexternalSemaphore}, Ptr{CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS}, Cuint, CUstream), extSemArray, paramsArray, numExtSems, stream)
    end

@checked function cuWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)
        initialize_context()
        ccall((:cuWaitExternalSemaphoresAsync, libcuda), CUresult, (Ptr{CUexternalSemaphore}, Ptr{CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS}, Cuint, CUstream), extSemArray, paramsArray, numExtSems, stream)
    end

@checked function cuDestroyExternalSemaphore(extSem)
        initialize_context()
        ccall((:cuDestroyExternalSemaphore, libcuda), CUresult, (CUexternalSemaphore,), extSem)
    end

@checked function cuStreamWaitValue32(stream, addr, value, flags)
        initialize_context()
        ccall((:cuStreamWaitValue32, libcuda), CUresult, (CUstream, CUdeviceptr, cuuint32_t, Cuint), stream, addr, value, flags)
    end

@checked function cuStreamWaitValue64(stream, addr, value, flags)
        initialize_context()
        ccall((:cuStreamWaitValue64, libcuda), CUresult, (CUstream, CUdeviceptr, cuuint64_t, Cuint), stream, addr, value, flags)
    end

@checked function cuStreamWriteValue32(stream, addr, value, flags)
        initialize_context()
        ccall((:cuStreamWriteValue32, libcuda), CUresult, (CUstream, CUdeviceptr, cuuint32_t, Cuint), stream, addr, value, flags)
    end

@checked function cuStreamWriteValue64(stream, addr, value, flags)
        initialize_context()
        ccall((:cuStreamWriteValue64, libcuda), CUresult, (CUstream, CUdeviceptr, cuuint64_t, Cuint), stream, addr, value, flags)
    end

@checked function cuStreamBatchMemOp(stream, count, paramArray, flags)
        initialize_context()
        ccall((:cuStreamBatchMemOp, libcuda), CUresult, (CUstream, Cuint, Ptr{CUstreamBatchMemOpParams}, Cuint), stream, count, paramArray, flags)
    end

@checked function cuStreamWaitValue32_v2(stream, addr, value, flags)
        initialize_context()
        ccall((:cuStreamWaitValue32_v2, libcuda), CUresult, (CUstream, CUdeviceptr, cuuint32_t, Cuint), stream, addr, value, flags)
    end

@checked function cuStreamWaitValue64_v2(stream, addr, value, flags)
        initialize_context()
        ccall((:cuStreamWaitValue64_v2, libcuda), CUresult, (CUstream, CUdeviceptr, cuuint64_t, Cuint), stream, addr, value, flags)
    end

@checked function cuStreamWriteValue32_v2(stream, addr, value, flags)
        initialize_context()
        ccall((:cuStreamWriteValue32_v2, libcuda), CUresult, (CUstream, CUdeviceptr, cuuint32_t, Cuint), stream, addr, value, flags)
    end

@checked function cuStreamWriteValue64_v2(stream, addr, value, flags)
        initialize_context()
        ccall((:cuStreamWriteValue64_v2, libcuda), CUresult, (CUstream, CUdeviceptr, cuuint64_t, Cuint), stream, addr, value, flags)
    end

@checked function cuStreamBatchMemOp_v2(stream, count, paramArray, flags)
        initialize_context()
        ccall((:cuStreamBatchMemOp_v2, libcuda), CUresult, (CUstream, Cuint, Ptr{CUstreamBatchMemOpParams}, Cuint), stream, count, paramArray, flags)
    end

@checked function cuFuncGetAttribute(pi, attrib, hfunc)
        initialize_context()
        ccall((:cuFuncGetAttribute, libcuda), CUresult, (Ptr{Cint}, CUfunction_attribute, CUfunction), pi, attrib, hfunc)
    end

@checked function cuFuncSetAttribute(hfunc, attrib, value)
        initialize_context()
        ccall((:cuFuncSetAttribute, libcuda), CUresult, (CUfunction, CUfunction_attribute, Cint), hfunc, attrib, value)
    end

@checked function cuFuncSetCacheConfig(hfunc, config)
        initialize_context()
        ccall((:cuFuncSetCacheConfig, libcuda), CUresult, (CUfunction, CUfunc_cache), hfunc, config)
    end

@checked function cuFuncSetSharedMemConfig(hfunc, config)
        initialize_context()
        ccall((:cuFuncSetSharedMemConfig, libcuda), CUresult, (CUfunction, CUsharedconfig), hfunc, config)
    end

@checked function cuFuncGetModule(hmod, hfunc)
        initialize_context()
        ccall((:cuFuncGetModule, libcuda), CUresult, (Ptr{CUmodule}, CUfunction), hmod, hfunc)
    end

@checked function cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra)
        initialize_context()
        ccall((:cuLaunchKernel, libcuda), CUresult, (CUfunction, Cuint, Cuint, Cuint, Cuint, Cuint, Cuint, Cuint, CUstream, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}), f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra)
    end

@checked function cuLaunchKernelEx(config, f, kernelParams, extra)
        initialize_context()
        ccall((:cuLaunchKernelEx, libcuda), CUresult, (Ptr{CUlaunchConfig}, CUfunction, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}), config, f, kernelParams, extra)
    end

@checked function cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams)
        initialize_context()
        ccall((:cuLaunchCooperativeKernel, libcuda), CUresult, (CUfunction, Cuint, Cuint, Cuint, Cuint, Cuint, Cuint, Cuint, CUstream, Ptr{Ptr{Cvoid}}), f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams)
    end

@checked function cuLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags)
        initialize_context()
        ccall((:cuLaunchCooperativeKernelMultiDevice, libcuda), CUresult, (Ptr{CUDA_LAUNCH_PARAMS}, Cuint, Cuint), launchParamsList, numDevices, flags)
    end

@checked function cuLaunchHostFunc(hStream, fn, userData)
        initialize_context()
        ccall((:cuLaunchHostFunc, libcuda), CUresult, (CUstream, CUhostFn, Ptr{Cvoid}), hStream, fn, userData)
    end

@checked function cuFuncSetBlockShape(hfunc, x, y, z)
        initialize_context()
        ccall((:cuFuncSetBlockShape, libcuda), CUresult, (CUfunction, Cint, Cint, Cint), hfunc, x, y, z)
    end

@checked function cuFuncSetSharedSize(hfunc, bytes)
        initialize_context()
        ccall((:cuFuncSetSharedSize, libcuda), CUresult, (CUfunction, Cuint), hfunc, bytes)
    end

@checked function cuParamSetSize(hfunc, numbytes)
        initialize_context()
        ccall((:cuParamSetSize, libcuda), CUresult, (CUfunction, Cuint), hfunc, numbytes)
    end

@checked function cuParamSeti(hfunc, offset, value)
        initialize_context()
        ccall((:cuParamSeti, libcuda), CUresult, (CUfunction, Cint, Cuint), hfunc, offset, value)
    end

@checked function cuParamSetf(hfunc, offset, value)
        initialize_context()
        ccall((:cuParamSetf, libcuda), CUresult, (CUfunction, Cint, Cfloat), hfunc, offset, value)
    end

@checked function cuParamSetv(hfunc, offset, ptr, numbytes)
        initialize_context()
        ccall((:cuParamSetv, libcuda), CUresult, (CUfunction, Cint, Ptr{Cvoid}, Cuint), hfunc, offset, ptr, numbytes)
    end

@checked function cuLaunch(f)
        initialize_context()
        ccall((:cuLaunch, libcuda), CUresult, (CUfunction,), f)
    end

@checked function cuLaunchGrid(f, grid_width, grid_height)
        initialize_context()
        ccall((:cuLaunchGrid, libcuda), CUresult, (CUfunction, Cint, Cint), f, grid_width, grid_height)
    end

@checked function cuLaunchGridAsync(f, grid_width, grid_height, hStream)
        initialize_context()
        ccall((:cuLaunchGridAsync, libcuda), CUresult, (CUfunction, Cint, Cint, CUstream), f, grid_width, grid_height, hStream)
    end

@checked function cuParamSetTexRef(hfunc, texunit, hTexRef)
        initialize_context()
        ccall((:cuParamSetTexRef, libcuda), CUresult, (CUfunction, Cint, CUtexref), hfunc, texunit, hTexRef)
    end

@checked function cuGraphCreate(phGraph, flags)
        initialize_context()
        ccall((:cuGraphCreate, libcuda), CUresult, (Ptr{CUgraph}, Cuint), phGraph, flags)
    end

@checked function cuGraphAddKernelNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
        initialize_context()
        ccall((:cuGraphAddKernelNode, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t, Ptr{CUDA_KERNEL_NODE_PARAMS}), phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    end

@checked function cuGraphKernelNodeGetParams(hNode, nodeParams)
        initialize_context()
        ccall((:cuGraphKernelNodeGetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_KERNEL_NODE_PARAMS}), hNode, nodeParams)
    end

@checked function cuGraphKernelNodeSetParams(hNode, nodeParams)
        initialize_context()
        ccall((:cuGraphKernelNodeSetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_KERNEL_NODE_PARAMS}), hNode, nodeParams)
    end

@checked function cuGraphAddMemcpyNode(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx)
        initialize_context()
        ccall((:cuGraphAddMemcpyNode, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t, Ptr{CUDA_MEMCPY3D}, CUcontext), phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx)
    end

@checked function cuGraphMemcpyNodeGetParams(hNode, nodeParams)
        initialize_context()
        ccall((:cuGraphMemcpyNodeGetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_MEMCPY3D}), hNode, nodeParams)
    end

@checked function cuGraphMemcpyNodeSetParams(hNode, nodeParams)
        initialize_context()
        ccall((:cuGraphMemcpyNodeSetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_MEMCPY3D}), hNode, nodeParams)
    end

@checked function cuGraphAddMemsetNode(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx)
        initialize_context()
        ccall((:cuGraphAddMemsetNode, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t, Ptr{CUDA_MEMSET_NODE_PARAMS}, CUcontext), phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx)
    end

@checked function cuGraphMemsetNodeGetParams(hNode, nodeParams)
        initialize_context()
        ccall((:cuGraphMemsetNodeGetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_MEMSET_NODE_PARAMS}), hNode, nodeParams)
    end

@checked function cuGraphMemsetNodeSetParams(hNode, nodeParams)
        initialize_context()
        ccall((:cuGraphMemsetNodeSetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_MEMSET_NODE_PARAMS}), hNode, nodeParams)
    end

@checked function cuGraphAddHostNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
        initialize_context()
        ccall((:cuGraphAddHostNode, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t, Ptr{CUDA_HOST_NODE_PARAMS}), phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    end

@checked function cuGraphHostNodeGetParams(hNode, nodeParams)
        initialize_context()
        ccall((:cuGraphHostNodeGetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_HOST_NODE_PARAMS}), hNode, nodeParams)
    end

@checked function cuGraphHostNodeSetParams(hNode, nodeParams)
        initialize_context()
        ccall((:cuGraphHostNodeSetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_HOST_NODE_PARAMS}), hNode, nodeParams)
    end

@checked function cuGraphAddChildGraphNode(phGraphNode, hGraph, dependencies, numDependencies, childGraph)
        initialize_context()
        ccall((:cuGraphAddChildGraphNode, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t, CUgraph), phGraphNode, hGraph, dependencies, numDependencies, childGraph)
    end

@checked function cuGraphChildGraphNodeGetGraph(hNode, phGraph)
        initialize_context()
        ccall((:cuGraphChildGraphNodeGetGraph, libcuda), CUresult, (CUgraphNode, Ptr{CUgraph}), hNode, phGraph)
    end

@checked function cuGraphAddEmptyNode(phGraphNode, hGraph, dependencies, numDependencies)
        initialize_context()
        ccall((:cuGraphAddEmptyNode, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t), phGraphNode, hGraph, dependencies, numDependencies)
    end

@checked function cuGraphAddEventRecordNode(phGraphNode, hGraph, dependencies, numDependencies, event)
        initialize_context()
        ccall((:cuGraphAddEventRecordNode, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t, CUevent), phGraphNode, hGraph, dependencies, numDependencies, event)
    end

@checked function cuGraphEventRecordNodeGetEvent(hNode, event_out)
        initialize_context()
        ccall((:cuGraphEventRecordNodeGetEvent, libcuda), CUresult, (CUgraphNode, Ptr{CUevent}), hNode, event_out)
    end

@checked function cuGraphEventRecordNodeSetEvent(hNode, event)
        initialize_context()
        ccall((:cuGraphEventRecordNodeSetEvent, libcuda), CUresult, (CUgraphNode, CUevent), hNode, event)
    end

@checked function cuGraphAddEventWaitNode(phGraphNode, hGraph, dependencies, numDependencies, event)
        initialize_context()
        ccall((:cuGraphAddEventWaitNode, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t, CUevent), phGraphNode, hGraph, dependencies, numDependencies, event)
    end

@checked function cuGraphEventWaitNodeGetEvent(hNode, event_out)
        initialize_context()
        ccall((:cuGraphEventWaitNodeGetEvent, libcuda), CUresult, (CUgraphNode, Ptr{CUevent}), hNode, event_out)
    end

@checked function cuGraphEventWaitNodeSetEvent(hNode, event)
        initialize_context()
        ccall((:cuGraphEventWaitNodeSetEvent, libcuda), CUresult, (CUgraphNode, CUevent), hNode, event)
    end

@checked function cuGraphAddExternalSemaphoresSignalNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
        initialize_context()
        ccall((:cuGraphAddExternalSemaphoresSignalNode, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t, Ptr{CUDA_EXT_SEM_SIGNAL_NODE_PARAMS}), phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    end

@checked function cuGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out)
        initialize_context()
        ccall((:cuGraphExternalSemaphoresSignalNodeGetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_EXT_SEM_SIGNAL_NODE_PARAMS}), hNode, params_out)
    end

@checked function cuGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams)
        initialize_context()
        ccall((:cuGraphExternalSemaphoresSignalNodeSetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_EXT_SEM_SIGNAL_NODE_PARAMS}), hNode, nodeParams)
    end

@checked function cuGraphAddExternalSemaphoresWaitNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
        initialize_context()
        ccall((:cuGraphAddExternalSemaphoresWaitNode, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t, Ptr{CUDA_EXT_SEM_WAIT_NODE_PARAMS}), phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    end

@checked function cuGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out)
        initialize_context()
        ccall((:cuGraphExternalSemaphoresWaitNodeGetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_EXT_SEM_WAIT_NODE_PARAMS}), hNode, params_out)
    end

@checked function cuGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams)
        initialize_context()
        ccall((:cuGraphExternalSemaphoresWaitNodeSetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_EXT_SEM_WAIT_NODE_PARAMS}), hNode, nodeParams)
    end

@checked function cuGraphAddBatchMemOpNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
        initialize_context()
        ccall((:cuGraphAddBatchMemOpNode, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t, Ptr{CUDA_BATCH_MEM_OP_NODE_PARAMS}), phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    end

@checked function cuGraphBatchMemOpNodeGetParams(hNode, nodeParams_out)
        initialize_context()
        ccall((:cuGraphBatchMemOpNodeGetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_BATCH_MEM_OP_NODE_PARAMS}), hNode, nodeParams_out)
    end

@checked function cuGraphBatchMemOpNodeSetParams(hNode, nodeParams)
        initialize_context()
        ccall((:cuGraphBatchMemOpNodeSetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_BATCH_MEM_OP_NODE_PARAMS}), hNode, nodeParams)
    end

@checked function cuGraphExecBatchMemOpNodeSetParams(hGraphExec, hNode, nodeParams)
        initialize_context()
        ccall((:cuGraphExecBatchMemOpNodeSetParams, libcuda), CUresult, (CUgraphExec, CUgraphNode, Ptr{CUDA_BATCH_MEM_OP_NODE_PARAMS}), hGraphExec, hNode, nodeParams)
    end

@checked function cuGraphAddMemAllocNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
        initialize_context()
        ccall((:cuGraphAddMemAllocNode, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t, Ptr{CUDA_MEM_ALLOC_NODE_PARAMS}), phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    end

@checked function cuGraphMemAllocNodeGetParams(hNode, params_out)
        initialize_context()
        ccall((:cuGraphMemAllocNodeGetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_MEM_ALLOC_NODE_PARAMS}), hNode, params_out)
    end

@checked function cuGraphAddMemFreeNode(phGraphNode, hGraph, dependencies, numDependencies, dptr)
        initialize_context()
        ccall((:cuGraphAddMemFreeNode, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t, CUdeviceptr), phGraphNode, hGraph, dependencies, numDependencies, dptr)
    end

@checked function cuGraphMemFreeNodeGetParams(hNode, dptr_out)
        initialize_context()
        ccall((:cuGraphMemFreeNodeGetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUdeviceptr}), hNode, dptr_out)
    end

@checked function cuDeviceGraphMemTrim(device)
        initialize_context()
        ccall((:cuDeviceGraphMemTrim, libcuda), CUresult, (CUdevice,), device)
    end

@checked function cuDeviceGetGraphMemAttribute(device, attr, value)
        initialize_context()
        ccall((:cuDeviceGetGraphMemAttribute, libcuda), CUresult, (CUdevice, CUgraphMem_attribute, Ptr{Cvoid}), device, attr, value)
    end

@checked function cuDeviceSetGraphMemAttribute(device, attr, value)
        initialize_context()
        ccall((:cuDeviceSetGraphMemAttribute, libcuda), CUresult, (CUdevice, CUgraphMem_attribute, Ptr{Cvoid}), device, attr, value)
    end

@checked function cuGraphClone(phGraphClone, originalGraph)
        initialize_context()
        ccall((:cuGraphClone, libcuda), CUresult, (Ptr{CUgraph}, CUgraph), phGraphClone, originalGraph)
    end

@checked function cuGraphNodeFindInClone(phNode, hOriginalNode, hClonedGraph)
        initialize_context()
        ccall((:cuGraphNodeFindInClone, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraphNode, CUgraph), phNode, hOriginalNode, hClonedGraph)
    end

@checked function cuGraphNodeGetType(hNode, type)
        initialize_context()
        ccall((:cuGraphNodeGetType, libcuda), CUresult, (CUgraphNode, Ptr{CUgraphNodeType}), hNode, type)
    end

@checked function cuGraphGetNodes(hGraph, nodes, numNodes)
        initialize_context()
        ccall((:cuGraphGetNodes, libcuda), CUresult, (CUgraph, Ptr{CUgraphNode}, Ptr{Csize_t}), hGraph, nodes, numNodes)
    end

@checked function cuGraphGetRootNodes(hGraph, rootNodes, numRootNodes)
        initialize_context()
        ccall((:cuGraphGetRootNodes, libcuda), CUresult, (CUgraph, Ptr{CUgraphNode}, Ptr{Csize_t}), hGraph, rootNodes, numRootNodes)
    end

@checked function cuGraphGetEdges(hGraph, from, to, numEdges)
        initialize_context()
        ccall((:cuGraphGetEdges, libcuda), CUresult, (CUgraph, Ptr{CUgraphNode}, Ptr{CUgraphNode}, Ptr{Csize_t}), hGraph, from, to, numEdges)
    end

@checked function cuGraphNodeGetDependencies(hNode, dependencies, numDependencies)
        initialize_context()
        ccall((:cuGraphNodeGetDependencies, libcuda), CUresult, (CUgraphNode, Ptr{CUgraphNode}, Ptr{Csize_t}), hNode, dependencies, numDependencies)
    end

@checked function cuGraphNodeGetDependentNodes(hNode, dependentNodes, numDependentNodes)
        initialize_context()
        ccall((:cuGraphNodeGetDependentNodes, libcuda), CUresult, (CUgraphNode, Ptr{CUgraphNode}, Ptr{Csize_t}), hNode, dependentNodes, numDependentNodes)
    end

@checked function cuGraphAddDependencies(hGraph, from, to, numDependencies)
        initialize_context()
        ccall((:cuGraphAddDependencies, libcuda), CUresult, (CUgraph, Ptr{CUgraphNode}, Ptr{CUgraphNode}, Csize_t), hGraph, from, to, numDependencies)
    end

@checked function cuGraphRemoveDependencies(hGraph, from, to, numDependencies)
        initialize_context()
        ccall((:cuGraphRemoveDependencies, libcuda), CUresult, (CUgraph, Ptr{CUgraphNode}, Ptr{CUgraphNode}, Csize_t), hGraph, from, to, numDependencies)
    end

@checked function cuGraphDestroyNode(hNode)
        initialize_context()
        ccall((:cuGraphDestroyNode, libcuda), CUresult, (CUgraphNode,), hNode)
    end

@checked function cuGraphInstantiateWithFlags(phGraphExec, hGraph, flags)
        initialize_context()
        ccall((:cuGraphInstantiateWithFlags, libcuda), CUresult, (Ptr{CUgraphExec}, CUgraph, Culonglong), phGraphExec, hGraph, flags)
    end

@checked function cuGraphExecKernelNodeSetParams(hGraphExec, hNode, nodeParams)
        initialize_context()
        ccall((:cuGraphExecKernelNodeSetParams, libcuda), CUresult, (CUgraphExec, CUgraphNode, Ptr{CUDA_KERNEL_NODE_PARAMS}), hGraphExec, hNode, nodeParams)
    end

@checked function cuGraphExecMemcpyNodeSetParams(hGraphExec, hNode, copyParams, ctx)
        initialize_context()
        ccall((:cuGraphExecMemcpyNodeSetParams, libcuda), CUresult, (CUgraphExec, CUgraphNode, Ptr{CUDA_MEMCPY3D}, CUcontext), hGraphExec, hNode, copyParams, ctx)
    end

@checked function cuGraphExecMemsetNodeSetParams(hGraphExec, hNode, memsetParams, ctx)
        initialize_context()
        ccall((:cuGraphExecMemsetNodeSetParams, libcuda), CUresult, (CUgraphExec, CUgraphNode, Ptr{CUDA_MEMSET_NODE_PARAMS}, CUcontext), hGraphExec, hNode, memsetParams, ctx)
    end

@checked function cuGraphExecHostNodeSetParams(hGraphExec, hNode, nodeParams)
        initialize_context()
        ccall((:cuGraphExecHostNodeSetParams, libcuda), CUresult, (CUgraphExec, CUgraphNode, Ptr{CUDA_HOST_NODE_PARAMS}), hGraphExec, hNode, nodeParams)
    end

@checked function cuGraphExecChildGraphNodeSetParams(hGraphExec, hNode, childGraph)
        initialize_context()
        ccall((:cuGraphExecChildGraphNodeSetParams, libcuda), CUresult, (CUgraphExec, CUgraphNode, CUgraph), hGraphExec, hNode, childGraph)
    end

@checked function cuGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event)
        initialize_context()
        ccall((:cuGraphExecEventRecordNodeSetEvent, libcuda), CUresult, (CUgraphExec, CUgraphNode, CUevent), hGraphExec, hNode, event)
    end

@checked function cuGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event)
        initialize_context()
        ccall((:cuGraphExecEventWaitNodeSetEvent, libcuda), CUresult, (CUgraphExec, CUgraphNode, CUevent), hGraphExec, hNode, event)
    end

@checked function cuGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams)
        initialize_context()
        ccall((:cuGraphExecExternalSemaphoresSignalNodeSetParams, libcuda), CUresult, (CUgraphExec, CUgraphNode, Ptr{CUDA_EXT_SEM_SIGNAL_NODE_PARAMS}), hGraphExec, hNode, nodeParams)
    end

@checked function cuGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams)
        initialize_context()
        ccall((:cuGraphExecExternalSemaphoresWaitNodeSetParams, libcuda), CUresult, (CUgraphExec, CUgraphNode, Ptr{CUDA_EXT_SEM_WAIT_NODE_PARAMS}), hGraphExec, hNode, nodeParams)
    end

@checked function cuGraphNodeSetEnabled(hGraphExec, hNode, isEnabled)
        initialize_context()
        ccall((:cuGraphNodeSetEnabled, libcuda), CUresult, (CUgraphExec, CUgraphNode, Cuint), hGraphExec, hNode, isEnabled)
    end

@checked function cuGraphNodeGetEnabled(hGraphExec, hNode, isEnabled)
        initialize_context()
        ccall((:cuGraphNodeGetEnabled, libcuda), CUresult, (CUgraphExec, CUgraphNode, Ptr{Cuint}), hGraphExec, hNode, isEnabled)
    end

@checked function cuGraphUpload(hGraphExec, hStream)
        initialize_context()
        ccall((:cuGraphUpload, libcuda), CUresult, (CUgraphExec, CUstream), hGraphExec, hStream)
    end

@checked function cuGraphLaunch(hGraphExec, hStream)
        initialize_context()
        ccall((:cuGraphLaunch, libcuda), CUresult, (CUgraphExec, CUstream), hGraphExec, hStream)
    end

@checked function cuGraphExecDestroy(hGraphExec)
        initialize_context()
        ccall((:cuGraphExecDestroy, libcuda), CUresult, (CUgraphExec,), hGraphExec)
    end

@checked function cuGraphDestroy(hGraph)
        initialize_context()
        ccall((:cuGraphDestroy, libcuda), CUresult, (CUgraph,), hGraph)
    end

@checked function cuGraphExecUpdate(hGraphExec, hGraph, hErrorNode_out, updateResult_out)
        initialize_context()
        ccall((:cuGraphExecUpdate, libcuda), CUresult, (CUgraphExec, CUgraph, Ptr{CUgraphNode}, Ptr{CUgraphExecUpdateResult}), hGraphExec, hGraph, hErrorNode_out, updateResult_out)
    end

@checked function cuGraphKernelNodeCopyAttributes(dst, src)
        initialize_context()
        ccall((:cuGraphKernelNodeCopyAttributes, libcuda), CUresult, (CUgraphNode, CUgraphNode), dst, src)
    end

@checked function cuGraphKernelNodeGetAttribute(hNode, attr, value_out)
        initialize_context()
        ccall((:cuGraphKernelNodeGetAttribute, libcuda), CUresult, (CUgraphNode, CUkernelNodeAttrID, Ptr{CUkernelNodeAttrValue}), hNode, attr, value_out)
    end

@checked function cuGraphKernelNodeSetAttribute(hNode, attr, value)
        initialize_context()
        ccall((:cuGraphKernelNodeSetAttribute, libcuda), CUresult, (CUgraphNode, CUkernelNodeAttrID, Ptr{CUkernelNodeAttrValue}), hNode, attr, value)
    end

@checked function cuGraphDebugDotPrint(hGraph, path, flags)
        initialize_context()
        ccall((:cuGraphDebugDotPrint, libcuda), CUresult, (CUgraph, Cstring, Cuint), hGraph, path, flags)
    end

@checked function cuUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags)
        initialize_context()
        ccall((:cuUserObjectCreate, libcuda), CUresult, (Ptr{CUuserObject}, Ptr{Cvoid}, CUhostFn, Cuint, Cuint), object_out, ptr, destroy, initialRefcount, flags)
    end

@checked function cuUserObjectRetain(object, count)
        initialize_context()
        ccall((:cuUserObjectRetain, libcuda), CUresult, (CUuserObject, Cuint), object, count)
    end

@checked function cuUserObjectRelease(object, count)
        initialize_context()
        ccall((:cuUserObjectRelease, libcuda), CUresult, (CUuserObject, Cuint), object, count)
    end

@checked function cuGraphRetainUserObject(graph, object, count, flags)
        initialize_context()
        ccall((:cuGraphRetainUserObject, libcuda), CUresult, (CUgraph, CUuserObject, Cuint, Cuint), graph, object, count, flags)
    end

@checked function cuGraphReleaseUserObject(graph, object, count)
        initialize_context()
        ccall((:cuGraphReleaseUserObject, libcuda), CUresult, (CUgraph, CUuserObject, Cuint), graph, object, count)
    end

@checked function cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize)
        initialize_context()
        ccall((:cuOccupancyMaxActiveBlocksPerMultiprocessor, libcuda), CUresult, (Ptr{Cint}, CUfunction, Cint, Csize_t), numBlocks, func, blockSize, dynamicSMemSize)
    end

@checked function cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags)
        initialize_context()
        ccall((:cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, libcuda), CUresult, (Ptr{Cint}, CUfunction, Cint, Csize_t, Cuint), numBlocks, func, blockSize, dynamicSMemSize, flags)
    end

@checked function cuOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit)
        initialize_context()
        ccall((:cuOccupancyMaxPotentialBlockSize, libcuda), CUresult, (Ptr{Cint}, Ptr{Cint}, CUfunction, CUoccupancyB2DSize, Csize_t, Cint), minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit)
    end

@checked function cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags)
        initialize_context()
        ccall((:cuOccupancyMaxPotentialBlockSizeWithFlags, libcuda), CUresult, (Ptr{Cint}, Ptr{Cint}, CUfunction, CUoccupancyB2DSize, Csize_t, Cint, Cuint), minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags)
    end

@checked function cuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize)
        initialize_context()
        ccall((:cuOccupancyAvailableDynamicSMemPerBlock, libcuda), CUresult, (Ptr{Csize_t}, CUfunction, Cint, Cint), dynamicSmemSize, func, numBlocks, blockSize)
    end

@checked function cuOccupancyMaxPotentialClusterSize(clusterSize, func, config)
        initialize_context()
        ccall((:cuOccupancyMaxPotentialClusterSize, libcuda), CUresult, (Ptr{Cint}, CUfunction, Ptr{CUlaunchConfig}), clusterSize, func, config)
    end

@checked function cuOccupancyMaxActiveClusters(numClusters, func, config)
        initialize_context()
        ccall((:cuOccupancyMaxActiveClusters, libcuda), CUresult, (Ptr{Cint}, CUfunction, Ptr{CUlaunchConfig}), numClusters, func, config)
    end

@checked function cuTexRefSetArray(hTexRef, hArray, Flags)
        initialize_context()
        ccall((:cuTexRefSetArray, libcuda), CUresult, (CUtexref, CUarray, Cuint), hTexRef, hArray, Flags)
    end

@checked function cuTexRefSetMipmappedArray(hTexRef, hMipmappedArray, Flags)
        initialize_context()
        ccall((:cuTexRefSetMipmappedArray, libcuda), CUresult, (CUtexref, CUmipmappedArray, Cuint), hTexRef, hMipmappedArray, Flags)
    end

@checked function cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents)
        initialize_context()
        ccall((:cuTexRefSetFormat, libcuda), CUresult, (CUtexref, CUarray_format, Cint), hTexRef, fmt, NumPackedComponents)
    end

@checked function cuTexRefSetAddressMode(hTexRef, dim, am)
        initialize_context()
        ccall((:cuTexRefSetAddressMode, libcuda), CUresult, (CUtexref, Cint, CUaddress_mode), hTexRef, dim, am)
    end

@checked function cuTexRefSetFilterMode(hTexRef, fm)
        initialize_context()
        ccall((:cuTexRefSetFilterMode, libcuda), CUresult, (CUtexref, CUfilter_mode), hTexRef, fm)
    end

@checked function cuTexRefSetMipmapFilterMode(hTexRef, fm)
        initialize_context()
        ccall((:cuTexRefSetMipmapFilterMode, libcuda), CUresult, (CUtexref, CUfilter_mode), hTexRef, fm)
    end

@checked function cuTexRefSetMipmapLevelBias(hTexRef, bias)
        initialize_context()
        ccall((:cuTexRefSetMipmapLevelBias, libcuda), CUresult, (CUtexref, Cfloat), hTexRef, bias)
    end

@checked function cuTexRefSetMipmapLevelClamp(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp)
        initialize_context()
        ccall((:cuTexRefSetMipmapLevelClamp, libcuda), CUresult, (CUtexref, Cfloat, Cfloat), hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp)
    end

@checked function cuTexRefSetMaxAnisotropy(hTexRef, maxAniso)
        initialize_context()
        ccall((:cuTexRefSetMaxAnisotropy, libcuda), CUresult, (CUtexref, Cuint), hTexRef, maxAniso)
    end

@checked function cuTexRefSetBorderColor(hTexRef, pBorderColor)
        initialize_context()
        ccall((:cuTexRefSetBorderColor, libcuda), CUresult, (CUtexref, Ptr{Cfloat}), hTexRef, pBorderColor)
    end

@checked function cuTexRefSetFlags(hTexRef, Flags)
        initialize_context()
        ccall((:cuTexRefSetFlags, libcuda), CUresult, (CUtexref, Cuint), hTexRef, Flags)
    end

@checked function cuTexRefGetArray(phArray, hTexRef)
        initialize_context()
        ccall((:cuTexRefGetArray, libcuda), CUresult, (Ptr{CUarray}, CUtexref), phArray, hTexRef)
    end

@checked function cuTexRefGetMipmappedArray(phMipmappedArray, hTexRef)
        initialize_context()
        ccall((:cuTexRefGetMipmappedArray, libcuda), CUresult, (Ptr{CUmipmappedArray}, CUtexref), phMipmappedArray, hTexRef)
    end

@checked function cuTexRefGetAddressMode(pam, hTexRef, dim)
        initialize_context()
        ccall((:cuTexRefGetAddressMode, libcuda), CUresult, (Ptr{CUaddress_mode}, CUtexref, Cint), pam, hTexRef, dim)
    end

@checked function cuTexRefGetFilterMode(pfm, hTexRef)
        initialize_context()
        ccall((:cuTexRefGetFilterMode, libcuda), CUresult, (Ptr{CUfilter_mode}, CUtexref), pfm, hTexRef)
    end

@checked function cuTexRefGetFormat(pFormat, pNumChannels, hTexRef)
        initialize_context()
        ccall((:cuTexRefGetFormat, libcuda), CUresult, (Ptr{CUarray_format}, Ptr{Cint}, CUtexref), pFormat, pNumChannels, hTexRef)
    end

@checked function cuTexRefGetMipmapFilterMode(pfm, hTexRef)
        initialize_context()
        ccall((:cuTexRefGetMipmapFilterMode, libcuda), CUresult, (Ptr{CUfilter_mode}, CUtexref), pfm, hTexRef)
    end

@checked function cuTexRefGetMipmapLevelBias(pbias, hTexRef)
        initialize_context()
        ccall((:cuTexRefGetMipmapLevelBias, libcuda), CUresult, (Ptr{Cfloat}, CUtexref), pbias, hTexRef)
    end

@checked function cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef)
        initialize_context()
        ccall((:cuTexRefGetMipmapLevelClamp, libcuda), CUresult, (Ptr{Cfloat}, Ptr{Cfloat}, CUtexref), pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef)
    end

@checked function cuTexRefGetMaxAnisotropy(pmaxAniso, hTexRef)
        initialize_context()
        ccall((:cuTexRefGetMaxAnisotropy, libcuda), CUresult, (Ptr{Cint}, CUtexref), pmaxAniso, hTexRef)
    end

@checked function cuTexRefGetBorderColor(pBorderColor, hTexRef)
        initialize_context()
        ccall((:cuTexRefGetBorderColor, libcuda), CUresult, (Ptr{Cfloat}, CUtexref), pBorderColor, hTexRef)
    end

@checked function cuTexRefGetFlags(pFlags, hTexRef)
        initialize_context()
        ccall((:cuTexRefGetFlags, libcuda), CUresult, (Ptr{Cuint}, CUtexref), pFlags, hTexRef)
    end

@checked function cuTexRefCreate(pTexRef)
        initialize_context()
        ccall((:cuTexRefCreate, libcuda), CUresult, (Ptr{CUtexref},), pTexRef)
    end

@checked function cuTexRefDestroy(hTexRef)
        initialize_context()
        ccall((:cuTexRefDestroy, libcuda), CUresult, (CUtexref,), hTexRef)
    end

@checked function cuSurfRefSetArray(hSurfRef, hArray, Flags)
        initialize_context()
        ccall((:cuSurfRefSetArray, libcuda), CUresult, (CUsurfref, CUarray, Cuint), hSurfRef, hArray, Flags)
    end

@checked function cuSurfRefGetArray(phArray, hSurfRef)
        initialize_context()
        ccall((:cuSurfRefGetArray, libcuda), CUresult, (Ptr{CUarray}, CUsurfref), phArray, hSurfRef)
    end

@checked function cuTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc)
        initialize_context()
        ccall((:cuTexObjectCreate, libcuda), CUresult, (Ptr{CUtexObject}, Ptr{CUDA_RESOURCE_DESC}, Ptr{CUDA_TEXTURE_DESC}, Ptr{CUDA_RESOURCE_VIEW_DESC}), pTexObject, pResDesc, pTexDesc, pResViewDesc)
    end

@checked function cuTexObjectDestroy(texObject)
        initialize_context()
        ccall((:cuTexObjectDestroy, libcuda), CUresult, (CUtexObject,), texObject)
    end

@checked function cuTexObjectGetResourceDesc(pResDesc, texObject)
        initialize_context()
        ccall((:cuTexObjectGetResourceDesc, libcuda), CUresult, (Ptr{CUDA_RESOURCE_DESC}, CUtexObject), pResDesc, texObject)
    end

@checked function cuTexObjectGetTextureDesc(pTexDesc, texObject)
        initialize_context()
        ccall((:cuTexObjectGetTextureDesc, libcuda), CUresult, (Ptr{CUDA_TEXTURE_DESC}, CUtexObject), pTexDesc, texObject)
    end

@checked function cuTexObjectGetResourceViewDesc(pResViewDesc, texObject)
        initialize_context()
        ccall((:cuTexObjectGetResourceViewDesc, libcuda), CUresult, (Ptr{CUDA_RESOURCE_VIEW_DESC}, CUtexObject), pResViewDesc, texObject)
    end

@checked function cuSurfObjectCreate(pSurfObject, pResDesc)
        initialize_context()
        ccall((:cuSurfObjectCreate, libcuda), CUresult, (Ptr{CUsurfObject}, Ptr{CUDA_RESOURCE_DESC}), pSurfObject, pResDesc)
    end

@checked function cuSurfObjectDestroy(surfObject)
        initialize_context()
        ccall((:cuSurfObjectDestroy, libcuda), CUresult, (CUsurfObject,), surfObject)
    end

@checked function cuSurfObjectGetResourceDesc(pResDesc, surfObject)
        initialize_context()
        ccall((:cuSurfObjectGetResourceDesc, libcuda), CUresult, (Ptr{CUDA_RESOURCE_DESC}, CUsurfObject), pResDesc, surfObject)
    end

@checked function cuDeviceCanAccessPeer(canAccessPeer, dev, peerDev)
        initialize_context()
        ccall((:cuDeviceCanAccessPeer, libcuda), CUresult, (Ptr{Cint}, CUdevice, CUdevice), canAccessPeer, dev, peerDev)
    end

@checked function cuCtxEnablePeerAccess(peerContext, Flags)
        initialize_context()
        ccall((:cuCtxEnablePeerAccess, libcuda), CUresult, (CUcontext, Cuint), peerContext, Flags)
    end

@checked function cuCtxDisablePeerAccess(peerContext)
        initialize_context()
        ccall((:cuCtxDisablePeerAccess, libcuda), CUresult, (CUcontext,), peerContext)
    end

@checked function cuDeviceGetP2PAttribute(value, attrib, srcDevice, dstDevice)
        initialize_context()
        ccall((:cuDeviceGetP2PAttribute, libcuda), CUresult, (Ptr{Cint}, CUdevice_P2PAttribute, CUdevice, CUdevice), value, attrib, srcDevice, dstDevice)
    end

@checked function cuGraphicsUnregisterResource(resource)
        initialize_context()
        ccall((:cuGraphicsUnregisterResource, libcuda), CUresult, (CUgraphicsResource,), resource)
    end

@checked function cuGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel)
        initialize_context()
        ccall((:cuGraphicsSubResourceGetMappedArray, libcuda), CUresult, (Ptr{CUarray}, CUgraphicsResource, Cuint, Cuint), pArray, resource, arrayIndex, mipLevel)
    end

@checked function cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray, resource)
        initialize_context()
        ccall((:cuGraphicsResourceGetMappedMipmappedArray, libcuda), CUresult, (Ptr{CUmipmappedArray}, CUgraphicsResource), pMipmappedArray, resource)
    end

@checked function cuGraphicsMapResources(count, resources, hStream)
        initialize_context()
        ccall((:cuGraphicsMapResources, libcuda), CUresult, (Cuint, Ptr{CUgraphicsResource}, CUstream), count, resources, hStream)
    end

@checked function cuGraphicsUnmapResources(count, resources, hStream)
        initialize_context()
        ccall((:cuGraphicsUnmapResources, libcuda), CUresult, (Cuint, Ptr{CUgraphicsResource}, CUstream), count, resources, hStream)
    end

@checked function cuGetProcAddress(symbol, pfn, cudaVersion, flags)
        initialize_context()
        ccall((:cuGetProcAddress, libcuda), CUresult, (Cstring, Ptr{Ptr{Cvoid}}, Cint, cuuint64_t), symbol, pfn, cudaVersion, flags)
    end

@checked function cuGetExportTable(ppExportTable, pExportTableId)
        initialize_context()
        ccall((:cuGetExportTable, libcuda), CUresult, (Ptr{Ptr{Cvoid}}, Ptr{CUuuid}), ppExportTable, pExportTableId)
    end

@checked function cuGLCtxCreate_v2(pCtx, Flags, device)
        initialize_context()
        ccall((:cuGLCtxCreate_v2, libcuda), CUresult, (Ptr{CUcontext}, Cuint, CUdevice), pCtx, Flags, device)
    end

@checked function cuGLMapBufferObject_v2(dptr, size, buffer)
        initialize_context()
        ccall((:cuGLMapBufferObject_v2, libcuda), CUresult, (Ptr{CUdeviceptr}, Ptr{Csize_t}, GLuint), dptr, size, buffer)
    end

@checked function cuGLMapBufferObjectAsync_v2(dptr, size, buffer, hStream)
        initialize_context()
        ccall((:cuGLMapBufferObjectAsync_v2, libcuda), CUresult, (Ptr{CUdeviceptr}, Ptr{Csize_t}, GLuint, CUstream), dptr, size, buffer, hStream)
    end

@cenum CUGLDeviceList_enum::UInt32 begin
    CU_GL_DEVICE_LIST_ALL = 1
    CU_GL_DEVICE_LIST_CURRENT_FRAME = 2
    CU_GL_DEVICE_LIST_NEXT_FRAME = 3
end

const CUGLDeviceList = CUGLDeviceList_enum

@checked function cuGLGetDevices_v2(pCudaDeviceCount, pCudaDevices, cudaDeviceCount, deviceList)
        initialize_context()
        ccall((:cuGLGetDevices_v2, libcuda), CUresult, (Ptr{Cuint}, Ptr{CUdevice}, Cuint, CUGLDeviceList), pCudaDeviceCount, pCudaDevices, cudaDeviceCount, deviceList)
    end

@checked function cuGraphicsGLRegisterBuffer(pCudaResource, buffer, Flags)
        initialize_context()
        ccall((:cuGraphicsGLRegisterBuffer, libcuda), CUresult, (Ptr{CUgraphicsResource}, GLuint, Cuint), pCudaResource, buffer, Flags)
    end

@checked function cuGraphicsGLRegisterImage(pCudaResource, image, target, Flags)
        initialize_context()
        ccall((:cuGraphicsGLRegisterImage, libcuda), CUresult, (Ptr{CUgraphicsResource}, GLuint, GLenum, Cuint), pCudaResource, image, target, Flags)
    end

@cenum CUGLmap_flags_enum::UInt32 begin
    CU_GL_MAP_RESOURCE_FLAGS_NONE = 0
    CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY = 1
    CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 2
end

const CUGLmap_flags = CUGLmap_flags_enum

@checked function cuGLInit()
        initialize_context()
        ccall((:cuGLInit, libcuda), CUresult, ())
    end

@checked function cuGLRegisterBufferObject(buffer)
        initialize_context()
        ccall((:cuGLRegisterBufferObject, libcuda), CUresult, (GLuint,), buffer)
    end

@checked function cuGLUnmapBufferObject(buffer)
        initialize_context()
        ccall((:cuGLUnmapBufferObject, libcuda), CUresult, (GLuint,), buffer)
    end

@checked function cuGLUnregisterBufferObject(buffer)
        initialize_context()
        ccall((:cuGLUnregisterBufferObject, libcuda), CUresult, (GLuint,), buffer)
    end

@checked function cuGLSetBufferObjectMapFlags(buffer, Flags)
        initialize_context()
        ccall((:cuGLSetBufferObjectMapFlags, libcuda), CUresult, (GLuint, Cuint), buffer, Flags)
    end

@checked function cuGLUnmapBufferObjectAsync(buffer, hStream)
        initialize_context()
        ccall((:cuGLUnmapBufferObjectAsync, libcuda), CUresult, (GLuint, CUstream), buffer, hStream)
    end

@cenum CUoutput_mode_enum::UInt32 begin
    CU_OUT_KEY_VALUE_PAIR = 0
    CU_OUT_CSV = 1
end

const CUoutput_mode = CUoutput_mode_enum

@checked function cuProfilerInitialize(configFile, outputFile, outputMode)
        initialize_context()
        ccall((:cuProfilerInitialize, libcuda), CUresult, (Cstring, Cstring, CUoutput_mode), configFile, outputFile, outputMode)
    end

@checked function cuProfilerStart()
        initialize_context()
        ccall((:cuProfilerStart, libcuda), CUresult, ())
    end

@checked function cuProfilerStop()
        initialize_context()
        ccall((:cuProfilerStop, libcuda), CUresult, ())
    end

struct var"##Ctag#363"
    hArray::CUarray
end
function Base.getproperty(x::Ptr{var"##Ctag#363"}, f::Symbol)
    f === :hArray && return Ptr{CUarray}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#363", f::Symbol)
    r = Ref{var"##Ctag#363"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#363"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#363"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct var"##Ctag#364"
    hMipmappedArray::CUmipmappedArray
end
function Base.getproperty(x::Ptr{var"##Ctag#364"}, f::Symbol)
    f === :hMipmappedArray && return Ptr{CUmipmappedArray}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#364", f::Symbol)
    r = Ref{var"##Ctag#364"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#364"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#364"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct var"##Ctag#365"
    devPtr::CUdeviceptr
    format::CUarray_format
    numChannels::Cuint
    sizeInBytes::Csize_t
end
function Base.getproperty(x::Ptr{var"##Ctag#365"}, f::Symbol)
    f === :devPtr && return Ptr{CUdeviceptr}(x + 0)
    f === :format && return Ptr{CUarray_format}(x + 8)
    f === :numChannels && return Ptr{Cuint}(x + 12)
    f === :sizeInBytes && return Ptr{Csize_t}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#365", f::Symbol)
    r = Ref{var"##Ctag#365"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#365"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#365"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct var"##Ctag#366"
    devPtr::CUdeviceptr
    format::CUarray_format
    numChannels::Cuint
    width::Csize_t
    height::Csize_t
    pitchInBytes::Csize_t
end
function Base.getproperty(x::Ptr{var"##Ctag#366"}, f::Symbol)
    f === :devPtr && return Ptr{CUdeviceptr}(x + 0)
    f === :format && return Ptr{CUarray_format}(x + 8)
    f === :numChannels && return Ptr{Cuint}(x + 12)
    f === :width && return Ptr{Csize_t}(x + 16)
    f === :height && return Ptr{Csize_t}(x + 24)
    f === :pitchInBytes && return Ptr{Csize_t}(x + 32)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#366", f::Symbol)
    r = Ref{var"##Ctag#366"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#366"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#366"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct var"##Ctag#367"
    reserved::NTuple{32, Cint}
end
function Base.getproperty(x::Ptr{var"##Ctag#367"}, f::Symbol)
    f === :reserved && return Ptr{NTuple{32, Cint}}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#367", f::Symbol)
    r = Ref{var"##Ctag#367"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#367"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#367"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct var"##Ctag#369"
    handle::Ptr{Cvoid}
    name::Ptr{Cvoid}
end
function Base.getproperty(x::Ptr{var"##Ctag#369"}, f::Symbol)
    f === :handle && return Ptr{Ptr{Cvoid}}(x + 0)
    f === :name && return Ptr{Ptr{Cvoid}}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#369", f::Symbol)
    r = Ref{var"##Ctag#369"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#369"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#369"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct var"##Ctag#373"
    level::Cuint
    layer::Cuint
    offsetX::Cuint
    offsetY::Cuint
    offsetZ::Cuint
    extentWidth::Cuint
    extentHeight::Cuint
    extentDepth::Cuint
end
function Base.getproperty(x::Ptr{var"##Ctag#373"}, f::Symbol)
    f === :level && return Ptr{Cuint}(x + 0)
    f === :layer && return Ptr{Cuint}(x + 4)
    f === :offsetX && return Ptr{Cuint}(x + 8)
    f === :offsetY && return Ptr{Cuint}(x + 12)
    f === :offsetZ && return Ptr{Cuint}(x + 16)
    f === :extentWidth && return Ptr{Cuint}(x + 20)
    f === :extentHeight && return Ptr{Cuint}(x + 24)
    f === :extentDepth && return Ptr{Cuint}(x + 28)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#373", f::Symbol)
    r = Ref{var"##Ctag#373"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#373"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#373"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct var"##Ctag#374"
    layer::Cuint
    offset::Culonglong
    size::Culonglong
end
function Base.getproperty(x::Ptr{var"##Ctag#374"}, f::Symbol)
    f === :layer && return Ptr{Cuint}(x + 0)
    f === :offset && return Ptr{Culonglong}(x + 8)
    f === :size && return Ptr{Culonglong}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#374", f::Symbol)
    r = Ref{var"##Ctag#374"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#374"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#374"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct var"##Ctag#380"
    x::Cuint
    y::Cuint
    z::Cuint
end
function Base.getproperty(x::Ptr{var"##Ctag#380"}, f::Symbol)
    f === :x && return Ptr{Cuint}(x + 0)
    f === :y && return Ptr{Cuint}(x + 4)
    f === :z && return Ptr{Cuint}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#380", f::Symbol)
    r = Ref{var"##Ctag#380"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#380"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#380"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct var"##Ctag#381"
    event::CUevent
    flags::Cint
    triggerAtBlockStart::Cint
end
function Base.getproperty(x::Ptr{var"##Ctag#381"}, f::Symbol)
    f === :event && return Ptr{CUevent}(x + 0)
    f === :flags && return Ptr{Cint}(x + 8)
    f === :triggerAtBlockStart && return Ptr{Cint}(x + 12)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#381", f::Symbol)
    r = Ref{var"##Ctag#381"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#381"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#381"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct CUstreamMemOpWaitValueParams_st
    data::NTuple{40, UInt8}
end

function Base.getproperty(x::Ptr{CUstreamMemOpWaitValueParams_st}, f::Symbol)
    f === :operation && return Ptr{CUstreamBatchMemOpType}(x + 0)
    f === :address && return Ptr{CUdeviceptr}(x + 8)
    f === :value && return Ptr{cuuint32_t}(x + 16)
    f === :value64 && return Ptr{cuuint64_t}(x + 16)
    f === :flags && return Ptr{Cuint}(x + 24)
    f === :alias && return Ptr{CUdeviceptr}(x + 32)
    return getfield(x, f)
end

function Base.getproperty(x::CUstreamMemOpWaitValueParams_st, f::Symbol)
    r = Ref{CUstreamMemOpWaitValueParams_st}(x)
    ptr = Base.unsafe_convert(Ptr{CUstreamMemOpWaitValueParams_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{CUstreamMemOpWaitValueParams_st}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct CUstreamMemOpWriteValueParams_st
    data::NTuple{40, UInt8}
end

function Base.getproperty(x::Ptr{CUstreamMemOpWriteValueParams_st}, f::Symbol)
    f === :operation && return Ptr{CUstreamBatchMemOpType}(x + 0)
    f === :address && return Ptr{CUdeviceptr}(x + 8)
    f === :value && return Ptr{cuuint32_t}(x + 16)
    f === :value64 && return Ptr{cuuint64_t}(x + 16)
    f === :flags && return Ptr{Cuint}(x + 24)
    f === :alias && return Ptr{CUdeviceptr}(x + 32)
    return getfield(x, f)
end

function Base.getproperty(x::CUstreamMemOpWriteValueParams_st, f::Symbol)
    r = Ref{CUstreamMemOpWriteValueParams_st}(x)
    ptr = Base.unsafe_convert(Ptr{CUstreamMemOpWriteValueParams_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{CUstreamMemOpWriteValueParams_st}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct CUstreamMemOpFlushRemoteWritesParams_st
    operation::CUstreamBatchMemOpType
    flags::Cuint
end

struct CUstreamMemOpMemoryBarrierParams_st
    operation::CUstreamBatchMemOpType
    flags::Cuint
end

struct var"##Ctag#383"
    handle::Ptr{Cvoid}
    name::Ptr{Cvoid}
end
function Base.getproperty(x::Ptr{var"##Ctag#383"}, f::Symbol)
    f === :handle && return Ptr{Ptr{Cvoid}}(x + 0)
    f === :name && return Ptr{Ptr{Cvoid}}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#383", f::Symbol)
    r = Ref{var"##Ctag#383"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#383"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#383"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


# Skipping MacroDefinition: __CUDA_DEPRECATED __attribute__ ( ( deprecated ) )

const CU_IPC_HANDLE_SIZE = 64

const CU_STREAM_LEGACY = CUstream(0x01)

const CU_STREAM_PER_THREAD = CUstream(0x02)

const CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW

const CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE = CU_LAUNCH_ATTRIBUTE_COOPERATIVE

const CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION

const CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE

const CU_KERNEL_NODE_ATTRIBUTE_PRIORITY = CU_LAUNCH_ATTRIBUTE_PRIORITY

const CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW = CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW

const CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY = CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY

const CU_MEMHOSTALLOC_PORTABLE = 0x01

const CU_MEMHOSTALLOC_DEVICEMAP = 0x02

const CU_MEMHOSTALLOC_WRITECOMBINED = 0x04

const CU_MEMHOSTREGISTER_PORTABLE = 0x01

const CU_MEMHOSTREGISTER_DEVICEMAP = 0x02

const CU_MEMHOSTREGISTER_IOMEMORY = 0x04

const CU_MEMHOSTREGISTER_READ_ONLY = 0x08

const CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL = 0x01

const CUDA_EXTERNAL_MEMORY_DEDICATED = 0x01

const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC = 0x01

const CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC = 0x02

const CUDA_NVSCISYNC_ATTR_SIGNAL = 0x01

const CUDA_NVSCISYNC_ATTR_WAIT = 0x02

const CU_MEM_CREATE_USAGE_TILE_POOL = 0x01

const CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC = 0x01

const CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC = 0x02

const CUDA_ARRAY3D_LAYERED = 0x01

const CUDA_ARRAY3D_2DARRAY = 0x01

const CUDA_ARRAY3D_SURFACE_LDST = 0x02

const CUDA_ARRAY3D_CUBEMAP = 0x04

const CUDA_ARRAY3D_TEXTURE_GATHER = 0x08

const CUDA_ARRAY3D_DEPTH_TEXTURE = 0x10

const CUDA_ARRAY3D_COLOR_ATTACHMENT = 0x20

const CUDA_ARRAY3D_SPARSE = 0x40

const CUDA_ARRAY3D_DEFERRED_MAPPING = 0x80

const CU_TRSA_OVERRIDE_FORMAT = 0x01

const CU_TRSF_READ_AS_INTEGER = 0x01

const CU_TRSF_NORMALIZED_COORDINATES = 0x02

const CU_TRSF_SRGB = 0x10

const CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION = 0x20

const CU_TRSF_SEAMLESS_CUBEMAP = 0x40

const CU_LAUNCH_PARAM_END_AS_INT = 0x00

# Skipping MacroDefinition: CU_LAUNCH_PARAM_END ( ( void * ) CU_LAUNCH_PARAM_END_AS_INT )

const CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT = 0x01

# Skipping MacroDefinition: CU_LAUNCH_PARAM_BUFFER_POINTER ( ( void * ) CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT )

const CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT = 0x02

# Skipping MacroDefinition: CU_LAUNCH_PARAM_BUFFER_SIZE ( ( void * ) CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT )

const CU_PARAM_TR_DEFAULT = -1

const CU_DEVICE_CPU = CUdevice(-1)

const CU_DEVICE_INVALID = CUdevice(-2)

