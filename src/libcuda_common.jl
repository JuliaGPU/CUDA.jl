# Automatically generated using Clang.jl



# Skipping MacroDefinition: cuMemcpyHtoD __CUDA_API_PTDS ( cuMemcpyHtoD_v2 )
# Skipping MacroDefinition: cuMemcpyDtoH __CUDA_API_PTDS ( cuMemcpyDtoH_v2 )
# Skipping MacroDefinition: cuMemcpyDtoD __CUDA_API_PTDS ( cuMemcpyDtoD_v2 )
# Skipping MacroDefinition: cuMemcpyDtoA __CUDA_API_PTDS ( cuMemcpyDtoA_v2 )
# Skipping MacroDefinition: cuMemcpyAtoD __CUDA_API_PTDS ( cuMemcpyAtoD_v2 )
# Skipping MacroDefinition: cuMemcpyHtoA __CUDA_API_PTDS ( cuMemcpyHtoA_v2 )
# Skipping MacroDefinition: cuMemcpyAtoH __CUDA_API_PTDS ( cuMemcpyAtoH_v2 )
# Skipping MacroDefinition: cuMemcpyAtoA __CUDA_API_PTDS ( cuMemcpyAtoA_v2 )
# Skipping MacroDefinition: cuMemcpyHtoAAsync __CUDA_API_PTSZ ( cuMemcpyHtoAAsync_v2 )
# Skipping MacroDefinition: cuMemcpyAtoHAsync __CUDA_API_PTSZ ( cuMemcpyAtoHAsync_v2 )
# Skipping MacroDefinition: cuMemcpy2D __CUDA_API_PTDS ( cuMemcpy2D_v2 )
# Skipping MacroDefinition: cuMemcpy2DUnaligned __CUDA_API_PTDS ( cuMemcpy2DUnaligned_v2 )
# Skipping MacroDefinition: cuMemcpy3D __CUDA_API_PTDS ( cuMemcpy3D_v2 )
# Skipping MacroDefinition: cuMemcpyHtoDAsync __CUDA_API_PTSZ ( cuMemcpyHtoDAsync_v2 )
# Skipping MacroDefinition: cuMemcpyDtoHAsync __CUDA_API_PTSZ ( cuMemcpyDtoHAsync_v2 )
# Skipping MacroDefinition: cuMemcpyDtoDAsync __CUDA_API_PTSZ ( cuMemcpyDtoDAsync_v2 )
# Skipping MacroDefinition: cuMemcpy2DAsync __CUDA_API_PTSZ ( cuMemcpy2DAsync_v2 )
# Skipping MacroDefinition: cuMemcpy3DAsync __CUDA_API_PTSZ ( cuMemcpy3DAsync_v2 )
# Skipping MacroDefinition: cuMemsetD8 __CUDA_API_PTDS ( cuMemsetD8_v2 )
# Skipping MacroDefinition: cuMemsetD16 __CUDA_API_PTDS ( cuMemsetD16_v2 )
# Skipping MacroDefinition: cuMemsetD32 __CUDA_API_PTDS ( cuMemsetD32_v2 )
# Skipping MacroDefinition: cuMemsetD2D8 __CUDA_API_PTDS ( cuMemsetD2D8_v2 )
# Skipping MacroDefinition: cuMemsetD2D16 __CUDA_API_PTDS ( cuMemsetD2D16_v2 )
# Skipping MacroDefinition: cuMemsetD2D32 __CUDA_API_PTDS ( cuMemsetD2D32_v2 )


# Skipping MacroDefinition: cuStreamBeginCapture __CUDA_API_PTSZ ( cuStreamBeginCapture_v2 )

const CUDA_VERSION = 10010
const IPC_HANDLE_SIZE = 64

# Skipping MacroDefinition: STREAM_LEGACY ( ( CUstream ) 0x1 )
# Skipping MacroDefinition: STREAM_PER_THREAD ( ( CUstream ) 0x2 )

const MEMHOSTALLOC_PORTABLE = 0x01
const MEMHOSTALLOC_DEVICEMAP = 0x02
const MEMHOSTALLOC_WRITECOMBINED = 0x04
const MEMHOSTREGISTER_PORTABLE = 0x01
const MEMHOSTREGISTER_DEVICEMAP = 0x02
const MEMHOSTREGISTER_IOMEMORY = 0x04
const CUDA_EXTERNAL_MEMORY_DEDICATED = 0x01
const CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC = 0x01
const CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC = 0x02
const CUDA_ARRAY3D_LAYERED = 0x01
const CUDA_ARRAY3D_2DARRAY = 0x01
const CUDA_ARRAY3D_SURFACE_LDST = 0x02
const CUDA_ARRAY3D_CUBEMAP = 0x04
const CUDA_ARRAY3D_TEXTURE_GATHER = 0x08
const CUDA_ARRAY3D_DEPTH_TEXTURE = 0x10
const CUDA_ARRAY3D_COLOR_ATTACHMENT = 0x20
const TRSA_OVERRIDE_FORMAT = 0x01
const TRSF_READ_AS_INTEGER = 0x01
const TRSF_NORMALIZED_COORDINATES = 0x02
const TRSF_SRGB = 0x10

# Skipping MacroDefinition: LAUNCH_PARAM_END ( ( void * ) 0x00 )
# Skipping MacroDefinition: LAUNCH_PARAM_BUFFER_POINTER ( ( void * ) 0x01 )
# Skipping MacroDefinition: LAUNCH_PARAM_BUFFER_SIZE ( ( void * ) 0x02 )

const PARAM_TR_DEFAULT = -1

# Skipping MacroDefinition: DEVICE_CPU ( ( CUdevice ) - 1 )
# Skipping MacroDefinition: DEVICE_INVALID ( ( CUdevice ) - 2 )

const cuuint32_t = UInt32
const cuuint64_t = UInt64
const CUdeviceptr = Culonglong
const CUdevice = Cint
const CUctx_st = Cvoid
const CUcontext = Ptr{CUctx_st}
const CUmod_st = Cvoid
const CUmodule = Ptr{CUmod_st}
const CUfunc_st = Cvoid
const CUfunction = Ptr{CUfunc_st}
const CUarray_st = Cvoid
const CUarray = Ptr{CUarray_st}
const CUmipmappedArray_st = Cvoid
const CUmipmappedArray = Ptr{CUmipmappedArray_st}
const CUtexref_st = Cvoid
const CUtexref = Ptr{CUtexref_st}
const CUsurfref_st = Cvoid
const CUsurfref = Ptr{CUsurfref_st}
const CUevent_st = Cvoid
const CUevent = Ptr{CUevent_st}
const CUstream_st = Cvoid
const CUstream = Ptr{CUstream_st}
const CUgraphicsResource_st = Cvoid
const CUgraphicsResource = Ptr{CUgraphicsResource_st}
const CUtexObject = Culonglong
const CUsurfObject = Culonglong
const CUextMemory_st = Cvoid
const CUexternalMemory = Ptr{CUextMemory_st}
const CUextSemaphore_st = Cvoid
const CUexternalSemaphore = Ptr{CUextSemaphore_st}
const CUgraph_st = Cvoid
const CUgraph = Ptr{CUgraph_st}
const CUgraphNode_st = Cvoid
const CUgraphNode = Ptr{CUgraphNode_st}
const CUgraphExec_st = Cvoid
const CUgraphExec = Ptr{CUgraphExec_st}

struct CUuuid_st
    bytes::NTuple{16, UInt8}
end

const CUuuid = CUuuid_st

struct CUipcEventHandle_st
    reserved::NTuple{64, UInt8}
end

const CUipcEventHandle = CUipcEventHandle_st

struct CUipcMemHandle_st
    reserved::NTuple{64, UInt8}
end

const CUipcMemHandle = CUipcMemHandle_st

@cenum CUipcMem_flags_enum::UInt32 begin
    IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1
end


const CUipcMem_flags = CUipcMem_flags_enum

@cenum CUmemAttach_flags_enum::UInt32 begin
    MEM_ATTACH_GLOBAL = 1
    MEM_ATTACH_HOST = 2
    MEM_ATTACH_SINGLE = 4
end


const CUmemAttach_flags = CUmemAttach_flags_enum

@cenum CUctx_flags_enum::UInt32 begin
    CTX_SCHED_AUTO = 0
    CTX_SCHED_SPIN = 1
    CTX_SCHED_YIELD = 2
    CTX_SCHED_BLOCKING_SYNC = 4
    CTX_BLOCKING_SYNC = 4
    CTX_SCHED_MASK = 7
    CTX_MAP_HOST = 8
    CTX_LMEM_RESIZE_TO_MAX = 16
    CTX_FLAGS_MASK = 31
end


const CUctx_flags = CUctx_flags_enum

@cenum CUstream_flags_enum::UInt32 begin
    STREAM_DEFAULT = 0
    STREAM_NON_BLOCKING = 1
end


const CUstream_flags = CUstream_flags_enum

@cenum CUevent_flags_enum::UInt32 begin
    EVENT_DEFAULT = 0
    EVENT_BLOCKING_SYNC = 1
    EVENT_DISABLE_TIMING = 2
    EVENT_INTERPROCESS = 4
end


const CUevent_flags = CUevent_flags_enum

@cenum CUstreamWaitValue_flags_enum::UInt32 begin
    STREAM_WAIT_VALUE_GEQ = 0
    STREAM_WAIT_VALUE_EQ = 1
    STREAM_WAIT_VALUE_AND = 2
    STREAM_WAIT_VALUE_NOR = 3
    STREAM_WAIT_VALUE_FLUSH = 1073741824
end


const CUstreamWaitValue_flags = CUstreamWaitValue_flags_enum

@cenum CUstreamWriteValue_flags_enum::UInt32 begin
    STREAM_WRITE_VALUE_DEFAULT = 0
    STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 1
end


const CUstreamWriteValue_flags = CUstreamWriteValue_flags_enum

@cenum CUoccupancy_flags_enum::UInt32 begin
    OCCUPANCY_DEFAULT = 0
    OCCUPANCY_DISABLE_CACHING_OVERRIDE = 1
end


const CUoccupancy_flags = CUoccupancy_flags_enum

@cenum CUarray_format_enum::UInt32 begin
    AD_FORMAT_UNSIGNED_INT8 = 1
    AD_FORMAT_UNSIGNED_INT16 = 2
    AD_FORMAT_UNSIGNED_INT32 = 3
    AD_FORMAT_SIGNED_INT8 = 8
    AD_FORMAT_SIGNED_INT16 = 9
    AD_FORMAT_SIGNED_INT32 = 10
    AD_FORMAT_HALF = 16
    AD_FORMAT_FLOAT = 32
end


const CUarray_format = CUarray_format_enum

@cenum CUaddress_mode_enum::UInt32 begin
    TR_ADDRESS_MODE_WRAP = 0
    TR_ADDRESS_MODE_CLAMP = 1
    TR_ADDRESS_MODE_MIRROR = 2
    TR_ADDRESS_MODE_BORDER = 3
end


const CUaddress_mode = CUaddress_mode_enum

@cenum CUfilter_mode_enum::UInt32 begin
    TR_FILTER_MODE_POINT = 0
    TR_FILTER_MODE_LINEAR = 1
end


const CUfilter_mode = CUfilter_mode_enum

@cenum CUdevice_attribute_enum::UInt32 begin
    DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
    DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2
    DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3
    DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4
    DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5
    DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6
    DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7
    DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
    DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8
    DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9
    DEVICE_ATTRIBUTE_WARP_SIZE = 10
    DEVICE_ATTRIBUTE_MAX_PITCH = 11
    DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12
    DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12
    DEVICE_ATTRIBUTE_CLOCK_RATE = 13
    DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14
    DEVICE_ATTRIBUTE_GPU_OVERLAP = 15
    DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
    DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17
    DEVICE_ATTRIBUTE_INTEGRATED = 18
    DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19
    DEVICE_ATTRIBUTE_COMPUTE_MODE = 20
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29
    DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30
    DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31
    DEVICE_ATTRIBUTE_ECC_ENABLED = 32
    DEVICE_ATTRIBUTE_PCI_BUS_ID = 33
    DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34
    DEVICE_ATTRIBUTE_TCC_DRIVER = 35
    DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
    DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37
    DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38
    DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
    DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40
    DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43
    DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49
    DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50
    DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54
    DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55
    DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56
    DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57
    DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58
    DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59
    DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60
    DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61
    DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62
    DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63
    DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64
    DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65
    DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66
    DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67
    DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74
    DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
    DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
    DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77
    DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78
    DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79
    DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80
    DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81
    DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82
    DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83
    DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84
    DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85
    DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86
    DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87
    DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88
    DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89
    DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90
    DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91
    DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92
    DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93
    DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94
    DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95
    DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96
    DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
    DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98
    DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99
    DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100
    DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101
    DEVICE_ATTRIBUTE_MAX = 102
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

const CUdevprop = CUdevprop_st

@cenum CUpointer_attribute_enum::UInt32 begin
    POINTER_ATTRIBUTE_CONTEXT = 1
    POINTER_ATTRIBUTE_MEMORY_TYPE = 2
    POINTER_ATTRIBUTE_DEVICE_POINTER = 3
    POINTER_ATTRIBUTE_HOST_POINTER = 4
    POINTER_ATTRIBUTE_P2P_TOKENS = 5
    POINTER_ATTRIBUTE_SYNC_MEMOPS = 6
    POINTER_ATTRIBUTE_BUFFER_ID = 7
    POINTER_ATTRIBUTE_IS_MANAGED = 8
    POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9
end


const CUpointer_attribute = CUpointer_attribute_enum

@cenum CUfunction_attribute_enum::UInt32 begin
    FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
    FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
    FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2
    FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
    FUNC_ATTRIBUTE_NUM_REGS = 4
    FUNC_ATTRIBUTE_PTX_VERSION = 5
    FUNC_ATTRIBUTE_BINARY_VERSION = 6
    FUNC_ATTRIBUTE_CACHE_MODE_CA = 7
    FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
    FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9
    FUNC_ATTRIBUTE_MAX = 10
end


const CUfunction_attribute = CUfunction_attribute_enum

@cenum CUfunc_cache_enum::UInt32 begin
    FUNC_CACHE_PREFER_NONE = 0
    FUNC_CACHE_PREFER_SHARED = 1
    FUNC_CACHE_PREFER_L1 = 2
    FUNC_CACHE_PREFER_EQUAL = 3
end


const CUfunc_cache = CUfunc_cache_enum

@cenum CUsharedconfig_enum::UInt32 begin
    SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0
    SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 1
    SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 2
end


const CUsharedconfig = CUsharedconfig_enum

@cenum CUshared_carveout_enum::Int32 begin
    SHAREDMEM_CARVEOUT_DEFAULT = -1
    SHAREDMEM_CARVEOUT_MAX_SHARED = 100
    SHAREDMEM_CARVEOUT_MAX_L1 = 0
end


const CUshared_carveout = CUshared_carveout_enum

@cenum CUmemorytype_enum::UInt32 begin
    MEMORYTYPE_HOST = 1
    MEMORYTYPE_DEVICE = 2
    MEMORYTYPE_ARRAY = 3
    MEMORYTYPE_UNIFIED = 4
end


const CUmemorytype = CUmemorytype_enum

@cenum CUcomputemode_enum::UInt32 begin
    COMPUTEMODE_DEFAULT = 0
    COMPUTEMODE_PROHIBITED = 2
    COMPUTEMODE_EXCLUSIVE_PROCESS = 3
end


const CUcomputemode = CUcomputemode_enum

@cenum CUmem_advise_enum::UInt32 begin
    MEM_ADVISE_SET_READ_MOSTLY = 1
    MEM_ADVISE_UNSET_READ_MOSTLY = 2
    MEM_ADVISE_SET_PREFERRED_LOCATION = 3
    MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4
    MEM_ADVISE_SET_ACCESSED_BY = 5
    MEM_ADVISE_UNSET_ACCESSED_BY = 6
end


const CUmem_advise = CUmem_advise_enum

@cenum CUmem_range_attribute_enum::UInt32 begin
    MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1
    MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2
    MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3
    MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4
end


const CUmem_range_attribute = CUmem_range_attribute_enum

@cenum CUjit_option_enum::UInt32 begin
    JIT_MAX_REGISTERS = 0
    JIT_THREADS_PER_BLOCK = 1
    JIT_WALL_TIME = 2
    JIT_INFO_LOG_BUFFER = 3
    JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4
    JIT_ERROR_LOG_BUFFER = 5
    JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
    JIT_OPTIMIZATION_LEVEL = 7
    JIT_TARGET_FROM_CUCONTEXT = 8
    JIT_TARGET = 9
    JIT_FALLBACK_STRATEGY = 10
    JIT_GENERATE_DEBUG_INFO = 11
    JIT_LOG_VERBOSE = 12
    JIT_GENERATE_LINE_INFO = 13
    JIT_CACHE_MODE = 14
    JIT_NEW_SM3X_OPT = 15
    JIT_FAST_COMPILE = 16
    JIT_GLOBAL_SYMBOL_NAMES = 17
    JIT_GLOBAL_SYMBOL_ADDRESSES = 18
    JIT_GLOBAL_SYMBOL_COUNT = 19
    JIT_NUM_OPTIONS = 20
end


const CUjit_option = CUjit_option_enum

@cenum CUjit_target_enum::UInt32 begin
    TARGET_COMPUTE_20 = 20
    TARGET_COMPUTE_21 = 21
    TARGET_COMPUTE_30 = 30
    TARGET_COMPUTE_32 = 32
    TARGET_COMPUTE_35 = 35
    TARGET_COMPUTE_37 = 37
    TARGET_COMPUTE_50 = 50
    TARGET_COMPUTE_52 = 52
    TARGET_COMPUTE_53 = 53
    TARGET_COMPUTE_60 = 60
    TARGET_COMPUTE_61 = 61
    TARGET_COMPUTE_62 = 62
    TARGET_COMPUTE_70 = 70
    TARGET_COMPUTE_72 = 72
    TARGET_COMPUTE_75 = 75
end


const CUjit_target = CUjit_target_enum

@cenum CUjit_fallback_enum::UInt32 begin
    PREFER_PTX = 0
    PREFER_BINARY = 1
end


const CUjit_fallback = CUjit_fallback_enum

@cenum CUjit_cacheMode_enum::UInt32 begin
    JIT_CACHE_OPTION_NONE = 0
    JIT_CACHE_OPTION_CG = 1
    JIT_CACHE_OPTION_CA = 2
end


const CUjit_cacheMode = CUjit_cacheMode_enum

@cenum CUjitInputType_enum::UInt32 begin
    JIT_INPUT_CUBIN = 0
    JIT_INPUT_PTX = 1
    JIT_INPUT_FATBINARY = 2
    JIT_INPUT_OBJECT = 3
    JIT_INPUT_LIBRARY = 4
    JIT_NUM_INPUT_TYPES = 5
end


const CUjitInputType = CUjitInputType_enum
const CUlinkState_st = Cvoid
const CUlinkState = Ptr{CUlinkState_st}

@cenum CUgraphicsRegisterFlags_enum::UInt32 begin
    GRAPHICS_REGISTER_FLAGS_NONE = 0
    GRAPHICS_REGISTER_FLAGS_READ_ONLY = 1
    GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2
    GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 4
    GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 8
end


const CUgraphicsRegisterFlags = CUgraphicsRegisterFlags_enum

@cenum CUgraphicsMapResourceFlags_enum::UInt32 begin
    GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0
    GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 1
    GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 2
end


const CUgraphicsMapResourceFlags = CUgraphicsMapResourceFlags_enum

@cenum CUarray_cubemap_face_enum::UInt32 begin
    CUBEMAP_FACE_POSITIVE_X = 0
    CUBEMAP_FACE_NEGATIVE_X = 1
    CUBEMAP_FACE_POSITIVE_Y = 2
    CUBEMAP_FACE_NEGATIVE_Y = 3
    CUBEMAP_FACE_POSITIVE_Z = 4
    CUBEMAP_FACE_NEGATIVE_Z = 5
end


const CUarray_cubemap_face = CUarray_cubemap_face_enum

@cenum CUlimit_enum::UInt32 begin
    LIMIT_STACK_SIZE = 0
    LIMIT_PRINTF_FIFO_SIZE = 1
    LIMIT_MALLOC_HEAP_SIZE = 2
    LIMIT_DEV_RUNTIME_SYNC_DEPTH = 3
    LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 4
    LIMIT_MAX_L2_FETCH_GRANULARITY = 5
    LIMIT_MAX = 6
end


const CUlimit = CUlimit_enum

@cenum CUresourcetype_enum::UInt32 begin
    RESOURCE_TYPE_ARRAY = 0
    RESOURCE_TYPE_MIPMAPPED_ARRAY = 1
    RESOURCE_TYPE_LINEAR = 2
    RESOURCE_TYPE_PITCH2D = 3
end


const CUresourcetype = CUresourcetype_enum
const CUhostFn = Ptr{Cvoid}

struct CUDA_KERNEL_NODE_PARAMS_st
    func::CUfunction
    gridDimX::UInt32
    gridDimY::UInt32
    gridDimZ::UInt32
    blockDimX::UInt32
    blockDimY::UInt32
    blockDimZ::UInt32
    sharedMemBytes::UInt32
    kernelParams::Ptr{Ptr{Cvoid}}
    extra::Ptr{Ptr{Cvoid}}
end

const CUDA_KERNEL_NODE_PARAMS = CUDA_KERNEL_NODE_PARAMS_st

struct CUDA_MEMSET_NODE_PARAMS_st
    dst::CUdeviceptr
    pitch::Csize_t
    value::UInt32
    elementSize::UInt32
    width::Csize_t
    height::Csize_t
end

const CUDA_MEMSET_NODE_PARAMS = CUDA_MEMSET_NODE_PARAMS_st

struct CUDA_HOST_NODE_PARAMS_st
    fn::CUhostFn
    userData::Ptr{Cvoid}
end

const CUDA_HOST_NODE_PARAMS = CUDA_HOST_NODE_PARAMS_st

@cenum CUgraphNodeType_enum::UInt32 begin
    GRAPH_NODE_TYPE_KERNEL = 0
    GRAPH_NODE_TYPE_MEMCPY = 1
    GRAPH_NODE_TYPE_MEMSET = 2
    GRAPH_NODE_TYPE_HOST = 3
    GRAPH_NODE_TYPE_GRAPH = 4
    GRAPH_NODE_TYPE_EMPTY = 5
    GRAPH_NODE_TYPE_COUNT = 6
end


const CUgraphNodeType = CUgraphNodeType_enum

@cenum CUstreamCaptureStatus_enum::UInt32 begin
    STREAM_CAPTURE_STATUS_NONE = 0
    STREAM_CAPTURE_STATUS_ACTIVE = 1
    STREAM_CAPTURE_STATUS_INVALIDATED = 2
end


const CUstreamCaptureStatus = CUstreamCaptureStatus_enum

@cenum CUstreamCaptureMode_enum::UInt32 begin
    STREAM_CAPTURE_MODE_GLOBAL = 0
    STREAM_CAPTURE_MODE_THREAD_LOCAL = 1
    STREAM_CAPTURE_MODE_RELAXED = 2
end


const CUstreamCaptureMode = CUstreamCaptureMode_enum

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
    CUDA_ERROR_NO_DEVICE = 100
    CUDA_ERROR_INVALID_DEVICE = 101
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
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906
    CUDA_ERROR_CAPTURED_EVENT = 907
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908
    CUDA_ERROR_UNKNOWN = 999
end


const CUresult = cudaError_enum

@cenum CUdevice_P2PAttribute_enum::UInt32 begin
    DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 1
    DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 2
    DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 3
    DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = 4
    DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = 4
end


const CUdevice_P2PAttribute = CUdevice_P2PAttribute_enum
const CUstreamCallback = Ptr{Cvoid}
const CUoccupancyB2DSize = Ptr{Cvoid}

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

const CUDA_MEMCPY2D = CUDA_MEMCPY2D_st

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

const CUDA_MEMCPY3D = CUDA_MEMCPY3D_st

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

const CUDA_MEMCPY3D_PEER = CUDA_MEMCPY3D_PEER_st

struct CUDA_ARRAY_DESCRIPTOR_st
    Width::Csize_t
    Height::Csize_t
    Format::CUarray_format
    NumChannels::UInt32
end

const CUDA_ARRAY_DESCRIPTOR = CUDA_ARRAY_DESCRIPTOR_st

struct CUDA_ARRAY3D_DESCRIPTOR_st
    Width::Csize_t
    Height::Csize_t
    Depth::Csize_t
    Format::CUarray_format
    NumChannels::UInt32
    Flags::UInt32
end

const CUDA_ARRAY3D_DESCRIPTOR = CUDA_ARRAY3D_DESCRIPTOR_st

struct ANONYMOUS2_reserved
    reserved::NTuple{32, Cint}
end

struct ANONYMOUS1_res
    reserved::ANONYMOUS2_reserved
end

struct CUDA_RESOURCE_DESC_st
    resType::CUresourcetype
    res::ANONYMOUS1_res
    flags::UInt32
end

const CUDA_RESOURCE_DESC = CUDA_RESOURCE_DESC_st

struct CUDA_TEXTURE_DESC_st
    addressMode::NTuple{3, CUaddress_mode}
    filterMode::CUfilter_mode
    flags::UInt32
    maxAnisotropy::UInt32
    mipmapFilterMode::CUfilter_mode
    mipmapLevelBias::Cfloat
    minMipmapLevelClamp::Cfloat
    maxMipmapLevelClamp::Cfloat
    borderColor::NTuple{4, Cfloat}
    reserved::NTuple{12, Cint}
end

const CUDA_TEXTURE_DESC = CUDA_TEXTURE_DESC_st

@cenum CUresourceViewFormat_enum::UInt32 begin
    RES_VIEW_FORMAT_NONE = 0
    RES_VIEW_FORMAT_UINT_1X8 = 1
    RES_VIEW_FORMAT_UINT_2X8 = 2
    RES_VIEW_FORMAT_UINT_4X8 = 3
    RES_VIEW_FORMAT_SINT_1X8 = 4
    RES_VIEW_FORMAT_SINT_2X8 = 5
    RES_VIEW_FORMAT_SINT_4X8 = 6
    RES_VIEW_FORMAT_UINT_1X16 = 7
    RES_VIEW_FORMAT_UINT_2X16 = 8
    RES_VIEW_FORMAT_UINT_4X16 = 9
    RES_VIEW_FORMAT_SINT_1X16 = 10
    RES_VIEW_FORMAT_SINT_2X16 = 11
    RES_VIEW_FORMAT_SINT_4X16 = 12
    RES_VIEW_FORMAT_UINT_1X32 = 13
    RES_VIEW_FORMAT_UINT_2X32 = 14
    RES_VIEW_FORMAT_UINT_4X32 = 15
    RES_VIEW_FORMAT_SINT_1X32 = 16
    RES_VIEW_FORMAT_SINT_2X32 = 17
    RES_VIEW_FORMAT_SINT_4X32 = 18
    RES_VIEW_FORMAT_FLOAT_1X16 = 19
    RES_VIEW_FORMAT_FLOAT_2X16 = 20
    RES_VIEW_FORMAT_FLOAT_4X16 = 21
    RES_VIEW_FORMAT_FLOAT_1X32 = 22
    RES_VIEW_FORMAT_FLOAT_2X32 = 23
    RES_VIEW_FORMAT_FLOAT_4X32 = 24
    RES_VIEW_FORMAT_UNSIGNED_BC1 = 25
    RES_VIEW_FORMAT_UNSIGNED_BC2 = 26
    RES_VIEW_FORMAT_UNSIGNED_BC3 = 27
    RES_VIEW_FORMAT_UNSIGNED_BC4 = 28
    RES_VIEW_FORMAT_SIGNED_BC4 = 29
    RES_VIEW_FORMAT_UNSIGNED_BC5 = 30
    RES_VIEW_FORMAT_SIGNED_BC5 = 31
    RES_VIEW_FORMAT_UNSIGNED_BC6H = 32
    RES_VIEW_FORMAT_SIGNED_BC6H = 33
    RES_VIEW_FORMAT_UNSIGNED_BC7 = 34
end


const CUresourceViewFormat = CUresourceViewFormat_enum

struct CUDA_RESOURCE_VIEW_DESC_st
    format::CUresourceViewFormat
    width::Csize_t
    height::Csize_t
    depth::Csize_t
    firstMipmapLevel::UInt32
    lastMipmapLevel::UInt32
    firstLayer::UInt32
    lastLayer::UInt32
    reserved::NTuple{16, UInt32}
end

const CUDA_RESOURCE_VIEW_DESC = CUDA_RESOURCE_VIEW_DESC_st

struct CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st
    p2pToken::Culonglong
    vaSpaceToken::UInt32
end

const CUDA_POINTER_ATTRIBUTE_P2P_TOKENS = CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st

struct CUDA_LAUNCH_PARAMS_st
    _function::CUfunction
    gridDimX::UInt32
    gridDimY::UInt32
    gridDimZ::UInt32
    blockDimX::UInt32
    blockDimY::UInt32
    blockDimZ::UInt32
    sharedMemBytes::UInt32
    hStream::CUstream
    kernelParams::Ptr{Ptr{Cvoid}}
end

const CUDA_LAUNCH_PARAMS = CUDA_LAUNCH_PARAMS_st

@cenum CUexternalMemoryHandleType_enum::UInt32 begin
    EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1
    EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2
    EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3
    EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = 4
    EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5
end


const CUexternalMemoryHandleType = CUexternalMemoryHandleType_enum

struct ANONYMOUS4_win32
    handle::Ptr{Cvoid}
    name::Ptr{Cvoid}
end

struct ANONYMOUS3_handle
    win32::ANONYMOUS4_win32
end

struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
    type::CUexternalMemoryHandleType
    handle::ANONYMOUS3_handle
    size::Culonglong
    flags::UInt32
    reserved::NTuple{16, UInt32}
end

const CUDA_EXTERNAL_MEMORY_HANDLE_DESC = CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st

struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
    offset::Culonglong
    size::Culonglong
    flags::UInt32
    reserved::NTuple{16, UInt32}
end

const CUDA_EXTERNAL_MEMORY_BUFFER_DESC = CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st

struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st
    offset::Culonglong
    arrayDesc::CUDA_ARRAY3D_DESCRIPTOR
    numLevels::UInt32
    reserved::NTuple{16, UInt32}
end

const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC = CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st

@cenum CUexternalSemaphoreHandleType_enum::UInt32 begin
    EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = 1
    EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = 2
    EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3
    EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = 4
end


const CUexternalSemaphoreHandleType = CUexternalSemaphoreHandleType_enum

struct ANONYMOUS6_win32
    handle::Ptr{Cvoid}
    name::Ptr{Cvoid}
end

struct ANONYMOUS5_handle
    win32::ANONYMOUS6_win32
end

struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
    type::CUexternalSemaphoreHandleType
    handle::ANONYMOUS5_handle
    flags::UInt32
    reserved::NTuple{16, UInt32}
end

const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st

struct ANONYMOUS8_fence
    value::Culonglong
end

struct ANONYMOUS7_params
    fence::ANONYMOUS8_fence
    reserved::NTuple{16, UInt32}
end

struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
    params::ANONYMOUS7_params
    flags::UInt32
    reserved::NTuple{16, UInt32}
end

const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS = CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st

struct ANONYMOUS10_fence
    value::Culonglong
end

struct ANONYMOUS9_params
    fence::ANONYMOUS10_fence
    reserved::NTuple{16, UInt32}
end

struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
    params::ANONYMOUS9_params
    flags::UInt32
    reserved::NTuple{16, UInt32}
end

const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st

@cenum CUoutput_mode_enum::UInt32 begin
    OUT_KEY_VALUE_PAIR = 0
    OUT_CSV = 1
end


const CUoutput_mode = CUoutput_mode_enum

