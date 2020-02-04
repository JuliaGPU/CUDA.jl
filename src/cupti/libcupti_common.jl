# Automatically generated using Clang.jl


@cenum CUptiResult::UInt32 begin
    CUPTI_SUCCESS = 0
    CUPTI_ERROR_INVALID_PARAMETER = 1
    CUPTI_ERROR_INVALID_DEVICE = 2
    CUPTI_ERROR_INVALID_CONTEXT = 3
    CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID = 4
    CUPTI_ERROR_INVALID_EVENT_ID = 5
    CUPTI_ERROR_INVALID_EVENT_NAME = 6
    CUPTI_ERROR_INVALID_OPERATION = 7
    CUPTI_ERROR_OUT_OF_MEMORY = 8
    CUPTI_ERROR_HARDWARE = 9
    CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT = 10
    CUPTI_ERROR_API_NOT_IMPLEMENTED = 11
    CUPTI_ERROR_MAX_LIMIT_REACHED = 12
    CUPTI_ERROR_NOT_READY = 13
    CUPTI_ERROR_NOT_COMPATIBLE = 14
    CUPTI_ERROR_NOT_INITIALIZED = 15
    CUPTI_ERROR_INVALID_METRIC_ID = 16
    CUPTI_ERROR_INVALID_METRIC_NAME = 17
    CUPTI_ERROR_QUEUE_EMPTY = 18
    CUPTI_ERROR_INVALID_HANDLE = 19
    CUPTI_ERROR_INVALID_STREAM = 20
    CUPTI_ERROR_INVALID_KIND = 21
    CUPTI_ERROR_INVALID_EVENT_VALUE = 22
    CUPTI_ERROR_DISABLED = 23
    CUPTI_ERROR_INVALID_MODULE = 24
    CUPTI_ERROR_INVALID_METRIC_VALUE = 25
    CUPTI_ERROR_HARDWARE_BUSY = 26
    CUPTI_ERROR_NOT_SUPPORTED = 27
    CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED = 28
    CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE = 29
    CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES = 30
    CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_WITH_MPS = 31
    CUPTI_ERROR_CDP_TRACING_NOT_SUPPORTED = 32
    CUPTI_ERROR_VIRTUALIZED_DEVICE_NOT_SUPPORTED = 33
    CUPTI_ERROR_CUDA_COMPILER_NOT_COMPATIBLE = 34
    CUPTI_ERROR_INSUFFICIENT_PRIVILEGES = 35
    CUPTI_ERROR_OLD_PROFILER_API_INITIALIZED = 36
    CUPTI_ERROR_OPENACC_UNDEFINED_ROUTINE = 37
    CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED = 38
    CUPTI_ERROR_UNKNOWN = 999
    CUPTI_ERROR_FORCE_INT = 2147483647
end


const CUPTI_API_VERSION = 12
const CUPTILP64 = 1
const ACTIVITY_RECORD_ALIGNMENT = 8

# Skipping MacroDefinition: PACKED_ALIGNMENT __attribute__ ( ( __packed__ ) ) __attribute__ ( ( aligned ( ACTIVITY_RECORD_ALIGNMENT ) ) )
# Skipping MacroDefinition: CUPTI_UNIFIED_MEMORY_CPU_DEVICE_ID ( ( uint32_t ) 0xFFFFFFFFU )

const CUPTI_SOURCE_LOCATOR_ID_UNKNOWN = 0
const CUPTI_CORRELATION_ID_UNKNOWN = 0
const CUPTI_GRID_ID_UNKNOWN = Int64(0)
const CUPTI_TIMESTAMP_UNKNOWN = Int64(0)
const CUPTI_SYNCHRONIZATION_INVALID_VALUE = -1
const CUPTI_AUTO_BOOST_INVALID_CLIENT_PID = 0
const CUPTI_NVLINK_INVALID_PORT = -1
const CUPTI_MAX_NVLINK_PORTS = 16
const CUPTI_MAX_GPUS = 32

@cenum CUpti_ActivityKind::UInt32 begin
    CUPTI_ACTIVITY_KIND_INVALID = 0
    CUPTI_ACTIVITY_KIND_MEMCPY = 1
    CUPTI_ACTIVITY_KIND_MEMSET = 2
    CUPTI_ACTIVITY_KIND_KERNEL = 3
    CUPTI_ACTIVITY_KIND_DRIVER = 4
    CUPTI_ACTIVITY_KIND_RUNTIME = 5
    CUPTI_ACTIVITY_KIND_EVENT = 6
    CUPTI_ACTIVITY_KIND_METRIC = 7
    CUPTI_ACTIVITY_KIND_DEVICE = 8
    CUPTI_ACTIVITY_KIND_CONTEXT = 9
    CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10
    CUPTI_ACTIVITY_KIND_NAME = 11
    CUPTI_ACTIVITY_KIND_MARKER = 12
    CUPTI_ACTIVITY_KIND_MARKER_DATA = 13
    CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR = 14
    CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS = 15
    CUPTI_ACTIVITY_KIND_BRANCH = 16
    CUPTI_ACTIVITY_KIND_OVERHEAD = 17
    CUPTI_ACTIVITY_KIND_CDP_KERNEL = 18
    CUPTI_ACTIVITY_KIND_PREEMPTION = 19
    CUPTI_ACTIVITY_KIND_ENVIRONMENT = 20
    CUPTI_ACTIVITY_KIND_EVENT_INSTANCE = 21
    CUPTI_ACTIVITY_KIND_MEMCPY2 = 22
    CUPTI_ACTIVITY_KIND_METRIC_INSTANCE = 23
    CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION = 24
    CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER = 25
    CUPTI_ACTIVITY_KIND_FUNCTION = 26
    CUPTI_ACTIVITY_KIND_MODULE = 27
    CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE = 28
    CUPTI_ACTIVITY_KIND_SHARED_ACCESS = 29
    CUPTI_ACTIVITY_KIND_PC_SAMPLING = 30
    CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO = 31
    CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION = 32
    CUPTI_ACTIVITY_KIND_OPENACC_DATA = 33
    CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH = 34
    CUPTI_ACTIVITY_KIND_OPENACC_OTHER = 35
    CUPTI_ACTIVITY_KIND_CUDA_EVENT = 36
    CUPTI_ACTIVITY_KIND_STREAM = 37
    CUPTI_ACTIVITY_KIND_SYNCHRONIZATION = 38
    CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION = 39
    CUPTI_ACTIVITY_KIND_NVLINK = 40
    CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT = 41
    CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE = 42
    CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC = 43
    CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE = 44
    CUPTI_ACTIVITY_KIND_MEMORY = 45
    CUPTI_ACTIVITY_KIND_PCIE = 46
    CUPTI_ACTIVITY_KIND_OPENMP = 47
    CUPTI_ACTIVITY_KIND_COUNT = 48
    CUPTI_ACTIVITY_KIND_FORCE_INT = 2147483647
end

@cenum CUpti_ActivityObjectKind::UInt32 begin
    CUPTI_ACTIVITY_OBJECT_UNKNOWN = 0
    CUPTI_ACTIVITY_OBJECT_PROCESS = 1
    CUPTI_ACTIVITY_OBJECT_THREAD = 2
    CUPTI_ACTIVITY_OBJECT_DEVICE = 3
    CUPTI_ACTIVITY_OBJECT_CONTEXT = 4
    CUPTI_ACTIVITY_OBJECT_STREAM = 5
    CUPTI_ACTIVITY_OBJECT_FORCE_INT = 2147483647
end


struct ANONYMOUS1_dcs
    deviceId::UInt32
    contextId::UInt32
    streamId::UInt32
end

struct CUpti_ActivityObjectKindId
    dcs::ANONYMOUS1_dcs
end

@cenum CUpti_ActivityOverheadKind::UInt32 begin
    CUPTI_ACTIVITY_OVERHEAD_UNKNOWN = 0
    CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER = 1
    CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH = 65536
    CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION = 131072
    CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE = 196608
    CUPTI_ACTIVITY_OVERHEAD_FORCE_INT = 2147483647
end

@cenum CUpti_ActivityComputeApiKind::UInt32 begin
    CUPTI_ACTIVITY_COMPUTE_API_UNKNOWN = 0
    CUPTI_ACTIVITY_COMPUTE_API_CUDA = 1
    CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS = 2
    CUPTI_ACTIVITY_COMPUTE_API_FORCE_INT = 2147483647
end

@cenum CUpti_ActivityFlag::UInt32 begin
    CUPTI_ACTIVITY_FLAG_NONE = 0
    CUPTI_ACTIVITY_FLAG_DEVICE_CONCURRENT_KERNELS = 1
    CUPTI_ACTIVITY_FLAG_DEVICE_ATTRIBUTE_CUDEVICE = 1
    CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC = 1
    CUPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS = 1
    CUPTI_ACTIVITY_FLAG_MARKER_START = 2
    CUPTI_ACTIVITY_FLAG_MARKER_END = 4
    CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE = 8
    CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_SUCCESS = 16
    CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_FAILED = 32
    CUPTI_ACTIVITY_FLAG_MARKER_SYNC_RELEASE = 64
    CUPTI_ACTIVITY_FLAG_MARKER_COLOR_NONE = 1
    CUPTI_ACTIVITY_FLAG_MARKER_COLOR_ARGB = 2
    CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_SIZE_MASK = 255
    CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_LOAD = 256
    CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_CACHED = 512
    CUPTI_ACTIVITY_FLAG_METRIC_OVERFLOWED = 1
    CUPTI_ACTIVITY_FLAG_METRIC_VALUE_INVALID = 2
    CUPTI_ACTIVITY_FLAG_INSTRUCTION_VALUE_INVALID = 1
    CUPTI_ACTIVITY_FLAG_INSTRUCTION_CLASS_MASK = 510
    CUPTI_ACTIVITY_FLAG_FLUSH_FORCED = 1
    CUPTI_ACTIVITY_FLAG_SHARED_ACCESS_KIND_SIZE_MASK = 255
    CUPTI_ACTIVITY_FLAG_SHARED_ACCESS_KIND_LOAD = 256
    CUPTI_ACTIVITY_FLAG_MEMSET_ASYNC = 1
    CUPTI_ACTIVITY_FLAG_THRASHING_IN_CPU = 1
    CUPTI_ACTIVITY_FLAG_THROTTLING_IN_CPU = 1
    CUPTI_ACTIVITY_FLAG_FORCE_INT = 2147483647
end

@cenum CUpti_ActivityPCSamplingStallReason::UInt32 begin
    CUPTI_ACTIVITY_PC_SAMPLING_STALL_INVALID = 0
    CUPTI_ACTIVITY_PC_SAMPLING_STALL_NONE = 1
    CUPTI_ACTIVITY_PC_SAMPLING_STALL_INST_FETCH = 2
    CUPTI_ACTIVITY_PC_SAMPLING_STALL_EXEC_DEPENDENCY = 3
    CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_DEPENDENCY = 4
    CUPTI_ACTIVITY_PC_SAMPLING_STALL_TEXTURE = 5
    CUPTI_ACTIVITY_PC_SAMPLING_STALL_SYNC = 6
    CUPTI_ACTIVITY_PC_SAMPLING_STALL_CONSTANT_MEMORY_DEPENDENCY = 7
    CUPTI_ACTIVITY_PC_SAMPLING_STALL_PIPE_BUSY = 8
    CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_THROTTLE = 9
    CUPTI_ACTIVITY_PC_SAMPLING_STALL_NOT_SELECTED = 10
    CUPTI_ACTIVITY_PC_SAMPLING_STALL_OTHER = 11
    CUPTI_ACTIVITY_PC_SAMPLING_STALL_SLEEPING = 12
    CUPTI_ACTIVITY_PC_SAMPLING_STALL_FORCE_INT = 2147483647
end

@cenum CUpti_ActivityPCSamplingPeriod::UInt32 begin
    CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_INVALID = 0
    CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN = 1
    CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_LOW = 2
    CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MID = 3
    CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_HIGH = 4
    CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MAX = 5
    CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_FORCE_INT = 2147483647
end

@cenum CUpti_ActivityMemcpyKind::UInt32 begin
    CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN = 0
    CUPTI_ACTIVITY_MEMCPY_KIND_HTOD = 1
    CUPTI_ACTIVITY_MEMCPY_KIND_DTOH = 2
    CUPTI_ACTIVITY_MEMCPY_KIND_HTOA = 3
    CUPTI_ACTIVITY_MEMCPY_KIND_ATOH = 4
    CUPTI_ACTIVITY_MEMCPY_KIND_ATOA = 5
    CUPTI_ACTIVITY_MEMCPY_KIND_ATOD = 6
    CUPTI_ACTIVITY_MEMCPY_KIND_DTOA = 7
    CUPTI_ACTIVITY_MEMCPY_KIND_DTOD = 8
    CUPTI_ACTIVITY_MEMCPY_KIND_HTOH = 9
    CUPTI_ACTIVITY_MEMCPY_KIND_PTOP = 10
    CUPTI_ACTIVITY_MEMCPY_KIND_FORCE_INT = 2147483647
end

@cenum CUpti_ActivityMemoryKind::UInt32 begin
    CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN = 0
    CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE = 1
    CUPTI_ACTIVITY_MEMORY_KIND_PINNED = 2
    CUPTI_ACTIVITY_MEMORY_KIND_DEVICE = 3
    CUPTI_ACTIVITY_MEMORY_KIND_ARRAY = 4
    CUPTI_ACTIVITY_MEMORY_KIND_MANAGED = 5
    CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC = 6
    CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC = 7
    CUPTI_ACTIVITY_MEMORY_KIND_FORCE_INT = 2147483647
end

@cenum CUpti_ActivityPreemptionKind::UInt32 begin
    CUPTI_ACTIVITY_PREEMPTION_KIND_UNKNOWN = 0
    CUPTI_ACTIVITY_PREEMPTION_KIND_SAVE = 1
    CUPTI_ACTIVITY_PREEMPTION_KIND_RESTORE = 2
    CUPTI_ACTIVITY_PREEMPTION_KIND_FORCE_INT = 2147483647
end

@cenum CUpti_ActivityEnvironmentKind::UInt32 begin
    CUPTI_ACTIVITY_ENVIRONMENT_UNKNOWN = 0
    CUPTI_ACTIVITY_ENVIRONMENT_SPEED = 1
    CUPTI_ACTIVITY_ENVIRONMENT_TEMPERATURE = 2
    CUPTI_ACTIVITY_ENVIRONMENT_POWER = 3
    CUPTI_ACTIVITY_ENVIRONMENT_COOLING = 4
    CUPTI_ACTIVITY_ENVIRONMENT_COUNT = 5
    CUPTI_ACTIVITY_ENVIRONMENT_KIND_FORCE_INT = 2147483647
end

@cenum CUpti_EnvironmentClocksThrottleReason::UInt32 begin
    CUPTI_CLOCKS_THROTTLE_REASON_GPU_IDLE = 1
    CUPTI_CLOCKS_THROTTLE_REASON_USER_DEFINED_CLOCKS = 2
    CUPTI_CLOCKS_THROTTLE_REASON_SW_POWER_CAP = 4
    CUPTI_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN = 8
    CUPTI_CLOCKS_THROTTLE_REASON_UNKNOWN = 2147483648
    CUPTI_CLOCKS_THROTTLE_REASON_UNSUPPORTED = 1073741824
    CUPTI_CLOCKS_THROTTLE_REASON_NONE = 0
    CUPTI_CLOCKS_THROTTLE_REASON_FORCE_INT = 2147483647
end

@cenum CUpti_ActivityUnifiedMemoryCounterScope::UInt32 begin
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_UNKNOWN = 0
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE = 1
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_ALL_DEVICES = 2
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_COUNT = 3
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_FORCE_INT = 2147483647
end

@cenum CUpti_ActivityUnifiedMemoryCounterKind::UInt32 begin
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_UNKNOWN = 0
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD = 1
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH = 2
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT = 3
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT = 4
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING = 5
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING = 6
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP = 7
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD = 8
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_COUNT = 9
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_FORCE_INT = 2147483647
end

@cenum CUpti_ActivityUnifiedMemoryAccessType::UInt32 begin
    CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_UNKNOWN = 0
    CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_READ = 1
    CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_WRITE = 2
    CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_ATOMIC = 3
    CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_PREFETCH = 4
end

@cenum CUpti_ActivityUnifiedMemoryMigrationCause::UInt32 begin
    CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_UNKNOWN = 0
    CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_USER = 1
    CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_COHERENCE = 2
    CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_PREFETCH = 3
    CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_EVICTION = 4
    CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_ACCESS_COUNTERS = 5
end

@cenum CUpti_ActivityUnifiedMemoryRemoteMapCause::UInt32 begin
    CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_UNKNOWN = 0
    CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_COHERENCE = 1
    CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_THRASHING = 2
    CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_POLICY = 3
    CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_OUT_OF_MEMORY = 4
    CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_EVICTION = 5
end

@cenum CUpti_ActivityInstructionClass::UInt32 begin
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_UNKNOWN = 0
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_32 = 1
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_64 = 2
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_INTEGER = 3
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_BIT_CONVERSION = 4
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_CONTROL_FLOW = 5
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_GLOBAL = 6
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_SHARED = 7
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_LOCAL = 8
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_GENERIC = 9
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_SURFACE = 10
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_CONSTANT = 11
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_TEXTURE = 12
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_GLOBAL_ATOMIC = 13
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_SHARED_ATOMIC = 14
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_SURFACE_ATOMIC = 15
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_INTER_THREAD_COMMUNICATION = 16
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_BARRIER = 17
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_MISCELLANEOUS = 18
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_16 = 19
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_UNIFORM = 20
    CUPTI_ACTIVITY_INSTRUCTION_CLASS_KIND_FORCE_INT = 2147483647
end

@cenum CUpti_ActivityPartitionedGlobalCacheConfig::UInt32 begin
    CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_UNKNOWN = 0
    CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_NOT_SUPPORTED = 1
    CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_OFF = 2
    CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_ON = 3
    CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_FORCE_INT = 2147483647
end

@cenum CUpti_ActivitySynchronizationType::UInt32 begin
    CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_UNKNOWN = 0
    CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE = 1
    CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT = 2
    CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE = 3
    CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE = 4
    CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_FORCE_INT = 2147483647
end

@cenum CUpti_ActivityStreamFlag::UInt32 begin
    CUPTI_ACTIVITY_STREAM_CREATE_FLAG_UNKNOWN = 0
    CUPTI_ACTIVITY_STREAM_CREATE_FLAG_DEFAULT = 1
    CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NON_BLOCKING = 2
    CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NULL = 3
    CUPTI_ACTIVITY_STREAM_CREATE_MASK = 65535
    CUPTI_ACTIVITY_STREAM_CREATE_FLAG_FORCE_INT = 2147483647
end

@cenum CUpti_LinkFlag::UInt32 begin
    CUPTI_LINK_FLAG_INVALID = 0
    CUPTI_LINK_FLAG_PEER_ACCESS = 2
    CUPTI_LINK_FLAG_SYSMEM_ACCESS = 4
    CUPTI_LINK_FLAG_PEER_ATOMICS = 8
    CUPTI_LINK_FLAG_SYSMEM_ATOMICS = 16
    CUPTI_LINK_FLAG_FORCE_INT = 2147483647
end

@cenum CUpti_DeviceSupport::UInt32 begin
    CUPTI_DEVICE_UNSUPPORTED = 0
    CUPTI_DEVICE_SUPPORTED = 1
    CUPTI_DEVICE_VIRTUAL = 2
end


struct CUpti_ActivityUnifiedMemoryCounterConfig
    scope::CUpti_ActivityUnifiedMemoryCounterScope
    kind::CUpti_ActivityUnifiedMemoryCounterKind
    deviceId::UInt32
    enable::UInt32
end

struct CUpti_ActivityAutoBoostState
    enabled::UInt32
    pid::UInt32
end

struct CUpti_ActivityPCSamplingConfig
    size::UInt32
    samplingPeriod::CUpti_ActivityPCSamplingPeriod
    samplingPeriod2::UInt32
end

struct CUpti_Activity
    kind::CUpti_ActivityKind
end

struct CUpti_ActivityMemcpy
    kind::CUpti_ActivityKind
    copyKind::UInt8
    srcKind::UInt8
    dstKind::UInt8
    flags::UInt8
    bytes::UInt64
    start::UInt64
    _end::UInt64
    deviceId::UInt32
    contextId::UInt32
    streamId::UInt32
    correlationId::UInt32
    runtimeCorrelationId::UInt32
    pad::UInt32
    reserved0::Ptr{Cvoid}
end

struct CUpti_ActivityMemcpy2
    kind::CUpti_ActivityKind
    copyKind::UInt8
    srcKind::UInt8
    dstKind::UInt8
    flags::UInt8
    bytes::UInt64
    start::UInt64
    _end::UInt64
    deviceId::UInt32
    contextId::UInt32
    streamId::UInt32
    srcDeviceId::UInt32
    srcContextId::UInt32
    dstDeviceId::UInt32
    dstContextId::UInt32
    correlationId::UInt32
    reserved0::Ptr{Cvoid}
end

struct CUpti_ActivityMemset
    kind::CUpti_ActivityKind
    value::UInt32
    bytes::UInt64
    start::UInt64
    _end::UInt64
    deviceId::UInt32
    contextId::UInt32
    streamId::UInt32
    correlationId::UInt32
    flags::UInt16
    memoryKind::UInt16
    pad::UInt32
    reserved0::Ptr{Cvoid}
end

struct CUpti_ActivityMemory
    kind::CUpti_ActivityKind
    memoryKind::CUpti_ActivityMemoryKind
    address::UInt64
    bytes::UInt64
    start::UInt64
    _end::UInt64
    allocPC::UInt64
    freePC::UInt64
    processId::UInt32
    deviceId::UInt32
    contextId::UInt32
    pad::UInt32
    name::Cstring
end

struct CUpti_ActivityKernel
    kind::CUpti_ActivityKind
    cacheConfigRequested::UInt8
    cacheConfigExecuted::UInt8
    registersPerThread::UInt16
    start::UInt64
    _end::UInt64
    deviceId::UInt32
    contextId::UInt32
    streamId::UInt32
    gridX::Int32
    gridY::Int32
    gridZ::Int32
    blockX::Int32
    blockY::Int32
    blockZ::Int32
    staticSharedMemory::Int32
    dynamicSharedMemory::Int32
    localMemoryPerThread::UInt32
    localMemoryTotal::UInt32
    correlationId::UInt32
    runtimeCorrelationId::UInt32
    pad::UInt32
    name::Cstring
    reserved0::Ptr{Cvoid}
end

struct ANONYMOUS2_cacheConfig
    both::UInt8
end

struct CUpti_ActivityKernel2
    kind::CUpti_ActivityKind
    cacheConfig::ANONYMOUS2_cacheConfig
    sharedMemoryConfig::UInt8
    registersPerThread::UInt16
    start::UInt64
    _end::UInt64
    completed::UInt64
    deviceId::UInt32
    contextId::UInt32
    streamId::UInt32
    gridX::Int32
    gridY::Int32
    gridZ::Int32
    blockX::Int32
    blockY::Int32
    blockZ::Int32
    staticSharedMemory::Int32
    dynamicSharedMemory::Int32
    localMemoryPerThread::UInt32
    localMemoryTotal::UInt32
    correlationId::UInt32
    gridId::Int64
    name::Cstring
    reserved0::Ptr{Cvoid}
end

struct ANONYMOUS3_cacheConfig
    both::UInt8
end

struct CUpti_ActivityKernel3
    kind::CUpti_ActivityKind
    cacheConfig::ANONYMOUS3_cacheConfig
    sharedMemoryConfig::UInt8
    registersPerThread::UInt16
    partitionedGlobalCacheRequested::CUpti_ActivityPartitionedGlobalCacheConfig
    partitionedGlobalCacheExecuted::CUpti_ActivityPartitionedGlobalCacheConfig
    start::UInt64
    _end::UInt64
    completed::UInt64
    deviceId::UInt32
    contextId::UInt32
    streamId::UInt32
    gridX::Int32
    gridY::Int32
    gridZ::Int32
    blockX::Int32
    blockY::Int32
    blockZ::Int32
    staticSharedMemory::Int32
    dynamicSharedMemory::Int32
    localMemoryPerThread::UInt32
    localMemoryTotal::UInt32
    correlationId::UInt32
    gridId::Int64
    name::Cstring
    reserved0::Ptr{Cvoid}
end

@cenum CUpti_ActivityLaunchType::UInt32 begin
    CUPTI_ACTIVITY_LAUNCH_TYPE_REGULAR = 0
    CUPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_SINGLE_DEVICE = 1
    CUPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_MULTI_DEVICE = 2
end


struct ANONYMOUS4_cacheConfig
    both::UInt8
end

struct CUpti_ActivityKernel4
    kind::CUpti_ActivityKind
    cacheConfig::ANONYMOUS4_cacheConfig
    sharedMemoryConfig::UInt8
    registersPerThread::UInt16
    partitionedGlobalCacheRequested::CUpti_ActivityPartitionedGlobalCacheConfig
    partitionedGlobalCacheExecuted::CUpti_ActivityPartitionedGlobalCacheConfig
    start::UInt64
    _end::UInt64
    completed::UInt64
    deviceId::UInt32
    contextId::UInt32
    streamId::UInt32
    gridX::Int32
    gridY::Int32
    gridZ::Int32
    blockX::Int32
    blockY::Int32
    blockZ::Int32
    staticSharedMemory::Int32
    dynamicSharedMemory::Int32
    localMemoryPerThread::UInt32
    localMemoryTotal::UInt32
    correlationId::UInt32
    gridId::Int64
    name::Cstring
    reserved0::Ptr{Cvoid}
    queued::UInt64
    submitted::UInt64
    launchType::UInt8
    isSharedMemoryCarveoutRequested::UInt8
    sharedMemoryCarveoutRequested::UInt8
    padding::UInt8
    sharedMemoryExecuted::UInt32
end

struct ANONYMOUS5_cacheConfig
    both::UInt8
end

struct CUpti_ActivityCdpKernel
    kind::CUpti_ActivityKind
    cacheConfig::ANONYMOUS5_cacheConfig
    sharedMemoryConfig::UInt8
    registersPerThread::UInt16
    start::UInt64
    _end::UInt64
    deviceId::UInt32
    contextId::UInt32
    streamId::UInt32
    gridX::Int32
    gridY::Int32
    gridZ::Int32
    blockX::Int32
    blockY::Int32
    blockZ::Int32
    staticSharedMemory::Int32
    dynamicSharedMemory::Int32
    localMemoryPerThread::UInt32
    localMemoryTotal::UInt32
    correlationId::UInt32
    gridId::Int64
    parentGridId::Int64
    queued::UInt64
    submitted::UInt64
    completed::UInt64
    parentBlockX::UInt32
    parentBlockY::UInt32
    parentBlockZ::UInt32
    pad::UInt32
    name::Cstring
end

struct CUpti_ActivityPreemption
    kind::CUpti_ActivityKind
    preemptionKind::CUpti_ActivityPreemptionKind
    timestamp::UInt64
    gridId::Int64
    blockX::UInt32
    blockY::UInt32
    blockZ::UInt32
    pad::UInt32
end

const CUpti_CallbackId = UInt32

struct CUpti_ActivityAPI
    kind::CUpti_ActivityKind
    cbid::CUpti_CallbackId
    start::UInt64
    _end::UInt64
    processId::UInt32
    threadId::UInt32
    correlationId::UInt32
    returnValue::UInt32
end

const CUpti_EventID = UInt32
const CUpti_EventDomainID = UInt32

struct CUpti_ActivityEvent
    kind::CUpti_ActivityKind
    id::CUpti_EventID
    value::UInt64
    domain::CUpti_EventDomainID
    correlationId::UInt32
end

struct CUpti_ActivityEventInstance
    kind::CUpti_ActivityKind
    id::CUpti_EventID
    domain::CUpti_EventDomainID
    instance::UInt32
    value::UInt64
    correlationId::UInt32
    pad::UInt32
end

const CUpti_MetricID = UInt32

struct CUpti_MetricValue
    metricValueDouble::Cdouble
end

struct CUpti_ActivityMetric
    kind::CUpti_ActivityKind
    id::CUpti_MetricID
    value::CUpti_MetricValue
    correlationId::UInt32
    flags::UInt8
    pad::NTuple{3, UInt8}
end

struct CUpti_ActivityMetricInstance
    kind::CUpti_ActivityKind
    id::CUpti_MetricID
    value::CUpti_MetricValue
    instance::UInt32
    correlationId::UInt32
    flags::UInt8
    pad::NTuple{7, UInt8}
end

struct CUpti_ActivitySourceLocator
    kind::CUpti_ActivityKind
    id::UInt32
    lineNumber::UInt32
    pad::UInt32
    fileName::Cstring
end

struct CUpti_ActivityGlobalAccess
    kind::CUpti_ActivityKind
    flags::CUpti_ActivityFlag
    sourceLocatorId::UInt32
    correlationId::UInt32
    pcOffset::UInt32
    executed::UInt32
    threadsExecuted::UInt64
    l2_transactions::UInt64
end

struct CUpti_ActivityGlobalAccess2
    kind::CUpti_ActivityKind
    flags::CUpti_ActivityFlag
    sourceLocatorId::UInt32
    correlationId::UInt32
    functionId::UInt32
    pcOffset::UInt32
    threadsExecuted::UInt64
    l2_transactions::UInt64
    theoreticalL2Transactions::UInt64
    executed::UInt32
    pad::UInt32
end

struct CUpti_ActivityGlobalAccess3
    kind::CUpti_ActivityKind
    flags::CUpti_ActivityFlag
    sourceLocatorId::UInt32
    correlationId::UInt32
    functionId::UInt32
    executed::UInt32
    pcOffset::UInt64
    threadsExecuted::UInt64
    l2_transactions::UInt64
    theoreticalL2Transactions::UInt64
end

struct CUpti_ActivityBranch
    kind::CUpti_ActivityKind
    sourceLocatorId::UInt32
    correlationId::UInt32
    pcOffset::UInt32
    executed::UInt32
    diverged::UInt32
    threadsExecuted::UInt64
end

struct CUpti_ActivityBranch2
    kind::CUpti_ActivityKind
    sourceLocatorId::UInt32
    correlationId::UInt32
    functionId::UInt32
    pcOffset::UInt32
    diverged::UInt32
    threadsExecuted::UInt64
    executed::UInt32
    pad::UInt32
end

struct CUpti_ActivityDevice
    kind::CUpti_ActivityKind
    flags::CUpti_ActivityFlag
    globalMemoryBandwidth::UInt64
    globalMemorySize::UInt64
    constantMemorySize::UInt32
    l2CacheSize::UInt32
    numThreadsPerWarp::UInt32
    coreClockRate::UInt32
    numMemcpyEngines::UInt32
    numMultiprocessors::UInt32
    maxIPC::UInt32
    maxWarpsPerMultiprocessor::UInt32
    maxBlocksPerMultiprocessor::UInt32
    maxRegistersPerBlock::UInt32
    maxSharedMemoryPerBlock::UInt32
    maxThreadsPerBlock::UInt32
    maxBlockDimX::UInt32
    maxBlockDimY::UInt32
    maxBlockDimZ::UInt32
    maxGridDimX::UInt32
    maxGridDimY::UInt32
    maxGridDimZ::UInt32
    computeCapabilityMajor::UInt32
    computeCapabilityMinor::UInt32
    id::UInt32
    pad::UInt32
    name::Cstring
end

struct CUpti_ActivityDevice2
    kind::CUpti_ActivityKind
    flags::CUpti_ActivityFlag
    globalMemoryBandwidth::UInt64
    globalMemorySize::UInt64
    constantMemorySize::UInt32
    l2CacheSize::UInt32
    numThreadsPerWarp::UInt32
    coreClockRate::UInt32
    numMemcpyEngines::UInt32
    numMultiprocessors::UInt32
    maxIPC::UInt32
    maxWarpsPerMultiprocessor::UInt32
    maxBlocksPerMultiprocessor::UInt32
    maxSharedMemoryPerMultiprocessor::UInt32
    maxRegistersPerMultiprocessor::UInt32
    maxRegistersPerBlock::UInt32
    maxSharedMemoryPerBlock::UInt32
    maxThreadsPerBlock::UInt32
    maxBlockDimX::UInt32
    maxBlockDimY::UInt32
    maxBlockDimZ::UInt32
    maxGridDimX::UInt32
    maxGridDimY::UInt32
    maxGridDimZ::UInt32
    computeCapabilityMajor::UInt32
    computeCapabilityMinor::UInt32
    id::UInt32
    eccEnabled::UInt32
    uuid::CUuuid
    name::Cstring
end

struct ANONYMOUS6_attribute
    cu::CUdevice_attribute
end

struct ANONYMOUS7_value
    vDouble::Cdouble
end

struct CUpti_ActivityDeviceAttribute
    kind::CUpti_ActivityKind
    flags::CUpti_ActivityFlag
    deviceId::UInt32
    attribute::ANONYMOUS6_attribute
    value::ANONYMOUS7_value
end

struct CUpti_ActivityContext
    kind::CUpti_ActivityKind
    contextId::UInt32
    deviceId::UInt32
    computeApiKind::UInt16
    nullStreamId::UInt16
end

struct CUpti_ActivityName
    kind::CUpti_ActivityKind
    objectKind::CUpti_ActivityObjectKind
    objectId::CUpti_ActivityObjectKindId
    pad::UInt32
    name::Cstring
end

struct CUpti_ActivityMarker
    kind::CUpti_ActivityKind
    flags::CUpti_ActivityFlag
    timestamp::UInt64
    id::UInt32
    objectKind::CUpti_ActivityObjectKind
    objectId::CUpti_ActivityObjectKindId
    pad::UInt32
    name::Cstring
end

struct CUpti_ActivityMarker2
    kind::CUpti_ActivityKind
    flags::CUpti_ActivityFlag
    timestamp::UInt64
    id::UInt32
    objectKind::CUpti_ActivityObjectKind
    objectId::CUpti_ActivityObjectKindId
    pad::UInt32
    name::Cstring
    domain::Cstring
end

@cenum CUpti_MetricValueKind::UInt32 begin
    CUPTI_METRIC_VALUE_KIND_DOUBLE = 0
    CUPTI_METRIC_VALUE_KIND_UINT64 = 1
    CUPTI_METRIC_VALUE_KIND_PERCENT = 2
    CUPTI_METRIC_VALUE_KIND_THROUGHPUT = 3
    CUPTI_METRIC_VALUE_KIND_INT64 = 4
    CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL = 5
    CUPTI_METRIC_VALUE_KIND_FORCE_INT = 2147483647
end


struct CUpti_ActivityMarkerData
    kind::CUpti_ActivityKind
    flags::CUpti_ActivityFlag
    id::UInt32
    payloadKind::CUpti_MetricValueKind
    payload::CUpti_MetricValue
    color::UInt32
    category::UInt32
end

struct CUpti_ActivityOverhead
    kind::CUpti_ActivityKind
    overheadKind::CUpti_ActivityOverheadKind
    objectKind::CUpti_ActivityObjectKind
    objectId::CUpti_ActivityObjectKindId
    start::UInt64
    _end::UInt64
end

struct ANONYMOUS9_speed
    smClock::UInt32
    memoryClock::UInt32
    pcieLinkGen::UInt32
    pcieLinkWidth::UInt32
    clocksThrottleReasons::CUpti_EnvironmentClocksThrottleReason
end

struct ANONYMOUS8_data
    speed::ANONYMOUS9_speed
end

struct CUpti_ActivityEnvironment
    kind::CUpti_ActivityKind
    deviceId::UInt32
    timestamp::UInt64
    environmentKind::CUpti_ActivityEnvironmentKind
    data::ANONYMOUS8_data
end

struct CUpti_ActivityInstructionExecution
    kind::CUpti_ActivityKind
    flags::CUpti_ActivityFlag
    sourceLocatorId::UInt32
    correlationId::UInt32
    functionId::UInt32
    pcOffset::UInt32
    threadsExecuted::UInt64
    notPredOffThreadsExecuted::UInt64
    executed::UInt32
    pad::UInt32
end

struct CUpti_ActivityPCSampling
    kind::CUpti_ActivityKind
    flags::CUpti_ActivityFlag
    sourceLocatorId::UInt32
    correlationId::UInt32
    functionId::UInt32
    pcOffset::UInt32
    samples::UInt32
    stallReason::CUpti_ActivityPCSamplingStallReason
end

struct CUpti_ActivityPCSampling2
    kind::CUpti_ActivityKind
    flags::CUpti_ActivityFlag
    sourceLocatorId::UInt32
    correlationId::UInt32
    functionId::UInt32
    pcOffset::UInt32
    latencySamples::UInt32
    samples::UInt32
    stallReason::CUpti_ActivityPCSamplingStallReason
    pad::UInt32
end

struct CUpti_ActivityPCSampling3
    kind::CUpti_ActivityKind
    flags::CUpti_ActivityFlag
    sourceLocatorId::UInt32
    correlationId::UInt32
    functionId::UInt32
    latencySamples::UInt32
    samples::UInt32
    stallReason::CUpti_ActivityPCSamplingStallReason
    pcOffset::UInt64
end

struct CUpti_ActivityPCSamplingRecordInfo
    kind::CUpti_ActivityKind
    correlationId::UInt32
    totalSamples::UInt64
    droppedSamples::UInt64
    samplingPeriodInCycles::UInt64
end

struct CUpti_ActivityUnifiedMemoryCounter
    kind::CUpti_ActivityKind
    counterKind::CUpti_ActivityUnifiedMemoryCounterKind
    scope::CUpti_ActivityUnifiedMemoryCounterScope
    deviceId::UInt32
    value::UInt64
    timestamp::UInt64
    processId::UInt32
    pad::UInt32
end

struct CUpti_ActivityUnifiedMemoryCounter2
    kind::CUpti_ActivityKind
    counterKind::CUpti_ActivityUnifiedMemoryCounterKind
    value::UInt64
    start::UInt64
    _end::UInt64
    address::UInt64
    srcId::UInt32
    dstId::UInt32
    streamId::UInt32
    processId::UInt32
    flags::UInt32
    pad::UInt32
end

struct CUpti_ActivityFunction
    kind::CUpti_ActivityKind
    id::UInt32
    contextId::UInt32
    moduleId::UInt32
    functionIndex::UInt32
    pad::UInt32
    name::Cstring
end

struct CUpti_ActivityModule
    kind::CUpti_ActivityKind
    contextId::UInt32
    id::UInt32
    cubinSize::UInt32
    cubin::Ptr{Cvoid}
end

struct CUpti_ActivitySharedAccess
    kind::CUpti_ActivityKind
    flags::CUpti_ActivityFlag
    sourceLocatorId::UInt32
    correlationId::UInt32
    functionId::UInt32
    pcOffset::UInt32
    threadsExecuted::UInt64
    sharedTransactions::UInt64
    theoreticalSharedTransactions::UInt64
    executed::UInt32
    pad::UInt32
end

struct CUpti_ActivityCudaEvent
    kind::CUpti_ActivityKind
    correlationId::UInt32
    contextId::UInt32
    streamId::UInt32
    eventId::UInt32
    pad::UInt32
end

struct CUpti_ActivityStream
    kind::CUpti_ActivityKind
    contextId::UInt32
    streamId::UInt32
    priority::UInt32
    flag::CUpti_ActivityStreamFlag
    correlationId::UInt32
end

struct CUpti_ActivitySynchronization
    kind::CUpti_ActivityKind
    type::CUpti_ActivitySynchronizationType
    start::UInt64
    _end::UInt64
    correlationId::UInt32
    contextId::UInt32
    streamId::UInt32
    cudaEventId::UInt32
end

struct CUpti_ActivityInstructionCorrelation
    kind::CUpti_ActivityKind
    flags::CUpti_ActivityFlag
    sourceLocatorId::UInt32
    functionId::UInt32
    pcOffset::UInt32
    pad::UInt32
end

@cenum CUpti_OpenAccEventKind::UInt32 begin
    CUPTI_OPENACC_EVENT_KIND_INVALID = 0
    CUPTI_OPENACC_EVENT_KIND_DEVICE_INIT = 1
    CUPTI_OPENACC_EVENT_KIND_DEVICE_SHUTDOWN = 2
    CUPTI_OPENACC_EVENT_KIND_RUNTIME_SHUTDOWN = 3
    CUPTI_OPENACC_EVENT_KIND_ENQUEUE_LAUNCH = 4
    CUPTI_OPENACC_EVENT_KIND_ENQUEUE_UPLOAD = 5
    CUPTI_OPENACC_EVENT_KIND_ENQUEUE_DOWNLOAD = 6
    CUPTI_OPENACC_EVENT_KIND_WAIT = 7
    CUPTI_OPENACC_EVENT_KIND_IMPLICIT_WAIT = 8
    CUPTI_OPENACC_EVENT_KIND_COMPUTE_CONSTRUCT = 9
    CUPTI_OPENACC_EVENT_KIND_UPDATE = 10
    CUPTI_OPENACC_EVENT_KIND_ENTER_DATA = 11
    CUPTI_OPENACC_EVENT_KIND_EXIT_DATA = 12
    CUPTI_OPENACC_EVENT_KIND_CREATE = 13
    CUPTI_OPENACC_EVENT_KIND_DELETE = 14
    CUPTI_OPENACC_EVENT_KIND_ALLOC = 15
    CUPTI_OPENACC_EVENT_KIND_FREE = 16
    CUPTI_OPENACC_EVENT_KIND_FORCE_INT = 2147483647
end

@cenum CUpti_OpenAccConstructKind::UInt32 begin
    CUPTI_OPENACC_CONSTRUCT_KIND_UNKNOWN = 0
    CUPTI_OPENACC_CONSTRUCT_KIND_PARALLEL = 1
    CUPTI_OPENACC_CONSTRUCT_KIND_KERNELS = 2
    CUPTI_OPENACC_CONSTRUCT_KIND_LOOP = 3
    CUPTI_OPENACC_CONSTRUCT_KIND_DATA = 4
    CUPTI_OPENACC_CONSTRUCT_KIND_ENTER_DATA = 5
    CUPTI_OPENACC_CONSTRUCT_KIND_EXIT_DATA = 6
    CUPTI_OPENACC_CONSTRUCT_KIND_HOST_DATA = 7
    CUPTI_OPENACC_CONSTRUCT_KIND_ATOMIC = 8
    CUPTI_OPENACC_CONSTRUCT_KIND_DECLARE = 9
    CUPTI_OPENACC_CONSTRUCT_KIND_INIT = 10
    CUPTI_OPENACC_CONSTRUCT_KIND_SHUTDOWN = 11
    CUPTI_OPENACC_CONSTRUCT_KIND_SET = 12
    CUPTI_OPENACC_CONSTRUCT_KIND_UPDATE = 13
    CUPTI_OPENACC_CONSTRUCT_KIND_ROUTINE = 14
    CUPTI_OPENACC_CONSTRUCT_KIND_WAIT = 15
    CUPTI_OPENACC_CONSTRUCT_KIND_RUNTIME_API = 16
    CUPTI_OPENACC_CONSTRUCT_KIND_FORCE_INT = 2147483647
end

@cenum CUpti_OpenMpEventKind::UInt32 begin
    CUPTI_OPENMP_EVENT_KIND_INVALID = 0
    CUPTI_OPENMP_EVENT_KIND_PARALLEL = 1
    CUPTI_OPENMP_EVENT_KIND_TASK = 2
    CUPTI_OPENMP_EVENT_KIND_THREAD = 3
    CUPTI_OPENMP_EVENT_KIND_IDLE = 4
    CUPTI_OPENMP_EVENT_KIND_WAIT_BARRIER = 5
    CUPTI_OPENMP_EVENT_KIND_WAIT_TASKWAIT = 6
    CUPTI_OPENMP_EVENT_KIND_FORCE_INT = 2147483647
end


struct CUpti_ActivityOpenAcc
    kind::CUpti_ActivityKind
    eventKind::CUpti_OpenAccEventKind
    parentConstruct::CUpti_OpenAccConstructKind
    version::UInt32
    implicit::UInt32
    deviceType::UInt32
    deviceNumber::UInt32
    threadId::UInt32
    async::UInt64
    asyncMap::UInt64
    lineNo::UInt32
    endLineNo::UInt32
    funcLineNo::UInt32
    funcEndLineNo::UInt32
    start::UInt64
    _end::UInt64
    cuDeviceId::UInt32
    cuContextId::UInt32
    cuStreamId::UInt32
    cuProcessId::UInt32
    cuThreadId::UInt32
    externalId::UInt32
    srcFile::Cstring
    funcName::Cstring
end

struct CUpti_ActivityOpenAccData
    kind::CUpti_ActivityKind
    eventKind::CUpti_OpenAccEventKind
    parentConstruct::CUpti_OpenAccConstructKind
    version::UInt32
    implicit::UInt32
    deviceType::UInt32
    deviceNumber::UInt32
    threadId::UInt32
    async::UInt64
    asyncMap::UInt64
    lineNo::UInt32
    endLineNo::UInt32
    funcLineNo::UInt32
    funcEndLineNo::UInt32
    start::UInt64
    _end::UInt64
    cuDeviceId::UInt32
    cuContextId::UInt32
    cuStreamId::UInt32
    cuProcessId::UInt32
    cuThreadId::UInt32
    externalId::UInt32
    srcFile::Cstring
    funcName::Cstring
    bytes::UInt64
    hostPtr::UInt64
    devicePtr::UInt64
    varName::Cstring
end

struct CUpti_ActivityOpenAccLaunch
    kind::CUpti_ActivityKind
    eventKind::CUpti_OpenAccEventKind
    parentConstruct::CUpti_OpenAccConstructKind
    version::UInt32
    implicit::UInt32
    deviceType::UInt32
    deviceNumber::UInt32
    threadId::UInt32
    async::UInt64
    asyncMap::UInt64
    lineNo::UInt32
    endLineNo::UInt32
    funcLineNo::UInt32
    funcEndLineNo::UInt32
    start::UInt64
    _end::UInt64
    cuDeviceId::UInt32
    cuContextId::UInt32
    cuStreamId::UInt32
    cuProcessId::UInt32
    cuThreadId::UInt32
    externalId::UInt32
    srcFile::Cstring
    funcName::Cstring
    numGangs::UInt64
    numWorkers::UInt64
    vectorLength::UInt64
    kernelName::Cstring
end

struct CUpti_ActivityOpenAccOther
    kind::CUpti_ActivityKind
    eventKind::CUpti_OpenAccEventKind
    parentConstruct::CUpti_OpenAccConstructKind
    version::UInt32
    implicit::UInt32
    deviceType::UInt32
    deviceNumber::UInt32
    threadId::UInt32
    async::UInt64
    asyncMap::UInt64
    lineNo::UInt32
    endLineNo::UInt32
    funcLineNo::UInt32
    funcEndLineNo::UInt32
    start::UInt64
    _end::UInt64
    cuDeviceId::UInt32
    cuContextId::UInt32
    cuStreamId::UInt32
    cuProcessId::UInt32
    cuThreadId::UInt32
    externalId::UInt32
    srcFile::Cstring
    funcName::Cstring
end

struct CUpti_ActivityOpenMp
    kind::CUpti_ActivityKind
    eventKind::CUpti_OpenMpEventKind
    version::UInt32
    threadId::UInt32
    start::UInt64
    _end::UInt64
    cuProcessId::UInt32
    cuThreadId::UInt32
end

@cenum CUpti_ExternalCorrelationKind::UInt32 begin
    CUPTI_EXTERNAL_CORRELATION_KIND_INVALID = 0
    CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN = 1
    CUPTI_EXTERNAL_CORRELATION_KIND_OPENACC = 2
    CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0 = 3
    CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1 = 4
    CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2 = 5
    CUPTI_EXTERNAL_CORRELATION_KIND_SIZE = 6
    CUPTI_EXTERNAL_CORRELATION_KIND_FORCE_INT = 2147483647
end


struct CUpti_ActivityExternalCorrelation
    kind::CUpti_ActivityKind
    externalKind::CUpti_ExternalCorrelationKind
    externalId::UInt64
    correlationId::UInt32
    reserved::UInt32
end

@cenum CUpti_DevType::UInt32 begin
    CUPTI_DEV_TYPE_INVALID = 0
    CUPTI_DEV_TYPE_GPU = 1
    CUPTI_DEV_TYPE_NPU = 2
    CUPTI_DEV_TYPE_FORCE_INT = 2147483647
end


struct ANONYMOUS10_idDev0
    uuidDev::CUuuid
end

struct ANONYMOUS11_idDev1
    uuidDev::CUuuid
end

struct CUpti_ActivityNvLink
    kind::CUpti_ActivityKind
    nvlinkVersion::UInt32
    typeDev0::CUpti_DevType
    typeDev1::CUpti_DevType
    idDev0::ANONYMOUS10_idDev0
    idDev1::ANONYMOUS11_idDev1
    flag::UInt32
    physicalNvLinkCount::UInt32
    portDev0::NTuple{4, Int8}
    portDev1::NTuple{4, Int8}
    bandwidth::UInt64
end

struct ANONYMOUS12_idDev0
    uuidDev::CUuuid
end

struct ANONYMOUS13_idDev1
    uuidDev::CUuuid
end

struct CUpti_ActivityNvLink2
    kind::CUpti_ActivityKind
    nvlinkVersion::UInt32
    typeDev0::CUpti_DevType
    typeDev1::CUpti_DevType
    idDev0::ANONYMOUS12_idDev0
    idDev1::ANONYMOUS13_idDev1
    flag::UInt32
    physicalNvLinkCount::UInt32
    portDev0::NTuple{16, Int8}
    portDev1::NTuple{16, Int8}
    bandwidth::UInt64
end

struct ANONYMOUS14_idDev0
    uuidDev::CUuuid
end

struct ANONYMOUS15_idDev1
    uuidDev::CUuuid
end

struct CUpti_ActivityNvLink3
    kind::CUpti_ActivityKind
    nvlinkVersion::UInt32
    typeDev0::CUpti_DevType
    typeDev1::CUpti_DevType
    idDev0::ANONYMOUS14_idDev0
    idDev1::ANONYMOUS15_idDev1
    flag::UInt32
    physicalNvLinkCount::UInt32
    portDev0::NTuple{16, Int8}
    portDev1::NTuple{16, Int8}
    bandwidth::UInt64
    nvswitchConnected::UInt8
    pad::NTuple{7, UInt8}
end

@cenum CUpti_PcieDeviceType::UInt32 begin
    CUPTI_PCIE_DEVICE_TYPE_GPU = 0
    CUPTI_PCIE_DEVICE_TYPE_BRIDGE = 1
    CUPTI_PCIE_DEVICE_TYPE_FORCE_INT = 2147483647
end


struct ANONYMOUS16_id
    devId::CUdevice
end

struct ANONYMOUS18_gpuAttr
    uuidDev::CUuuid
    peerDev::NTuple{32, CUdevice}
end

struct ANONYMOUS17_attr
    gpuAttr::ANONYMOUS18_gpuAttr
end

struct CUpti_ActivityPcie
    kind::CUpti_ActivityKind
    type::CUpti_PcieDeviceType
    id::ANONYMOUS16_id
    domain::UInt32
    pcieGeneration::UInt16
    linkRate::UInt16
    linkWidth::UInt16
    upstreamBus::UInt16
    attr::ANONYMOUS17_attr
end

@cenum CUpti_PcieGen::UInt32 begin
    CUPTI_PCIE_GEN_GEN1 = 1
    CUPTI_PCIE_GEN_GEN2 = 2
    CUPTI_PCIE_GEN_GEN3 = 3
    CUPTI_PCIE_GEN_GEN4 = 4
    CUPTI_PCIE_GEN_FORCE_INT = 2147483647
end


struct CUpti_ActivityInstantaneousEvent
    kind::CUpti_ActivityKind
    id::CUpti_EventID
    value::UInt64
    timestamp::UInt64
    deviceId::UInt32
    reserved::UInt32
end

struct CUpti_ActivityInstantaneousEventInstance
    kind::CUpti_ActivityKind
    id::CUpti_EventID
    value::UInt64
    timestamp::UInt64
    deviceId::UInt32
    instance::UInt8
    pad::NTuple{3, UInt8}
end

struct CUpti_ActivityInstantaneousMetric
    kind::CUpti_ActivityKind
    id::CUpti_MetricID
    value::CUpti_MetricValue
    timestamp::UInt64
    deviceId::UInt32
    flags::UInt8
    pad::NTuple{3, UInt8}
end

struct CUpti_ActivityInstantaneousMetricInstance
    kind::CUpti_ActivityKind
    id::CUpti_MetricID
    value::CUpti_MetricValue
    timestamp::UInt64
    deviceId::UInt32
    flags::UInt8
    instance::UInt8
    pad::NTuple{2, UInt8}
end

@cenum CUpti_ActivityAttribute::UInt32 begin
    CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE = 0
    CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_CDP = 1
    CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT = 2
    CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE = 3
    CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_LIMIT = 4
    CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_FORCE_INT = 2147483647
end

@cenum CUpti_ActivityThreadIdType::UInt32 begin
    CUPTI_ACTIVITY_THREAD_ID_TYPE_DEFAULT = 0
    CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM = 1
    CUPTI_ACTIVITY_THREAD_ID_TYPE_FORCE_INT = 2147483647
end


const CUpti_BuffersCallbackRequestFunc = Ptr{Cvoid}
const CUpti_BuffersCallbackCompleteFunc = Ptr{Cvoid}

@cenum CUpti_ApiCallbackSite::UInt32 begin
    CUPTI_API_ENTER = 0
    CUPTI_API_EXIT = 1
    CUPTI_API_CBSITE_FORCE_INT = 2147483647
end

@cenum CUpti_CallbackDomain::UInt32 begin
    CUPTI_CB_DOMAIN_INVALID = 0
    CUPTI_CB_DOMAIN_DRIVER_API = 1
    CUPTI_CB_DOMAIN_RUNTIME_API = 2
    CUPTI_CB_DOMAIN_RESOURCE = 3
    CUPTI_CB_DOMAIN_SYNCHRONIZE = 4
    CUPTI_CB_DOMAIN_NVTX = 5
    CUPTI_CB_DOMAIN_SIZE = 6
    CUPTI_CB_DOMAIN_FORCE_INT = 2147483647
end

@cenum CUpti_CallbackIdResource::UInt32 begin
    CUPTI_CBID_RESOURCE_INVALID = 0
    CUPTI_CBID_RESOURCE_CONTEXT_CREATED = 1
    CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING = 2
    CUPTI_CBID_RESOURCE_STREAM_CREATED = 3
    CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING = 4
    CUPTI_CBID_RESOURCE_CU_INIT_FINISHED = 5
    CUPTI_CBID_RESOURCE_MODULE_LOADED = 6
    CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING = 7
    CUPTI_CBID_RESOURCE_MODULE_PROFILED = 8
    CUPTI_CBID_RESOURCE_GRAPH_CREATED = 9
    CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING = 10
    CUPTI_CBID_RESOURCE_GRAPH_CLONED = 11
    CUPTI_CBID_RESOURCE_GRAPHNODE_CREATE_STARTING = 12
    CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED = 13
    CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING = 14
    CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_CREATED = 15
    CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_DESTROY_STARTING = 16
    CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATE_STARTING = 17
    CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED = 18
    CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING = 19
    CUPTI_CBID_RESOURCE_SIZE = 20
    CUPTI_CBID_RESOURCE_FORCE_INT = 2147483647
end

@cenum CUpti_CallbackIdSync::UInt32 begin
    CUPTI_CBID_SYNCHRONIZE_INVALID = 0
    CUPTI_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED = 1
    CUPTI_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED = 2
    CUPTI_CBID_SYNCHRONIZE_SIZE = 3
    CUPTI_CBID_SYNCHRONIZE_FORCE_INT = 2147483647
end


struct CUpti_CallbackData
    callbackSite::CUpti_ApiCallbackSite
    functionName::Cstring
    functionParams::Ptr{Cvoid}
    functionReturnValue::Ptr{Cvoid}
    symbolName::Cstring
    context::CUcontext
    contextUid::UInt32
    correlationData::Ptr{UInt64}
    correlationId::UInt32
end

struct ANONYMOUS19_resourceHandle
    stream::CUstream
end

struct CUpti_ResourceData
    context::CUcontext
    resourceHandle::ANONYMOUS19_resourceHandle
    resourceDescriptor::Ptr{Cvoid}
end

struct CUpti_ModuleResourceData
    moduleId::UInt32
    cubinSize::Csize_t
    pCubin::Cstring
end

struct CUpti_GraphData
    graph::CUgraph
    originalGraph::CUgraph
    node::CUgraphNode
    nodeType::CUgraphNodeType
    dependency::CUgraphNode
    graphExec::CUgraphExec
end

struct CUpti_SynchronizeData
    context::CUcontext
    stream::CUstream
end

struct CUpti_NvtxData
    functionName::Cstring
    functionParams::Ptr{Cvoid}
end

const CUpti_CallbackFunc = Ptr{Cvoid}
const CUpti_Subscriber_st = Cvoid
const CUpti_SubscriberHandle = Ptr{CUpti_Subscriber_st}
const CUpti_DomainTable = Ptr{CUpti_CallbackDomain}

# Skipping MacroDefinition: CUPTI_EVENT_OVERFLOW ( ( uint64_t ) 0xFFFFFFFFFFFFFFFFULL )
# Skipping MacroDefinition: CUPTI_EVENT_INVALID ( ( uint64_t ) 0xFFFFFFFFFFFFFFFEULL )

const CUpti_EventGroup = Ptr{Cvoid}

@cenum CUpti_DeviceAttributeDeviceClass::UInt32 begin
    CUPTI_DEVICE_ATTR_DEVICE_CLASS_TESLA = 0
    CUPTI_DEVICE_ATTR_DEVICE_CLASS_QUADRO = 1
    CUPTI_DEVICE_ATTR_DEVICE_CLASS_GEFORCE = 2
    CUPTI_DEVICE_ATTR_DEVICE_CLASS_TEGRA = 3
end

@cenum CUpti_DeviceAttribute::UInt32 begin
    CUPTI_DEVICE_ATTR_MAX_EVENT_ID = 1
    CUPTI_DEVICE_ATTR_MAX_EVENT_DOMAIN_ID = 2
    CUPTI_DEVICE_ATTR_GLOBAL_MEMORY_BANDWIDTH = 3
    CUPTI_DEVICE_ATTR_INSTRUCTION_PER_CYCLE = 4
    CUPTI_DEVICE_ATTR_INSTRUCTION_THROUGHPUT_SINGLE_PRECISION = 5
    CUPTI_DEVICE_ATTR_MAX_FRAME_BUFFERS = 6
    CUPTI_DEVICE_ATTR_PCIE_LINK_RATE = 7
    CUPTI_DEVICE_ATTR_PCIE_LINK_WIDTH = 8
    CUPTI_DEVICE_ATTR_PCIE_GEN = 9
    CUPTI_DEVICE_ATTR_DEVICE_CLASS = 10
    CUPTI_DEVICE_ATTR_FLOP_SP_PER_CYCLE = 11
    CUPTI_DEVICE_ATTR_FLOP_DP_PER_CYCLE = 12
    CUPTI_DEVICE_ATTR_MAX_L2_UNITS = 13
    CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_SHARED = 14
    CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_L1 = 15
    CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_EQUAL = 16
    CUPTI_DEVICE_ATTR_FLOP_HP_PER_CYCLE = 17
    CUPTI_DEVICE_ATTR_NVLINK_PRESENT = 18
    CUPTI_DEVICE_ATTR_GPU_CPU_NVLINK_BW = 19
    CUPTI_DEVICE_ATTR_NVSWITCH_PRESENT = 20
    CUPTI_DEVICE_ATTR_FORCE_INT = 2147483647
end

@cenum CUpti_EventDomainAttribute::UInt32 begin
    CUPTI_EVENT_DOMAIN_ATTR_NAME = 0
    CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT = 1
    CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT = 3
    CUPTI_EVENT_DOMAIN_ATTR_COLLECTION_METHOD = 4
    CUPTI_EVENT_DOMAIN_ATTR_FORCE_INT = 2147483647
end

@cenum CUpti_EventCollectionMethod::UInt32 begin
    CUPTI_EVENT_COLLECTION_METHOD_PM = 0
    CUPTI_EVENT_COLLECTION_METHOD_SM = 1
    CUPTI_EVENT_COLLECTION_METHOD_INSTRUMENTED = 2
    CUPTI_EVENT_COLLECTION_METHOD_NVLINK_TC = 3
    CUPTI_EVENT_COLLECTION_METHOD_FORCE_INT = 2147483647
end

@cenum CUpti_EventGroupAttribute::UInt32 begin
    CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID = 0
    CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES = 1
    CUPTI_EVENT_GROUP_ATTR_USER_DATA = 2
    CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS = 3
    CUPTI_EVENT_GROUP_ATTR_EVENTS = 4
    CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT = 5
    CUPTI_EVENT_GROUP_ATTR_PROFILING_SCOPE = 6
    CUPTI_EVENT_GROUP_ATTR_FORCE_INT = 2147483647
end

@cenum CUpti_EventProfilingScope::UInt32 begin
    CUPTI_EVENT_PROFILING_SCOPE_CONTEXT = 0
    CUPTI_EVENT_PROFILING_SCOPE_DEVICE = 1
    CUPTI_EVENT_PROFILING_SCOPE_BOTH = 2
    CUPTI_EVENT_PROFILING_SCOPE_FORCE_INT = 2147483647
end

@cenum CUpti_EventAttribute::UInt32 begin
    CUPTI_EVENT_ATTR_NAME = 0
    CUPTI_EVENT_ATTR_SHORT_DESCRIPTION = 1
    CUPTI_EVENT_ATTR_LONG_DESCRIPTION = 2
    CUPTI_EVENT_ATTR_CATEGORY = 3
    CUPTI_EVENT_ATTR_PROFILING_SCOPE = 5
    CUPTI_EVENT_ATTR_FORCE_INT = 2147483647
end

@cenum CUpti_EventCollectionMode::UInt32 begin
    CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS = 0
    CUPTI_EVENT_COLLECTION_MODE_KERNEL = 1
    CUPTI_EVENT_COLLECTION_MODE_FORCE_INT = 2147483647
end

@cenum CUpti_EventCategory::UInt32 begin
    CUPTI_EVENT_CATEGORY_INSTRUCTION = 0
    CUPTI_EVENT_CATEGORY_MEMORY = 1
    CUPTI_EVENT_CATEGORY_CACHE = 2
    CUPTI_EVENT_CATEGORY_PROFILE_TRIGGER = 3
    CUPTI_EVENT_CATEGORY_SYSTEM = 4
    CUPTI_EVENT_CATEGORY_FORCE_INT = 2147483647
end

@cenum CUpti_ReadEventFlags::UInt32 begin
    CUPTI_EVENT_READ_FLAG_NONE = 0
    CUPTI_EVENT_READ_FLAG_FORCE_INT = 2147483647
end


struct CUpti_EventGroupSet
    numEventGroups::UInt32
    eventGroups::Ptr{CUpti_EventGroup}
end

struct CUpti_EventGroupSets
    numSets::UInt32
    sets::Ptr{CUpti_EventGroupSet}
end

const CUpti_KernelReplayUpdateFunc = Ptr{Cvoid}

@cenum CUpti_MetricCategory::UInt32 begin
    CUPTI_METRIC_CATEGORY_MEMORY = 0
    CUPTI_METRIC_CATEGORY_INSTRUCTION = 1
    CUPTI_METRIC_CATEGORY_MULTIPROCESSOR = 2
    CUPTI_METRIC_CATEGORY_CACHE = 3
    CUPTI_METRIC_CATEGORY_TEXTURE = 4
    CUPTI_METRIC_CATEGORY_NVLINK = 5
    CUPTI_METRIC_CATEGORY_PCIE = 6
    CUPTI_METRIC_CATEGORY_FORCE_INT = 2147483647
end

@cenum CUpti_MetricEvaluationMode::UInt32 begin
    CUPTI_METRIC_EVALUATION_MODE_PER_INSTANCE = 1
    CUPTI_METRIC_EVALUATION_MODE_AGGREGATE = 2
    CUPTI_METRIC_EVALUATION_MODE_FORCE_INT = 2147483647
end

@cenum CUpti_MetricValueUtilizationLevel::UInt32 begin
    CUPTI_METRIC_VALUE_UTILIZATION_IDLE = 0
    CUPTI_METRIC_VALUE_UTILIZATION_LOW = 2
    CUPTI_METRIC_VALUE_UTILIZATION_MID = 5
    CUPTI_METRIC_VALUE_UTILIZATION_HIGH = 8
    CUPTI_METRIC_VALUE_UTILIZATION_MAX = 10
    CUPTI_METRIC_VALUE_UTILIZATION_FORCE_INT = 2147483647
end

@cenum CUpti_MetricAttribute::UInt32 begin
    CUPTI_METRIC_ATTR_NAME = 0
    CUPTI_METRIC_ATTR_SHORT_DESCRIPTION = 1
    CUPTI_METRIC_ATTR_LONG_DESCRIPTION = 2
    CUPTI_METRIC_ATTR_CATEGORY = 3
    CUPTI_METRIC_ATTR_VALUE_KIND = 4
    CUPTI_METRIC_ATTR_EVALUATION_MODE = 5
    CUPTI_METRIC_ATTR_FORCE_INT = 2147483647
end

@cenum CUpti_MetricPropertyDeviceClass::UInt32 begin
    CUPTI_METRIC_PROPERTY_DEVICE_CLASS_TESLA = 0
    CUPTI_METRIC_PROPERTY_DEVICE_CLASS_QUADRO = 1
    CUPTI_METRIC_PROPERTY_DEVICE_CLASS_GEFORCE = 2
    CUPTI_METRIC_PROPERTY_DEVICE_CLASS_TEGRA = 3
end

@cenum CUpti_MetricPropertyID::UInt32 begin
    CUPTI_METRIC_PROPERTY_MULTIPROCESSOR_COUNT = 0
    CUPTI_METRIC_PROPERTY_WARPS_PER_MULTIPROCESSOR = 1
    CUPTI_METRIC_PROPERTY_KERNEL_GPU_TIME = 2
    CUPTI_METRIC_PROPERTY_CLOCK_RATE = 3
    CUPTI_METRIC_PROPERTY_FRAME_BUFFER_COUNT = 4
    CUPTI_METRIC_PROPERTY_GLOBAL_MEMORY_BANDWIDTH = 5
    CUPTI_METRIC_PROPERTY_PCIE_LINK_RATE = 6
    CUPTI_METRIC_PROPERTY_PCIE_LINK_WIDTH = 7
    CUPTI_METRIC_PROPERTY_PCIE_GEN = 8
    CUPTI_METRIC_PROPERTY_DEVICE_CLASS = 9
    CUPTI_METRIC_PROPERTY_FLOP_SP_PER_CYCLE = 10
    CUPTI_METRIC_PROPERTY_FLOP_DP_PER_CYCLE = 11
    CUPTI_METRIC_PROPERTY_L2_UNITS = 12
    CUPTI_METRIC_PROPERTY_ECC_ENABLED = 13
    CUPTI_METRIC_PROPERTY_FLOP_HP_PER_CYCLE = 14
    CUPTI_METRIC_PROPERTY_GPU_CPU_NVLINK_BANDWIDTH = 15
end


# Skipping MacroDefinition: CUPTI_PROFILER_STRUCT_SIZE ( type_ , lastfield_ ) ( offsetof ( type_ , lastfield_ ) + sizeof ( ( ( type_ * ) 0 ) -> lastfield_ ) )
# Skipping MacroDefinition: CUpti_Profiler_Initialize_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE ( CUpti_Profiler_Initialize_Params , pPriv )
# Skipping MacroDefinition: CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE ( CUpti_Profiler_DeInitialize_Params , pPriv )
# Skipping MacroDefinition: CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE ( CUpti_Profiler_CounterDataImageOptions , maxRangeNameLength )
# Skipping MacroDefinition: CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE ( CUpti_Profiler_CounterDataImage_CalculateSize_Params , counterDataImageSize )
# Skipping MacroDefinition: CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE ( CUpti_Profiler_CounterDataImage_Initialize_Params , pCounterDataImage )
# Skipping MacroDefinition: CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE ( CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params , counterDataScratchBufferSize )
# Skipping MacroDefinition: CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE ( CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params , pCounterDataScratchBuffer )
# Skipping MacroDefinition: CUpti_Profiler_BeginSession_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE ( CUpti_Profiler_BeginSession_Params , maxLaunchesPerPass )
# Skipping MacroDefinition: CUpti_Profiler_EndSession_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE ( CUpti_Profiler_EndSession_Params , ctx )
# Skipping MacroDefinition: CUpti_Profiler_SetConfig_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE ( CUpti_Profiler_SetConfig_Params , passIndex )
# Skipping MacroDefinition: CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE ( CUpti_Profiler_UnsetConfig_Params , ctx )
# Skipping MacroDefinition: CUpti_Profiler_BeginPass_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE ( CUpti_Profiler_BeginPass_Params , ctx )
# Skipping MacroDefinition: CUpti_Profiler_EndPass_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE ( CUpti_Profiler_EndPass_Params , ctx )
# Skipping MacroDefinition: CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE ( CUpti_Profiler_EnableProfiling_Params , ctx )
# Skipping MacroDefinition: CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE ( CUpti_Profiler_DisableProfiling_Params , ctx )
# Skipping MacroDefinition: CUpti_Profiler_IsPassCollected_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE ( CUpti_Profiler_IsPassCollected_Params , ctx )
# Skipping MacroDefinition: CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE ( CUpti_Profiler_FlushCounterData_Params , ctx )
# Skipping MacroDefinition: CUpti_Profiler_PushRange_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE ( CUpti_Profiler_PushRange_Params , rangeNameLength )
# Skipping MacroDefinition: CUpti_Profiler_PopRange_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE ( CUpti_Profiler_PopRange_Params , ctx )

@cenum CUpti_ProfilerRange::UInt32 begin
    CUPTI_Range_INVALID = 0
    CUPTI_AutoRange = 1
    CUPTI_UserRange = 2
    CUPTI_Range_COUNT = 3
end

@cenum CUpti_ProfilerReplayMode::UInt32 begin
    CUPTI_Replay_INVALID = 0
    CUPTI_ApplicationReplay = 1
    CUPTI_KernelReplay = 2
    CUPTI_UserReplay = 3
    CUPTI_Replay_COUNT = 4
end


struct CUpti_Profiler_Initialize_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
end

struct CUpti_Profiler_DeInitialize_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
end

struct CUpti_Profiler_CounterDataImageOptions
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pCounterDataPrefix::Ptr{UInt8}
    counterDataPrefixSize::Csize_t
    maxNumRanges::UInt32
    maxNumRangeTreeNodes::UInt32
    maxRangeNameLength::UInt32
end

struct CUpti_Profiler_CounterDataImage_CalculateSize_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    sizeofCounterDataImageOptions::Csize_t
    pOptions::Ptr{CUpti_Profiler_CounterDataImageOptions}
    counterDataImageSize::Csize_t
end

struct CUpti_Profiler_CounterDataImage_Initialize_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    sizeofCounterDataImageOptions::Csize_t
    pOptions::Ptr{CUpti_Profiler_CounterDataImageOptions}
    counterDataImageSize::Csize_t
    pCounterDataImage::Ptr{UInt8}
end

struct CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    counterDataImageSize::Csize_t
    pCounterDataImage::Ptr{UInt8}
    counterDataScratchBufferSize::Csize_t
end

struct CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    counterDataImageSize::Csize_t
    pCounterDataImage::Ptr{UInt8}
    counterDataScratchBufferSize::Csize_t
    pCounterDataScratchBuffer::Ptr{UInt8}
end

struct CUpti_Profiler_BeginSession_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    ctx::CUcontext
    counterDataImageSize::Csize_t
    pCounterDataImage::Ptr{UInt8}
    counterDataScratchBufferSize::Csize_t
    pCounterDataScratchBuffer::Ptr{UInt8}
    bDumpCounterDataInFile::UInt8
    pCounterDataFilePath::Cstring
    range::CUpti_ProfilerRange
    replayMode::CUpti_ProfilerReplayMode
    maxRangesPerPass::Csize_t
    maxLaunchesPerPass::Csize_t
end

struct CUpti_Profiler_EndSession_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    ctx::CUcontext
end

struct CUpti_Profiler_SetConfig_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    ctx::CUcontext
    pConfig::Ptr{UInt8}
    configSize::Csize_t
    minNestingLevel::UInt16
    numNestingLevels::UInt16
    passIndex::Csize_t
    targetNestingLevel::UInt16
end

struct CUpti_Profiler_UnsetConfig_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    ctx::CUcontext
end

struct CUpti_Profiler_BeginPass_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    ctx::CUcontext
end

struct CUpti_Profiler_EndPass_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    ctx::CUcontext
    targetNestingLevel::UInt16
    passIndex::Csize_t
    allPassesSubmitted::UInt8
end

struct CUpti_Profiler_EnableProfiling_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    ctx::CUcontext
end

struct CUpti_Profiler_DisableProfiling_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    ctx::CUcontext
end

struct CUpti_Profiler_IsPassCollected_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    ctx::CUcontext
    numRangesDropped::Csize_t
    numTraceBytesDropped::Csize_t
    onePassCollected::UInt8
    allPassesCollected::UInt8
end

struct CUpti_Profiler_FlushCounterData_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    ctx::CUcontext
    numRangesDropped::Csize_t
    numTraceBytesDropped::Csize_t
end

struct CUpti_Profiler_PushRange_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    ctx::CUcontext
    pRangeName::Cstring
    rangeNameLength::Csize_t
end

struct CUpti_Profiler_PopRange_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    ctx::CUcontext
end
