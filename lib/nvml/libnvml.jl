using CEnum

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(NVMLError(res))
end

const initialized = Ref(false)
function initialize_context()
    if !initialized[]
        res = unsafe_nvmlInitWithFlags(0)
        if res !== NVML_SUCCESS
            # NOTE: we can't call nvmlErrorString during initialization
            error("NVML could not be initialized ($res)")
        end
        atexit() do
            return nvmlShutdown()
        end
        initialized[] = true
    end
end

function check(f)
    res = f()
    if res != NVML_SUCCESS
        throw_api_error(res)
    end

    return
end

@cenum nvmlReturn_enum::UInt32 begin
    NVML_SUCCESS = 0
    NVML_ERROR_UNINITIALIZED = 1
    NVML_ERROR_INVALID_ARGUMENT = 2
    NVML_ERROR_NOT_SUPPORTED = 3
    NVML_ERROR_NO_PERMISSION = 4
    NVML_ERROR_ALREADY_INITIALIZED = 5
    NVML_ERROR_NOT_FOUND = 6
    NVML_ERROR_INSUFFICIENT_SIZE = 7
    NVML_ERROR_INSUFFICIENT_POWER = 8
    NVML_ERROR_DRIVER_NOT_LOADED = 9
    NVML_ERROR_TIMEOUT = 10
    NVML_ERROR_IRQ_ISSUE = 11
    NVML_ERROR_LIBRARY_NOT_FOUND = 12
    NVML_ERROR_FUNCTION_NOT_FOUND = 13
    NVML_ERROR_CORRUPTED_INFOROM = 14
    NVML_ERROR_GPU_IS_LOST = 15
    NVML_ERROR_RESET_REQUIRED = 16
    NVML_ERROR_OPERATING_SYSTEM = 17
    NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18
    NVML_ERROR_IN_USE = 19
    NVML_ERROR_MEMORY = 20
    NVML_ERROR_NO_DATA = 21
    NVML_ERROR_VGPU_ECC_NOT_SUPPORTED = 22
    NVML_ERROR_INSUFFICIENT_RESOURCES = 23
    NVML_ERROR_FREQ_NOT_SUPPORTED = 24
    NVML_ERROR_ARGUMENT_VERSION_MISMATCH = 25
    NVML_ERROR_DEPRECATED = 26
    NVML_ERROR_UNKNOWN = 999
end

const nvmlReturn_t = nvmlReturn_enum

@checked function nvmlInit_v2()
    @ccall (libnvml()).nvmlInit_v2()::nvmlReturn_t
end

mutable struct nvmlDevice_st end

const nvmlDevice_t = Ptr{nvmlDevice_st}

struct nvmlPciInfo_st
    busIdLegacy::NTuple{16,Cchar}
    domain::Cuint
    bus::Cuint
    device::Cuint
    pciDeviceId::Cuint
    pciSubSystemId::Cuint
    busId::NTuple{32,Cchar}
end

const nvmlPciInfo_t = nvmlPciInfo_st

@checked function nvmlDeviceGetPciInfo_v3(device, pci)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetPciInfo_v3(device::nvmlDevice_t,
                                               pci::Ptr{nvmlPciInfo_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetCount_v2(deviceCount)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetCount_v2(deviceCount::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetHandleByIndex_v2(index, device)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetHandleByIndex_v2(index::Cuint,
                                                     device::Ptr{nvmlDevice_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetHandleByPciBusId_v2(pciBusId, device)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetHandleByPciBusId_v2(pciBusId::Cstring,
                                                        device::Ptr{nvmlDevice_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetNvLinkRemotePciInfo_v2(device, link, pci)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetNvLinkRemotePciInfo_v2(device::nvmlDevice_t,
                                                           link::Cuint,
                                                           pci::Ptr{nvmlPciInfo_t})::nvmlReturn_t
end

@cenum nvmlDetachGpuState_enum::UInt32 begin
    NVML_DETACH_GPU_KEEP = 0
    NVML_DETACH_GPU_REMOVE = 1
end

const nvmlDetachGpuState_t = nvmlDetachGpuState_enum

@cenum nvmlPcieLinkState_enum::UInt32 begin
    NVML_PCIE_LINK_KEEP = 0
    NVML_PCIE_LINK_SHUT_DOWN = 1
end

const nvmlPcieLinkState_t = nvmlPcieLinkState_enum

@checked function nvmlDeviceRemoveGpu_v2(pciInfo, gpuState, linkState)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceRemoveGpu_v2(pciInfo::Ptr{nvmlPciInfo_t},
                                              gpuState::nvmlDetachGpuState_t,
                                              linkState::nvmlPcieLinkState_t)::nvmlReturn_t
end

@cenum nvmlGridLicenseFeatureCode_t::UInt32 begin
    NVML_GRID_LICENSE_FEATURE_CODE_UNKNOWN = 0
    NVML_GRID_LICENSE_FEATURE_CODE_VGPU = 1
    NVML_GRID_LICENSE_FEATURE_CODE_NVIDIA_RTX = 2
    NVML_GRID_LICENSE_FEATURE_CODE_VWORKSTATION = 2
    NVML_GRID_LICENSE_FEATURE_CODE_GAMING = 3
    NVML_GRID_LICENSE_FEATURE_CODE_COMPUTE = 4
end

struct nvmlGridLicenseExpiry_st
    year::Cuint
    month::Cushort
    day::Cushort
    hour::Cushort
    min::Cushort
    sec::Cushort
    status::Cuchar
end

const nvmlGridLicenseExpiry_t = nvmlGridLicenseExpiry_st

struct nvmlGridLicensableFeature_st
    featureCode::nvmlGridLicenseFeatureCode_t
    featureState::Cuint
    licenseInfo::NTuple{128,Cchar}
    productName::NTuple{128,Cchar}
    featureEnabled::Cuint
    licenseExpiry::nvmlGridLicenseExpiry_t
end

const nvmlGridLicensableFeature_t = nvmlGridLicensableFeature_st

struct nvmlGridLicensableFeatures_st
    isGridLicenseSupported::Cint
    licensableFeaturesCount::Cuint
    gridLicensableFeatures::NTuple{3,nvmlGridLicensableFeature_t}
end

const nvmlGridLicensableFeatures_t = nvmlGridLicensableFeatures_st

@checked function nvmlDeviceGetGridLicensableFeatures_v4(device, pGridLicensableFeatures)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetGridLicensableFeatures_v4(device::nvmlDevice_t,
                                                              pGridLicensableFeatures::Ptr{nvmlGridLicensableFeatures_t})::nvmlReturn_t
end

mutable struct nvmlEventSet_st end

const nvmlEventSet_t = Ptr{nvmlEventSet_st}

struct nvmlEventData_st
    device::nvmlDevice_t
    eventType::Culonglong
    eventData::Culonglong
    gpuInstanceId::Cuint
    computeInstanceId::Cuint
end

const nvmlEventData_t = nvmlEventData_st

@checked function nvmlEventSetWait_v2(set, data, timeoutms)
    initialize_context()
    @ccall (libnvml()).nvmlEventSetWait_v2(set::nvmlEventSet_t, data::Ptr{nvmlEventData_t},
                                           timeoutms::Cuint)::nvmlReturn_t
end

struct nvmlDeviceAttributes_st
    multiprocessorCount::Cuint
    sharedCopyEngineCount::Cuint
    sharedDecoderCount::Cuint
    sharedEncoderCount::Cuint
    sharedJpegCount::Cuint
    sharedOfaCount::Cuint
    gpuInstanceSliceCount::Cuint
    computeInstanceSliceCount::Cuint
    memorySizeMB::Culonglong
end

const nvmlDeviceAttributes_t = nvmlDeviceAttributes_st

@checked function nvmlDeviceGetAttributes_v2(device, attributes)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetAttributes_v2(device::nvmlDevice_t,
                                                  attributes::Ptr{nvmlDeviceAttributes_t})::nvmlReturn_t
end

mutable struct nvmlComputeInstance_st end

const nvmlComputeInstance_t = Ptr{nvmlComputeInstance_st}

mutable struct nvmlGpuInstance_st end

const nvmlGpuInstance_t = Ptr{nvmlGpuInstance_st}

struct nvmlComputeInstancePlacement_st
    start::Cuint
    size::Cuint
end

const nvmlComputeInstancePlacement_t = nvmlComputeInstancePlacement_st

struct nvmlComputeInstanceInfo_st
    device::nvmlDevice_t
    gpuInstance::nvmlGpuInstance_t
    id::Cuint
    profileId::Cuint
    placement::nvmlComputeInstancePlacement_t
end

const nvmlComputeInstanceInfo_t = nvmlComputeInstanceInfo_st

@checked function nvmlComputeInstanceGetInfo_v2(computeInstance, info)
    initialize_context()
    @ccall (libnvml()).nvmlComputeInstanceGetInfo_v2(computeInstance::nvmlComputeInstance_t,
                                                     info::Ptr{nvmlComputeInstanceInfo_t})::nvmlReturn_t
end

struct nvmlProcessInfo_st
    pid::Cuint
    usedGpuMemory::Culonglong
    gpuInstanceId::Cuint
    computeInstanceId::Cuint
end

const nvmlProcessInfo_t = nvmlProcessInfo_st

@checked function nvmlDeviceGetComputeRunningProcesses_v3(device, infoCount, infos)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetComputeRunningProcesses_v3(device::nvmlDevice_t,
                                                               infoCount::Ptr{Cuint},
                                                               infos::Ptr{nvmlProcessInfo_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetGraphicsRunningProcesses_v3(device, infoCount, infos)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetGraphicsRunningProcesses_v3(device::nvmlDevice_t,
                                                                infoCount::Ptr{Cuint},
                                                                infos::Ptr{nvmlProcessInfo_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetMPSComputeRunningProcesses_v3(device, infoCount, infos)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetMPSComputeRunningProcesses_v3(device::nvmlDevice_t,
                                                                  infoCount::Ptr{Cuint},
                                                                  infos::Ptr{nvmlProcessInfo_t})::nvmlReturn_t
end

struct nvmlExcludedDeviceInfo_st
    pciInfo::nvmlPciInfo_t
    uuid::NTuple{80,Cchar}
end

const nvmlExcludedDeviceInfo_t = nvmlExcludedDeviceInfo_st

@checked function nvmlGetExcludedDeviceCount(deviceCount)
    initialize_context()
    @ccall (libnvml()).nvmlGetExcludedDeviceCount(deviceCount::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlGetExcludedDeviceInfoByIndex(index, info)
    initialize_context()
    @ccall (libnvml()).nvmlGetExcludedDeviceInfoByIndex(index::Cuint,
                                                        info::Ptr{nvmlExcludedDeviceInfo_t})::nvmlReturn_t
end

struct nvmlGpuInstancePlacement_st
    start::Cuint
    size::Cuint
end

const nvmlGpuInstancePlacement_t = nvmlGpuInstancePlacement_st

@checked function nvmlDeviceGetGpuInstancePossiblePlacements_v2(device, profileId,
                                                                placements, count)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetGpuInstancePossiblePlacements_v2(device::nvmlDevice_t,
                                                                     profileId::Cuint,
                                                                     placements::Ptr{nvmlGpuInstancePlacement_t},
                                                                     count::Ptr{Cuint})::nvmlReturn_t
end

const nvmlVgpuInstance_t = Cuint

struct nvmlVgpuLicenseExpiry_st
    year::Cuint
    month::Cushort
    day::Cushort
    hour::Cushort
    min::Cushort
    sec::Cushort
    status::Cuchar
end

const nvmlVgpuLicenseExpiry_t = nvmlVgpuLicenseExpiry_st

struct nvmlVgpuLicenseInfo_st
    isLicensed::Cuchar
    licenseExpiry::nvmlVgpuLicenseExpiry_t
    currentState::Cuint
end

const nvmlVgpuLicenseInfo_t = nvmlVgpuLicenseInfo_st

@checked function nvmlVgpuInstanceGetLicenseInfo_v2(vgpuInstance, licenseInfo)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetLicenseInfo_v2(vgpuInstance::nvmlVgpuInstance_t,
                                                         licenseInfo::Ptr{nvmlVgpuLicenseInfo_t})::nvmlReturn_t
end

@cenum nvmlMemoryErrorType_enum::UInt32 begin
    NVML_MEMORY_ERROR_TYPE_CORRECTED = 0
    NVML_MEMORY_ERROR_TYPE_UNCORRECTED = 1
    NVML_MEMORY_ERROR_TYPE_COUNT = 2
end

const nvmlMemoryErrorType_t = nvmlMemoryErrorType_enum

struct nvmlEccErrorCounts_st
    l1Cache::Culonglong
    l2Cache::Culonglong
    deviceMemory::Culonglong
    registerFile::Culonglong
end

const nvmlEccErrorCounts_t = nvmlEccErrorCounts_st

struct nvmlUtilization_st
    gpu::Cuint
    memory::Cuint
end

const nvmlUtilization_t = nvmlUtilization_st

struct nvmlMemory_st
    total::Culonglong
    free::Culonglong
    used::Culonglong
end

const nvmlMemory_t = nvmlMemory_st

struct nvmlMemory_v2_st
    version::Cuint
    total::Culonglong
    reserved::Culonglong
    free::Culonglong
    used::Culonglong
end

const nvmlMemory_v2_t = nvmlMemory_v2_st

struct nvmlBAR1Memory_st
    bar1Total::Culonglong
    bar1Free::Culonglong
    bar1Used::Culonglong
end

const nvmlBAR1Memory_t = nvmlBAR1Memory_st

struct nvmlProcessInfo_v1_st
    pid::Cuint
    usedGpuMemory::Culonglong
end

const nvmlProcessInfo_v1_t = nvmlProcessInfo_v1_st

struct nvmlProcessInfo_v2_st
    pid::Cuint
    usedGpuMemory::Culonglong
    gpuInstanceId::Cuint
    computeInstanceId::Cuint
end

const nvmlProcessInfo_v2_t = nvmlProcessInfo_v2_st

struct nvmlRowRemapperHistogramValues_st
    max::Cuint
    high::Cuint
    partial::Cuint
    low::Cuint
    none::Cuint
end

const nvmlRowRemapperHistogramValues_t = nvmlRowRemapperHistogramValues_st

@cenum nvmlBridgeChipType_enum::UInt32 begin
    NVML_BRIDGE_CHIP_PLX = 0
    NVML_BRIDGE_CHIP_BRO4 = 1
end

const nvmlBridgeChipType_t = nvmlBridgeChipType_enum

@cenum nvmlNvLinkUtilizationCountUnits_enum::UInt32 begin
    NVML_NVLINK_COUNTER_UNIT_CYCLES = 0
    NVML_NVLINK_COUNTER_UNIT_PACKETS = 1
    NVML_NVLINK_COUNTER_UNIT_BYTES = 2
    NVML_NVLINK_COUNTER_UNIT_RESERVED = 3
    NVML_NVLINK_COUNTER_UNIT_COUNT = 4
end

const nvmlNvLinkUtilizationCountUnits_t = nvmlNvLinkUtilizationCountUnits_enum

@cenum nvmlNvLinkUtilizationCountPktTypes_enum::UInt32 begin
    NVML_NVLINK_COUNTER_PKTFILTER_NOP = 1
    NVML_NVLINK_COUNTER_PKTFILTER_READ = 2
    NVML_NVLINK_COUNTER_PKTFILTER_WRITE = 4
    NVML_NVLINK_COUNTER_PKTFILTER_RATOM = 8
    NVML_NVLINK_COUNTER_PKTFILTER_NRATOM = 16
    NVML_NVLINK_COUNTER_PKTFILTER_FLUSH = 32
    NVML_NVLINK_COUNTER_PKTFILTER_RESPDATA = 64
    NVML_NVLINK_COUNTER_PKTFILTER_RESPNODATA = 128
    NVML_NVLINK_COUNTER_PKTFILTER_ALL = 255
end

const nvmlNvLinkUtilizationCountPktTypes_t = nvmlNvLinkUtilizationCountPktTypes_enum

struct nvmlNvLinkUtilizationControl_st
    units::nvmlNvLinkUtilizationCountUnits_t
    pktfilter::nvmlNvLinkUtilizationCountPktTypes_t
end

const nvmlNvLinkUtilizationControl_t = nvmlNvLinkUtilizationControl_st

@cenum nvmlNvLinkCapability_enum::UInt32 begin
    NVML_NVLINK_CAP_P2P_SUPPORTED = 0
    NVML_NVLINK_CAP_SYSMEM_ACCESS = 1
    NVML_NVLINK_CAP_P2P_ATOMICS = 2
    NVML_NVLINK_CAP_SYSMEM_ATOMICS = 3
    NVML_NVLINK_CAP_SLI_BRIDGE = 4
    NVML_NVLINK_CAP_VALID = 5
    NVML_NVLINK_CAP_COUNT = 6
end

const nvmlNvLinkCapability_t = nvmlNvLinkCapability_enum

@cenum nvmlNvLinkErrorCounter_enum::UInt32 begin
    NVML_NVLINK_ERROR_DL_REPLAY = 0
    NVML_NVLINK_ERROR_DL_RECOVERY = 1
    NVML_NVLINK_ERROR_DL_CRC_FLIT = 2
    NVML_NVLINK_ERROR_DL_CRC_DATA = 3
    NVML_NVLINK_ERROR_DL_ECC_DATA = 4
    NVML_NVLINK_ERROR_COUNT = 5
end

const nvmlNvLinkErrorCounter_t = nvmlNvLinkErrorCounter_enum

@cenum nvmlIntNvLinkDeviceType_enum::UInt32 begin
    NVML_NVLINK_DEVICE_TYPE_GPU = 0
    NVML_NVLINK_DEVICE_TYPE_IBMNPU = 1
    NVML_NVLINK_DEVICE_TYPE_SWITCH = 2
    NVML_NVLINK_DEVICE_TYPE_UNKNOWN = 255
end

const nvmlIntNvLinkDeviceType_t = nvmlIntNvLinkDeviceType_enum

@cenum nvmlGpuLevel_enum::UInt32 begin
    NVML_TOPOLOGY_INTERNAL = 0
    NVML_TOPOLOGY_SINGLE = 10
    NVML_TOPOLOGY_MULTIPLE = 20
    NVML_TOPOLOGY_HOSTBRIDGE = 30
    NVML_TOPOLOGY_NODE = 40
    NVML_TOPOLOGY_SYSTEM = 50
end

const nvmlGpuTopologyLevel_t = nvmlGpuLevel_enum

@cenum nvmlGpuP2PStatus_enum::UInt32 begin
    NVML_P2P_STATUS_OK = 0
    NVML_P2P_STATUS_CHIPSET_NOT_SUPPORED = 1
    NVML_P2P_STATUS_CHIPSET_NOT_SUPPORTED = 1
    NVML_P2P_STATUS_GPU_NOT_SUPPORTED = 2
    NVML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED = 3
    NVML_P2P_STATUS_DISABLED_BY_REGKEY = 4
    NVML_P2P_STATUS_NOT_SUPPORTED = 5
    NVML_P2P_STATUS_UNKNOWN = 6
end

const nvmlGpuP2PStatus_t = nvmlGpuP2PStatus_enum

@cenum nvmlGpuP2PCapsIndex_enum::UInt32 begin
    NVML_P2P_CAPS_INDEX_READ = 0
    NVML_P2P_CAPS_INDEX_WRITE = 1
    NVML_P2P_CAPS_INDEX_NVLINK = 2
    NVML_P2P_CAPS_INDEX_ATOMICS = 3
    NVML_P2P_CAPS_INDEX_PROP = 4
    NVML_P2P_CAPS_INDEX_UNKNOWN = 5
end

const nvmlGpuP2PCapsIndex_t = nvmlGpuP2PCapsIndex_enum

struct nvmlBridgeChipInfo_st
    type::nvmlBridgeChipType_t
    fwVersion::Cuint
end

const nvmlBridgeChipInfo_t = nvmlBridgeChipInfo_st

struct nvmlBridgeChipHierarchy_st
    bridgeCount::Cuchar
    bridgeChipInfo::NTuple{128,nvmlBridgeChipInfo_t}
end

const nvmlBridgeChipHierarchy_t = nvmlBridgeChipHierarchy_st

@cenum nvmlSamplingType_enum::UInt32 begin
    NVML_TOTAL_POWER_SAMPLES = 0
    NVML_GPU_UTILIZATION_SAMPLES = 1
    NVML_MEMORY_UTILIZATION_SAMPLES = 2
    NVML_ENC_UTILIZATION_SAMPLES = 3
    NVML_DEC_UTILIZATION_SAMPLES = 4
    NVML_PROCESSOR_CLK_SAMPLES = 5
    NVML_MEMORY_CLK_SAMPLES = 6
    NVML_SAMPLINGTYPE_COUNT = 7
end

const nvmlSamplingType_t = nvmlSamplingType_enum

@cenum nvmlPcieUtilCounter_enum::UInt32 begin
    NVML_PCIE_UTIL_TX_BYTES = 0
    NVML_PCIE_UTIL_RX_BYTES = 1
    NVML_PCIE_UTIL_COUNT = 2
end

const nvmlPcieUtilCounter_t = nvmlPcieUtilCounter_enum

@cenum nvmlValueType_enum::UInt32 begin
    NVML_VALUE_TYPE_DOUBLE = 0
    NVML_VALUE_TYPE_UNSIGNED_INT = 1
    NVML_VALUE_TYPE_UNSIGNED_LONG = 2
    NVML_VALUE_TYPE_UNSIGNED_LONG_LONG = 3
    NVML_VALUE_TYPE_SIGNED_LONG_LONG = 4
    NVML_VALUE_TYPE_COUNT = 5
end

const nvmlValueType_t = nvmlValueType_enum

struct nvmlValue_st
    data::NTuple{8,UInt8}
end

function Base.getproperty(x::Ptr{nvmlValue_st}, f::Symbol)
    f === :dVal && return Ptr{Cdouble}(x + 0)
    f === :uiVal && return Ptr{Cuint}(x + 0)
    f === :ulVal && return Ptr{Culong}(x + 0)
    f === :ullVal && return Ptr{Culonglong}(x + 0)
    f === :sllVal && return Ptr{Clonglong}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::nvmlValue_st, f::Symbol)
    r = Ref{nvmlValue_st}(x)
    ptr = Base.unsafe_convert(Ptr{nvmlValue_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{nvmlValue_st}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

const nvmlValue_t = nvmlValue_st

struct nvmlSample_st
    timeStamp::Culonglong
    sampleValue::nvmlValue_t
end

const nvmlSample_t = nvmlSample_st

@cenum nvmlPerfPolicyType_enum::UInt32 begin
    NVML_PERF_POLICY_POWER = 0
    NVML_PERF_POLICY_THERMAL = 1
    NVML_PERF_POLICY_SYNC_BOOST = 2
    NVML_PERF_POLICY_BOARD_LIMIT = 3
    NVML_PERF_POLICY_LOW_UTILIZATION = 4
    NVML_PERF_POLICY_RELIABILITY = 5
    NVML_PERF_POLICY_TOTAL_APP_CLOCKS = 10
    NVML_PERF_POLICY_TOTAL_BASE_CLOCKS = 11
    NVML_PERF_POLICY_COUNT = 12
end

const nvmlPerfPolicyType_t = nvmlPerfPolicyType_enum

struct nvmlViolationTime_st
    referenceTime::Culonglong
    violationTime::Culonglong
end

const nvmlViolationTime_t = nvmlViolationTime_st

@cenum nvmlThermalTarget_t::Int32 begin
    NVML_THERMAL_TARGET_NONE = 0
    NVML_THERMAL_TARGET_GPU = 1
    NVML_THERMAL_TARGET_MEMORY = 2
    NVML_THERMAL_TARGET_POWER_SUPPLY = 4
    NVML_THERMAL_TARGET_BOARD = 8
    NVML_THERMAL_TARGET_VCD_BOARD = 9
    NVML_THERMAL_TARGET_VCD_INLET = 10
    NVML_THERMAL_TARGET_VCD_OUTLET = 11
    NVML_THERMAL_TARGET_ALL = 15
    NVML_THERMAL_TARGET_UNKNOWN = -1
end

@cenum nvmlThermalController_t::Int32 begin
    NVML_THERMAL_CONTROLLER_NONE = 0
    NVML_THERMAL_CONTROLLER_GPU_INTERNAL = 1
    NVML_THERMAL_CONTROLLER_ADM1032 = 2
    NVML_THERMAL_CONTROLLER_ADT7461 = 3
    NVML_THERMAL_CONTROLLER_MAX6649 = 4
    NVML_THERMAL_CONTROLLER_MAX1617 = 5
    NVML_THERMAL_CONTROLLER_LM99 = 6
    NVML_THERMAL_CONTROLLER_LM89 = 7
    NVML_THERMAL_CONTROLLER_LM64 = 8
    NVML_THERMAL_CONTROLLER_G781 = 9
    NVML_THERMAL_CONTROLLER_ADT7473 = 10
    NVML_THERMAL_CONTROLLER_SBMAX6649 = 11
    NVML_THERMAL_CONTROLLER_VBIOSEVT = 12
    NVML_THERMAL_CONTROLLER_OS = 13
    NVML_THERMAL_CONTROLLER_NVSYSCON_CANOAS = 14
    NVML_THERMAL_CONTROLLER_NVSYSCON_E551 = 15
    NVML_THERMAL_CONTROLLER_MAX6649R = 16
    NVML_THERMAL_CONTROLLER_ADT7473S = 17
    NVML_THERMAL_CONTROLLER_UNKNOWN = -1
end

struct var"##Ctag#378"
    controller::nvmlThermalController_t
    defaultMinTemp::Cint
    defaultMaxTemp::Cint
    currentTemp::Cint
    target::nvmlThermalTarget_t
end
function Base.getproperty(x::Ptr{var"##Ctag#378"}, f::Symbol)
    f === :controller && return Ptr{nvmlThermalController_t}(x + 0)
    f === :defaultMinTemp && return Ptr{Cint}(x + 4)
    f === :defaultMaxTemp && return Ptr{Cint}(x + 8)
    f === :currentTemp && return Ptr{Cint}(x + 12)
    f === :target && return Ptr{nvmlThermalTarget_t}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#378", f::Symbol)
    r = Ref{var"##Ctag#378"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#378"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#378"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct nvmlGpuThermalSettings_t
    data::NTuple{64,UInt8}
end

function Base.getproperty(x::Ptr{nvmlGpuThermalSettings_t}, f::Symbol)
    f === :count && return Ptr{Cuint}(x + 0)
    f === :sensor && return Ptr{NTuple{3,var"##Ctag#378"}}(x + 4)
    return getfield(x, f)
end

function Base.getproperty(x::nvmlGpuThermalSettings_t, f::Symbol)
    r = Ref{nvmlGpuThermalSettings_t}(x)
    ptr = Base.unsafe_convert(Ptr{nvmlGpuThermalSettings_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{nvmlGpuThermalSettings_t}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

@cenum nvmlEnableState_enum::UInt32 begin
    NVML_FEATURE_DISABLED = 0
    NVML_FEATURE_ENABLED = 1
end

const nvmlEnableState_t = nvmlEnableState_enum

@cenum nvmlBrandType_enum::UInt32 begin
    NVML_BRAND_UNKNOWN = 0
    NVML_BRAND_QUADRO = 1
    NVML_BRAND_TESLA = 2
    NVML_BRAND_NVS = 3
    NVML_BRAND_GRID = 4
    NVML_BRAND_GEFORCE = 5
    NVML_BRAND_TITAN = 6
    NVML_BRAND_NVIDIA_VAPPS = 7
    NVML_BRAND_NVIDIA_VPC = 8
    NVML_BRAND_NVIDIA_VCS = 9
    NVML_BRAND_NVIDIA_VWS = 10
    NVML_BRAND_NVIDIA_CLOUD_GAMING = 11
    NVML_BRAND_NVIDIA_VGAMING = 11
    NVML_BRAND_QUADRO_RTX = 12
    NVML_BRAND_NVIDIA_RTX = 13
    NVML_BRAND_NVIDIA = 14
    NVML_BRAND_GEFORCE_RTX = 15
    NVML_BRAND_TITAN_RTX = 16
    NVML_BRAND_COUNT = 17
end

const nvmlBrandType_t = nvmlBrandType_enum

@cenum nvmlTemperatureThresholds_enum::UInt32 begin
    NVML_TEMPERATURE_THRESHOLD_SHUTDOWN = 0
    NVML_TEMPERATURE_THRESHOLD_SLOWDOWN = 1
    NVML_TEMPERATURE_THRESHOLD_MEM_MAX = 2
    NVML_TEMPERATURE_THRESHOLD_GPU_MAX = 3
    NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MIN = 4
    NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_CURR = 5
    NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MAX = 6
    NVML_TEMPERATURE_THRESHOLD_COUNT = 7
end

const nvmlTemperatureThresholds_t = nvmlTemperatureThresholds_enum

@cenum nvmlTemperatureSensors_enum::UInt32 begin
    NVML_TEMPERATURE_GPU = 0
    NVML_TEMPERATURE_COUNT = 1
end

const nvmlTemperatureSensors_t = nvmlTemperatureSensors_enum

@cenum nvmlComputeMode_enum::UInt32 begin
    NVML_COMPUTEMODE_DEFAULT = 0
    NVML_COMPUTEMODE_EXCLUSIVE_THREAD = 1
    NVML_COMPUTEMODE_PROHIBITED = 2
    NVML_COMPUTEMODE_EXCLUSIVE_PROCESS = 3
    NVML_COMPUTEMODE_COUNT = 4
end

const nvmlComputeMode_t = nvmlComputeMode_enum

struct nvmlClkMonFaultInfo_struct
    clkApiDomain::Cuint
    clkDomainFaultMask::Cuint
end

const nvmlClkMonFaultInfo_t = nvmlClkMonFaultInfo_struct

struct nvmlClkMonStatus_status
    bGlobalStatus::Cuint
    clkMonListSize::Cuint
    clkMonList::NTuple{32,nvmlClkMonFaultInfo_t}
end

const nvmlClkMonStatus_t = nvmlClkMonStatus_status

@cenum nvmlEccCounterType_enum::UInt32 begin
    NVML_VOLATILE_ECC = 0
    NVML_AGGREGATE_ECC = 1
    NVML_ECC_COUNTER_TYPE_COUNT = 2
end

const nvmlEccCounterType_t = nvmlEccCounterType_enum

@cenum nvmlClockType_enum::UInt32 begin
    NVML_CLOCK_GRAPHICS = 0
    NVML_CLOCK_SM = 1
    NVML_CLOCK_MEM = 2
    NVML_CLOCK_VIDEO = 3
    NVML_CLOCK_COUNT = 4
end

const nvmlClockType_t = nvmlClockType_enum

@cenum nvmlClockId_enum::UInt32 begin
    NVML_CLOCK_ID_CURRENT = 0
    NVML_CLOCK_ID_APP_CLOCK_TARGET = 1
    NVML_CLOCK_ID_APP_CLOCK_DEFAULT = 2
    NVML_CLOCK_ID_CUSTOMER_BOOST_MAX = 3
    NVML_CLOCK_ID_COUNT = 4
end

const nvmlClockId_t = nvmlClockId_enum

@cenum nvmlDriverModel_enum::UInt32 begin
    NVML_DRIVER_WDDM = 0
    NVML_DRIVER_WDM = 1
end

const nvmlDriverModel_t = nvmlDriverModel_enum

@cenum nvmlPStates_enum::UInt32 begin
    NVML_PSTATE_0 = 0
    NVML_PSTATE_1 = 1
    NVML_PSTATE_2 = 2
    NVML_PSTATE_3 = 3
    NVML_PSTATE_4 = 4
    NVML_PSTATE_5 = 5
    NVML_PSTATE_6 = 6
    NVML_PSTATE_7 = 7
    NVML_PSTATE_8 = 8
    NVML_PSTATE_9 = 9
    NVML_PSTATE_10 = 10
    NVML_PSTATE_11 = 11
    NVML_PSTATE_12 = 12
    NVML_PSTATE_13 = 13
    NVML_PSTATE_14 = 14
    NVML_PSTATE_15 = 15
    NVML_PSTATE_UNKNOWN = 32
end

const nvmlPstates_t = nvmlPStates_enum

@cenum nvmlGom_enum::UInt32 begin
    NVML_GOM_ALL_ON = 0
    NVML_GOM_COMPUTE = 1
    NVML_GOM_LOW_DP = 2
end

const nvmlGpuOperationMode_t = nvmlGom_enum

@cenum nvmlInforomObject_enum::UInt32 begin
    NVML_INFOROM_OEM = 0
    NVML_INFOROM_ECC = 1
    NVML_INFOROM_POWER = 2
    NVML_INFOROM_COUNT = 3
end

const nvmlInforomObject_t = nvmlInforomObject_enum

@cenum nvmlMemoryLocation_enum::UInt32 begin
    NVML_MEMORY_LOCATION_L1_CACHE = 0
    NVML_MEMORY_LOCATION_L2_CACHE = 1
    NVML_MEMORY_LOCATION_DRAM = 2
    NVML_MEMORY_LOCATION_DEVICE_MEMORY = 2
    NVML_MEMORY_LOCATION_REGISTER_FILE = 3
    NVML_MEMORY_LOCATION_TEXTURE_MEMORY = 4
    NVML_MEMORY_LOCATION_TEXTURE_SHM = 5
    NVML_MEMORY_LOCATION_CBU = 6
    NVML_MEMORY_LOCATION_SRAM = 7
    NVML_MEMORY_LOCATION_COUNT = 8
end

const nvmlMemoryLocation_t = nvmlMemoryLocation_enum

@cenum nvmlPageRetirementCause_enum::UInt32 begin
    NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS = 0
    NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR = 1
    NVML_PAGE_RETIREMENT_CAUSE_COUNT = 2
end

const nvmlPageRetirementCause_t = nvmlPageRetirementCause_enum

@cenum nvmlRestrictedAPI_enum::UInt32 begin
    NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS = 0
    NVML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS = 1
    NVML_RESTRICTED_API_COUNT = 2
end

const nvmlRestrictedAPI_t = nvmlRestrictedAPI_enum

@cenum nvmlGpuVirtualizationMode::UInt32 begin
    NVML_GPU_VIRTUALIZATION_MODE_NONE = 0
    NVML_GPU_VIRTUALIZATION_MODE_PASSTHROUGH = 1
    NVML_GPU_VIRTUALIZATION_MODE_VGPU = 2
    NVML_GPU_VIRTUALIZATION_MODE_HOST_VGPU = 3
    NVML_GPU_VIRTUALIZATION_MODE_HOST_VSGA = 4
end

const nvmlGpuVirtualizationMode_t = nvmlGpuVirtualizationMode

@cenum nvmlHostVgpuMode_enum::UInt32 begin
    NVML_HOST_VGPU_MODE_NON_SRIOV = 0
    NVML_HOST_VGPU_MODE_SRIOV = 1
end

const nvmlHostVgpuMode_t = nvmlHostVgpuMode_enum

@cenum nvmlVgpuVmIdType::UInt32 begin
    NVML_VGPU_VM_ID_DOMAIN_ID = 0
    NVML_VGPU_VM_ID_UUID = 1
end

const nvmlVgpuVmIdType_t = nvmlVgpuVmIdType

@cenum nvmlVgpuGuestInfoState_enum::UInt32 begin
    NVML_VGPU_INSTANCE_GUEST_INFO_STATE_UNINITIALIZED = 0
    NVML_VGPU_INSTANCE_GUEST_INFO_STATE_INITIALIZED = 1
end

const nvmlVgpuGuestInfoState_t = nvmlVgpuGuestInfoState_enum

@cenum nvmlVgpuCapability_enum::UInt32 begin
    NVML_VGPU_CAP_NVLINK_P2P = 0
    NVML_VGPU_CAP_GPUDIRECT = 1
    NVML_VGPU_CAP_MULTI_VGPU_EXCLUSIVE = 2
    NVML_VGPU_CAP_EXCLUSIVE_TYPE = 3
    NVML_VGPU_CAP_EXCLUSIVE_SIZE = 4
    NVML_VGPU_CAP_COUNT = 5
end

const nvmlVgpuCapability_t = nvmlVgpuCapability_enum

@cenum nvmlVgpuDriverCapability_enum::UInt32 begin
    NVML_VGPU_DRIVER_CAP_HETEROGENEOUS_MULTI_VGPU = 0
    NVML_VGPU_DRIVER_CAP_COUNT = 1
end

const nvmlVgpuDriverCapability_t = nvmlVgpuDriverCapability_enum

@cenum nvmlDeviceVgpuCapability_enum::UInt32 begin
    NVML_DEVICE_VGPU_CAP_FRACTIONAL_MULTI_VGPU = 0
    NVML_DEVICE_VGPU_CAP_HETEROGENEOUS_TIMESLICE_PROFILES = 1
    NVML_DEVICE_VGPU_CAP_HETEROGENEOUS_TIMESLICE_SIZES = 2
    NVML_DEVICE_VGPU_CAP_READ_DEVICE_BUFFER_BW = 3
    NVML_DEVICE_VGPU_CAP_WRITE_DEVICE_BUFFER_BW = 4
    NVML_DEVICE_VGPU_CAP_COUNT = 5
end

const nvmlDeviceVgpuCapability_t = nvmlDeviceVgpuCapability_enum

const nvmlVgpuTypeId_t = Cuint

struct nvmlVgpuInstanceUtilizationSample_st
    vgpuInstance::nvmlVgpuInstance_t
    timeStamp::Culonglong
    smUtil::nvmlValue_t
    memUtil::nvmlValue_t
    encUtil::nvmlValue_t
    decUtil::nvmlValue_t
end

const nvmlVgpuInstanceUtilizationSample_t = nvmlVgpuInstanceUtilizationSample_st

struct nvmlVgpuProcessUtilizationSample_st
    vgpuInstance::nvmlVgpuInstance_t
    pid::Cuint
    processName::NTuple{64,Cchar}
    timeStamp::Culonglong
    smUtil::Cuint
    memUtil::Cuint
    encUtil::Cuint
    decUtil::Cuint
end

const nvmlVgpuProcessUtilizationSample_t = nvmlVgpuProcessUtilizationSample_st

struct nvmlProcessUtilizationSample_st
    pid::Cuint
    timeStamp::Culonglong
    smUtil::Cuint
    memUtil::Cuint
    encUtil::Cuint
    decUtil::Cuint
end

const nvmlProcessUtilizationSample_t = nvmlProcessUtilizationSample_st

const nvmlDeviceArchitecture_t = Cuint

const nvmlBusType_t = Cuint

const nvmlFanControlPolicy_t = Cuint

const nvmlPowerSource_t = Cuint

@cenum nvmlGpuUtilizationDomainId_t::UInt32 begin
    NVML_GPU_UTILIZATION_DOMAIN_GPU = 0
    NVML_GPU_UTILIZATION_DOMAIN_FB = 1
    NVML_GPU_UTILIZATION_DOMAIN_VID = 2
    NVML_GPU_UTILIZATION_DOMAIN_BUS = 3
end

struct var"##Ctag#379"
    bIsPresent::Cuint
    percentage::Cuint
    incThreshold::Cuint
    decThreshold::Cuint
end
function Base.getproperty(x::Ptr{var"##Ctag#379"}, f::Symbol)
    f === :bIsPresent && return Ptr{Cuint}(x + 0)
    f === :percentage && return Ptr{Cuint}(x + 4)
    f === :incThreshold && return Ptr{Cuint}(x + 8)
    f === :decThreshold && return Ptr{Cuint}(x + 12)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#379", f::Symbol)
    r = Ref{var"##Ctag#379"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#379"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#379"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct nvmlGpuDynamicPstatesInfo_st
    data::NTuple{132,UInt8}
end

function Base.getproperty(x::Ptr{nvmlGpuDynamicPstatesInfo_st}, f::Symbol)
    f === :flags && return Ptr{Cuint}(x + 0)
    f === :utilization && return Ptr{NTuple{8,var"##Ctag#379"}}(x + 4)
    return getfield(x, f)
end

function Base.getproperty(x::nvmlGpuDynamicPstatesInfo_st, f::Symbol)
    r = Ref{nvmlGpuDynamicPstatesInfo_st}(x)
    ptr = Base.unsafe_convert(Ptr{nvmlGpuDynamicPstatesInfo_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{nvmlGpuDynamicPstatesInfo_st}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

const nvmlGpuDynamicPstatesInfo_t = nvmlGpuDynamicPstatesInfo_st

struct nvmlFieldValue_st
    fieldId::Cuint
    scopeId::Cuint
    timestamp::Clonglong
    latencyUsec::Clonglong
    valueType::nvmlValueType_t
    nvmlReturn::nvmlReturn_t
    value::nvmlValue_t
end

const nvmlFieldValue_t = nvmlFieldValue_st

mutable struct nvmlUnit_st end

const nvmlUnit_t = Ptr{nvmlUnit_st}

struct nvmlHwbcEntry_st
    hwbcId::Cuint
    firmwareVersion::NTuple{32,Cchar}
end

const nvmlHwbcEntry_t = nvmlHwbcEntry_st

@cenum nvmlFanState_enum::UInt32 begin
    NVML_FAN_NORMAL = 0
    NVML_FAN_FAILED = 1
end

const nvmlFanState_t = nvmlFanState_enum

@cenum nvmlLedColor_enum::UInt32 begin
    NVML_LED_COLOR_GREEN = 0
    NVML_LED_COLOR_AMBER = 1
end

const nvmlLedColor_t = nvmlLedColor_enum

struct nvmlLedState_st
    cause::NTuple{256,Cchar}
    color::nvmlLedColor_t
end

const nvmlLedState_t = nvmlLedState_st

struct nvmlUnitInfo_st
    name::NTuple{96,Cchar}
    id::NTuple{96,Cchar}
    serial::NTuple{96,Cchar}
    firmwareVersion::NTuple{96,Cchar}
end

const nvmlUnitInfo_t = nvmlUnitInfo_st

struct nvmlPSUInfo_st
    state::NTuple{256,Cchar}
    current::Cuint
    voltage::Cuint
    power::Cuint
end

const nvmlPSUInfo_t = nvmlPSUInfo_st

struct nvmlUnitFanInfo_st
    speed::Cuint
    state::nvmlFanState_t
end

const nvmlUnitFanInfo_t = nvmlUnitFanInfo_st

struct nvmlUnitFanSpeeds_st
    fans::NTuple{24,nvmlUnitFanInfo_t}
    count::Cuint
end

const nvmlUnitFanSpeeds_t = nvmlUnitFanSpeeds_st

struct nvmlAccountingStats_st
    gpuUtilization::Cuint
    memoryUtilization::Cuint
    maxMemoryUsage::Culonglong
    time::Culonglong
    startTime::Culonglong
    isRunning::Cuint
    reserved::NTuple{5,Cuint}
end

const nvmlAccountingStats_t = nvmlAccountingStats_st

@cenum nvmlEncoderQueryType_enum::UInt32 begin
    NVML_ENCODER_QUERY_H264 = 0
    NVML_ENCODER_QUERY_HEVC = 1
end

const nvmlEncoderType_t = nvmlEncoderQueryType_enum

struct nvmlEncoderSessionInfo_st
    sessionId::Cuint
    pid::Cuint
    vgpuInstance::nvmlVgpuInstance_t
    codecType::nvmlEncoderType_t
    hResolution::Cuint
    vResolution::Cuint
    averageFps::Cuint
    averageLatency::Cuint
end

const nvmlEncoderSessionInfo_t = nvmlEncoderSessionInfo_st

@cenum nvmlFBCSessionType_enum::UInt32 begin
    NVML_FBC_SESSION_TYPE_UNKNOWN = 0
    NVML_FBC_SESSION_TYPE_TOSYS = 1
    NVML_FBC_SESSION_TYPE_CUDA = 2
    NVML_FBC_SESSION_TYPE_VID = 3
    NVML_FBC_SESSION_TYPE_HWENC = 4
end

const nvmlFBCSessionType_t = nvmlFBCSessionType_enum

struct nvmlFBCStats_st
    sessionsCount::Cuint
    averageFPS::Cuint
    averageLatency::Cuint
end

const nvmlFBCStats_t = nvmlFBCStats_st

struct nvmlFBCSessionInfo_st
    sessionId::Cuint
    pid::Cuint
    vgpuInstance::nvmlVgpuInstance_t
    displayOrdinal::Cuint
    sessionType::nvmlFBCSessionType_t
    sessionFlags::Cuint
    hMaxResolution::Cuint
    vMaxResolution::Cuint
    hResolution::Cuint
    vResolution::Cuint
    averageFPS::Cuint
    averageLatency::Cuint
end

const nvmlFBCSessionInfo_t = nvmlFBCSessionInfo_st

const nvmlGpuFabricState_t = Cuchar

struct nvmlGpuFabricInfo_t
    clusterUuid::NTuple{16,Cchar}
    status::nvmlReturn_t
    partitionId::Cuint
    state::nvmlGpuFabricState_t
end

@checked function nvmlInitWithFlags(flags)
    @ccall (libnvml()).nvmlInitWithFlags(flags::Cuint)::nvmlReturn_t
end

@checked function nvmlShutdown()
    @ccall (libnvml()).nvmlShutdown()::nvmlReturn_t
end

function nvmlErrorString(result)
    @ccall (libnvml()).nvmlErrorString(result::nvmlReturn_t)::Cstring
end

@checked function nvmlSystemGetDriverVersion(version, length)
    initialize_context()
    @ccall (libnvml()).nvmlSystemGetDriverVersion(version::Cstring,
                                                  length::Cuint)::nvmlReturn_t
end

@checked function nvmlSystemGetNVMLVersion(version, length)
    initialize_context()
    @ccall (libnvml()).nvmlSystemGetNVMLVersion(version::Cstring,
                                                length::Cuint)::nvmlReturn_t
end

@checked function nvmlSystemGetCudaDriverVersion(cudaDriverVersion)
    initialize_context()
    @ccall (libnvml()).nvmlSystemGetCudaDriverVersion(cudaDriverVersion::Ptr{Cint})::nvmlReturn_t
end

@checked function nvmlSystemGetCudaDriverVersion_v2(cudaDriverVersion)
    initialize_context()
    @ccall (libnvml()).nvmlSystemGetCudaDriverVersion_v2(cudaDriverVersion::Ptr{Cint})::nvmlReturn_t
end

@checked function nvmlSystemGetProcessName(pid, name, length)
    initialize_context()
    @ccall (libnvml()).nvmlSystemGetProcessName(pid::Cuint, name::Cstring,
                                                length::Cuint)::nvmlReturn_t
end

@checked function nvmlUnitGetCount(unitCount)
    initialize_context()
    @ccall (libnvml()).nvmlUnitGetCount(unitCount::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlUnitGetHandleByIndex(index, unit)
    initialize_context()
    @ccall (libnvml()).nvmlUnitGetHandleByIndex(index::Cuint,
                                                unit::Ptr{nvmlUnit_t})::nvmlReturn_t
end

@checked function nvmlUnitGetUnitInfo(unit, info)
    initialize_context()
    @ccall (libnvml()).nvmlUnitGetUnitInfo(unit::nvmlUnit_t,
                                           info::Ptr{nvmlUnitInfo_t})::nvmlReturn_t
end

@checked function nvmlUnitGetLedState(unit, state)
    initialize_context()
    @ccall (libnvml()).nvmlUnitGetLedState(unit::nvmlUnit_t,
                                           state::Ptr{nvmlLedState_t})::nvmlReturn_t
end

@checked function nvmlUnitGetPsuInfo(unit, psu)
    initialize_context()
    @ccall (libnvml()).nvmlUnitGetPsuInfo(unit::nvmlUnit_t,
                                          psu::Ptr{nvmlPSUInfo_t})::nvmlReturn_t
end

@checked function nvmlUnitGetTemperature(unit, type, temp)
    initialize_context()
    @ccall (libnvml()).nvmlUnitGetTemperature(unit::nvmlUnit_t, type::Cuint,
                                              temp::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlUnitGetFanSpeedInfo(unit, fanSpeeds)
    initialize_context()
    @ccall (libnvml()).nvmlUnitGetFanSpeedInfo(unit::nvmlUnit_t,
                                               fanSpeeds::Ptr{nvmlUnitFanSpeeds_t})::nvmlReturn_t
end

@checked function nvmlUnitGetDevices(unit, deviceCount, devices)
    initialize_context()
    @ccall (libnvml()).nvmlUnitGetDevices(unit::nvmlUnit_t, deviceCount::Ptr{Cuint},
                                          devices::Ptr{nvmlDevice_t})::nvmlReturn_t
end

@checked function nvmlSystemGetHicVersion(hwbcCount, hwbcEntries)
    initialize_context()
    @ccall (libnvml()).nvmlSystemGetHicVersion(hwbcCount::Ptr{Cuint},
                                               hwbcEntries::Ptr{nvmlHwbcEntry_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetHandleBySerial(serial, device)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetHandleBySerial(serial::Cstring,
                                                   device::Ptr{nvmlDevice_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetHandleByUUID(uuid, device)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetHandleByUUID(uuid::Cstring,
                                                 device::Ptr{nvmlDevice_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetName(device, name, length)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetName(device::nvmlDevice_t, name::Cstring,
                                         length::Cuint)::nvmlReturn_t
end

@checked function nvmlDeviceGetBrand(device, type)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetBrand(device::nvmlDevice_t,
                                          type::Ptr{nvmlBrandType_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetIndex(device, index)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetIndex(device::nvmlDevice_t,
                                          index::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetSerial(device, serial, length)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetSerial(device::nvmlDevice_t, serial::Cstring,
                                           length::Cuint)::nvmlReturn_t
end

const nvmlAffinityScope_t = Cuint

@checked function nvmlDeviceGetMemoryAffinity(device, nodeSetSize, nodeSet, scope)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetMemoryAffinity(device::nvmlDevice_t, nodeSetSize::Cuint,
                                                   nodeSet::Ptr{Culong},
                                                   scope::nvmlAffinityScope_t)::nvmlReturn_t
end

@checked function nvmlDeviceGetCpuAffinityWithinScope(device, cpuSetSize, cpuSet, scope)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetCpuAffinityWithinScope(device::nvmlDevice_t,
                                                           cpuSetSize::Cuint,
                                                           cpuSet::Ptr{Culong},
                                                           scope::nvmlAffinityScope_t)::nvmlReturn_t
end

@checked function nvmlDeviceGetCpuAffinity(device, cpuSetSize, cpuSet)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetCpuAffinity(device::nvmlDevice_t, cpuSetSize::Cuint,
                                                cpuSet::Ptr{Culong})::nvmlReturn_t
end

@checked function nvmlDeviceSetCpuAffinity(device)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetCpuAffinity(device::nvmlDevice_t)::nvmlReturn_t
end

@checked function nvmlDeviceClearCpuAffinity(device)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceClearCpuAffinity(device::nvmlDevice_t)::nvmlReturn_t
end

@checked function nvmlDeviceGetTopologyCommonAncestor(device1, device2, pathInfo)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetTopologyCommonAncestor(device1::nvmlDevice_t,
                                                           device2::nvmlDevice_t,
                                                           pathInfo::Ptr{nvmlGpuTopologyLevel_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetTopologyNearestGpus(device, level, count, deviceArray)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetTopologyNearestGpus(device::nvmlDevice_t,
                                                        level::nvmlGpuTopologyLevel_t,
                                                        count::Ptr{Cuint},
                                                        deviceArray::Ptr{nvmlDevice_t})::nvmlReturn_t
end

@checked function nvmlSystemGetTopologyGpuSet(cpuNumber, count, deviceArray)
    initialize_context()
    @ccall (libnvml()).nvmlSystemGetTopologyGpuSet(cpuNumber::Cuint, count::Ptr{Cuint},
                                                   deviceArray::Ptr{nvmlDevice_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetP2PStatus(device1, device2, p2pIndex, p2pStatus)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetP2PStatus(device1::nvmlDevice_t, device2::nvmlDevice_t,
                                              p2pIndex::nvmlGpuP2PCapsIndex_t,
                                              p2pStatus::Ptr{nvmlGpuP2PStatus_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetUUID(device, uuid, length)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetUUID(device::nvmlDevice_t, uuid::Cstring,
                                         length::Cuint)::nvmlReturn_t
end

@checked function nvmlVgpuInstanceGetMdevUUID(vgpuInstance, mdevUuid, size)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetMdevUUID(vgpuInstance::nvmlVgpuInstance_t,
                                                   mdevUuid::Cstring,
                                                   size::Cuint)::nvmlReturn_t
end

@checked function nvmlDeviceGetMinorNumber(device, minorNumber)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetMinorNumber(device::nvmlDevice_t,
                                                minorNumber::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetBoardPartNumber(device, partNumber, length)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetBoardPartNumber(device::nvmlDevice_t,
                                                    partNumber::Cstring,
                                                    length::Cuint)::nvmlReturn_t
end

@checked function nvmlDeviceGetInforomVersion(device, object, version, length)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetInforomVersion(device::nvmlDevice_t,
                                                   object::nvmlInforomObject_t,
                                                   version::Cstring,
                                                   length::Cuint)::nvmlReturn_t
end

@checked function nvmlDeviceGetInforomImageVersion(device, version, length)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetInforomImageVersion(device::nvmlDevice_t,
                                                        version::Cstring,
                                                        length::Cuint)::nvmlReturn_t
end

@checked function nvmlDeviceGetInforomConfigurationChecksum(device, checksum)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetInforomConfigurationChecksum(device::nvmlDevice_t,
                                                                 checksum::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceValidateInforom(device)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceValidateInforom(device::nvmlDevice_t)::nvmlReturn_t
end

@checked function nvmlDeviceGetDisplayMode(device, display)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetDisplayMode(device::nvmlDevice_t,
                                                display::Ptr{nvmlEnableState_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetDisplayActive(device, isActive)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetDisplayActive(device::nvmlDevice_t,
                                                  isActive::Ptr{nvmlEnableState_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetPersistenceMode(device, mode)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetPersistenceMode(device::nvmlDevice_t,
                                                    mode::Ptr{nvmlEnableState_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetMaxPcieLinkGeneration(device, maxLinkGen)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetMaxPcieLinkGeneration(device::nvmlDevice_t,
                                                          maxLinkGen::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetGpuMaxPcieLinkGeneration(device, maxLinkGenDevice)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetGpuMaxPcieLinkGeneration(device::nvmlDevice_t,
                                                             maxLinkGenDevice::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetMaxPcieLinkWidth(device, maxLinkWidth)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetMaxPcieLinkWidth(device::nvmlDevice_t,
                                                     maxLinkWidth::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetCurrPcieLinkGeneration(device, currLinkGen)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetCurrPcieLinkGeneration(device::nvmlDevice_t,
                                                           currLinkGen::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetCurrPcieLinkWidth(device, currLinkWidth)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetCurrPcieLinkWidth(device::nvmlDevice_t,
                                                      currLinkWidth::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetPcieThroughput(device, counter, value)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetPcieThroughput(device::nvmlDevice_t,
                                                   counter::nvmlPcieUtilCounter_t,
                                                   value::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetPcieReplayCounter(device, value)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetPcieReplayCounter(device::nvmlDevice_t,
                                                      value::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetClockInfo(device, type, clock)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetClockInfo(device::nvmlDevice_t, type::nvmlClockType_t,
                                              clock::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetMaxClockInfo(device, type, clock)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetMaxClockInfo(device::nvmlDevice_t,
                                                 type::nvmlClockType_t,
                                                 clock::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetApplicationsClock(device, clockType, clockMHz)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetApplicationsClock(device::nvmlDevice_t,
                                                      clockType::nvmlClockType_t,
                                                      clockMHz::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetDefaultApplicationsClock(device, clockType, clockMHz)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetDefaultApplicationsClock(device::nvmlDevice_t,
                                                             clockType::nvmlClockType_t,
                                                             clockMHz::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceResetApplicationsClocks(device)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceResetApplicationsClocks(device::nvmlDevice_t)::nvmlReturn_t
end

@checked function nvmlDeviceGetClock(device, clockType, clockId, clockMHz)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetClock(device::nvmlDevice_t, clockType::nvmlClockType_t,
                                          clockId::nvmlClockId_t,
                                          clockMHz::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetMaxCustomerBoostClock(device, clockType, clockMHz)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetMaxCustomerBoostClock(device::nvmlDevice_t,
                                                          clockType::nvmlClockType_t,
                                                          clockMHz::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetSupportedMemoryClocks(device, count, clocksMHz)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetSupportedMemoryClocks(device::nvmlDevice_t,
                                                          count::Ptr{Cuint},
                                                          clocksMHz::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetSupportedGraphicsClocks(device, memoryClockMHz, count,
                                                       clocksMHz)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetSupportedGraphicsClocks(device::nvmlDevice_t,
                                                            memoryClockMHz::Cuint,
                                                            count::Ptr{Cuint},
                                                            clocksMHz::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetAutoBoostedClocksEnabled(device, isEnabled, defaultIsEnabled)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetAutoBoostedClocksEnabled(device::nvmlDevice_t,
                                                             isEnabled::Ptr{nvmlEnableState_t},
                                                             defaultIsEnabled::Ptr{nvmlEnableState_t})::nvmlReturn_t
end

@checked function nvmlDeviceSetAutoBoostedClocksEnabled(device, enabled)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetAutoBoostedClocksEnabled(device::nvmlDevice_t,
                                                             enabled::nvmlEnableState_t)::nvmlReturn_t
end

@checked function nvmlDeviceSetDefaultAutoBoostedClocksEnabled(device, enabled, flags)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetDefaultAutoBoostedClocksEnabled(device::nvmlDevice_t,
                                                                    enabled::nvmlEnableState_t,
                                                                    flags::Cuint)::nvmlReturn_t
end

@checked function nvmlDeviceGetFanSpeed(device, speed)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetFanSpeed(device::nvmlDevice_t,
                                             speed::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetFanSpeed_v2(device, fan, speed)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetFanSpeed_v2(device::nvmlDevice_t, fan::Cuint,
                                                speed::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetTargetFanSpeed(device, fan, targetSpeed)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetTargetFanSpeed(device::nvmlDevice_t, fan::Cuint,
                                                   targetSpeed::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceSetDefaultFanSpeed_v2(device, fan)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetDefaultFanSpeed_v2(device::nvmlDevice_t,
                                                       fan::Cuint)::nvmlReturn_t
end

@checked function nvmlDeviceGetMinMaxFanSpeed(device, minSpeed, maxSpeed)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetMinMaxFanSpeed(device::nvmlDevice_t,
                                                   minSpeed::Ptr{Cuint},
                                                   maxSpeed::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetFanControlPolicy_v2(device, fan, policy)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetFanControlPolicy_v2(device::nvmlDevice_t, fan::Cuint,
                                                        policy::Ptr{nvmlFanControlPolicy_t})::nvmlReturn_t
end

@checked function nvmlDeviceSetFanControlPolicy(device, fan, policy)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetFanControlPolicy(device::nvmlDevice_t, fan::Cuint,
                                                     policy::nvmlFanControlPolicy_t)::nvmlReturn_t
end

@checked function nvmlDeviceGetNumFans(device, numFans)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetNumFans(device::nvmlDevice_t,
                                            numFans::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetTemperature(device, sensorType, temp)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetTemperature(device::nvmlDevice_t,
                                                sensorType::nvmlTemperatureSensors_t,
                                                temp::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetTemperatureThreshold(device, thresholdType, temp)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetTemperatureThreshold(device::nvmlDevice_t,
                                                         thresholdType::nvmlTemperatureThresholds_t,
                                                         temp::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceSetTemperatureThreshold(device, thresholdType, temp)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetTemperatureThreshold(device::nvmlDevice_t,
                                                         thresholdType::nvmlTemperatureThresholds_t,
                                                         temp::Ptr{Cint})::nvmlReturn_t
end

@checked function nvmlDeviceGetThermalSettings(device, sensorIndex, pThermalSettings)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetThermalSettings(device::nvmlDevice_t,
                                                    sensorIndex::Cuint,
                                                    pThermalSettings::Ptr{nvmlGpuThermalSettings_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetPerformanceState(device, pState)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetPerformanceState(device::nvmlDevice_t,
                                                     pState::Ptr{nvmlPstates_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetCurrentClocksThrottleReasons(device, clocksThrottleReasons)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetCurrentClocksThrottleReasons(device::nvmlDevice_t,
                                                                 clocksThrottleReasons::Ptr{Culonglong})::nvmlReturn_t
end

@checked function nvmlDeviceGetSupportedClocksThrottleReasons(device,
                                                              supportedClocksThrottleReasons)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetSupportedClocksThrottleReasons(device::nvmlDevice_t,
                                                                   supportedClocksThrottleReasons::Ptr{Culonglong})::nvmlReturn_t
end

@checked function nvmlDeviceGetPowerState(device, pState)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetPowerState(device::nvmlDevice_t,
                                               pState::Ptr{nvmlPstates_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetPowerManagementMode(device, mode)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetPowerManagementMode(device::nvmlDevice_t,
                                                        mode::Ptr{nvmlEnableState_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetPowerManagementLimit(device, limit)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetPowerManagementLimit(device::nvmlDevice_t,
                                                         limit::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetPowerManagementLimitConstraints(device, minLimit, maxLimit)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetPowerManagementLimitConstraints(device::nvmlDevice_t,
                                                                    minLimit::Ptr{Cuint},
                                                                    maxLimit::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetPowerManagementDefaultLimit(device, defaultLimit)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetPowerManagementDefaultLimit(device::nvmlDevice_t,
                                                                defaultLimit::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetPowerUsage(device, power)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetPowerUsage(device::nvmlDevice_t,
                                               power::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetTotalEnergyConsumption(device, energy)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetTotalEnergyConsumption(device::nvmlDevice_t,
                                                           energy::Ptr{Culonglong})::nvmlReturn_t
end

@checked function nvmlDeviceGetEnforcedPowerLimit(device, limit)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetEnforcedPowerLimit(device::nvmlDevice_t,
                                                       limit::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetGpuOperationMode(device, current, pending)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetGpuOperationMode(device::nvmlDevice_t,
                                                     current::Ptr{nvmlGpuOperationMode_t},
                                                     pending::Ptr{nvmlGpuOperationMode_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetMemoryInfo(device, memory)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetMemoryInfo(device::nvmlDevice_t,
                                               memory::Ptr{nvmlMemory_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetMemoryInfo_v2(device, memory)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetMemoryInfo_v2(device::nvmlDevice_t,
                                                  memory::Ptr{nvmlMemory_v2_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetComputeMode(device, mode)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetComputeMode(device::nvmlDevice_t,
                                                mode::Ptr{nvmlComputeMode_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetCudaComputeCapability(device, major, minor)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetCudaComputeCapability(device::nvmlDevice_t,
                                                          major::Ptr{Cint},
                                                          minor::Ptr{Cint})::nvmlReturn_t
end

@checked function nvmlDeviceGetEccMode(device, current, pending)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetEccMode(device::nvmlDevice_t,
                                            current::Ptr{nvmlEnableState_t},
                                            pending::Ptr{nvmlEnableState_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetDefaultEccMode(device, defaultMode)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetDefaultEccMode(device::nvmlDevice_t,
                                                   defaultMode::Ptr{nvmlEnableState_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetBoardId(device, boardId)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetBoardId(device::nvmlDevice_t,
                                            boardId::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetMultiGpuBoard(device, multiGpuBool)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetMultiGpuBoard(device::nvmlDevice_t,
                                                  multiGpuBool::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetTotalEccErrors(device, errorType, counterType, eccCounts)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetTotalEccErrors(device::nvmlDevice_t,
                                                   errorType::nvmlMemoryErrorType_t,
                                                   counterType::nvmlEccCounterType_t,
                                                   eccCounts::Ptr{Culonglong})::nvmlReturn_t
end

@checked function nvmlDeviceGetDetailedEccErrors(device, errorType, counterType, eccCounts)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetDetailedEccErrors(device::nvmlDevice_t,
                                                      errorType::nvmlMemoryErrorType_t,
                                                      counterType::nvmlEccCounterType_t,
                                                      eccCounts::Ptr{nvmlEccErrorCounts_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetMemoryErrorCounter(device, errorType, counterType,
                                                  locationType, count)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetMemoryErrorCounter(device::nvmlDevice_t,
                                                       errorType::nvmlMemoryErrorType_t,
                                                       counterType::nvmlEccCounterType_t,
                                                       locationType::nvmlMemoryLocation_t,
                                                       count::Ptr{Culonglong})::nvmlReturn_t
end

@checked function nvmlDeviceGetUtilizationRates(device, utilization)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetUtilizationRates(device::nvmlDevice_t,
                                                     utilization::Ptr{nvmlUtilization_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetEncoderUtilization(device, utilization, samplingPeriodUs)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetEncoderUtilization(device::nvmlDevice_t,
                                                       utilization::Ptr{Cuint},
                                                       samplingPeriodUs::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetEncoderCapacity(device, encoderQueryType, encoderCapacity)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetEncoderCapacity(device::nvmlDevice_t,
                                                    encoderQueryType::nvmlEncoderType_t,
                                                    encoderCapacity::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetEncoderStats(device, sessionCount, averageFps,
                                            averageLatency)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetEncoderStats(device::nvmlDevice_t,
                                                 sessionCount::Ptr{Cuint},
                                                 averageFps::Ptr{Cuint},
                                                 averageLatency::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetEncoderSessions(device, sessionCount, sessionInfos)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetEncoderSessions(device::nvmlDevice_t,
                                                    sessionCount::Ptr{Cuint},
                                                    sessionInfos::Ptr{nvmlEncoderSessionInfo_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetDecoderUtilization(device, utilization, samplingPeriodUs)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetDecoderUtilization(device::nvmlDevice_t,
                                                       utilization::Ptr{Cuint},
                                                       samplingPeriodUs::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetFBCStats(device, fbcStats)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetFBCStats(device::nvmlDevice_t,
                                             fbcStats::Ptr{nvmlFBCStats_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetFBCSessions(device, sessionCount, sessionInfo)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetFBCSessions(device::nvmlDevice_t,
                                                sessionCount::Ptr{Cuint},
                                                sessionInfo::Ptr{nvmlFBCSessionInfo_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetDriverModel(device, current, pending)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetDriverModel(device::nvmlDevice_t,
                                                current::Ptr{nvmlDriverModel_t},
                                                pending::Ptr{nvmlDriverModel_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetVbiosVersion(device, version, length)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetVbiosVersion(device::nvmlDevice_t, version::Cstring,
                                                 length::Cuint)::nvmlReturn_t
end

@checked function nvmlDeviceGetBridgeChipInfo(device, bridgeHierarchy)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetBridgeChipInfo(device::nvmlDevice_t,
                                                   bridgeHierarchy::Ptr{nvmlBridgeChipHierarchy_t})::nvmlReturn_t
end

@checked function nvmlDeviceOnSameBoard(device1, device2, onSameBoard)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceOnSameBoard(device1::nvmlDevice_t, device2::nvmlDevice_t,
                                             onSameBoard::Ptr{Cint})::nvmlReturn_t
end

@checked function nvmlDeviceGetAPIRestriction(device, apiType, isRestricted)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetAPIRestriction(device::nvmlDevice_t,
                                                   apiType::nvmlRestrictedAPI_t,
                                                   isRestricted::Ptr{nvmlEnableState_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetSamples(device, type, lastSeenTimeStamp, sampleValType,
                                       sampleCount, samples)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetSamples(device::nvmlDevice_t, type::nvmlSamplingType_t,
                                            lastSeenTimeStamp::Culonglong,
                                            sampleValType::Ptr{nvmlValueType_t},
                                            sampleCount::Ptr{Cuint},
                                            samples::Ptr{nvmlSample_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetBAR1MemoryInfo(device, bar1Memory)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetBAR1MemoryInfo(device::nvmlDevice_t,
                                                   bar1Memory::Ptr{nvmlBAR1Memory_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetViolationStatus(device, perfPolicyType, violTime)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetViolationStatus(device::nvmlDevice_t,
                                                    perfPolicyType::nvmlPerfPolicyType_t,
                                                    violTime::Ptr{nvmlViolationTime_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetIrqNum(device, irqNum)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetIrqNum(device::nvmlDevice_t,
                                           irqNum::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetNumGpuCores(device, numCores)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetNumGpuCores(device::nvmlDevice_t,
                                                numCores::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetPowerSource(device, powerSource)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetPowerSource(device::nvmlDevice_t,
                                                powerSource::Ptr{nvmlPowerSource_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetMemoryBusWidth(device, busWidth)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetMemoryBusWidth(device::nvmlDevice_t,
                                                   busWidth::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetPcieLinkMaxSpeed(device, maxSpeed)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetPcieLinkMaxSpeed(device::nvmlDevice_t,
                                                     maxSpeed::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetPcieSpeed(device, pcieSpeed)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetPcieSpeed(device::nvmlDevice_t,
                                              pcieSpeed::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetAdaptiveClockInfoStatus(device, adaptiveClockStatus)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetAdaptiveClockInfoStatus(device::nvmlDevice_t,
                                                            adaptiveClockStatus::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetAccountingMode(device, mode)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetAccountingMode(device::nvmlDevice_t,
                                                   mode::Ptr{nvmlEnableState_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetAccountingStats(device, pid, stats)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetAccountingStats(device::nvmlDevice_t, pid::Cuint,
                                                    stats::Ptr{nvmlAccountingStats_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetAccountingPids(device, count, pids)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetAccountingPids(device::nvmlDevice_t, count::Ptr{Cuint},
                                                   pids::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetAccountingBufferSize(device, bufferSize)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetAccountingBufferSize(device::nvmlDevice_t,
                                                         bufferSize::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetRetiredPages(device, cause, pageCount, addresses)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetRetiredPages(device::nvmlDevice_t,
                                                 cause::nvmlPageRetirementCause_t,
                                                 pageCount::Ptr{Cuint},
                                                 addresses::Ptr{Culonglong})::nvmlReturn_t
end

@checked function nvmlDeviceGetRetiredPages_v2(device, cause, pageCount, addresses,
                                               timestamps)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetRetiredPages_v2(device::nvmlDevice_t,
                                                    cause::nvmlPageRetirementCause_t,
                                                    pageCount::Ptr{Cuint},
                                                    addresses::Ptr{Culonglong},
                                                    timestamps::Ptr{Culonglong})::nvmlReturn_t
end

@checked function nvmlDeviceGetRetiredPagesPendingStatus(device, isPending)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetRetiredPagesPendingStatus(device::nvmlDevice_t,
                                                              isPending::Ptr{nvmlEnableState_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetRemappedRows(device, corrRows, uncRows, isPending,
                                            failureOccurred)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetRemappedRows(device::nvmlDevice_t, corrRows::Ptr{Cuint},
                                                 uncRows::Ptr{Cuint}, isPending::Ptr{Cuint},
                                                 failureOccurred::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetRowRemapperHistogram(device, values)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetRowRemapperHistogram(device::nvmlDevice_t,
                                                         values::Ptr{nvmlRowRemapperHistogramValues_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetArchitecture(device, arch)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetArchitecture(device::nvmlDevice_t,
                                                 arch::Ptr{nvmlDeviceArchitecture_t})::nvmlReturn_t
end

@checked function nvmlUnitSetLedState(unit, color)
    initialize_context()
    @ccall (libnvml()).nvmlUnitSetLedState(unit::nvmlUnit_t,
                                           color::nvmlLedColor_t)::nvmlReturn_t
end

@checked function nvmlDeviceSetPersistenceMode(device, mode)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetPersistenceMode(device::nvmlDevice_t,
                                                    mode::nvmlEnableState_t)::nvmlReturn_t
end

@checked function nvmlDeviceSetComputeMode(device, mode)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetComputeMode(device::nvmlDevice_t,
                                                mode::nvmlComputeMode_t)::nvmlReturn_t
end

@checked function nvmlDeviceSetEccMode(device, ecc)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetEccMode(device::nvmlDevice_t,
                                            ecc::nvmlEnableState_t)::nvmlReturn_t
end

@checked function nvmlDeviceClearEccErrorCounts(device, counterType)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceClearEccErrorCounts(device::nvmlDevice_t,
                                                     counterType::nvmlEccCounterType_t)::nvmlReturn_t
end

@checked function nvmlDeviceSetDriverModel(device, driverModel, flags)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetDriverModel(device::nvmlDevice_t,
                                                driverModel::nvmlDriverModel_t,
                                                flags::Cuint)::nvmlReturn_t
end

@cenum nvmlClockLimitId_enum::UInt32 begin
    NVML_CLOCK_LIMIT_ID_RANGE_START = 0x00000000ffffff00
    NVML_CLOCK_LIMIT_ID_TDP = 0x00000000ffffff01
    NVML_CLOCK_LIMIT_ID_UNLIMITED = 0x00000000ffffff02
end

const nvmlClockLimitId_t = nvmlClockLimitId_enum

@checked function nvmlDeviceSetGpuLockedClocks(device, minGpuClockMHz, maxGpuClockMHz)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetGpuLockedClocks(device::nvmlDevice_t,
                                                    minGpuClockMHz::Cuint,
                                                    maxGpuClockMHz::Cuint)::nvmlReturn_t
end

@checked function nvmlDeviceResetGpuLockedClocks(device)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceResetGpuLockedClocks(device::nvmlDevice_t)::nvmlReturn_t
end

@checked function nvmlDeviceSetMemoryLockedClocks(device, minMemClockMHz, maxMemClockMHz)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetMemoryLockedClocks(device::nvmlDevice_t,
                                                       minMemClockMHz::Cuint,
                                                       maxMemClockMHz::Cuint)::nvmlReturn_t
end

@checked function nvmlDeviceResetMemoryLockedClocks(device)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceResetMemoryLockedClocks(device::nvmlDevice_t)::nvmlReturn_t
end

@checked function nvmlDeviceSetApplicationsClocks(device, memClockMHz, graphicsClockMHz)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetApplicationsClocks(device::nvmlDevice_t,
                                                       memClockMHz::Cuint,
                                                       graphicsClockMHz::Cuint)::nvmlReturn_t
end

@checked function nvmlDeviceGetClkMonStatus(device, status)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetClkMonStatus(device::nvmlDevice_t,
                                                 status::Ptr{nvmlClkMonStatus_t})::nvmlReturn_t
end

@checked function nvmlDeviceSetPowerManagementLimit(device, limit)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetPowerManagementLimit(device::nvmlDevice_t,
                                                         limit::Cuint)::nvmlReturn_t
end

@checked function nvmlDeviceSetGpuOperationMode(device, mode)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetGpuOperationMode(device::nvmlDevice_t,
                                                     mode::nvmlGpuOperationMode_t)::nvmlReturn_t
end

@checked function nvmlDeviceSetAPIRestriction(device, apiType, isRestricted)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetAPIRestriction(device::nvmlDevice_t,
                                                   apiType::nvmlRestrictedAPI_t,
                                                   isRestricted::nvmlEnableState_t)::nvmlReturn_t
end

@checked function nvmlDeviceSetAccountingMode(device, mode)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetAccountingMode(device::nvmlDevice_t,
                                                   mode::nvmlEnableState_t)::nvmlReturn_t
end

@checked function nvmlDeviceClearAccountingPids(device)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceClearAccountingPids(device::nvmlDevice_t)::nvmlReturn_t
end

@checked function nvmlDeviceGetNvLinkState(device, link, isActive)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetNvLinkState(device::nvmlDevice_t, link::Cuint,
                                                isActive::Ptr{nvmlEnableState_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetNvLinkVersion(device, link, version)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetNvLinkVersion(device::nvmlDevice_t, link::Cuint,
                                                  version::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetNvLinkCapability(device, link, capability, capResult)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetNvLinkCapability(device::nvmlDevice_t, link::Cuint,
                                                     capability::nvmlNvLinkCapability_t,
                                                     capResult::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetNvLinkErrorCounter(device, link, counter, counterValue)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetNvLinkErrorCounter(device::nvmlDevice_t, link::Cuint,
                                                       counter::nvmlNvLinkErrorCounter_t,
                                                       counterValue::Ptr{Culonglong})::nvmlReturn_t
end

@checked function nvmlDeviceResetNvLinkErrorCounters(device, link)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceResetNvLinkErrorCounters(device::nvmlDevice_t,
                                                          link::Cuint)::nvmlReturn_t
end

@checked function nvmlDeviceSetNvLinkUtilizationControl(device, link, counter, control,
                                                        reset)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetNvLinkUtilizationControl(device::nvmlDevice_t,
                                                             link::Cuint, counter::Cuint,
                                                             control::Ptr{nvmlNvLinkUtilizationControl_t},
                                                             reset::Cuint)::nvmlReturn_t
end

@checked function nvmlDeviceGetNvLinkUtilizationControl(device, link, counter, control)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetNvLinkUtilizationControl(device::nvmlDevice_t,
                                                             link::Cuint, counter::Cuint,
                                                             control::Ptr{nvmlNvLinkUtilizationControl_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetNvLinkUtilizationCounter(device, link, counter, rxcounter,
                                                        txcounter)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetNvLinkUtilizationCounter(device::nvmlDevice_t,
                                                             link::Cuint, counter::Cuint,
                                                             rxcounter::Ptr{Culonglong},
                                                             txcounter::Ptr{Culonglong})::nvmlReturn_t
end

@checked function nvmlDeviceFreezeNvLinkUtilizationCounter(device, link, counter, freeze)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceFreezeNvLinkUtilizationCounter(device::nvmlDevice_t,
                                                                link::Cuint, counter::Cuint,
                                                                freeze::nvmlEnableState_t)::nvmlReturn_t
end

@checked function nvmlDeviceResetNvLinkUtilizationCounter(device, link, counter)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceResetNvLinkUtilizationCounter(device::nvmlDevice_t,
                                                               link::Cuint,
                                                               counter::Cuint)::nvmlReturn_t
end

@checked function nvmlDeviceGetNvLinkRemoteDeviceType(device, link, pNvLinkDeviceType)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetNvLinkRemoteDeviceType(device::nvmlDevice_t,
                                                           link::Cuint,
                                                           pNvLinkDeviceType::Ptr{nvmlIntNvLinkDeviceType_t})::nvmlReturn_t
end

@checked function nvmlEventSetCreate(set)
    initialize_context()
    @ccall (libnvml()).nvmlEventSetCreate(set::Ptr{nvmlEventSet_t})::nvmlReturn_t
end

@checked function nvmlDeviceRegisterEvents(device, eventTypes, set)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceRegisterEvents(device::nvmlDevice_t,
                                                eventTypes::Culonglong,
                                                set::nvmlEventSet_t)::nvmlReturn_t
end

@checked function nvmlDeviceGetSupportedEventTypes(device, eventTypes)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetSupportedEventTypes(device::nvmlDevice_t,
                                                        eventTypes::Ptr{Culonglong})::nvmlReturn_t
end

@checked function nvmlEventSetFree(set)
    initialize_context()
    @ccall (libnvml()).nvmlEventSetFree(set::nvmlEventSet_t)::nvmlReturn_t
end

@checked function nvmlDeviceModifyDrainState(pciInfo, newState)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceModifyDrainState(pciInfo::Ptr{nvmlPciInfo_t},
                                                  newState::nvmlEnableState_t)::nvmlReturn_t
end

@checked function nvmlDeviceQueryDrainState(pciInfo, currentState)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceQueryDrainState(pciInfo::Ptr{nvmlPciInfo_t},
                                                 currentState::Ptr{nvmlEnableState_t})::nvmlReturn_t
end

@checked function nvmlDeviceDiscoverGpus(pciInfo)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceDiscoverGpus(pciInfo::Ptr{nvmlPciInfo_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetFieldValues(device, valuesCount, values)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetFieldValues(device::nvmlDevice_t, valuesCount::Cint,
                                                values::Ptr{nvmlFieldValue_t})::nvmlReturn_t
end

@checked function nvmlDeviceClearFieldValues(device, valuesCount, values)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceClearFieldValues(device::nvmlDevice_t, valuesCount::Cint,
                                                  values::Ptr{nvmlFieldValue_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetVirtualizationMode(device, pVirtualMode)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetVirtualizationMode(device::nvmlDevice_t,
                                                       pVirtualMode::Ptr{nvmlGpuVirtualizationMode_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetHostVgpuMode(device, pHostVgpuMode)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetHostVgpuMode(device::nvmlDevice_t,
                                                 pHostVgpuMode::Ptr{nvmlHostVgpuMode_t})::nvmlReturn_t
end

@checked function nvmlDeviceSetVirtualizationMode(device, virtualMode)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetVirtualizationMode(device::nvmlDevice_t,
                                                       virtualMode::nvmlGpuVirtualizationMode_t)::nvmlReturn_t
end

@checked function nvmlDeviceGetProcessUtilization(device, utilization, processSamplesCount,
                                                  lastSeenTimeStamp)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetProcessUtilization(device::nvmlDevice_t,
                                                       utilization::Ptr{nvmlProcessUtilizationSample_t},
                                                       processSamplesCount::Ptr{Cuint},
                                                       lastSeenTimeStamp::Culonglong)::nvmlReturn_t
end

@checked function nvmlDeviceGetGspFirmwareVersion(device, version)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetGspFirmwareVersion(device::nvmlDevice_t,
                                                       version::Cstring)::nvmlReturn_t
end

@checked function nvmlDeviceGetGspFirmwareMode(device, isEnabled, defaultMode)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetGspFirmwareMode(device::nvmlDevice_t,
                                                    isEnabled::Ptr{Cuint},
                                                    defaultMode::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlGetVgpuDriverCapabilities(capability, capResult)
    initialize_context()
    @ccall (libnvml()).nvmlGetVgpuDriverCapabilities(capability::nvmlVgpuDriverCapability_t,
                                                     capResult::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetVgpuCapabilities(device, capability, capResult)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetVgpuCapabilities(device::nvmlDevice_t,
                                                     capability::nvmlDeviceVgpuCapability_t,
                                                     capResult::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetSupportedVgpus(device, vgpuCount, vgpuTypeIds)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetSupportedVgpus(device::nvmlDevice_t,
                                                   vgpuCount::Ptr{Cuint},
                                                   vgpuTypeIds::Ptr{nvmlVgpuTypeId_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetCreatableVgpus(device, vgpuCount, vgpuTypeIds)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetCreatableVgpus(device::nvmlDevice_t,
                                                   vgpuCount::Ptr{Cuint},
                                                   vgpuTypeIds::Ptr{nvmlVgpuTypeId_t})::nvmlReturn_t
end

@checked function nvmlVgpuTypeGetClass(vgpuTypeId, vgpuTypeClass, size)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuTypeGetClass(vgpuTypeId::nvmlVgpuTypeId_t,
                                            vgpuTypeClass::Cstring,
                                            size::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlVgpuTypeGetName(vgpuTypeId, vgpuTypeName, size)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuTypeGetName(vgpuTypeId::nvmlVgpuTypeId_t,
                                           vgpuTypeName::Cstring,
                                           size::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlVgpuTypeGetGpuInstanceProfileId(vgpuTypeId, gpuInstanceProfileId)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuTypeGetGpuInstanceProfileId(vgpuTypeId::nvmlVgpuTypeId_t,
                                                           gpuInstanceProfileId::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlVgpuTypeGetDeviceID(vgpuTypeId, deviceID, subsystemID)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuTypeGetDeviceID(vgpuTypeId::nvmlVgpuTypeId_t,
                                               deviceID::Ptr{Culonglong},
                                               subsystemID::Ptr{Culonglong})::nvmlReturn_t
end

@checked function nvmlVgpuTypeGetFramebufferSize(vgpuTypeId, fbSize)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuTypeGetFramebufferSize(vgpuTypeId::nvmlVgpuTypeId_t,
                                                      fbSize::Ptr{Culonglong})::nvmlReturn_t
end

@checked function nvmlVgpuTypeGetNumDisplayHeads(vgpuTypeId, numDisplayHeads)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuTypeGetNumDisplayHeads(vgpuTypeId::nvmlVgpuTypeId_t,
                                                      numDisplayHeads::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlVgpuTypeGetResolution(vgpuTypeId, displayIndex, xdim, ydim)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuTypeGetResolution(vgpuTypeId::nvmlVgpuTypeId_t,
                                                 displayIndex::Cuint, xdim::Ptr{Cuint},
                                                 ydim::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlVgpuTypeGetLicense(vgpuTypeId, vgpuTypeLicenseString, size)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuTypeGetLicense(vgpuTypeId::nvmlVgpuTypeId_t,
                                              vgpuTypeLicenseString::Cstring,
                                              size::Cuint)::nvmlReturn_t
end

@checked function nvmlVgpuTypeGetFrameRateLimit(vgpuTypeId, frameRateLimit)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuTypeGetFrameRateLimit(vgpuTypeId::nvmlVgpuTypeId_t,
                                                     frameRateLimit::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlVgpuTypeGetMaxInstances(device, vgpuTypeId, vgpuInstanceCount)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuTypeGetMaxInstances(device::nvmlDevice_t,
                                                   vgpuTypeId::nvmlVgpuTypeId_t,
                                                   vgpuInstanceCount::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlVgpuTypeGetMaxInstancesPerVm(vgpuTypeId, vgpuInstanceCountPerVm)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuTypeGetMaxInstancesPerVm(vgpuTypeId::nvmlVgpuTypeId_t,
                                                        vgpuInstanceCountPerVm::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetActiveVgpus(device, vgpuCount, vgpuInstances)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetActiveVgpus(device::nvmlDevice_t, vgpuCount::Ptr{Cuint},
                                                vgpuInstances::Ptr{nvmlVgpuInstance_t})::nvmlReturn_t
end

@checked function nvmlVgpuInstanceGetVmID(vgpuInstance, vmId, size, vmIdType)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetVmID(vgpuInstance::nvmlVgpuInstance_t,
                                               vmId::Cstring, size::Cuint,
                                               vmIdType::Ptr{nvmlVgpuVmIdType_t})::nvmlReturn_t
end

@checked function nvmlVgpuInstanceGetUUID(vgpuInstance, uuid, size)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetUUID(vgpuInstance::nvmlVgpuInstance_t,
                                               uuid::Cstring, size::Cuint)::nvmlReturn_t
end

@checked function nvmlVgpuInstanceGetVmDriverVersion(vgpuInstance, version, length)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetVmDriverVersion(vgpuInstance::nvmlVgpuInstance_t,
                                                          version::Cstring,
                                                          length::Cuint)::nvmlReturn_t
end

@checked function nvmlVgpuInstanceGetFbUsage(vgpuInstance, fbUsage)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetFbUsage(vgpuInstance::nvmlVgpuInstance_t,
                                                  fbUsage::Ptr{Culonglong})::nvmlReturn_t
end

@checked function nvmlVgpuInstanceGetLicenseStatus(vgpuInstance, licensed)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetLicenseStatus(vgpuInstance::nvmlVgpuInstance_t,
                                                        licensed::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlVgpuInstanceGetType(vgpuInstance, vgpuTypeId)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetType(vgpuInstance::nvmlVgpuInstance_t,
                                               vgpuTypeId::Ptr{nvmlVgpuTypeId_t})::nvmlReturn_t
end

@checked function nvmlVgpuInstanceGetFrameRateLimit(vgpuInstance, frameRateLimit)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetFrameRateLimit(vgpuInstance::nvmlVgpuInstance_t,
                                                         frameRateLimit::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlVgpuInstanceGetEccMode(vgpuInstance, eccMode)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetEccMode(vgpuInstance::nvmlVgpuInstance_t,
                                                  eccMode::Ptr{nvmlEnableState_t})::nvmlReturn_t
end

@checked function nvmlVgpuInstanceGetEncoderCapacity(vgpuInstance, encoderCapacity)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetEncoderCapacity(vgpuInstance::nvmlVgpuInstance_t,
                                                          encoderCapacity::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlVgpuInstanceSetEncoderCapacity(vgpuInstance, encoderCapacity)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceSetEncoderCapacity(vgpuInstance::nvmlVgpuInstance_t,
                                                          encoderCapacity::Cuint)::nvmlReturn_t
end

@checked function nvmlVgpuInstanceGetEncoderStats(vgpuInstance, sessionCount, averageFps,
                                                  averageLatency)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetEncoderStats(vgpuInstance::nvmlVgpuInstance_t,
                                                       sessionCount::Ptr{Cuint},
                                                       averageFps::Ptr{Cuint},
                                                       averageLatency::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlVgpuInstanceGetEncoderSessions(vgpuInstance, sessionCount,
                                                     sessionInfo)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetEncoderSessions(vgpuInstance::nvmlVgpuInstance_t,
                                                          sessionCount::Ptr{Cuint},
                                                          sessionInfo::Ptr{nvmlEncoderSessionInfo_t})::nvmlReturn_t
end

@checked function nvmlVgpuInstanceGetFBCStats(vgpuInstance, fbcStats)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetFBCStats(vgpuInstance::nvmlVgpuInstance_t,
                                                   fbcStats::Ptr{nvmlFBCStats_t})::nvmlReturn_t
end

@checked function nvmlVgpuInstanceGetFBCSessions(vgpuInstance, sessionCount, sessionInfo)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetFBCSessions(vgpuInstance::nvmlVgpuInstance_t,
                                                      sessionCount::Ptr{Cuint},
                                                      sessionInfo::Ptr{nvmlFBCSessionInfo_t})::nvmlReturn_t
end

@checked function nvmlVgpuInstanceGetGpuInstanceId(vgpuInstance, gpuInstanceId)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetGpuInstanceId(vgpuInstance::nvmlVgpuInstance_t,
                                                        gpuInstanceId::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlVgpuInstanceGetGpuPciId(vgpuInstance, vgpuPciId, length)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetGpuPciId(vgpuInstance::nvmlVgpuInstance_t,
                                                   vgpuPciId::Cstring,
                                                   length::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlVgpuTypeGetCapabilities(vgpuTypeId, capability, capResult)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuTypeGetCapabilities(vgpuTypeId::nvmlVgpuTypeId_t,
                                                   capability::nvmlVgpuCapability_t,
                                                   capResult::Ptr{Cuint})::nvmlReturn_t
end

struct nvmlVgpuVersion_st
    minVersion::Cuint
    maxVersion::Cuint
end

const nvmlVgpuVersion_t = nvmlVgpuVersion_st

struct nvmlVgpuMetadata_st
    version::Cuint
    revision::Cuint
    guestInfoState::nvmlVgpuGuestInfoState_t
    guestDriverVersion::NTuple{80,Cchar}
    hostDriverVersion::NTuple{80,Cchar}
    reserved::NTuple{6,Cuint}
    vgpuVirtualizationCaps::Cuint
    guestVgpuVersion::Cuint
    opaqueDataSize::Cuint
    opaqueData::NTuple{4,Cchar}
end

const nvmlVgpuMetadata_t = nvmlVgpuMetadata_st

struct nvmlVgpuPgpuMetadata_st
    version::Cuint
    revision::Cuint
    hostDriverVersion::NTuple{80,Cchar}
    pgpuVirtualizationCaps::Cuint
    reserved::NTuple{5,Cuint}
    hostSupportedVgpuRange::nvmlVgpuVersion_t
    opaqueDataSize::Cuint
    opaqueData::NTuple{4,Cchar}
end

const nvmlVgpuPgpuMetadata_t = nvmlVgpuPgpuMetadata_st

@cenum nvmlVgpuVmCompatibility_enum::UInt32 begin
    NVML_VGPU_VM_COMPATIBILITY_NONE = 0
    NVML_VGPU_VM_COMPATIBILITY_COLD = 1
    NVML_VGPU_VM_COMPATIBILITY_HIBERNATE = 2
    NVML_VGPU_VM_COMPATIBILITY_SLEEP = 4
    NVML_VGPU_VM_COMPATIBILITY_LIVE = 8
end

const nvmlVgpuVmCompatibility_t = nvmlVgpuVmCompatibility_enum

@cenum nvmlVgpuPgpuCompatibilityLimitCode_enum::UInt32 begin
    NVML_VGPU_COMPATIBILITY_LIMIT_NONE = 0
    NVML_VGPU_COMPATIBILITY_LIMIT_HOST_DRIVER = 1
    NVML_VGPU_COMPATIBILITY_LIMIT_GUEST_DRIVER = 2
    NVML_VGPU_COMPATIBILITY_LIMIT_GPU = 4
    NVML_VGPU_COMPATIBILITY_LIMIT_OTHER = 0x0000000080000000
end

const nvmlVgpuPgpuCompatibilityLimitCode_t = nvmlVgpuPgpuCompatibilityLimitCode_enum

struct nvmlVgpuPgpuCompatibility_st
    vgpuVmCompatibility::nvmlVgpuVmCompatibility_t
    compatibilityLimitCode::nvmlVgpuPgpuCompatibilityLimitCode_t
end

const nvmlVgpuPgpuCompatibility_t = nvmlVgpuPgpuCompatibility_st

@checked function nvmlVgpuInstanceGetMetadata(vgpuInstance, vgpuMetadata, bufferSize)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetMetadata(vgpuInstance::nvmlVgpuInstance_t,
                                                   vgpuMetadata::Ptr{nvmlVgpuMetadata_t},
                                                   bufferSize::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetVgpuMetadata(device, pgpuMetadata, bufferSize)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetVgpuMetadata(device::nvmlDevice_t,
                                                 pgpuMetadata::Ptr{nvmlVgpuPgpuMetadata_t},
                                                 bufferSize::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlGetVgpuCompatibility(vgpuMetadata, pgpuMetadata, compatibilityInfo)
    initialize_context()
    @ccall (libnvml()).nvmlGetVgpuCompatibility(vgpuMetadata::Ptr{nvmlVgpuMetadata_t},
                                                pgpuMetadata::Ptr{nvmlVgpuPgpuMetadata_t},
                                                compatibilityInfo::Ptr{nvmlVgpuPgpuCompatibility_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetPgpuMetadataString(device, pgpuMetadata, bufferSize)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetPgpuMetadataString(device::nvmlDevice_t,
                                                       pgpuMetadata::Cstring,
                                                       bufferSize::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlGetVgpuVersion(supported, current)
    initialize_context()
    @ccall (libnvml()).nvmlGetVgpuVersion(supported::Ptr{nvmlVgpuVersion_t},
                                          current::Ptr{nvmlVgpuVersion_t})::nvmlReturn_t
end

@checked function nvmlSetVgpuVersion(vgpuVersion)
    initialize_context()
    @ccall (libnvml()).nvmlSetVgpuVersion(vgpuVersion::Ptr{nvmlVgpuVersion_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetVgpuUtilization(device, lastSeenTimeStamp, sampleValType,
                                               vgpuInstanceSamplesCount, utilizationSamples)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetVgpuUtilization(device::nvmlDevice_t,
                                                    lastSeenTimeStamp::Culonglong,
                                                    sampleValType::Ptr{nvmlValueType_t},
                                                    vgpuInstanceSamplesCount::Ptr{Cuint},
                                                    utilizationSamples::Ptr{nvmlVgpuInstanceUtilizationSample_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetVgpuProcessUtilization(device, lastSeenTimeStamp,
                                                      vgpuProcessSamplesCount,
                                                      utilizationSamples)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetVgpuProcessUtilization(device::nvmlDevice_t,
                                                           lastSeenTimeStamp::Culonglong,
                                                           vgpuProcessSamplesCount::Ptr{Cuint},
                                                           utilizationSamples::Ptr{nvmlVgpuProcessUtilizationSample_t})::nvmlReturn_t
end

@checked function nvmlVgpuInstanceGetAccountingMode(vgpuInstance, mode)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetAccountingMode(vgpuInstance::nvmlVgpuInstance_t,
                                                         mode::Ptr{nvmlEnableState_t})::nvmlReturn_t
end

@checked function nvmlVgpuInstanceGetAccountingPids(vgpuInstance, count, pids)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetAccountingPids(vgpuInstance::nvmlVgpuInstance_t,
                                                         count::Ptr{Cuint},
                                                         pids::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlVgpuInstanceGetAccountingStats(vgpuInstance, pid, stats)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceGetAccountingStats(vgpuInstance::nvmlVgpuInstance_t,
                                                          pid::Cuint,
                                                          stats::Ptr{nvmlAccountingStats_t})::nvmlReturn_t
end

@checked function nvmlVgpuInstanceClearAccountingPids(vgpuInstance)
    initialize_context()
    @ccall (libnvml()).nvmlVgpuInstanceClearAccountingPids(vgpuInstance::nvmlVgpuInstance_t)::nvmlReturn_t
end

struct nvmlGpuInstanceProfileInfo_st
    id::Cuint
    isP2pSupported::Cuint
    sliceCount::Cuint
    instanceCount::Cuint
    multiprocessorCount::Cuint
    copyEngineCount::Cuint
    decoderCount::Cuint
    encoderCount::Cuint
    jpegCount::Cuint
    ofaCount::Cuint
    memorySizeMB::Culonglong
end

const nvmlGpuInstanceProfileInfo_t = nvmlGpuInstanceProfileInfo_st

struct nvmlGpuInstanceProfileInfo_v2_st
    version::Cuint
    id::Cuint
    isP2pSupported::Cuint
    sliceCount::Cuint
    instanceCount::Cuint
    multiprocessorCount::Cuint
    copyEngineCount::Cuint
    decoderCount::Cuint
    encoderCount::Cuint
    jpegCount::Cuint
    ofaCount::Cuint
    memorySizeMB::Culonglong
    name::NTuple{96,Cchar}
end

const nvmlGpuInstanceProfileInfo_v2_t = nvmlGpuInstanceProfileInfo_v2_st

struct nvmlGpuInstanceInfo_st
    device::nvmlDevice_t
    id::Cuint
    profileId::Cuint
    placement::nvmlGpuInstancePlacement_t
end

const nvmlGpuInstanceInfo_t = nvmlGpuInstanceInfo_st

struct nvmlComputeInstanceProfileInfo_st
    id::Cuint
    sliceCount::Cuint
    instanceCount::Cuint
    multiprocessorCount::Cuint
    sharedCopyEngineCount::Cuint
    sharedDecoderCount::Cuint
    sharedEncoderCount::Cuint
    sharedJpegCount::Cuint
    sharedOfaCount::Cuint
end

const nvmlComputeInstanceProfileInfo_t = nvmlComputeInstanceProfileInfo_st

struct nvmlComputeInstanceProfileInfo_v2_st
    version::Cuint
    id::Cuint
    sliceCount::Cuint
    instanceCount::Cuint
    multiprocessorCount::Cuint
    sharedCopyEngineCount::Cuint
    sharedDecoderCount::Cuint
    sharedEncoderCount::Cuint
    sharedJpegCount::Cuint
    sharedOfaCount::Cuint
    name::NTuple{96,Cchar}
end

const nvmlComputeInstanceProfileInfo_v2_t = nvmlComputeInstanceProfileInfo_v2_st

@checked function nvmlDeviceSetMigMode(device, mode, activationStatus)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetMigMode(device::nvmlDevice_t, mode::Cuint,
                                            activationStatus::Ptr{nvmlReturn_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetMigMode(device, currentMode, pendingMode)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetMigMode(device::nvmlDevice_t, currentMode::Ptr{Cuint},
                                            pendingMode::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetGpuInstanceProfileInfo(device, profile, info)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetGpuInstanceProfileInfo(device::nvmlDevice_t,
                                                           profile::Cuint,
                                                           info::Ptr{nvmlGpuInstanceProfileInfo_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetGpuInstanceProfileInfoV(device, profile, info)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetGpuInstanceProfileInfoV(device::nvmlDevice_t,
                                                            profile::Cuint,
                                                            info::Ptr{nvmlGpuInstanceProfileInfo_v2_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetGpuInstanceRemainingCapacity(device, profileId, count)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetGpuInstanceRemainingCapacity(device::nvmlDevice_t,
                                                                 profileId::Cuint,
                                                                 count::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceCreateGpuInstance(device, profileId, gpuInstance)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceCreateGpuInstance(device::nvmlDevice_t, profileId::Cuint,
                                                   gpuInstance::Ptr{nvmlGpuInstance_t})::nvmlReturn_t
end

@checked function nvmlDeviceCreateGpuInstanceWithPlacement(device, profileId, placement,
                                                           gpuInstance)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceCreateGpuInstanceWithPlacement(device::nvmlDevice_t,
                                                                profileId::Cuint,
                                                                placement::Ptr{nvmlGpuInstancePlacement_t},
                                                                gpuInstance::Ptr{nvmlGpuInstance_t})::nvmlReturn_t
end

@checked function nvmlGpuInstanceDestroy(gpuInstance)
    initialize_context()
    @ccall (libnvml()).nvmlGpuInstanceDestroy(gpuInstance::nvmlGpuInstance_t)::nvmlReturn_t
end

@checked function nvmlDeviceGetGpuInstances(device, profileId, gpuInstances, count)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetGpuInstances(device::nvmlDevice_t, profileId::Cuint,
                                                 gpuInstances::Ptr{nvmlGpuInstance_t},
                                                 count::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetGpuInstanceById(device, id, gpuInstance)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetGpuInstanceById(device::nvmlDevice_t, id::Cuint,
                                                    gpuInstance::Ptr{nvmlGpuInstance_t})::nvmlReturn_t
end

@checked function nvmlGpuInstanceGetInfo(gpuInstance, info)
    initialize_context()
    @ccall (libnvml()).nvmlGpuInstanceGetInfo(gpuInstance::nvmlGpuInstance_t,
                                              info::Ptr{nvmlGpuInstanceInfo_t})::nvmlReturn_t
end

@checked function nvmlGpuInstanceGetComputeInstanceProfileInfo(gpuInstance, profile,
                                                               engProfile, info)
    initialize_context()
    @ccall (libnvml()).nvmlGpuInstanceGetComputeInstanceProfileInfo(gpuInstance::nvmlGpuInstance_t,
                                                                    profile::Cuint,
                                                                    engProfile::Cuint,
                                                                    info::Ptr{nvmlComputeInstanceProfileInfo_t})::nvmlReturn_t
end

@checked function nvmlGpuInstanceGetComputeInstanceProfileInfoV(gpuInstance, profile,
                                                                engProfile, info)
    initialize_context()
    @ccall (libnvml()).nvmlGpuInstanceGetComputeInstanceProfileInfoV(gpuInstance::nvmlGpuInstance_t,
                                                                     profile::Cuint,
                                                                     engProfile::Cuint,
                                                                     info::Ptr{nvmlComputeInstanceProfileInfo_v2_t})::nvmlReturn_t
end

@checked function nvmlGpuInstanceGetComputeInstanceRemainingCapacity(gpuInstance, profileId,
                                                                     count)
    initialize_context()
    @ccall (libnvml()).nvmlGpuInstanceGetComputeInstanceRemainingCapacity(gpuInstance::nvmlGpuInstance_t,
                                                                          profileId::Cuint,
                                                                          count::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlGpuInstanceGetComputeInstancePossiblePlacements(gpuInstance,
                                                                      profileId, placements,
                                                                      count)
    initialize_context()
    @ccall (libnvml()).nvmlGpuInstanceGetComputeInstancePossiblePlacements(gpuInstance::nvmlGpuInstance_t,
                                                                           profileId::Cuint,
                                                                           placements::Ptr{nvmlComputeInstancePlacement_t},
                                                                           count::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlGpuInstanceCreateComputeInstance(gpuInstance, profileId,
                                                       computeInstance)
    initialize_context()
    @ccall (libnvml()).nvmlGpuInstanceCreateComputeInstance(gpuInstance::nvmlGpuInstance_t,
                                                            profileId::Cuint,
                                                            computeInstance::Ptr{nvmlComputeInstance_t})::nvmlReturn_t
end

@checked function nvmlGpuInstanceCreateComputeInstanceWithPlacement(gpuInstance, profileId,
                                                                    placement,
                                                                    computeInstance)
    initialize_context()
    @ccall (libnvml()).nvmlGpuInstanceCreateComputeInstanceWithPlacement(gpuInstance::nvmlGpuInstance_t,
                                                                         profileId::Cuint,
                                                                         placement::Ptr{nvmlComputeInstancePlacement_t},
                                                                         computeInstance::Ptr{nvmlComputeInstance_t})::nvmlReturn_t
end

@checked function nvmlComputeInstanceDestroy(computeInstance)
    initialize_context()
    @ccall (libnvml()).nvmlComputeInstanceDestroy(computeInstance::nvmlComputeInstance_t)::nvmlReturn_t
end

@checked function nvmlGpuInstanceGetComputeInstances(gpuInstance, profileId,
                                                     computeInstances, count)
    initialize_context()
    @ccall (libnvml()).nvmlGpuInstanceGetComputeInstances(gpuInstance::nvmlGpuInstance_t,
                                                          profileId::Cuint,
                                                          computeInstances::Ptr{nvmlComputeInstance_t},
                                                          count::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlGpuInstanceGetComputeInstanceById(gpuInstance, id, computeInstance)
    initialize_context()
    @ccall (libnvml()).nvmlGpuInstanceGetComputeInstanceById(gpuInstance::nvmlGpuInstance_t,
                                                             id::Cuint,
                                                             computeInstance::Ptr{nvmlComputeInstance_t})::nvmlReturn_t
end

@checked function nvmlDeviceIsMigDeviceHandle(device, isMigDevice)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceIsMigDeviceHandle(device::nvmlDevice_t,
                                                   isMigDevice::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetGpuInstanceId(device, id)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetGpuInstanceId(device::nvmlDevice_t,
                                                  id::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetComputeInstanceId(device, id)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetComputeInstanceId(device::nvmlDevice_t,
                                                      id::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetMaxMigDeviceCount(device, count)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetMaxMigDeviceCount(device::nvmlDevice_t,
                                                      count::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetMigDeviceHandleByIndex(device, index, migDevice)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetMigDeviceHandleByIndex(device::nvmlDevice_t,
                                                           index::Cuint,
                                                           migDevice::Ptr{nvmlDevice_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetDeviceHandleFromMigDeviceHandle(migDevice, device)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetDeviceHandleFromMigDeviceHandle(migDevice::nvmlDevice_t,
                                                                    device::Ptr{nvmlDevice_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetBusType(device, type)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetBusType(device::nvmlDevice_t,
                                            type::Ptr{nvmlBusType_t})::nvmlReturn_t
end

@checked function nvmlDeviceGetDynamicPstatesInfo(device, pDynamicPstatesInfo)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetDynamicPstatesInfo(device::nvmlDevice_t,
                                                       pDynamicPstatesInfo::Ptr{nvmlGpuDynamicPstatesInfo_t})::nvmlReturn_t
end

@checked function nvmlDeviceSetFanSpeed_v2(device, fan, speed)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetFanSpeed_v2(device::nvmlDevice_t, fan::Cuint,
                                                speed::Cuint)::nvmlReturn_t
end

@checked function nvmlDeviceGetGpcClkVfOffset(device, offset)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetGpcClkVfOffset(device::nvmlDevice_t,
                                                   offset::Ptr{Cint})::nvmlReturn_t
end

@checked function nvmlDeviceSetGpcClkVfOffset(device, offset)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetGpcClkVfOffset(device::nvmlDevice_t,
                                                   offset::Cint)::nvmlReturn_t
end

@checked function nvmlDeviceGetMemClkVfOffset(device, offset)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetMemClkVfOffset(device::nvmlDevice_t,
                                                   offset::Ptr{Cint})::nvmlReturn_t
end

@checked function nvmlDeviceSetMemClkVfOffset(device, offset)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetMemClkVfOffset(device::nvmlDevice_t,
                                                   offset::Cint)::nvmlReturn_t
end

@checked function nvmlDeviceGetMinMaxClockOfPState(device, type, pstate, minClockMHz,
                                                   maxClockMHz)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetMinMaxClockOfPState(device::nvmlDevice_t,
                                                        type::nvmlClockType_t,
                                                        pstate::nvmlPstates_t,
                                                        minClockMHz::Ptr{Cuint},
                                                        maxClockMHz::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceGetSupportedPerformanceStates(device, pstates, size)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetSupportedPerformanceStates(device::nvmlDevice_t,
                                                               pstates::Ptr{nvmlPstates_t},
                                                               size::Cuint)::nvmlReturn_t
end

@checked function nvmlDeviceGetGpcClkMinMaxVfOffset(device, minOffset, maxOffset)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetGpcClkMinMaxVfOffset(device::nvmlDevice_t,
                                                         minOffset::Ptr{Cint},
                                                         maxOffset::Ptr{Cint})::nvmlReturn_t
end

@checked function nvmlDeviceGetMemClkMinMaxVfOffset(device, minOffset, maxOffset)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetMemClkMinMaxVfOffset(device::nvmlDevice_t,
                                                         minOffset::Ptr{Cint},
                                                         maxOffset::Ptr{Cint})::nvmlReturn_t
end

@checked function nvmlDeviceGetGpuFabricInfo(device, gpuFabricInfo)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceGetGpuFabricInfo(device::nvmlDevice_t,
                                                  gpuFabricInfo::Ptr{nvmlGpuFabricInfo_t})::nvmlReturn_t
end

@cenum nvmlGpmMetricId_t::UInt32 begin
    NVML_GPM_METRIC_GRAPHICS_UTIL = 1
    NVML_GPM_METRIC_SM_UTIL = 2
    NVML_GPM_METRIC_SM_OCCUPANCY = 3
    NVML_GPM_METRIC_INTEGER_UTIL = 4
    NVML_GPM_METRIC_ANY_TENSOR_UTIL = 5
    NVML_GPM_METRIC_DFMA_TENSOR_UTIL = 6
    NVML_GPM_METRIC_HMMA_TENSOR_UTIL = 7
    NVML_GPM_METRIC_IMMA_TENSOR_UTIL = 9
    NVML_GPM_METRIC_DRAM_BW_UTIL = 10
    NVML_GPM_METRIC_FP64_UTIL = 11
    NVML_GPM_METRIC_FP32_UTIL = 12
    NVML_GPM_METRIC_FP16_UTIL = 13
    NVML_GPM_METRIC_PCIE_TX_PER_SEC = 20
    NVML_GPM_METRIC_PCIE_RX_PER_SEC = 21
    NVML_GPM_METRIC_NVDEC_0_UTIL = 30
    NVML_GPM_METRIC_NVDEC_1_UTIL = 31
    NVML_GPM_METRIC_NVDEC_2_UTIL = 32
    NVML_GPM_METRIC_NVDEC_3_UTIL = 33
    NVML_GPM_METRIC_NVDEC_4_UTIL = 34
    NVML_GPM_METRIC_NVDEC_5_UTIL = 35
    NVML_GPM_METRIC_NVDEC_6_UTIL = 36
    NVML_GPM_METRIC_NVDEC_7_UTIL = 37
    NVML_GPM_METRIC_NVJPG_0_UTIL = 40
    NVML_GPM_METRIC_NVJPG_1_UTIL = 41
    NVML_GPM_METRIC_NVJPG_2_UTIL = 42
    NVML_GPM_METRIC_NVJPG_3_UTIL = 43
    NVML_GPM_METRIC_NVJPG_4_UTIL = 44
    NVML_GPM_METRIC_NVJPG_5_UTIL = 45
    NVML_GPM_METRIC_NVJPG_6_UTIL = 46
    NVML_GPM_METRIC_NVJPG_7_UTIL = 47
    NVML_GPM_METRIC_NVOFA_0_UTIL = 50
    NVML_GPM_METRIC_NVLINK_TOTAL_RX_PER_SEC = 60
    NVML_GPM_METRIC_NVLINK_TOTAL_TX_PER_SEC = 61
    NVML_GPM_METRIC_NVLINK_L0_RX_PER_SEC = 62
    NVML_GPM_METRIC_NVLINK_L0_TX_PER_SEC = 63
    NVML_GPM_METRIC_NVLINK_L1_RX_PER_SEC = 64
    NVML_GPM_METRIC_NVLINK_L1_TX_PER_SEC = 65
    NVML_GPM_METRIC_NVLINK_L2_RX_PER_SEC = 66
    NVML_GPM_METRIC_NVLINK_L2_TX_PER_SEC = 67
    NVML_GPM_METRIC_NVLINK_L3_RX_PER_SEC = 68
    NVML_GPM_METRIC_NVLINK_L3_TX_PER_SEC = 69
    NVML_GPM_METRIC_NVLINK_L4_RX_PER_SEC = 70
    NVML_GPM_METRIC_NVLINK_L4_TX_PER_SEC = 71
    NVML_GPM_METRIC_NVLINK_L5_RX_PER_SEC = 72
    NVML_GPM_METRIC_NVLINK_L5_TX_PER_SEC = 73
    NVML_GPM_METRIC_NVLINK_L6_RX_PER_SEC = 74
    NVML_GPM_METRIC_NVLINK_L6_TX_PER_SEC = 75
    NVML_GPM_METRIC_NVLINK_L7_RX_PER_SEC = 76
    NVML_GPM_METRIC_NVLINK_L7_TX_PER_SEC = 77
    NVML_GPM_METRIC_NVLINK_L8_RX_PER_SEC = 78
    NVML_GPM_METRIC_NVLINK_L8_TX_PER_SEC = 79
    NVML_GPM_METRIC_NVLINK_L9_RX_PER_SEC = 80
    NVML_GPM_METRIC_NVLINK_L9_TX_PER_SEC = 81
    NVML_GPM_METRIC_NVLINK_L10_RX_PER_SEC = 82
    NVML_GPM_METRIC_NVLINK_L10_TX_PER_SEC = 83
    NVML_GPM_METRIC_NVLINK_L11_RX_PER_SEC = 84
    NVML_GPM_METRIC_NVLINK_L11_TX_PER_SEC = 85
    NVML_GPM_METRIC_NVLINK_L12_RX_PER_SEC = 86
    NVML_GPM_METRIC_NVLINK_L12_TX_PER_SEC = 87
    NVML_GPM_METRIC_NVLINK_L13_RX_PER_SEC = 88
    NVML_GPM_METRIC_NVLINK_L13_TX_PER_SEC = 89
    NVML_GPM_METRIC_NVLINK_L14_RX_PER_SEC = 90
    NVML_GPM_METRIC_NVLINK_L14_TX_PER_SEC = 91
    NVML_GPM_METRIC_NVLINK_L15_RX_PER_SEC = 92
    NVML_GPM_METRIC_NVLINK_L15_TX_PER_SEC = 93
    NVML_GPM_METRIC_NVLINK_L16_RX_PER_SEC = 94
    NVML_GPM_METRIC_NVLINK_L16_TX_PER_SEC = 95
    NVML_GPM_METRIC_NVLINK_L17_RX_PER_SEC = 96
    NVML_GPM_METRIC_NVLINK_L17_TX_PER_SEC = 97
    NVML_GPM_METRIC_MAX = 98
end

mutable struct nvmlGpmSample_st end

const nvmlGpmSample_t = Ptr{nvmlGpmSample_st}

struct var"##Ctag#380"
    shortName::Cstring
    longName::Cstring
    unit::Cstring
end
function Base.getproperty(x::Ptr{var"##Ctag#380"}, f::Symbol)
    f === :shortName && return Ptr{Cstring}(x + 0)
    f === :longName && return Ptr{Cstring}(x + 8)
    f === :unit && return Ptr{Cstring}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#380", f::Symbol)
    r = Ref{var"##Ctag#380"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#380"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#380"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct nvmlGpmMetric_t
    data::NTuple{40,UInt8}
end

function Base.getproperty(x::Ptr{nvmlGpmMetric_t}, f::Symbol)
    f === :metricId && return Ptr{Cuint}(x + 0)
    f === :nvmlReturn && return Ptr{nvmlReturn_t}(x + 4)
    f === :value && return Ptr{Cdouble}(x + 8)
    f === :metricInfo && return Ptr{var"##Ctag#380"}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::nvmlGpmMetric_t, f::Symbol)
    r = Ref{nvmlGpmMetric_t}(x)
    ptr = Base.unsafe_convert(Ptr{nvmlGpmMetric_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{nvmlGpmMetric_t}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct nvmlGpmMetricsGet_t
    version::Cuint
    numMetrics::Cuint
    sample1::nvmlGpmSample_t
    sample2::nvmlGpmSample_t
    metrics::NTuple{98,nvmlGpmMetric_t}
end

struct nvmlGpmSupport_t
    version::Cuint
    isSupportedDevice::Cuint
end

@checked function nvmlGpmMetricsGet(metricsGet)
    initialize_context()
    @ccall (libnvml()).nvmlGpmMetricsGet(metricsGet::Ptr{nvmlGpmMetricsGet_t})::nvmlReturn_t
end

@checked function nvmlGpmSampleFree(gpmSample)
    initialize_context()
    @ccall (libnvml()).nvmlGpmSampleFree(gpmSample::nvmlGpmSample_t)::nvmlReturn_t
end

@checked function nvmlGpmSampleAlloc(gpmSample)
    initialize_context()
    @ccall (libnvml()).nvmlGpmSampleAlloc(gpmSample::Ptr{nvmlGpmSample_t})::nvmlReturn_t
end

@checked function nvmlGpmSampleGet(device, gpmSample)
    initialize_context()
    @ccall (libnvml()).nvmlGpmSampleGet(device::nvmlDevice_t,
                                        gpmSample::nvmlGpmSample_t)::nvmlReturn_t
end

@checked function nvmlGpmMigSampleGet(device, gpuInstanceId, gpmSample)
    initialize_context()
    @ccall (libnvml()).nvmlGpmMigSampleGet(device::nvmlDevice_t, gpuInstanceId::Cuint,
                                           gpmSample::nvmlGpmSample_t)::nvmlReturn_t
end

@checked function nvmlGpmQueryDeviceSupport(device, gpmSupport)
    initialize_context()
    @ccall (libnvml()).nvmlGpmQueryDeviceSupport(device::nvmlDevice_t,
                                                 gpmSupport::Ptr{nvmlGpmSupport_t})::nvmlReturn_t
end

@checked function nvmlDeviceCcuGetStreamState(device, state)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceCcuGetStreamState(device::nvmlDevice_t,
                                                   state::Ptr{Cuint})::nvmlReturn_t
end

@checked function nvmlDeviceCcuSetStreamState(device, state)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceCcuSetStreamState(device::nvmlDevice_t,
                                                   state::Cuint)::nvmlReturn_t
end

struct nvmlNvLinkPowerThres_st
    lowPwrThreshold::Cuint
end

const nvmlNvLinkPowerThres_t = nvmlNvLinkPowerThres_st

@checked function nvmlDeviceSetNvLinkDeviceLowPowerThreshold(device, info)
    initialize_context()
    @ccall (libnvml()).nvmlDeviceSetNvLinkDeviceLowPowerThreshold(device::nvmlDevice_t,
                                                                  info::Ptr{nvmlNvLinkPowerThres_t})::nvmlReturn_t
end

const NVML_API_VERSION = 12

const NVML_API_VERSION_STR = "12"

const NVML_VALUE_NOT_AVAILABLE = -1

const NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE = 32

const NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE = 16

const NVML_DEVICE_PCI_BUS_ID_LEGACY_FMT = "%04X:%02X:%02X.0"

const NVML_DEVICE_PCI_BUS_ID_FMT = "%08X:%02X:%02X.0"

const NVML_NVLINK_MAX_LINKS = 18

const NVML_TOPOLOGY_CPU = NVML_TOPOLOGY_NODE

const NVML_MAX_PHYSICAL_BRIDGE = 128

const NVML_MAX_THERMAL_SENSORS_PER_GPU = 3

const nvmlFlagDefault = 0x00

const nvmlFlagForce = 0x01

const MAX_CLK_DOMAINS = 32

const NVML_SINGLE_BIT_ECC = NVML_MEMORY_ERROR_TYPE_CORRECTED

const NVML_DOUBLE_BIT_ECC = NVML_MEMORY_ERROR_TYPE_UNCORRECTED

const NVML_MAX_GPU_PERF_PSTATES = 16

const NVML_GRID_LICENSE_EXPIRY_NOT_AVAILABLE = 0

const NVML_GRID_LICENSE_EXPIRY_INVALID = 1

const NVML_GRID_LICENSE_EXPIRY_VALID = 2

const NVML_GRID_LICENSE_EXPIRY_NOT_APPLICABLE = 3

const NVML_GRID_LICENSE_EXPIRY_PERMANENT = 4

const NVML_GRID_LICENSE_BUFFER_SIZE = 128

const NVML_VGPU_NAME_BUFFER_SIZE = 64

const NVML_GRID_LICENSE_FEATURE_MAX_COUNT = 3

const INVALID_GPU_INSTANCE_PROFILE_ID = 0xffffffff

const INVALID_GPU_INSTANCE_ID = 0xffffffff

const NVML_VGPU_VIRTUALIZATION_CAP_MIGRATION = 0:0

const NVML_VGPU_VIRTUALIZATION_CAP_MIGRATION_NO = 0x00

const NVML_VGPU_VIRTUALIZATION_CAP_MIGRATION_YES = 0x01

const NVML_VGPU_PGPU_VIRTUALIZATION_CAP_MIGRATION = 0:0

const NVML_VGPU_PGPU_VIRTUALIZATION_CAP_MIGRATION_NO = 0x00

const NVML_VGPU_PGPU_VIRTUALIZATION_CAP_MIGRATION_YES = 0x01

const NVML_GRID_LICENSE_STATE_UNKNOWN = 0

const NVML_GRID_LICENSE_STATE_UNINITIALIZED = 1

const NVML_GRID_LICENSE_STATE_UNLICENSED_UNRESTRICTED = 2

const NVML_GRID_LICENSE_STATE_UNLICENSED_RESTRICTED = 3

const NVML_GRID_LICENSE_STATE_UNLICENSED = 4

const NVML_GRID_LICENSE_STATE_LICENSED = 5

const NVML_GSP_FIRMWARE_VERSION_BUF_SIZE = 0x40

const NVML_DEVICE_ARCH_KEPLER = 2

const NVML_DEVICE_ARCH_MAXWELL = 3

const NVML_DEVICE_ARCH_PASCAL = 4

const NVML_DEVICE_ARCH_VOLTA = 5

const NVML_DEVICE_ARCH_TURING = 6

const NVML_DEVICE_ARCH_AMPERE = 7

const NVML_DEVICE_ARCH_ADA = 8

const NVML_DEVICE_ARCH_HOPPER = 9

const NVML_DEVICE_ARCH_UNKNOWN = 0xffffffff

const NVML_BUS_TYPE_UNKNOWN = 0

const NVML_BUS_TYPE_PCI = 1

const NVML_BUS_TYPE_PCIE = 2

const NVML_BUS_TYPE_FPCI = 3

const NVML_BUS_TYPE_AGP = 4

const NVML_FAN_POLICY_TEMPERATURE_CONTINOUS_SW = 0

const NVML_FAN_POLICY_MANUAL = 1

const NVML_POWER_SOURCE_AC = 0x00000000

const NVML_POWER_SOURCE_BATTERY = 0x00000001

const NVML_PCIE_LINK_MAX_SPEED_INVALID = 0x00000000

const NVML_PCIE_LINK_MAX_SPEED_2500MBPS = 0x00000001

const NVML_PCIE_LINK_MAX_SPEED_5000MBPS = 0x00000002

const NVML_PCIE_LINK_MAX_SPEED_8000MBPS = 0x00000003

const NVML_PCIE_LINK_MAX_SPEED_16000MBPS = 0x00000004

const NVML_PCIE_LINK_MAX_SPEED_32000MBPS = 0x00000005

const NVML_PCIE_LINK_MAX_SPEED_64000MBPS = 0x00000006

const NVML_ADAPTIVE_CLOCKING_INFO_STATUS_DISABLED = 0x00000000

const NVML_ADAPTIVE_CLOCKING_INFO_STATUS_ENABLED = 0x00000001

const NVML_MAX_GPU_UTILIZATIONS = 8

const NVML_FI_DEV_ECC_CURRENT = 1

const NVML_FI_DEV_ECC_PENDING = 2

const NVML_FI_DEV_ECC_SBE_VOL_TOTAL = 3

const NVML_FI_DEV_ECC_DBE_VOL_TOTAL = 4

const NVML_FI_DEV_ECC_SBE_AGG_TOTAL = 5

const NVML_FI_DEV_ECC_DBE_AGG_TOTAL = 6

const NVML_FI_DEV_ECC_SBE_VOL_L1 = 7

const NVML_FI_DEV_ECC_DBE_VOL_L1 = 8

const NVML_FI_DEV_ECC_SBE_VOL_L2 = 9

const NVML_FI_DEV_ECC_DBE_VOL_L2 = 10

const NVML_FI_DEV_ECC_SBE_VOL_DEV = 11

const NVML_FI_DEV_ECC_DBE_VOL_DEV = 12

const NVML_FI_DEV_ECC_SBE_VOL_REG = 13

const NVML_FI_DEV_ECC_DBE_VOL_REG = 14

const NVML_FI_DEV_ECC_SBE_VOL_TEX = 15

const NVML_FI_DEV_ECC_DBE_VOL_TEX = 16

const NVML_FI_DEV_ECC_DBE_VOL_CBU = 17

const NVML_FI_DEV_ECC_SBE_AGG_L1 = 18

const NVML_FI_DEV_ECC_DBE_AGG_L1 = 19

const NVML_FI_DEV_ECC_SBE_AGG_L2 = 20

const NVML_FI_DEV_ECC_DBE_AGG_L2 = 21

const NVML_FI_DEV_ECC_SBE_AGG_DEV = 22

const NVML_FI_DEV_ECC_DBE_AGG_DEV = 23

const NVML_FI_DEV_ECC_SBE_AGG_REG = 24

const NVML_FI_DEV_ECC_DBE_AGG_REG = 25

const NVML_FI_DEV_ECC_SBE_AGG_TEX = 26

const NVML_FI_DEV_ECC_DBE_AGG_TEX = 27

const NVML_FI_DEV_ECC_DBE_AGG_CBU = 28

const NVML_FI_DEV_RETIRED_SBE = 29

const NVML_FI_DEV_RETIRED_DBE = 30

const NVML_FI_DEV_RETIRED_PENDING = 31

const NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L0 = 32

const NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L1 = 33

const NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L2 = 34

const NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L3 = 35

const NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L4 = 36

const NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L5 = 37

const NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL = 38

const NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L0 = 39

const NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L1 = 40

const NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L2 = 41

const NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L3 = 42

const NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L4 = 43

const NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L5 = 44

const NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL = 45

const NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L0 = 46

const NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L1 = 47

const NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L2 = 48

const NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L3 = 49

const NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L4 = 50

const NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L5 = 51

const NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL = 52

const NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L0 = 53

const NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L1 = 54

const NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L2 = 55

const NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L3 = 56

const NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L4 = 57

const NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L5 = 58

const NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL = 59

const NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L0 = 60

const NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L1 = 61

const NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L2 = 62

const NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L3 = 63

const NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L4 = 64

const NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L5 = 65

const NVML_FI_DEV_NVLINK_BANDWIDTH_C0_TOTAL = 66

const NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L0 = 67

const NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L1 = 68

const NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L2 = 69

const NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L3 = 70

const NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L4 = 71

const NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L5 = 72

const NVML_FI_DEV_NVLINK_BANDWIDTH_C1_TOTAL = 73

const NVML_FI_DEV_PERF_POLICY_POWER = 74

const NVML_FI_DEV_PERF_POLICY_THERMAL = 75

const NVML_FI_DEV_PERF_POLICY_SYNC_BOOST = 76

const NVML_FI_DEV_PERF_POLICY_BOARD_LIMIT = 77

const NVML_FI_DEV_PERF_POLICY_LOW_UTILIZATION = 78

const NVML_FI_DEV_PERF_POLICY_RELIABILITY = 79

const NVML_FI_DEV_PERF_POLICY_TOTAL_APP_CLOCKS = 80

const NVML_FI_DEV_PERF_POLICY_TOTAL_BASE_CLOCKS = 81

const NVML_FI_DEV_MEMORY_TEMP = 82

const NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION = 83

const NVML_FI_DEV_NVLINK_SPEED_MBPS_L0 = 84

const NVML_FI_DEV_NVLINK_SPEED_MBPS_L1 = 85

const NVML_FI_DEV_NVLINK_SPEED_MBPS_L2 = 86

const NVML_FI_DEV_NVLINK_SPEED_MBPS_L3 = 87

const NVML_FI_DEV_NVLINK_SPEED_MBPS_L4 = 88

const NVML_FI_DEV_NVLINK_SPEED_MBPS_L5 = 89

const NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON = 90

const NVML_FI_DEV_NVLINK_LINK_COUNT = 91

const NVML_FI_DEV_RETIRED_PENDING_SBE = 92

const NVML_FI_DEV_RETIRED_PENDING_DBE = 93

const NVML_FI_DEV_PCIE_REPLAY_COUNTER = 94

const NVML_FI_DEV_PCIE_REPLAY_ROLLOVER_COUNTER = 95

const NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L6 = 96

const NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L7 = 97

const NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L8 = 98

const NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L9 = 99

const NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L10 = 100

const NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L11 = 101

const NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L6 = 102

const NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L7 = 103

const NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L8 = 104

const NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L9 = 105

const NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L10 = 106

const NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L11 = 107

const NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L6 = 108

const NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L7 = 109

const NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L8 = 110

const NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L9 = 111

const NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L10 = 112

const NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L11 = 113

const NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L6 = 114

const NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L7 = 115

const NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L8 = 116

const NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L9 = 117

const NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L10 = 118

const NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L11 = 119

const NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L6 = 120

const NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L7 = 121

const NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L8 = 122

const NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L9 = 123

const NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L10 = 124

const NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L11 = 125

const NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L6 = 126

const NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L7 = 127

const NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L8 = 128

const NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L9 = 129

const NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L10 = 130

const NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L11 = 131

const NVML_FI_DEV_NVLINK_SPEED_MBPS_L6 = 132

const NVML_FI_DEV_NVLINK_SPEED_MBPS_L7 = 133

const NVML_FI_DEV_NVLINK_SPEED_MBPS_L8 = 134

const NVML_FI_DEV_NVLINK_SPEED_MBPS_L9 = 135

const NVML_FI_DEV_NVLINK_SPEED_MBPS_L10 = 136

const NVML_FI_DEV_NVLINK_SPEED_MBPS_L11 = 137

const NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX = 138

const NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX = 139

const NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_TX = 140

const NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_RX = 141

const NVML_FI_DEV_REMAPPED_COR = 142

const NVML_FI_DEV_REMAPPED_UNC = 143

const NVML_FI_DEV_REMAPPED_PENDING = 144

const NVML_FI_DEV_REMAPPED_FAILURE = 145

const NVML_FI_DEV_NVLINK_REMOTE_NVLINK_ID = 146

const NVML_FI_DEV_NVSWITCH_CONNECTED_LINK_COUNT = 147

const NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L0 = 148

const NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L1 = 149

const NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L2 = 150

const NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L3 = 151

const NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L4 = 152

const NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L5 = 153

const NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L6 = 154

const NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L7 = 155

const NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L8 = 156

const NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L9 = 157

const NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L10 = 158

const NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L11 = 159

const NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_TOTAL = 160

const NVML_FI_DEV_NVLINK_ERROR_DL_REPLAY = 161

const NVML_FI_DEV_NVLINK_ERROR_DL_RECOVERY = 162

const NVML_FI_DEV_NVLINK_ERROR_DL_CRC = 163

const NVML_FI_DEV_NVLINK_GET_SPEED = 164

const NVML_FI_DEV_NVLINK_GET_STATE = 165

const NVML_FI_DEV_NVLINK_GET_VERSION = 166

const NVML_FI_DEV_NVLINK_GET_POWER_STATE = 167

const NVML_FI_DEV_NVLINK_GET_POWER_THRESHOLD = 168

const NVML_FI_DEV_PCIE_L0_TO_RECOVERY_COUNTER = 169

const NVML_FI_MAX = 173

const nvmlEventTypeSingleBitEccError = Clonglong(0x0000000000000001)

const nvmlEventTypeDoubleBitEccError = Clonglong(0x0000000000000002)

const nvmlEventTypePState = Clonglong(0x0000000000000004)

const nvmlEventTypeXidCriticalError = Clonglong(0x0000000000000008)

const nvmlEventTypeClock = Clonglong(0x0000000000000010)

const nvmlEventTypePowerSourceChange = Clonglong(0x0000000000000080)

const nvmlEventMigConfigChange = Clonglong(0x0000000000000100)

const nvmlEventTypeNone = Clonglong(0x0000000000000000)

const nvmlEventTypeAll = ((((((nvmlEventTypeNone | nvmlEventTypeSingleBitEccError) |
                              nvmlEventTypeDoubleBitEccError) | nvmlEventTypePState) |
                            nvmlEventTypeClock) | nvmlEventTypeXidCriticalError) |
                          nvmlEventTypePowerSourceChange) | nvmlEventMigConfigChange

const nvmlClocksThrottleReasonGpuIdle = Clonglong(0x0000000000000001)

const nvmlClocksThrottleReasonApplicationsClocksSetting = Clonglong(0x0000000000000002)

const nvmlClocksThrottleReasonSwPowerCap = Clonglong(0x0000000000000004)

const nvmlClocksThrottleReasonHwSlowdown = Clonglong(0x0000000000000008)

const nvmlClocksThrottleReasonSyncBoost = Clonglong(0x0000000000000010)

const nvmlClocksThrottleReasonSwThermalSlowdown = Clonglong(0x0000000000000020)

const nvmlClocksThrottleReasonHwThermalSlowdown = Clonglong(0x0000000000000040)

const nvmlClocksThrottleReasonHwPowerBrakeSlowdown = Clonglong(0x0000000000000080)

const nvmlClocksThrottleReasonDisplayClockSetting = Clonglong(0x0000000000000100)

const nvmlClocksThrottleReasonNone = Clonglong(0x0000000000000000)

# Skipping MacroDefinition: nvmlClocksThrottleReasonAll ( nvmlClocksThrottleReasonNone | nvmlClocksThrottleReasonGpuIdle | nvmlClocksThrottleReasonApplicationsClocksSetting | nvmlClocksThrottleReasonSwPowerCap | nvmlClocksThrottleReasonHwSlowdown | nvmlClocksThrottleReasonSyncBoost | nvmlClocksThrottleReasonSwThermalSlowdown | nvmlClocksThrottleReasonHwThermalSlowdown | nvmlClocksThrottleReasonHwPowerBrakeSlowdown | nvmlClocksThrottleReasonDisplayClockSetting \
#)

const NVML_NVFBC_SESSION_FLAG_DIFFMAP_ENABLED = 0x00000001

const NVML_NVFBC_SESSION_FLAG_CLASSIFICATIONMAP_ENABLED = 0x00000002

const NVML_NVFBC_SESSION_FLAG_CAPTURE_WITH_WAIT_NO_WAIT = 0x00000004

const NVML_NVFBC_SESSION_FLAG_CAPTURE_WITH_WAIT_INFINITE = 0x00000008

const NVML_NVFBC_SESSION_FLAG_CAPTURE_WITH_WAIT_TIMEOUT = 0x00000010

const NVML_GPU_FABRIC_UUID_LEN = 16

const NVML_GPU_FABRIC_STATE_NOT_SUPPORTED = 0

const NVML_GPU_FABRIC_STATE_NOT_STARTED = 1

const NVML_GPU_FABRIC_STATE_IN_PROGRESS = 2

const NVML_GPU_FABRIC_STATE_COMPLETED = 3

const NVML_INIT_FLAG_NO_GPUS = 1

const NVML_INIT_FLAG_NO_ATTACH = 2

const NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE = 16

const NVML_DEVICE_UUID_BUFFER_SIZE = 80

const NVML_DEVICE_UUID_V2_BUFFER_SIZE = 96

const NVML_DEVICE_PART_NUMBER_BUFFER_SIZE = 80

const NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE = 80

const NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE = 80

const NVML_DEVICE_NAME_BUFFER_SIZE = 64

const NVML_DEVICE_NAME_V2_BUFFER_SIZE = 96

const NVML_DEVICE_SERIAL_BUFFER_SIZE = 30

const NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE = 32

const NVML_AFFINITY_SCOPE_NODE = 0

const NVML_AFFINITY_SCOPE_SOCKET = 1

const NVML_DEVICE_MIG_DISABLE = 0x00

const NVML_DEVICE_MIG_ENABLE = 0x01

const NVML_GPU_INSTANCE_PROFILE_1_SLICE = 0x00

const NVML_GPU_INSTANCE_PROFILE_2_SLICE = 0x01

const NVML_GPU_INSTANCE_PROFILE_3_SLICE = 0x02

const NVML_GPU_INSTANCE_PROFILE_4_SLICE = 0x03

const NVML_GPU_INSTANCE_PROFILE_7_SLICE = 0x04

const NVML_GPU_INSTANCE_PROFILE_8_SLICE = 0x05

const NVML_GPU_INSTANCE_PROFILE_6_SLICE = 0x06

const NVML_GPU_INSTANCE_PROFILE_1_SLICE_REV1 = 0x07

const NVML_GPU_INSTANCE_PROFILE_2_SLICE_REV1 = 0x08

const NVML_GPU_INSTANCE_PROFILE_1_SLICE_REV2 = 0x09

const NVML_GPU_INSTANCE_PROFILE_COUNT = 0x0a

const NVML_COMPUTE_INSTANCE_PROFILE_1_SLICE = 0x00

const NVML_COMPUTE_INSTANCE_PROFILE_2_SLICE = 0x01

const NVML_COMPUTE_INSTANCE_PROFILE_3_SLICE = 0x02

const NVML_COMPUTE_INSTANCE_PROFILE_4_SLICE = 0x03

const NVML_COMPUTE_INSTANCE_PROFILE_7_SLICE = 0x04

const NVML_COMPUTE_INSTANCE_PROFILE_8_SLICE = 0x05

const NVML_COMPUTE_INSTANCE_PROFILE_6_SLICE = 0x06

const NVML_COMPUTE_INSTANCE_PROFILE_1_SLICE_REV1 = 0x07

const NVML_COMPUTE_INSTANCE_PROFILE_COUNT = 0x08

const NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED = 0x00

const NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_COUNT = 0x01

const NVML_GPM_METRICS_GET_VERSION = 1

const NVML_GPM_SUPPORT_VERSION = 1

const NVML_COUNTER_COLLECTION_UNIT_STREAM_STATE_DISABLE = 0

const NVML_COUNTER_COLLECTION_UNIT_STREAM_STATE_ENABLE = 1

const NVML_NVLINK_POWER_STATE_HIGH_SPEED = 0x00

const NVML_NVLINK_POWER_STATE_LOW = 0x01

const NVML_NVLINK_LOW_POWER_THRESHOLD_MIN = 0x01

const NVML_NVLINK_LOW_POWER_THRESHOLD_MAX = 0x1fff

const NVML_NVLINK_LOW_POWER_THRESHOLD_RESET = 0xffffffff
