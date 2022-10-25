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
            nvmlShutdown()
        end
        initialized[] = true
    end
end

macro check(ex)
    quote
        res = $(esc(ex))
        if res != NVML_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
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
    NVML_ERROR_UNKNOWN = 999
end

const nvmlReturn_t = nvmlReturn_enum

@checked function nvmlInit_v2()
    ccall((:nvmlInit_v2, libnvml()), nvmlReturn_t, ())
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
    ccall((:nvmlDeviceGetPciInfo_v3, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlPciInfo_t}), device, pci)
end

@checked function nvmlDeviceGetCount_v2(deviceCount)
    initialize_context()
    ccall((:nvmlDeviceGetCount_v2, libnvml()), nvmlReturn_t, (Ptr{Cuint},), deviceCount)
end

@checked function nvmlDeviceGetHandleByIndex_v2(index, device)
    initialize_context()
    ccall((:nvmlDeviceGetHandleByIndex_v2, libnvml()), nvmlReturn_t,
          (Cuint, Ptr{nvmlDevice_t}), index, device)
end

@checked function nvmlDeviceGetHandleByPciBusId_v2(pciBusId, device)
    initialize_context()
    ccall((:nvmlDeviceGetHandleByPciBusId_v2, libnvml()), nvmlReturn_t,
          (Cstring, Ptr{nvmlDevice_t}), pciBusId, device)
end

@checked function nvmlDeviceGetNvLinkRemotePciInfo_v2(device, link, pci)
    initialize_context()
    ccall((:nvmlDeviceGetNvLinkRemotePciInfo_v2, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{nvmlPciInfo_t}), device, link, pci)
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
    ccall((:nvmlDeviceRemoveGpu_v2, libnvml()), nvmlReturn_t,
          (Ptr{nvmlPciInfo_t}, nvmlDetachGpuState_t, nvmlPcieLinkState_t), pciInfo,
          gpuState, linkState)
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
    ccall((:nvmlDeviceGetGridLicensableFeatures_v4, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlGridLicensableFeatures_t}), device,
          pGridLicensableFeatures)
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
    ccall((:nvmlEventSetWait_v2, libnvml()), nvmlReturn_t,
          (nvmlEventSet_t, Ptr{nvmlEventData_t}, Cuint), set, data, timeoutms)
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
    ccall((:nvmlDeviceGetAttributes_v2, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlDeviceAttributes_t}), device, attributes)
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
    ccall((:nvmlComputeInstanceGetInfo_v2, libnvml()), nvmlReturn_t,
          (nvmlComputeInstance_t, Ptr{nvmlComputeInstanceInfo_t}), computeInstance, info)
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
    ccall((:nvmlDeviceGetComputeRunningProcesses_v3, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}, Ptr{nvmlProcessInfo_t}), device, infoCount, infos)
end

@checked function nvmlDeviceGetGraphicsRunningProcesses_v3(device, infoCount, infos)
    initialize_context()
    ccall((:nvmlDeviceGetGraphicsRunningProcesses_v3, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}, Ptr{nvmlProcessInfo_t}), device, infoCount, infos)
end

@checked function nvmlDeviceGetMPSComputeRunningProcesses_v3(device, infoCount, infos)
    initialize_context()
    ccall((:nvmlDeviceGetMPSComputeRunningProcesses_v3, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}, Ptr{nvmlProcessInfo_t}), device, infoCount, infos)
end

struct nvmlExcludedDeviceInfo_st
    pciInfo::nvmlPciInfo_t
    uuid::NTuple{80,Cchar}
end

const nvmlExcludedDeviceInfo_t = nvmlExcludedDeviceInfo_st

@checked function nvmlGetExcludedDeviceCount(deviceCount)
    initialize_context()
    ccall((:nvmlGetExcludedDeviceCount, libnvml()), nvmlReturn_t, (Ptr{Cuint},),
          deviceCount)
end

@checked function nvmlGetExcludedDeviceInfoByIndex(index, info)
    initialize_context()
    ccall((:nvmlGetExcludedDeviceInfoByIndex, libnvml()), nvmlReturn_t,
          (Cuint, Ptr{nvmlExcludedDeviceInfo_t}), index, info)
end

struct nvmlGpuInstancePlacement_st
    start::Cuint
    size::Cuint
end

const nvmlGpuInstancePlacement_t = nvmlGpuInstancePlacement_st

@checked function nvmlDeviceGetGpuInstancePossiblePlacements_v2(device, profileId,
                                                                placements, count)
    initialize_context()
    ccall((:nvmlDeviceGetGpuInstancePossiblePlacements_v2, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{nvmlGpuInstancePlacement_t}, Ptr{Cuint}), device,
          profileId, placements, count)
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
    ccall((:nvmlVgpuInstanceGetLicenseInfo_v2, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Ptr{nvmlVgpuLicenseInfo_t}), vgpuInstance, licenseInfo)
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
    unsafe_store!(getproperty(x, f), v)
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

struct var"##Ctag#493"
    controller::nvmlThermalController_t
    defaultMinTemp::Cint
    defaultMaxTemp::Cint
    currentTemp::Cint
    target::nvmlThermalTarget_t
end
function Base.getproperty(x::Ptr{var"##Ctag#493"}, f::Symbol)
    f === :controller && return Ptr{nvmlThermalController_t}(x + 0)
    f === :defaultMinTemp && return Ptr{Cint}(x + 4)
    f === :defaultMaxTemp && return Ptr{Cint}(x + 8)
    f === :currentTemp && return Ptr{Cint}(x + 12)
    f === :target && return Ptr{nvmlThermalTarget_t}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#493", f::Symbol)
    r = Ref{var"##Ctag#493"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#493"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#493"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct nvmlGpuThermalSettings_t
    data::NTuple{64,UInt8}
end

function Base.getproperty(x::Ptr{nvmlGpuThermalSettings_t}, f::Symbol)
    f === :count && return Ptr{Cuint}(x + 0)
    f === :sensor && return Ptr{NTuple{3,var"##Ctag#493"}}(x + 4)
    return getfield(x, f)
end

function Base.getproperty(x::nvmlGpuThermalSettings_t, f::Symbol)
    r = Ref{nvmlGpuThermalSettings_t}(x)
    ptr = Base.unsafe_convert(Ptr{nvmlGpuThermalSettings_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{nvmlGpuThermalSettings_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
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
    NVML_VGPU_CAP_COUNT = 2
end

const nvmlVgpuCapability_t = nvmlVgpuCapability_enum

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

const nvmlPowerSource_t = Cuint

@cenum nvmlGpuUtilizationDomainId_t::UInt32 begin
    NVML_GPU_UTILIZATION_DOMAIN_GPU = 0
    NVML_GPU_UTILIZATION_DOMAIN_FB = 1
    NVML_GPU_UTILIZATION_DOMAIN_VID = 2
    NVML_GPU_UTILIZATION_DOMAIN_BUS = 3
end

struct var"##Ctag#494"
    bIsPresent::Cuint
    percentage::Cuint
    incThreshold::Cuint
    decThreshold::Cuint
end
function Base.getproperty(x::Ptr{var"##Ctag#494"}, f::Symbol)
    f === :bIsPresent && return Ptr{Cuint}(x + 0)
    f === :percentage && return Ptr{Cuint}(x + 4)
    f === :incThreshold && return Ptr{Cuint}(x + 8)
    f === :decThreshold && return Ptr{Cuint}(x + 12)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#494", f::Symbol)
    r = Ref{var"##Ctag#494"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#494"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#494"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct nvmlGpuDynamicPstatesInfo_st
    data::NTuple{132,UInt8}
end

function Base.getproperty(x::Ptr{nvmlGpuDynamicPstatesInfo_st}, f::Symbol)
    f === :flags && return Ptr{Cuint}(x + 0)
    f === :utilization && return Ptr{NTuple{8,var"##Ctag#494"}}(x + 4)
    return getfield(x, f)
end

function Base.getproperty(x::nvmlGpuDynamicPstatesInfo_st, f::Symbol)
    r = Ref{nvmlGpuDynamicPstatesInfo_st}(x)
    ptr = Base.unsafe_convert(Ptr{nvmlGpuDynamicPstatesInfo_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{nvmlGpuDynamicPstatesInfo_st}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
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

@checked function nvmlInitWithFlags(flags)
    ccall((:nvmlInitWithFlags, libnvml()), nvmlReturn_t, (Cuint,), flags)
end

@checked function nvmlShutdown()
    ccall((:nvmlShutdown, libnvml()), nvmlReturn_t, ())
end

function nvmlErrorString(result)
    ccall((:nvmlErrorString, libnvml()), Cstring, (nvmlReturn_t,), result)
end

@checked function nvmlSystemGetDriverVersion(version, length)
    initialize_context()
    ccall((:nvmlSystemGetDriverVersion, libnvml()), nvmlReturn_t, (Cstring, Cuint), version,
          length)
end

@checked function nvmlSystemGetNVMLVersion(version, length)
    initialize_context()
    ccall((:nvmlSystemGetNVMLVersion, libnvml()), nvmlReturn_t, (Cstring, Cuint), version,
          length)
end

@checked function nvmlSystemGetCudaDriverVersion(cudaDriverVersion)
    initialize_context()
    ccall((:nvmlSystemGetCudaDriverVersion, libnvml()), nvmlReturn_t, (Ptr{Cint},),
          cudaDriverVersion)
end

@checked function nvmlSystemGetCudaDriverVersion_v2(cudaDriverVersion)
    initialize_context()
    ccall((:nvmlSystemGetCudaDriverVersion_v2, libnvml()), nvmlReturn_t, (Ptr{Cint},),
          cudaDriverVersion)
end

@checked function nvmlSystemGetProcessName(pid, name, length)
    initialize_context()
    ccall((:nvmlSystemGetProcessName, libnvml()), nvmlReturn_t, (Cuint, Cstring, Cuint),
          pid, name, length)
end

@checked function nvmlUnitGetCount(unitCount)
    initialize_context()
    ccall((:nvmlUnitGetCount, libnvml()), nvmlReturn_t, (Ptr{Cuint},), unitCount)
end

@checked function nvmlUnitGetHandleByIndex(index, unit)
    initialize_context()
    ccall((:nvmlUnitGetHandleByIndex, libnvml()), nvmlReturn_t, (Cuint, Ptr{nvmlUnit_t}),
          index, unit)
end

@checked function nvmlUnitGetUnitInfo(unit, info)
    initialize_context()
    ccall((:nvmlUnitGetUnitInfo, libnvml()), nvmlReturn_t,
          (nvmlUnit_t, Ptr{nvmlUnitInfo_t}), unit, info)
end

@checked function nvmlUnitGetLedState(unit, state)
    initialize_context()
    ccall((:nvmlUnitGetLedState, libnvml()), nvmlReturn_t,
          (nvmlUnit_t, Ptr{nvmlLedState_t}), unit, state)
end

@checked function nvmlUnitGetPsuInfo(unit, psu)
    initialize_context()
    ccall((:nvmlUnitGetPsuInfo, libnvml()), nvmlReturn_t, (nvmlUnit_t, Ptr{nvmlPSUInfo_t}),
          unit, psu)
end

@checked function nvmlUnitGetTemperature(unit, type, temp)
    initialize_context()
    ccall((:nvmlUnitGetTemperature, libnvml()), nvmlReturn_t,
          (nvmlUnit_t, Cuint, Ptr{Cuint}), unit, type, temp)
end

@checked function nvmlUnitGetFanSpeedInfo(unit, fanSpeeds)
    initialize_context()
    ccall((:nvmlUnitGetFanSpeedInfo, libnvml()), nvmlReturn_t,
          (nvmlUnit_t, Ptr{nvmlUnitFanSpeeds_t}), unit, fanSpeeds)
end

@checked function nvmlUnitGetDevices(unit, deviceCount, devices)
    initialize_context()
    ccall((:nvmlUnitGetDevices, libnvml()), nvmlReturn_t,
          (nvmlUnit_t, Ptr{Cuint}, Ptr{nvmlDevice_t}), unit, deviceCount, devices)
end

@checked function nvmlSystemGetHicVersion(hwbcCount, hwbcEntries)
    initialize_context()
    ccall((:nvmlSystemGetHicVersion, libnvml()), nvmlReturn_t,
          (Ptr{Cuint}, Ptr{nvmlHwbcEntry_t}), hwbcCount, hwbcEntries)
end

@checked function nvmlDeviceGetHandleBySerial(serial, device)
    initialize_context()
    ccall((:nvmlDeviceGetHandleBySerial, libnvml()), nvmlReturn_t,
          (Cstring, Ptr{nvmlDevice_t}), serial, device)
end

@checked function nvmlDeviceGetHandleByUUID(uuid, device)
    initialize_context()
    ccall((:nvmlDeviceGetHandleByUUID, libnvml()), nvmlReturn_t,
          (Cstring, Ptr{nvmlDevice_t}), uuid, device)
end

@checked function nvmlDeviceGetName(device, name, length)
    initialize_context()
    ccall((:nvmlDeviceGetName, libnvml()), nvmlReturn_t, (nvmlDevice_t, Cstring, Cuint),
          device, name, length)
end

@checked function nvmlDeviceGetBrand(device, type)
    initialize_context()
    ccall((:nvmlDeviceGetBrand, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlBrandType_t}), device, type)
end

@checked function nvmlDeviceGetIndex(device, index)
    initialize_context()
    ccall((:nvmlDeviceGetIndex, libnvml()), nvmlReturn_t, (nvmlDevice_t, Ptr{Cuint}),
          device, index)
end

@checked function nvmlDeviceGetSerial(device, serial, length)
    initialize_context()
    ccall((:nvmlDeviceGetSerial, libnvml()), nvmlReturn_t, (nvmlDevice_t, Cstring, Cuint),
          device, serial, length)
end

const nvmlAffinityScope_t = Cuint

@checked function nvmlDeviceGetMemoryAffinity(device, nodeSetSize, nodeSet, scope)
    initialize_context()
    ccall((:nvmlDeviceGetMemoryAffinity, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{Culong}, nvmlAffinityScope_t), device, nodeSetSize,
          nodeSet, scope)
end

@checked function nvmlDeviceGetCpuAffinityWithinScope(device, cpuSetSize, cpuSet, scope)
    initialize_context()
    ccall((:nvmlDeviceGetCpuAffinityWithinScope, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{Culong}, nvmlAffinityScope_t), device, cpuSetSize,
          cpuSet, scope)
end

@checked function nvmlDeviceGetCpuAffinity(device, cpuSetSize, cpuSet)
    initialize_context()
    ccall((:nvmlDeviceGetCpuAffinity, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{Culong}), device, cpuSetSize, cpuSet)
end

@checked function nvmlDeviceSetCpuAffinity(device)
    initialize_context()
    ccall((:nvmlDeviceSetCpuAffinity, libnvml()), nvmlReturn_t, (nvmlDevice_t,), device)
end

@checked function nvmlDeviceClearCpuAffinity(device)
    initialize_context()
    ccall((:nvmlDeviceClearCpuAffinity, libnvml()), nvmlReturn_t, (nvmlDevice_t,), device)
end

@checked function nvmlDeviceGetTopologyCommonAncestor(device1, device2, pathInfo)
    initialize_context()
    ccall((:nvmlDeviceGetTopologyCommonAncestor, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlDevice_t, Ptr{nvmlGpuTopologyLevel_t}), device1, device2,
          pathInfo)
end

@checked function nvmlDeviceGetTopologyNearestGpus(device, level, count, deviceArray)
    initialize_context()
    ccall((:nvmlDeviceGetTopologyNearestGpus, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlGpuTopologyLevel_t, Ptr{Cuint}, Ptr{nvmlDevice_t}), device,
          level, count, deviceArray)
end

@checked function nvmlSystemGetTopologyGpuSet(cpuNumber, count, deviceArray)
    initialize_context()
    ccall((:nvmlSystemGetTopologyGpuSet, libnvml()), nvmlReturn_t,
          (Cuint, Ptr{Cuint}, Ptr{nvmlDevice_t}), cpuNumber, count, deviceArray)
end

@checked function nvmlDeviceGetP2PStatus(device1, device2, p2pIndex, p2pStatus)
    initialize_context()
    ccall((:nvmlDeviceGetP2PStatus, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlDevice_t, nvmlGpuP2PCapsIndex_t, Ptr{nvmlGpuP2PStatus_t}),
          device1, device2, p2pIndex, p2pStatus)
end

@checked function nvmlDeviceGetUUID(device, uuid, length)
    initialize_context()
    ccall((:nvmlDeviceGetUUID, libnvml()), nvmlReturn_t, (nvmlDevice_t, Cstring, Cuint),
          device, uuid, length)
end

@checked function nvmlVgpuInstanceGetMdevUUID(vgpuInstance, mdevUuid, size)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetMdevUUID, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Cstring, Cuint), vgpuInstance, mdevUuid, size)
end

@checked function nvmlDeviceGetMinorNumber(device, minorNumber)
    initialize_context()
    ccall((:nvmlDeviceGetMinorNumber, libnvml()), nvmlReturn_t, (nvmlDevice_t, Ptr{Cuint}),
          device, minorNumber)
end

@checked function nvmlDeviceGetBoardPartNumber(device, partNumber, length)
    initialize_context()
    ccall((:nvmlDeviceGetBoardPartNumber, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cstring, Cuint), device, partNumber, length)
end

@checked function nvmlDeviceGetInforomVersion(device, object, version, length)
    initialize_context()
    ccall((:nvmlDeviceGetInforomVersion, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlInforomObject_t, Cstring, Cuint), device, object, version,
          length)
end

@checked function nvmlDeviceGetInforomImageVersion(device, version, length)
    initialize_context()
    ccall((:nvmlDeviceGetInforomImageVersion, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cstring, Cuint), device, version, length)
end

@checked function nvmlDeviceGetInforomConfigurationChecksum(device, checksum)
    initialize_context()
    ccall((:nvmlDeviceGetInforomConfigurationChecksum, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}), device, checksum)
end

@checked function nvmlDeviceValidateInforom(device)
    initialize_context()
    ccall((:nvmlDeviceValidateInforom, libnvml()), nvmlReturn_t, (nvmlDevice_t,), device)
end

@checked function nvmlDeviceGetDisplayMode(device, display)
    initialize_context()
    ccall((:nvmlDeviceGetDisplayMode, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlEnableState_t}), device, display)
end

@checked function nvmlDeviceGetDisplayActive(device, isActive)
    initialize_context()
    ccall((:nvmlDeviceGetDisplayActive, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlEnableState_t}), device, isActive)
end

@checked function nvmlDeviceGetPersistenceMode(device, mode)
    initialize_context()
    ccall((:nvmlDeviceGetPersistenceMode, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlEnableState_t}), device, mode)
end

@checked function nvmlDeviceGetMaxPcieLinkGeneration(device, maxLinkGen)
    initialize_context()
    ccall((:nvmlDeviceGetMaxPcieLinkGeneration, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}), device, maxLinkGen)
end

@checked function nvmlDeviceGetMaxPcieLinkWidth(device, maxLinkWidth)
    initialize_context()
    ccall((:nvmlDeviceGetMaxPcieLinkWidth, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}), device, maxLinkWidth)
end

@checked function nvmlDeviceGetCurrPcieLinkGeneration(device, currLinkGen)
    initialize_context()
    ccall((:nvmlDeviceGetCurrPcieLinkGeneration, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}), device, currLinkGen)
end

@checked function nvmlDeviceGetCurrPcieLinkWidth(device, currLinkWidth)
    initialize_context()
    ccall((:nvmlDeviceGetCurrPcieLinkWidth, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}), device, currLinkWidth)
end

@checked function nvmlDeviceGetPcieThroughput(device, counter, value)
    initialize_context()
    ccall((:nvmlDeviceGetPcieThroughput, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlPcieUtilCounter_t, Ptr{Cuint}), device, counter, value)
end

@checked function nvmlDeviceGetPcieReplayCounter(device, value)
    initialize_context()
    ccall((:nvmlDeviceGetPcieReplayCounter, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}), device, value)
end

@checked function nvmlDeviceGetClockInfo(device, type, clock)
    initialize_context()
    ccall((:nvmlDeviceGetClockInfo, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlClockType_t, Ptr{Cuint}), device, type, clock)
end

@checked function nvmlDeviceGetMaxClockInfo(device, type, clock)
    initialize_context()
    ccall((:nvmlDeviceGetMaxClockInfo, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlClockType_t, Ptr{Cuint}), device, type, clock)
end

@checked function nvmlDeviceGetApplicationsClock(device, clockType, clockMHz)
    initialize_context()
    ccall((:nvmlDeviceGetApplicationsClock, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlClockType_t, Ptr{Cuint}), device, clockType, clockMHz)
end

@checked function nvmlDeviceGetDefaultApplicationsClock(device, clockType, clockMHz)
    initialize_context()
    ccall((:nvmlDeviceGetDefaultApplicationsClock, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlClockType_t, Ptr{Cuint}), device, clockType, clockMHz)
end

@checked function nvmlDeviceResetApplicationsClocks(device)
    initialize_context()
    ccall((:nvmlDeviceResetApplicationsClocks, libnvml()), nvmlReturn_t, (nvmlDevice_t,),
          device)
end

@checked function nvmlDeviceGetClock(device, clockType, clockId, clockMHz)
    initialize_context()
    ccall((:nvmlDeviceGetClock, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlClockType_t, nvmlClockId_t, Ptr{Cuint}), device, clockType,
          clockId, clockMHz)
end

@checked function nvmlDeviceGetMaxCustomerBoostClock(device, clockType, clockMHz)
    initialize_context()
    ccall((:nvmlDeviceGetMaxCustomerBoostClock, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlClockType_t, Ptr{Cuint}), device, clockType, clockMHz)
end

@checked function nvmlDeviceGetSupportedMemoryClocks(device, count, clocksMHz)
    initialize_context()
    ccall((:nvmlDeviceGetSupportedMemoryClocks, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}, Ptr{Cuint}), device, count, clocksMHz)
end

@checked function nvmlDeviceGetSupportedGraphicsClocks(device, memoryClockMHz, count,
                                                       clocksMHz)
    initialize_context()
    ccall((:nvmlDeviceGetSupportedGraphicsClocks, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{Cuint}, Ptr{Cuint}), device, memoryClockMHz, count,
          clocksMHz)
end

@checked function nvmlDeviceGetAutoBoostedClocksEnabled(device, isEnabled, defaultIsEnabled)
    initialize_context()
    ccall((:nvmlDeviceGetAutoBoostedClocksEnabled, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlEnableState_t}, Ptr{nvmlEnableState_t}), device, isEnabled,
          defaultIsEnabled)
end

@checked function nvmlDeviceSetAutoBoostedClocksEnabled(device, enabled)
    initialize_context()
    ccall((:nvmlDeviceSetAutoBoostedClocksEnabled, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlEnableState_t), device, enabled)
end

@checked function nvmlDeviceSetDefaultAutoBoostedClocksEnabled(device, enabled, flags)
    initialize_context()
    ccall((:nvmlDeviceSetDefaultAutoBoostedClocksEnabled, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlEnableState_t, Cuint), device, enabled, flags)
end

@checked function nvmlDeviceGetFanSpeed(device, speed)
    initialize_context()
    ccall((:nvmlDeviceGetFanSpeed, libnvml()), nvmlReturn_t, (nvmlDevice_t, Ptr{Cuint}),
          device, speed)
end

@checked function nvmlDeviceGetFanSpeed_v2(device, fan, speed)
    initialize_context()
    ccall((:nvmlDeviceGetFanSpeed_v2, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{Cuint}), device, fan, speed)
end

@checked function nvmlDeviceGetTargetFanSpeed(device, fan, targetSpeed)
    initialize_context()
    ccall((:nvmlDeviceGetTargetFanSpeed, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{Cuint}), device, fan, targetSpeed)
end

@checked function nvmlDeviceSetDefaultFanSpeed_v2(device, fan)
    initialize_context()
    ccall((:nvmlDeviceSetDefaultFanSpeed_v2, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint), device, fan)
end

@checked function nvmlDeviceGetMinMaxFanSpeed(device, minSpeed, maxSpeed)
    initialize_context()
    ccall((:nvmlDeviceGetMinMaxFanSpeed, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}, Ptr{Cuint}), device, minSpeed, maxSpeed)
end

@checked function nvmlDeviceGetNumFans(device, numFans)
    initialize_context()
    ccall((:nvmlDeviceGetNumFans, libnvml()), nvmlReturn_t, (nvmlDevice_t, Ptr{Cuint}),
          device, numFans)
end

@checked function nvmlDeviceGetTemperature(device, sensorType, temp)
    initialize_context()
    ccall((:nvmlDeviceGetTemperature, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlTemperatureSensors_t, Ptr{Cuint}), device, sensorType, temp)
end

@checked function nvmlDeviceGetTemperatureThreshold(device, thresholdType, temp)
    initialize_context()
    ccall((:nvmlDeviceGetTemperatureThreshold, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlTemperatureThresholds_t, Ptr{Cuint}), device, thresholdType,
          temp)
end

@checked function nvmlDeviceSetTemperatureThreshold(device, thresholdType, temp)
    initialize_context()
    ccall((:nvmlDeviceSetTemperatureThreshold, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlTemperatureThresholds_t, Ptr{Cint}), device, thresholdType,
          temp)
end

@checked function nvmlDeviceGetThermalSettings(device, sensorIndex, pThermalSettings)
    initialize_context()
    ccall((:nvmlDeviceGetThermalSettings, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{nvmlGpuThermalSettings_t}), device, sensorIndex,
          pThermalSettings)
end

@checked function nvmlDeviceGetPerformanceState(device, pState)
    initialize_context()
    ccall((:nvmlDeviceGetPerformanceState, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlPstates_t}), device, pState)
end

@checked function nvmlDeviceGetCurrentClocksThrottleReasons(device, clocksThrottleReasons)
    initialize_context()
    ccall((:nvmlDeviceGetCurrentClocksThrottleReasons, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Culonglong}), device, clocksThrottleReasons)
end

@checked function nvmlDeviceGetSupportedClocksThrottleReasons(device,
                                                              supportedClocksThrottleReasons)
    initialize_context()
    ccall((:nvmlDeviceGetSupportedClocksThrottleReasons, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Culonglong}), device, supportedClocksThrottleReasons)
end

@checked function nvmlDeviceGetPowerState(device, pState)
    initialize_context()
    ccall((:nvmlDeviceGetPowerState, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlPstates_t}), device, pState)
end

@checked function nvmlDeviceGetPowerManagementMode(device, mode)
    initialize_context()
    ccall((:nvmlDeviceGetPowerManagementMode, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlEnableState_t}), device, mode)
end

@checked function nvmlDeviceGetPowerManagementLimit(device, limit)
    initialize_context()
    ccall((:nvmlDeviceGetPowerManagementLimit, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}), device, limit)
end

@checked function nvmlDeviceGetPowerManagementLimitConstraints(device, minLimit, maxLimit)
    initialize_context()
    ccall((:nvmlDeviceGetPowerManagementLimitConstraints, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}, Ptr{Cuint}), device, minLimit, maxLimit)
end

@checked function nvmlDeviceGetPowerManagementDefaultLimit(device, defaultLimit)
    initialize_context()
    ccall((:nvmlDeviceGetPowerManagementDefaultLimit, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}), device, defaultLimit)
end

@checked function nvmlDeviceGetPowerUsage(device, power)
    initialize_context()
    ccall((:nvmlDeviceGetPowerUsage, libnvml()), nvmlReturn_t, (nvmlDevice_t, Ptr{Cuint}),
          device, power)
end

@checked function nvmlDeviceGetPowerMode(device, powerModeId)
    initialize_context()
    ccall((:nvmlDeviceGetPowerMode, libnvml()), nvmlReturn_t, (nvmlDevice_t, Ptr{Cuint}),
          device, powerModeId)
end

@checked function nvmlDeviceGetSupportedPowerModes(device, supportedPowerModes)
    initialize_context()
    ccall((:nvmlDeviceGetSupportedPowerModes, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}), device, supportedPowerModes)
end

@checked function nvmlDeviceSetPowerMode(device, powerModeId)
    initialize_context()
    ccall((:nvmlDeviceSetPowerMode, libnvml()), nvmlReturn_t, (nvmlDevice_t, Cuint), device,
          powerModeId)
end

@checked function nvmlDeviceGetTotalEnergyConsumption(device, energy)
    initialize_context()
    ccall((:nvmlDeviceGetTotalEnergyConsumption, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Culonglong}), device, energy)
end

@checked function nvmlDeviceGetEnforcedPowerLimit(device, limit)
    initialize_context()
    ccall((:nvmlDeviceGetEnforcedPowerLimit, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}), device, limit)
end

@checked function nvmlDeviceGetGpuOperationMode(device, current, pending)
    initialize_context()
    ccall((:nvmlDeviceGetGpuOperationMode, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlGpuOperationMode_t}, Ptr{nvmlGpuOperationMode_t}), device,
          current, pending)
end

@checked function nvmlDeviceGetMemoryInfo(device, memory)
    initialize_context()
    ccall((:nvmlDeviceGetMemoryInfo, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlMemory_t}), device, memory)
end

@checked function nvmlDeviceGetMemoryInfo_v2(device, memory)
    initialize_context()
    ccall((:nvmlDeviceGetMemoryInfo_v2, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlMemory_v2_t}), device, memory)
end

@checked function nvmlDeviceGetComputeMode(device, mode)
    initialize_context()
    ccall((:nvmlDeviceGetComputeMode, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlComputeMode_t}), device, mode)
end

@checked function nvmlDeviceGetCudaComputeCapability(device, major, minor)
    initialize_context()
    ccall((:nvmlDeviceGetCudaComputeCapability, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cint}, Ptr{Cint}), device, major, minor)
end

@checked function nvmlDeviceGetEccMode(device, current, pending)
    initialize_context()
    ccall((:nvmlDeviceGetEccMode, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlEnableState_t}, Ptr{nvmlEnableState_t}), device, current,
          pending)
end

@checked function nvmlDeviceGetDefaultEccMode(device, defaultMode)
    initialize_context()
    ccall((:nvmlDeviceGetDefaultEccMode, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlEnableState_t}), device, defaultMode)
end

@checked function nvmlDeviceGetBoardId(device, boardId)
    initialize_context()
    ccall((:nvmlDeviceGetBoardId, libnvml()), nvmlReturn_t, (nvmlDevice_t, Ptr{Cuint}),
          device, boardId)
end

@checked function nvmlDeviceGetMultiGpuBoard(device, multiGpuBool)
    initialize_context()
    ccall((:nvmlDeviceGetMultiGpuBoard, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}), device, multiGpuBool)
end

@checked function nvmlDeviceGetTotalEccErrors(device, errorType, counterType, eccCounts)
    initialize_context()
    ccall((:nvmlDeviceGetTotalEccErrors, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlMemoryErrorType_t, nvmlEccCounterType_t, Ptr{Culonglong}),
          device, errorType, counterType, eccCounts)
end

@checked function nvmlDeviceGetDetailedEccErrors(device, errorType, counterType, eccCounts)
    initialize_context()
    ccall((:nvmlDeviceGetDetailedEccErrors, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlMemoryErrorType_t, nvmlEccCounterType_t,
           Ptr{nvmlEccErrorCounts_t}), device, errorType, counterType, eccCounts)
end

@checked function nvmlDeviceGetMemoryErrorCounter(device, errorType, counterType,
                                                  locationType, count)
    initialize_context()
    ccall((:nvmlDeviceGetMemoryErrorCounter, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlMemoryErrorType_t, nvmlEccCounterType_t, nvmlMemoryLocation_t,
           Ptr{Culonglong}), device, errorType, counterType, locationType, count)
end

@checked function nvmlDeviceGetUtilizationRates(device, utilization)
    initialize_context()
    ccall((:nvmlDeviceGetUtilizationRates, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlUtilization_t}), device, utilization)
end

@checked function nvmlDeviceGetEncoderUtilization(device, utilization, samplingPeriodUs)
    initialize_context()
    ccall((:nvmlDeviceGetEncoderUtilization, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}, Ptr{Cuint}), device, utilization, samplingPeriodUs)
end

@checked function nvmlDeviceGetEncoderCapacity(device, encoderQueryType, encoderCapacity)
    initialize_context()
    ccall((:nvmlDeviceGetEncoderCapacity, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlEncoderType_t, Ptr{Cuint}), device, encoderQueryType,
          encoderCapacity)
end

@checked function nvmlDeviceGetEncoderStats(device, sessionCount, averageFps,
                                            averageLatency)
    initialize_context()
    ccall((:nvmlDeviceGetEncoderStats, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}, Ptr{Cuint}, Ptr{Cuint}), device, sessionCount,
          averageFps, averageLatency)
end

@checked function nvmlDeviceGetEncoderSessions(device, sessionCount, sessionInfos)
    initialize_context()
    ccall((:nvmlDeviceGetEncoderSessions, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}, Ptr{nvmlEncoderSessionInfo_t}), device, sessionCount,
          sessionInfos)
end

@checked function nvmlDeviceGetDecoderUtilization(device, utilization, samplingPeriodUs)
    initialize_context()
    ccall((:nvmlDeviceGetDecoderUtilization, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}, Ptr{Cuint}), device, utilization, samplingPeriodUs)
end

@checked function nvmlDeviceGetFBCStats(device, fbcStats)
    initialize_context()
    ccall((:nvmlDeviceGetFBCStats, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlFBCStats_t}), device, fbcStats)
end

@checked function nvmlDeviceGetFBCSessions(device, sessionCount, sessionInfo)
    initialize_context()
    ccall((:nvmlDeviceGetFBCSessions, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}, Ptr{nvmlFBCSessionInfo_t}), device, sessionCount,
          sessionInfo)
end

@checked function nvmlDeviceGetDriverModel(device, current, pending)
    initialize_context()
    ccall((:nvmlDeviceGetDriverModel, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlDriverModel_t}, Ptr{nvmlDriverModel_t}), device, current,
          pending)
end

@checked function nvmlDeviceGetVbiosVersion(device, version, length)
    initialize_context()
    ccall((:nvmlDeviceGetVbiosVersion, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cstring, Cuint), device, version, length)
end

@checked function nvmlDeviceGetBridgeChipInfo(device, bridgeHierarchy)
    initialize_context()
    ccall((:nvmlDeviceGetBridgeChipInfo, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlBridgeChipHierarchy_t}), device, bridgeHierarchy)
end

@checked function nvmlDeviceOnSameBoard(device1, device2, onSameBoard)
    initialize_context()
    ccall((:nvmlDeviceOnSameBoard, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlDevice_t, Ptr{Cint}), device1, device2, onSameBoard)
end

@checked function nvmlDeviceGetAPIRestriction(device, apiType, isRestricted)
    initialize_context()
    ccall((:nvmlDeviceGetAPIRestriction, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlRestrictedAPI_t, Ptr{nvmlEnableState_t}), device, apiType,
          isRestricted)
end

@checked function nvmlDeviceGetSamples(device, type, lastSeenTimeStamp, sampleValType,
                                       sampleCount, samples)
    initialize_context()
    ccall((:nvmlDeviceGetSamples, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlSamplingType_t, Culonglong, Ptr{nvmlValueType_t}, Ptr{Cuint},
           Ptr{nvmlSample_t}), device, type, lastSeenTimeStamp, sampleValType, sampleCount,
          samples)
end

@checked function nvmlDeviceGetBAR1MemoryInfo(device, bar1Memory)
    initialize_context()
    ccall((:nvmlDeviceGetBAR1MemoryInfo, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlBAR1Memory_t}), device, bar1Memory)
end

@checked function nvmlDeviceGetViolationStatus(device, perfPolicyType, violTime)
    initialize_context()
    ccall((:nvmlDeviceGetViolationStatus, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlPerfPolicyType_t, Ptr{nvmlViolationTime_t}), device,
          perfPolicyType, violTime)
end

@checked function nvmlDeviceGetIrqNum(device, irqNum)
    initialize_context()
    ccall((:nvmlDeviceGetIrqNum, libnvml()), nvmlReturn_t, (nvmlDevice_t, Ptr{Cuint}),
          device, irqNum)
end

@checked function nvmlDeviceGetNumGpuCores(device, numCores)
    initialize_context()
    ccall((:nvmlDeviceGetNumGpuCores, libnvml()), nvmlReturn_t, (nvmlDevice_t, Ptr{Cuint}),
          device, numCores)
end

@checked function nvmlDeviceGetPowerSource(device, powerSource)
    initialize_context()
    ccall((:nvmlDeviceGetPowerSource, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlPowerSource_t}), device, powerSource)
end

@checked function nvmlDeviceGetMemoryBusWidth(device, busWidth)
    initialize_context()
    ccall((:nvmlDeviceGetMemoryBusWidth, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}), device, busWidth)
end

@checked function nvmlDeviceGetPcieLinkMaxSpeed(device, maxSpeed)
    initialize_context()
    ccall((:nvmlDeviceGetPcieLinkMaxSpeed, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}), device, maxSpeed)
end

@checked function nvmlDeviceGetPcieSpeed(device, pcieSpeed)
    initialize_context()
    ccall((:nvmlDeviceGetPcieSpeed, libnvml()), nvmlReturn_t, (nvmlDevice_t, Ptr{Cuint}),
          device, pcieSpeed)
end

@checked function nvmlDeviceGetAdaptiveClockInfoStatus(device, adaptiveClockStatus)
    initialize_context()
    ccall((:nvmlDeviceGetAdaptiveClockInfoStatus, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}), device, adaptiveClockStatus)
end

@checked function nvmlDeviceGetAccountingMode(device, mode)
    initialize_context()
    ccall((:nvmlDeviceGetAccountingMode, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlEnableState_t}), device, mode)
end

@checked function nvmlDeviceGetAccountingStats(device, pid, stats)
    initialize_context()
    ccall((:nvmlDeviceGetAccountingStats, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{nvmlAccountingStats_t}), device, pid, stats)
end

@checked function nvmlDeviceGetAccountingPids(device, count, pids)
    initialize_context()
    ccall((:nvmlDeviceGetAccountingPids, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}, Ptr{Cuint}), device, count, pids)
end

@checked function nvmlDeviceGetAccountingBufferSize(device, bufferSize)
    initialize_context()
    ccall((:nvmlDeviceGetAccountingBufferSize, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}), device, bufferSize)
end

@checked function nvmlDeviceGetRetiredPages(device, cause, pageCount, addresses)
    initialize_context()
    ccall((:nvmlDeviceGetRetiredPages, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlPageRetirementCause_t, Ptr{Cuint}, Ptr{Culonglong}), device,
          cause, pageCount, addresses)
end

@checked function nvmlDeviceGetRetiredPages_v2(device, cause, pageCount, addresses,
                                               timestamps)
    initialize_context()
    ccall((:nvmlDeviceGetRetiredPages_v2, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlPageRetirementCause_t, Ptr{Cuint}, Ptr{Culonglong},
           Ptr{Culonglong}), device, cause, pageCount, addresses, timestamps)
end

@checked function nvmlDeviceGetRetiredPagesPendingStatus(device, isPending)
    initialize_context()
    ccall((:nvmlDeviceGetRetiredPagesPendingStatus, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlEnableState_t}), device, isPending)
end

@checked function nvmlDeviceGetRemappedRows(device, corrRows, uncRows, isPending,
                                            failureOccurred)
    initialize_context()
    ccall((:nvmlDeviceGetRemappedRows, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}, Ptr{Cuint}, Ptr{Cuint}, Ptr{Cuint}), device, corrRows,
          uncRows, isPending, failureOccurred)
end

@checked function nvmlDeviceGetRowRemapperHistogram(device, values)
    initialize_context()
    ccall((:nvmlDeviceGetRowRemapperHistogram, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlRowRemapperHistogramValues_t}), device, values)
end

@checked function nvmlDeviceGetArchitecture(device, arch)
    initialize_context()
    ccall((:nvmlDeviceGetArchitecture, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlDeviceArchitecture_t}), device, arch)
end

@checked function nvmlUnitSetLedState(unit, color)
    initialize_context()
    ccall((:nvmlUnitSetLedState, libnvml()), nvmlReturn_t, (nvmlUnit_t, nvmlLedColor_t),
          unit, color)
end

@checked function nvmlDeviceSetPersistenceMode(device, mode)
    initialize_context()
    ccall((:nvmlDeviceSetPersistenceMode, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlEnableState_t), device, mode)
end

@checked function nvmlDeviceSetComputeMode(device, mode)
    initialize_context()
    ccall((:nvmlDeviceSetComputeMode, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlComputeMode_t), device, mode)
end

@checked function nvmlDeviceSetEccMode(device, ecc)
    initialize_context()
    ccall((:nvmlDeviceSetEccMode, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlEnableState_t), device, ecc)
end

@checked function nvmlDeviceClearEccErrorCounts(device, counterType)
    initialize_context()
    ccall((:nvmlDeviceClearEccErrorCounts, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlEccCounterType_t), device, counterType)
end

@checked function nvmlDeviceSetDriverModel(device, driverModel, flags)
    initialize_context()
    ccall((:nvmlDeviceSetDriverModel, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlDriverModel_t, Cuint), device, driverModel, flags)
end

@cenum nvmlClockLimitId_enum::UInt32 begin
    NVML_CLOCK_LIMIT_ID_RANGE_START = 0x00000000ffffff00
    NVML_CLOCK_LIMIT_ID_TDP = 0x00000000ffffff01
    NVML_CLOCK_LIMIT_ID_UNLIMITED = 0x00000000ffffff02
end

const nvmlClockLimitId_t = nvmlClockLimitId_enum

@checked function nvmlDeviceSetGpuLockedClocks(device, minGpuClockMHz, maxGpuClockMHz)
    initialize_context()
    ccall((:nvmlDeviceSetGpuLockedClocks, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Cuint), device, minGpuClockMHz, maxGpuClockMHz)
end

@checked function nvmlDeviceResetGpuLockedClocks(device)
    initialize_context()
    ccall((:nvmlDeviceResetGpuLockedClocks, libnvml()), nvmlReturn_t, (nvmlDevice_t,),
          device)
end

@checked function nvmlDeviceSetMemoryLockedClocks(device, minMemClockMHz, maxMemClockMHz)
    initialize_context()
    ccall((:nvmlDeviceSetMemoryLockedClocks, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Cuint), device, minMemClockMHz, maxMemClockMHz)
end

@checked function nvmlDeviceResetMemoryLockedClocks(device)
    initialize_context()
    ccall((:nvmlDeviceResetMemoryLockedClocks, libnvml()), nvmlReturn_t, (nvmlDevice_t,),
          device)
end

@checked function nvmlDeviceSetApplicationsClocks(device, memClockMHz, graphicsClockMHz)
    initialize_context()
    ccall((:nvmlDeviceSetApplicationsClocks, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Cuint), device, memClockMHz, graphicsClockMHz)
end

@checked function nvmlDeviceGetClkMonStatus(device, status)
    initialize_context()
    ccall((:nvmlDeviceGetClkMonStatus, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlClkMonStatus_t}), device, status)
end

@checked function nvmlDeviceSetPowerManagementLimit(device, limit)
    initialize_context()
    ccall((:nvmlDeviceSetPowerManagementLimit, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint), device, limit)
end

@checked function nvmlDeviceSetGpuOperationMode(device, mode)
    initialize_context()
    ccall((:nvmlDeviceSetGpuOperationMode, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlGpuOperationMode_t), device, mode)
end

@checked function nvmlDeviceSetAPIRestriction(device, apiType, isRestricted)
    initialize_context()
    ccall((:nvmlDeviceSetAPIRestriction, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlRestrictedAPI_t, nvmlEnableState_t), device, apiType,
          isRestricted)
end

@checked function nvmlDeviceSetAccountingMode(device, mode)
    initialize_context()
    ccall((:nvmlDeviceSetAccountingMode, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlEnableState_t), device, mode)
end

@checked function nvmlDeviceClearAccountingPids(device)
    initialize_context()
    ccall((:nvmlDeviceClearAccountingPids, libnvml()), nvmlReturn_t, (nvmlDevice_t,),
          device)
end

@checked function nvmlDeviceGetNvLinkState(device, link, isActive)
    initialize_context()
    ccall((:nvmlDeviceGetNvLinkState, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{nvmlEnableState_t}), device, link, isActive)
end

@checked function nvmlDeviceGetNvLinkVersion(device, link, version)
    initialize_context()
    ccall((:nvmlDeviceGetNvLinkVersion, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{Cuint}), device, link, version)
end

@checked function nvmlDeviceGetNvLinkCapability(device, link, capability, capResult)
    initialize_context()
    ccall((:nvmlDeviceGetNvLinkCapability, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, nvmlNvLinkCapability_t, Ptr{Cuint}), device, link,
          capability, capResult)
end

@checked function nvmlDeviceGetNvLinkErrorCounter(device, link, counter, counterValue)
    initialize_context()
    ccall((:nvmlDeviceGetNvLinkErrorCounter, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, nvmlNvLinkErrorCounter_t, Ptr{Culonglong}), device, link,
          counter, counterValue)
end

@checked function nvmlDeviceResetNvLinkErrorCounters(device, link)
    initialize_context()
    ccall((:nvmlDeviceResetNvLinkErrorCounters, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint), device, link)
end

@checked function nvmlDeviceSetNvLinkUtilizationControl(device, link, counter, control,
                                                        reset)
    initialize_context()
    ccall((:nvmlDeviceSetNvLinkUtilizationControl, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Cuint, Ptr{nvmlNvLinkUtilizationControl_t}, Cuint), device,
          link, counter, control, reset)
end

@checked function nvmlDeviceGetNvLinkUtilizationControl(device, link, counter, control)
    initialize_context()
    ccall((:nvmlDeviceGetNvLinkUtilizationControl, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Cuint, Ptr{nvmlNvLinkUtilizationControl_t}), device, link,
          counter, control)
end

@checked function nvmlDeviceGetNvLinkUtilizationCounter(device, link, counter, rxcounter,
                                                        txcounter)
    initialize_context()
    ccall((:nvmlDeviceGetNvLinkUtilizationCounter, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Cuint, Ptr{Culonglong}, Ptr{Culonglong}), device, link,
          counter, rxcounter, txcounter)
end

@checked function nvmlDeviceFreezeNvLinkUtilizationCounter(device, link, counter, freeze)
    initialize_context()
    ccall((:nvmlDeviceFreezeNvLinkUtilizationCounter, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Cuint, nvmlEnableState_t), device, link, counter, freeze)
end

@checked function nvmlDeviceResetNvLinkUtilizationCounter(device, link, counter)
    initialize_context()
    ccall((:nvmlDeviceResetNvLinkUtilizationCounter, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Cuint), device, link, counter)
end

@checked function nvmlDeviceGetNvLinkRemoteDeviceType(device, link, pNvLinkDeviceType)
    initialize_context()
    ccall((:nvmlDeviceGetNvLinkRemoteDeviceType, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{nvmlIntNvLinkDeviceType_t}), device, link,
          pNvLinkDeviceType)
end

@checked function nvmlEventSetCreate(set)
    initialize_context()
    ccall((:nvmlEventSetCreate, libnvml()), nvmlReturn_t, (Ptr{nvmlEventSet_t},), set)
end

@checked function nvmlDeviceRegisterEvents(device, eventTypes, set)
    initialize_context()
    ccall((:nvmlDeviceRegisterEvents, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Culonglong, nvmlEventSet_t), device, eventTypes, set)
end

@checked function nvmlDeviceGetSupportedEventTypes(device, eventTypes)
    initialize_context()
    ccall((:nvmlDeviceGetSupportedEventTypes, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Culonglong}), device, eventTypes)
end

@checked function nvmlEventSetFree(set)
    initialize_context()
    ccall((:nvmlEventSetFree, libnvml()), nvmlReturn_t, (nvmlEventSet_t,), set)
end

@checked function nvmlDeviceModifyDrainState(pciInfo, newState)
    initialize_context()
    ccall((:nvmlDeviceModifyDrainState, libnvml()), nvmlReturn_t,
          (Ptr{nvmlPciInfo_t}, nvmlEnableState_t), pciInfo, newState)
end

@checked function nvmlDeviceQueryDrainState(pciInfo, currentState)
    initialize_context()
    ccall((:nvmlDeviceQueryDrainState, libnvml()), nvmlReturn_t,
          (Ptr{nvmlPciInfo_t}, Ptr{nvmlEnableState_t}), pciInfo, currentState)
end

@checked function nvmlDeviceDiscoverGpus(pciInfo)
    initialize_context()
    ccall((:nvmlDeviceDiscoverGpus, libnvml()), nvmlReturn_t, (Ptr{nvmlPciInfo_t},),
          pciInfo)
end

@checked function nvmlDeviceGetFieldValues(device, valuesCount, values)
    initialize_context()
    ccall((:nvmlDeviceGetFieldValues, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cint, Ptr{nvmlFieldValue_t}), device, valuesCount, values)
end

@checked function nvmlDeviceGetVirtualizationMode(device, pVirtualMode)
    initialize_context()
    ccall((:nvmlDeviceGetVirtualizationMode, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlGpuVirtualizationMode_t}), device, pVirtualMode)
end

@checked function nvmlDeviceGetHostVgpuMode(device, pHostVgpuMode)
    initialize_context()
    ccall((:nvmlDeviceGetHostVgpuMode, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlHostVgpuMode_t}), device, pHostVgpuMode)
end

@checked function nvmlDeviceSetVirtualizationMode(device, virtualMode)
    initialize_context()
    ccall((:nvmlDeviceSetVirtualizationMode, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlGpuVirtualizationMode_t), device, virtualMode)
end

@checked function nvmlDeviceGetProcessUtilization(device, utilization, processSamplesCount,
                                                  lastSeenTimeStamp)
    initialize_context()
    ccall((:nvmlDeviceGetProcessUtilization, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlProcessUtilizationSample_t}, Ptr{Cuint}, Culonglong),
          device, utilization, processSamplesCount, lastSeenTimeStamp)
end

@checked function nvmlDeviceGetGspFirmwareVersion(device, version)
    initialize_context()
    ccall((:nvmlDeviceGetGspFirmwareVersion, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cstring), device, version)
end

@checked function nvmlDeviceGetGspFirmwareMode(device, isEnabled, defaultMode)
    initialize_context()
    ccall((:nvmlDeviceGetGspFirmwareMode, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}, Ptr{Cuint}), device, isEnabled, defaultMode)
end

@checked function nvmlDeviceGetSupportedVgpus(device, vgpuCount, vgpuTypeIds)
    initialize_context()
    ccall((:nvmlDeviceGetSupportedVgpus, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}, Ptr{nvmlVgpuTypeId_t}), device, vgpuCount, vgpuTypeIds)
end

@checked function nvmlDeviceGetCreatableVgpus(device, vgpuCount, vgpuTypeIds)
    initialize_context()
    ccall((:nvmlDeviceGetCreatableVgpus, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}, Ptr{nvmlVgpuTypeId_t}), device, vgpuCount, vgpuTypeIds)
end

@checked function nvmlVgpuTypeGetClass(vgpuTypeId, vgpuTypeClass, size)
    initialize_context()
    ccall((:nvmlVgpuTypeGetClass, libnvml()), nvmlReturn_t,
          (nvmlVgpuTypeId_t, Cstring, Ptr{Cuint}), vgpuTypeId, vgpuTypeClass, size)
end

@checked function nvmlVgpuTypeGetName(vgpuTypeId, vgpuTypeName, size)
    initialize_context()
    ccall((:nvmlVgpuTypeGetName, libnvml()), nvmlReturn_t,
          (nvmlVgpuTypeId_t, Cstring, Ptr{Cuint}), vgpuTypeId, vgpuTypeName, size)
end

@checked function nvmlVgpuTypeGetGpuInstanceProfileId(vgpuTypeId, gpuInstanceProfileId)
    initialize_context()
    ccall((:nvmlVgpuTypeGetGpuInstanceProfileId, libnvml()), nvmlReturn_t,
          (nvmlVgpuTypeId_t, Ptr{Cuint}), vgpuTypeId, gpuInstanceProfileId)
end

@checked function nvmlVgpuTypeGetDeviceID(vgpuTypeId, deviceID, subsystemID)
    initialize_context()
    ccall((:nvmlVgpuTypeGetDeviceID, libnvml()), nvmlReturn_t,
          (nvmlVgpuTypeId_t, Ptr{Culonglong}, Ptr{Culonglong}), vgpuTypeId, deviceID,
          subsystemID)
end

@checked function nvmlVgpuTypeGetFramebufferSize(vgpuTypeId, fbSize)
    initialize_context()
    ccall((:nvmlVgpuTypeGetFramebufferSize, libnvml()), nvmlReturn_t,
          (nvmlVgpuTypeId_t, Ptr{Culonglong}), vgpuTypeId, fbSize)
end

@checked function nvmlVgpuTypeGetNumDisplayHeads(vgpuTypeId, numDisplayHeads)
    initialize_context()
    ccall((:nvmlVgpuTypeGetNumDisplayHeads, libnvml()), nvmlReturn_t,
          (nvmlVgpuTypeId_t, Ptr{Cuint}), vgpuTypeId, numDisplayHeads)
end

@checked function nvmlVgpuTypeGetResolution(vgpuTypeId, displayIndex, xdim, ydim)
    initialize_context()
    ccall((:nvmlVgpuTypeGetResolution, libnvml()), nvmlReturn_t,
          (nvmlVgpuTypeId_t, Cuint, Ptr{Cuint}, Ptr{Cuint}), vgpuTypeId, displayIndex, xdim,
          ydim)
end

@checked function nvmlVgpuTypeGetLicense(vgpuTypeId, vgpuTypeLicenseString, size)
    initialize_context()
    ccall((:nvmlVgpuTypeGetLicense, libnvml()), nvmlReturn_t,
          (nvmlVgpuTypeId_t, Cstring, Cuint), vgpuTypeId, vgpuTypeLicenseString, size)
end

@checked function nvmlVgpuTypeGetFrameRateLimit(vgpuTypeId, frameRateLimit)
    initialize_context()
    ccall((:nvmlVgpuTypeGetFrameRateLimit, libnvml()), nvmlReturn_t,
          (nvmlVgpuTypeId_t, Ptr{Cuint}), vgpuTypeId, frameRateLimit)
end

@checked function nvmlVgpuTypeGetMaxInstances(device, vgpuTypeId, vgpuInstanceCount)
    initialize_context()
    ccall((:nvmlVgpuTypeGetMaxInstances, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlVgpuTypeId_t, Ptr{Cuint}), device, vgpuTypeId,
          vgpuInstanceCount)
end

@checked function nvmlVgpuTypeGetMaxInstancesPerVm(vgpuTypeId, vgpuInstanceCountPerVm)
    initialize_context()
    ccall((:nvmlVgpuTypeGetMaxInstancesPerVm, libnvml()), nvmlReturn_t,
          (nvmlVgpuTypeId_t, Ptr{Cuint}), vgpuTypeId, vgpuInstanceCountPerVm)
end

@checked function nvmlDeviceGetActiveVgpus(device, vgpuCount, vgpuInstances)
    initialize_context()
    ccall((:nvmlDeviceGetActiveVgpus, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}, Ptr{nvmlVgpuInstance_t}), device, vgpuCount,
          vgpuInstances)
end

@checked function nvmlVgpuInstanceGetVmID(vgpuInstance, vmId, size, vmIdType)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetVmID, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Cstring, Cuint, Ptr{nvmlVgpuVmIdType_t}), vgpuInstance, vmId,
          size, vmIdType)
end

@checked function nvmlVgpuInstanceGetUUID(vgpuInstance, uuid, size)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetUUID, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Cstring, Cuint), vgpuInstance, uuid, size)
end

@checked function nvmlVgpuInstanceGetVmDriverVersion(vgpuInstance, version, length)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetVmDriverVersion, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Cstring, Cuint), vgpuInstance, version, length)
end

@checked function nvmlVgpuInstanceGetFbUsage(vgpuInstance, fbUsage)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetFbUsage, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Ptr{Culonglong}), vgpuInstance, fbUsage)
end

@checked function nvmlVgpuInstanceGetLicenseStatus(vgpuInstance, licensed)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetLicenseStatus, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Ptr{Cuint}), vgpuInstance, licensed)
end

@checked function nvmlVgpuInstanceGetType(vgpuInstance, vgpuTypeId)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetType, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Ptr{nvmlVgpuTypeId_t}), vgpuInstance, vgpuTypeId)
end

@checked function nvmlVgpuInstanceGetFrameRateLimit(vgpuInstance, frameRateLimit)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetFrameRateLimit, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Ptr{Cuint}), vgpuInstance, frameRateLimit)
end

@checked function nvmlVgpuInstanceGetEccMode(vgpuInstance, eccMode)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetEccMode, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Ptr{nvmlEnableState_t}), vgpuInstance, eccMode)
end

@checked function nvmlVgpuInstanceGetEncoderCapacity(vgpuInstance, encoderCapacity)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetEncoderCapacity, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Ptr{Cuint}), vgpuInstance, encoderCapacity)
end

@checked function nvmlVgpuInstanceSetEncoderCapacity(vgpuInstance, encoderCapacity)
    initialize_context()
    ccall((:nvmlVgpuInstanceSetEncoderCapacity, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Cuint), vgpuInstance, encoderCapacity)
end

@checked function nvmlVgpuInstanceGetEncoderStats(vgpuInstance, sessionCount, averageFps,
                                                  averageLatency)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetEncoderStats, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Ptr{Cuint}, Ptr{Cuint}, Ptr{Cuint}), vgpuInstance,
          sessionCount, averageFps, averageLatency)
end

@checked function nvmlVgpuInstanceGetEncoderSessions(vgpuInstance, sessionCount,
                                                     sessionInfo)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetEncoderSessions, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Ptr{Cuint}, Ptr{nvmlEncoderSessionInfo_t}), vgpuInstance,
          sessionCount, sessionInfo)
end

@checked function nvmlVgpuInstanceGetFBCStats(vgpuInstance, fbcStats)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetFBCStats, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Ptr{nvmlFBCStats_t}), vgpuInstance, fbcStats)
end

@checked function nvmlVgpuInstanceGetFBCSessions(vgpuInstance, sessionCount, sessionInfo)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetFBCSessions, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Ptr{Cuint}, Ptr{nvmlFBCSessionInfo_t}), vgpuInstance,
          sessionCount, sessionInfo)
end

@checked function nvmlVgpuInstanceGetGpuInstanceId(vgpuInstance, gpuInstanceId)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetGpuInstanceId, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Ptr{Cuint}), vgpuInstance, gpuInstanceId)
end

@checked function nvmlVgpuInstanceGetGpuPciId(vgpuInstance, vgpuPciId, length)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetGpuPciId, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Cstring, Ptr{Cuint}), vgpuInstance, vgpuPciId, length)
end

@checked function nvmlVgpuTypeGetCapabilities(vgpuTypeId, capability, capResult)
    initialize_context()
    ccall((:nvmlVgpuTypeGetCapabilities, libnvml()), nvmlReturn_t,
          (nvmlVgpuTypeId_t, nvmlVgpuCapability_t, Ptr{Cuint}), vgpuTypeId, capability,
          capResult)
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
    ccall((:nvmlVgpuInstanceGetMetadata, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Ptr{nvmlVgpuMetadata_t}, Ptr{Cuint}), vgpuInstance,
          vgpuMetadata, bufferSize)
end

@checked function nvmlDeviceGetVgpuMetadata(device, pgpuMetadata, bufferSize)
    initialize_context()
    ccall((:nvmlDeviceGetVgpuMetadata, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlVgpuPgpuMetadata_t}, Ptr{Cuint}), device, pgpuMetadata,
          bufferSize)
end

@checked function nvmlGetVgpuCompatibility(vgpuMetadata, pgpuMetadata, compatibilityInfo)
    initialize_context()
    ccall((:nvmlGetVgpuCompatibility, libnvml()), nvmlReturn_t,
          (Ptr{nvmlVgpuMetadata_t}, Ptr{nvmlVgpuPgpuMetadata_t},
           Ptr{nvmlVgpuPgpuCompatibility_t}), vgpuMetadata, pgpuMetadata, compatibilityInfo)
end

@checked function nvmlDeviceGetPgpuMetadataString(device, pgpuMetadata, bufferSize)
    initialize_context()
    ccall((:nvmlDeviceGetPgpuMetadataString, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cstring, Ptr{Cuint}), device, pgpuMetadata, bufferSize)
end

@checked function nvmlGetVgpuVersion(supported, current)
    initialize_context()
    ccall((:nvmlGetVgpuVersion, libnvml()), nvmlReturn_t,
          (Ptr{nvmlVgpuVersion_t}, Ptr{nvmlVgpuVersion_t}), supported, current)
end

@checked function nvmlSetVgpuVersion(vgpuVersion)
    initialize_context()
    ccall((:nvmlSetVgpuVersion, libnvml()), nvmlReturn_t, (Ptr{nvmlVgpuVersion_t},),
          vgpuVersion)
end

@checked function nvmlDeviceGetVgpuUtilization(device, lastSeenTimeStamp, sampleValType,
                                               vgpuInstanceSamplesCount, utilizationSamples)
    initialize_context()
    ccall((:nvmlDeviceGetVgpuUtilization, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Culonglong, Ptr{nvmlValueType_t}, Ptr{Cuint},
           Ptr{nvmlVgpuInstanceUtilizationSample_t}), device, lastSeenTimeStamp,
          sampleValType, vgpuInstanceSamplesCount, utilizationSamples)
end

@checked function nvmlDeviceGetVgpuProcessUtilization(device, lastSeenTimeStamp,
                                                      vgpuProcessSamplesCount,
                                                      utilizationSamples)
    initialize_context()
    ccall((:nvmlDeviceGetVgpuProcessUtilization, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Culonglong, Ptr{Cuint}, Ptr{nvmlVgpuProcessUtilizationSample_t}),
          device, lastSeenTimeStamp, vgpuProcessSamplesCount, utilizationSamples)
end

@checked function nvmlVgpuInstanceGetAccountingMode(vgpuInstance, mode)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetAccountingMode, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Ptr{nvmlEnableState_t}), vgpuInstance, mode)
end

@checked function nvmlVgpuInstanceGetAccountingPids(vgpuInstance, count, pids)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetAccountingPids, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Ptr{Cuint}, Ptr{Cuint}), vgpuInstance, count, pids)
end

@checked function nvmlVgpuInstanceGetAccountingStats(vgpuInstance, pid, stats)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetAccountingStats, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t, Cuint, Ptr{nvmlAccountingStats_t}), vgpuInstance, pid, stats)
end

@checked function nvmlVgpuInstanceClearAccountingPids(vgpuInstance)
    initialize_context()
    ccall((:nvmlVgpuInstanceClearAccountingPids, libnvml()), nvmlReturn_t,
          (nvmlVgpuInstance_t,), vgpuInstance)
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
    ccall((:nvmlDeviceSetMigMode, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{nvmlReturn_t}), device, mode, activationStatus)
end

@checked function nvmlDeviceGetMigMode(device, currentMode, pendingMode)
    initialize_context()
    ccall((:nvmlDeviceGetMigMode, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}, Ptr{Cuint}), device, currentMode, pendingMode)
end

@checked function nvmlDeviceGetGpuInstanceProfileInfo(device, profile, info)
    initialize_context()
    ccall((:nvmlDeviceGetGpuInstanceProfileInfo, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{nvmlGpuInstanceProfileInfo_t}), device, profile, info)
end

@checked function nvmlDeviceGetGpuInstanceProfileInfoV(device, profile, info)
    initialize_context()
    ccall((:nvmlDeviceGetGpuInstanceProfileInfoV, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{nvmlGpuInstanceProfileInfo_v2_t}), device, profile,
          info)
end

@checked function nvmlDeviceGetGpuInstanceRemainingCapacity(device, profileId, count)
    initialize_context()
    ccall((:nvmlDeviceGetGpuInstanceRemainingCapacity, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{Cuint}), device, profileId, count)
end

@checked function nvmlDeviceCreateGpuInstance(device, profileId, gpuInstance)
    initialize_context()
    ccall((:nvmlDeviceCreateGpuInstance, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{nvmlGpuInstance_t}), device, profileId, gpuInstance)
end

@checked function nvmlDeviceCreateGpuInstanceWithPlacement(device, profileId, placement,
                                                           gpuInstance)
    initialize_context()
    ccall((:nvmlDeviceCreateGpuInstanceWithPlacement, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{nvmlGpuInstancePlacement_t}, Ptr{nvmlGpuInstance_t}),
          device, profileId, placement, gpuInstance)
end

@checked function nvmlGpuInstanceDestroy(gpuInstance)
    initialize_context()
    ccall((:nvmlGpuInstanceDestroy, libnvml()), nvmlReturn_t, (nvmlGpuInstance_t,),
          gpuInstance)
end

@checked function nvmlDeviceGetGpuInstances(device, profileId, gpuInstances, count)
    initialize_context()
    ccall((:nvmlDeviceGetGpuInstances, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{nvmlGpuInstance_t}, Ptr{Cuint}), device, profileId,
          gpuInstances, count)
end

@checked function nvmlDeviceGetGpuInstanceById(device, id, gpuInstance)
    initialize_context()
    ccall((:nvmlDeviceGetGpuInstanceById, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{nvmlGpuInstance_t}), device, id, gpuInstance)
end

@checked function nvmlGpuInstanceGetInfo(gpuInstance, info)
    initialize_context()
    ccall((:nvmlGpuInstanceGetInfo, libnvml()), nvmlReturn_t,
          (nvmlGpuInstance_t, Ptr{nvmlGpuInstanceInfo_t}), gpuInstance, info)
end

@checked function nvmlGpuInstanceGetComputeInstanceProfileInfo(gpuInstance, profile,
                                                               engProfile, info)
    initialize_context()
    ccall((:nvmlGpuInstanceGetComputeInstanceProfileInfo, libnvml()), nvmlReturn_t,
          (nvmlGpuInstance_t, Cuint, Cuint, Ptr{nvmlComputeInstanceProfileInfo_t}),
          gpuInstance, profile, engProfile, info)
end

@checked function nvmlGpuInstanceGetComputeInstanceProfileInfoV(gpuInstance, profile,
                                                                engProfile, info)
    initialize_context()
    ccall((:nvmlGpuInstanceGetComputeInstanceProfileInfoV, libnvml()), nvmlReturn_t,
          (nvmlGpuInstance_t, Cuint, Cuint, Ptr{nvmlComputeInstanceProfileInfo_v2_t}),
          gpuInstance, profile, engProfile, info)
end

@checked function nvmlGpuInstanceGetComputeInstanceRemainingCapacity(gpuInstance, profileId,
                                                                     count)
    initialize_context()
    ccall((:nvmlGpuInstanceGetComputeInstanceRemainingCapacity, libnvml()), nvmlReturn_t,
          (nvmlGpuInstance_t, Cuint, Ptr{Cuint}), gpuInstance, profileId, count)
end

@checked function nvmlGpuInstanceCreateComputeInstance(gpuInstance, profileId,
                                                       computeInstance)
    initialize_context()
    ccall((:nvmlGpuInstanceCreateComputeInstance, libnvml()), nvmlReturn_t,
          (nvmlGpuInstance_t, Cuint, Ptr{nvmlComputeInstance_t}), gpuInstance, profileId,
          computeInstance)
end

@checked function nvmlComputeInstanceDestroy(computeInstance)
    initialize_context()
    ccall((:nvmlComputeInstanceDestroy, libnvml()), nvmlReturn_t, (nvmlComputeInstance_t,),
          computeInstance)
end

@checked function nvmlGpuInstanceGetComputeInstances(gpuInstance, profileId,
                                                     computeInstances, count)
    initialize_context()
    ccall((:nvmlGpuInstanceGetComputeInstances, libnvml()), nvmlReturn_t,
          (nvmlGpuInstance_t, Cuint, Ptr{nvmlComputeInstance_t}, Ptr{Cuint}), gpuInstance,
          profileId, computeInstances, count)
end

@checked function nvmlGpuInstanceGetComputeInstanceById(gpuInstance, id, computeInstance)
    initialize_context()
    ccall((:nvmlGpuInstanceGetComputeInstanceById, libnvml()), nvmlReturn_t,
          (nvmlGpuInstance_t, Cuint, Ptr{nvmlComputeInstance_t}), gpuInstance, id,
          computeInstance)
end

@checked function nvmlDeviceIsMigDeviceHandle(device, isMigDevice)
    initialize_context()
    ccall((:nvmlDeviceIsMigDeviceHandle, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}), device, isMigDevice)
end

@checked function nvmlDeviceGetGpuInstanceId(device, id)
    initialize_context()
    ccall((:nvmlDeviceGetGpuInstanceId, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}), device, id)
end

@checked function nvmlDeviceGetComputeInstanceId(device, id)
    initialize_context()
    ccall((:nvmlDeviceGetComputeInstanceId, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}), device, id)
end

@checked function nvmlDeviceGetMaxMigDeviceCount(device, count)
    initialize_context()
    ccall((:nvmlDeviceGetMaxMigDeviceCount, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cuint}), device, count)
end

@checked function nvmlDeviceGetMigDeviceHandleByIndex(device, index, migDevice)
    initialize_context()
    ccall((:nvmlDeviceGetMigDeviceHandleByIndex, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Ptr{nvmlDevice_t}), device, index, migDevice)
end

@checked function nvmlDeviceGetDeviceHandleFromMigDeviceHandle(migDevice, device)
    initialize_context()
    ccall((:nvmlDeviceGetDeviceHandleFromMigDeviceHandle, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlDevice_t}), migDevice, device)
end

@checked function nvmlDeviceGetBusType(device, type)
    initialize_context()
    ccall((:nvmlDeviceGetBusType, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlBusType_t}), device, type)
end

@checked function nvmlDeviceGetDynamicPstatesInfo(device, pDynamicPstatesInfo)
    initialize_context()
    ccall((:nvmlDeviceGetDynamicPstatesInfo, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlGpuDynamicPstatesInfo_t}), device, pDynamicPstatesInfo)
end

@checked function nvmlDeviceSetFanSpeed_v2(device, fan, speed)
    initialize_context()
    ccall((:nvmlDeviceSetFanSpeed_v2, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Cuint, Cuint), device, fan, speed)
end

@checked function nvmlDeviceGetGpcClkVfOffset(device, offset)
    initialize_context()
    ccall((:nvmlDeviceGetGpcClkVfOffset, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cint}), device, offset)
end

@checked function nvmlDeviceSetGpcClkVfOffset(device, offset)
    initialize_context()
    ccall((:nvmlDeviceSetGpcClkVfOffset, libnvml()), nvmlReturn_t, (nvmlDevice_t, Cint),
          device, offset)
end

@checked function nvmlDeviceGetMemClkVfOffset(device, offset)
    initialize_context()
    ccall((:nvmlDeviceGetMemClkVfOffset, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cint}), device, offset)
end

@checked function nvmlDeviceSetMemClkVfOffset(device, offset)
    initialize_context()
    ccall((:nvmlDeviceSetMemClkVfOffset, libnvml()), nvmlReturn_t, (nvmlDevice_t, Cint),
          device, offset)
end

@checked function nvmlDeviceGetMinMaxClockOfPState(device, type, pstate, minClockMHz,
                                                   maxClockMHz)
    initialize_context()
    ccall((:nvmlDeviceGetMinMaxClockOfPState, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, nvmlClockType_t, nvmlPstates_t, Ptr{Cuint}, Ptr{Cuint}), device,
          type, pstate, minClockMHz, maxClockMHz)
end

@checked function nvmlDeviceGetSupportedPerformanceStates(device, pstates, size)
    initialize_context()
    ccall((:nvmlDeviceGetSupportedPerformanceStates, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlPstates_t}, Cuint), device, pstates, size)
end

@checked function nvmlDeviceGetGpcClkMinMaxVfOffset(device, minOffset, maxOffset)
    initialize_context()
    ccall((:nvmlDeviceGetGpcClkMinMaxVfOffset, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cint}, Ptr{Cint}), device, minOffset, maxOffset)
end

@checked function nvmlDeviceGetMemClkMinMaxVfOffset(device, minOffset, maxOffset)
    initialize_context()
    ccall((:nvmlDeviceGetMemClkMinMaxVfOffset, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{Cint}, Ptr{Cint}), device, minOffset, maxOffset)
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

struct var"##Ctag#495"
    shortName::Cstring
    longName::Cstring
    unit::Cstring
end
function Base.getproperty(x::Ptr{var"##Ctag#495"}, f::Symbol)
    f === :shortName && return Ptr{Cstring}(x + 0)
    f === :longName && return Ptr{Cstring}(x + 8)
    f === :unit && return Ptr{Cstring}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#495", f::Symbol)
    r = Ref{var"##Ctag#495"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#495"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#495"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct nvmlGpmMetric_t
    data::NTuple{40,UInt8}
end

function Base.getproperty(x::Ptr{nvmlGpmMetric_t}, f::Symbol)
    f === :metricId && return Ptr{Cuint}(x + 0)
    f === :nvmlReturn && return Ptr{nvmlReturn_t}(x + 4)
    f === :value && return Ptr{Cdouble}(x + 8)
    f === :metricInfo && return Ptr{var"##Ctag#495"}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::nvmlGpmMetric_t, f::Symbol)
    r = Ref{nvmlGpmMetric_t}(x)
    ptr = Base.unsafe_convert(Ptr{nvmlGpmMetric_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{nvmlGpmMetric_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
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
    ccall((:nvmlGpmMetricsGet, libnvml()), nvmlReturn_t, (Ptr{nvmlGpmMetricsGet_t},),
          metricsGet)
end

@checked function nvmlGpmSampleFree(gpmSample)
    initialize_context()
    ccall((:nvmlGpmSampleFree, libnvml()), nvmlReturn_t, (nvmlGpmSample_t,), gpmSample)
end

@checked function nvmlGpmSampleAlloc(gpmSample)
    initialize_context()
    ccall((:nvmlGpmSampleAlloc, libnvml()), nvmlReturn_t, (Ptr{nvmlGpmSample_t},),
          gpmSample)
end

@checked function nvmlGpmSampleGet(device, gpmSample)
    initialize_context()
    ccall((:nvmlGpmSampleGet, libnvml()), nvmlReturn_t, (nvmlDevice_t, nvmlGpmSample_t),
          device, gpmSample)
end

@checked function nvmlGpmQueryDeviceSupport(device, gpmSupport)
    initialize_context()
    ccall((:nvmlGpmQueryDeviceSupport, libnvml()), nvmlReturn_t,
          (nvmlDevice_t, Ptr{nvmlGpmSupport_t}), device, gpmSupport)
end

const NVML_API_VERSION = 11

const NVML_API_VERSION_STR = "11"

const nvmlBlacklistDeviceInfo_t = nvmlExcludedDeviceInfo_t

const nvmlGetBlacklistDeviceCount = nvmlGetExcludedDeviceCount

const nvmlGetBlacklistDeviceInfoByIndex = nvmlGetExcludedDeviceInfoByIndex

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

const nvmlEccBitType_t = nvmlMemoryErrorType_t

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

const NVML_POWER_MODE_ID_BALANCED = 0

const NVML_POWER_MODE_ID_MAX = 1

const NVML_POWER_SOURCE_AC = 0x00000000

const NVML_POWER_SOURCE_BATTERY = 0x00000001

const NVML_PCIE_LINK_MAX_SPEED_INVALID = 0x00000000

const NVML_PCIE_LINK_MAX_SPEED_2500MBPS = 0x00000001

const NVML_PCIE_LINK_MAX_SPEED_5000MBPS = 0x00000002

const NVML_PCIE_LINK_MAX_SPEED_8000MBPS = 0x00000003

const NVML_PCIE_LINK_MAX_SPEED_16000MBPS = 0x00000004

const NVML_PCIE_LINK_MAX_SPEED_32000MBPS = 0x00000005

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

const NVML_FI_MAX = 167

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

const nvmlClocksThrottleReasonUserDefinedClocks = nvmlClocksThrottleReasonApplicationsClocksSetting

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

const NVML_GPU_INSTANCE_PROFILE_COUNT = 0x08

const NVML_COMPUTE_INSTANCE_PROFILE_1_SLICE = 0x00

const NVML_COMPUTE_INSTANCE_PROFILE_2_SLICE = 0x01

const NVML_COMPUTE_INSTANCE_PROFILE_3_SLICE = 0x02

const NVML_COMPUTE_INSTANCE_PROFILE_4_SLICE = 0x03

const NVML_COMPUTE_INSTANCE_PROFILE_7_SLICE = 0x04

const NVML_COMPUTE_INSTANCE_PROFILE_8_SLICE = 0x05

const NVML_COMPUTE_INSTANCE_PROFILE_6_SLICE = 0x06

const NVML_COMPUTE_INSTANCE_PROFILE_COUNT = 0x07

const NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED = 0x00

const NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_COUNT = 0x01

const NVML_GPM_METRICS_GET_VERSION = 1

const NVML_GPM_SUPPORT_VERSION = 1
