# Automatically generated using Clang.jl

const NVML_API_VERSION = 11
const NVML_API_VERSION_STR = "11"

const NVML_VALUE_NOT_AVAILABLE = -1
const NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE = 32
const NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE = 16
const NVML_DEVICE_PCI_BUS_ID_LEGACY_FMT = "%04X:%02X:%02X.0"
const NVML_DEVICE_PCI_BUS_ID_FMT = "%08X:%02X:%02X.0"

# Skipping MacroDefinition: NVML_DEVICE_PCI_BUS_ID_FMT_ARGS ( pciInfo ) ( pciInfo ) -> domain , ( pciInfo ) -> bus , ( pciInfo ) -> device

const NVML_NVLINK_MAX_LINKS = 12

@cenum nvmlGpuLevel_enum::UInt32 begin
    NVML_TOPOLOGY_INTERNAL = 0
    NVML_TOPOLOGY_SINGLE = 10
    NVML_TOPOLOGY_MULTIPLE = 20
    NVML_TOPOLOGY_HOSTBRIDGE = 30
    NVML_TOPOLOGY_NODE = 40
    NVML_TOPOLOGY_SYSTEM = 50
end

const NVML_TOPOLOGY_CPU = NVML_TOPOLOGY_NODE
const NVML_MAX_PHYSICAL_BRIDGE = 128
const nvmlFlagDefault = 0x00
const nvmlFlagForce = 0x01

@cenum nvmlMemoryErrorType_enum::UInt32 begin
    NVML_MEMORY_ERROR_TYPE_CORRECTED = 0
    NVML_MEMORY_ERROR_TYPE_UNCORRECTED = 1
    NVML_MEMORY_ERROR_TYPE_COUNT = 2
end

const nvmlMemoryErrorType_t = nvmlMemoryErrorType_enum
const nvmlEccBitType_t = nvmlMemoryErrorType_t
const NVML_SINGLE_BIT_ECC = NVML_MEMORY_ERROR_TYPE_CORRECTED
const NVML_DOUBLE_BIT_ECC = NVML_MEMORY_ERROR_TYPE_UNCORRECTED
const NVML_GRID_LICENSE_BUFFER_SIZE = 128
const NVML_VGPU_NAME_BUFFER_SIZE = 64
const NVML_GRID_LICENSE_FEATURE_MAX_COUNT = 3
const INVALID_GPU_INSTANCE_PROFILE_ID = Float32(0x0fffffff)
const INVALID_GPU_INSTANCE_ID = Float32(0x0fffffff)

# Skipping MacroDefinition: NVML_VGPU_VIRTUALIZATION_CAP_MIGRATION 0 : 0

const NVML_VGPU_VIRTUALIZATION_CAP_MIGRATION_NO = 0x00
const NVML_VGPU_VIRTUALIZATION_CAP_MIGRATION_YES = 0x01

# Skipping MacroDefinition: NVML_VGPU_PGPU_VIRTUALIZATION_CAP_MIGRATION 0 : 0

const NVML_VGPU_PGPU_VIRTUALIZATION_CAP_MIGRATION_NO = 0x00
const NVML_VGPU_PGPU_VIRTUALIZATION_CAP_MIGRATION_YES = 0x01
const NVML_DEVICE_ARCH_KEPLER = 2
const NVML_DEVICE_ARCH_MAXWELL = 3
const NVML_DEVICE_ARCH_PASCAL = 4
const NVML_DEVICE_ARCH_VOLTA = 5
const NVML_DEVICE_ARCH_TURING = 6
const NVML_DEVICE_ARCH_AMPERE = 7
const NVML_DEVICE_ARCH_UNKNOWN = Float32(0x0fffffff)
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
const NVML_FI_MAX = 148
const nvmlEventTypeSingleBitEccError = Int64(0x0000000000000001)
const nvmlEventTypeDoubleBitEccError = Int64(0x0000000000000002)
const nvmlEventTypePState = Int64(0x0000000000000004)
const nvmlEventTypeXidCriticalError = Int64(0x0000000000000008)
const nvmlEventTypeClock = Int64(0x0000000000000010)
const nvmlEventTypePowerSourceChange = Int64(0x0000000000000080)
const nvmlEventMigConfigChange = Int64(0x0000000000000100)
const nvmlEventTypeNone = Int64(0x0000000000000000)
const nvmlEventTypeAll = ((((((nvmlEventTypeNone | nvmlEventTypeSingleBitEccError) | nvmlEventTypeDoubleBitEccError) | nvmlEventTypePState) | nvmlEventTypeClock) | nvmlEventTypeXidCriticalError) | nvmlEventTypePowerSourceChange) | nvmlEventMigConfigChange
const nvmlClocksThrottleReasonGpuIdle = Int64(0x0000000000000001)
const nvmlClocksThrottleReasonApplicationsClocksSetting = Int64(0x0000000000000002)
const nvmlClocksThrottleReasonUserDefinedClocks = nvmlClocksThrottleReasonApplicationsClocksSetting
const nvmlClocksThrottleReasonSwPowerCap = Int64(0x0000000000000004)
const nvmlClocksThrottleReasonHwSlowdown = Int64(0x0000000000000008)
const nvmlClocksThrottleReasonSyncBoost = Int64(0x0000000000000010)
const nvmlClocksThrottleReasonSwThermalSlowdown = Int64(0x0000000000000020)
const nvmlClocksThrottleReasonHwThermalSlowdown = Int64(0x0000000000000040)
const nvmlClocksThrottleReasonHwPowerBrakeSlowdown = Int64(0x0000000000000080)
const nvmlClocksThrottleReasonDisplayClockSetting = Int64(0x0000000000000100)
const nvmlClocksThrottleReasonNone = Int64(0x0000000000000000)

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

# Skipping MacroDefinition: NVML_CUDA_DRIVER_VERSION_MAJOR ( v ) ( ( v ) / 1000 )
# Skipping MacroDefinition: NVML_CUDA_DRIVER_VERSION_MINOR ( v ) ( ( ( v ) % 1000 ) / 10 )

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
const NVML_GPU_INSTANCE_PROFILE_COUNT = 0x06
const NVML_COMPUTE_INSTANCE_PROFILE_1_SLICE = 0x00
const NVML_COMPUTE_INSTANCE_PROFILE_2_SLICE = 0x01
const NVML_COMPUTE_INSTANCE_PROFILE_3_SLICE = 0x02
const NVML_COMPUTE_INSTANCE_PROFILE_4_SLICE = 0x03
const NVML_COMPUTE_INSTANCE_PROFILE_7_SLICE = 0x04
const NVML_COMPUTE_INSTANCE_PROFILE_8_SLICE = 0x05
const NVML_COMPUTE_INSTANCE_PROFILE_COUNT = 0x06
const NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED = 0x00
const NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_COUNT = 0x01
const nvmlDevice_st = Cvoid
const nvmlDevice_t = Ptr{nvmlDevice_st}

struct nvmlPciInfo_st
    busIdLegacy::NTuple{16, UInt8}
    domain::UInt32
    bus::UInt32
    device::UInt32
    pciDeviceId::UInt32
    pciSubSystemId::UInt32
    busId::NTuple{32, UInt8}
end

const nvmlPciInfo_t = nvmlPciInfo_st

struct nvmlEccErrorCounts_st
    l1Cache::Culonglong
    l2Cache::Culonglong
    deviceMemory::Culonglong
    registerFile::Culonglong
end

const nvmlEccErrorCounts_t = nvmlEccErrorCounts_st

struct nvmlUtilization_st
    gpu::UInt32
    memory::UInt32
end

const nvmlUtilization_t = nvmlUtilization_st

struct nvmlMemory_st
    total::Culonglong
    free::Culonglong
    used::Culonglong
end

const nvmlMemory_t = nvmlMemory_st

struct nvmlBAR1Memory_st
    bar1Total::Culonglong
    bar1Free::Culonglong
    bar1Used::Culonglong
end

const nvmlBAR1Memory_t = nvmlBAR1Memory_st

struct nvmlProcessInfo_st
    pid::UInt32
    usedGpuMemory::Culonglong
    gpuInstanceId::UInt32
    computeInstanceId::UInt32
end

const nvmlProcessInfo_t = nvmlProcessInfo_st

struct nvmlDeviceAttributes_st
    multiprocessorCount::UInt32
    sharedCopyEngineCount::UInt32
    sharedDecoderCount::UInt32
    sharedEncoderCount::UInt32
    sharedJpegCount::UInt32
    sharedOfaCount::UInt32
    gpuInstanceSliceCount::UInt32
    computeInstanceSliceCount::UInt32
    memorySizeMB::Culonglong
end

const nvmlDeviceAttributes_t = nvmlDeviceAttributes_st

struct nvmlRowRemapperHistogramValues_st
    max::UInt32
    high::UInt32
    partial::UInt32
    low::UInt32
    none::UInt32
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
    NVML_NVLINK_ERROR_COUNT = 4
end

const nvmlNvLinkErrorCounter_t = nvmlNvLinkErrorCounter_enum
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
    fwVersion::UInt32
end

const nvmlBridgeChipInfo_t = nvmlBridgeChipInfo_st

struct nvmlBridgeChipHierarchy_st
    bridgeCount::Cuchar
    bridgeChipInfo::NTuple{128, nvmlBridgeChipInfo_t}
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
    dVal::Cdouble
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
    NVML_BRAND_COUNT = 7
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
    NVML_ERROR_UNKNOWN = 999
end

const nvmlReturn_t = nvmlReturn_enum

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

@cenum nvmlGridLicenseFeatureCode_t::UInt32 begin
    NVML_GRID_LICENSE_FEATURE_CODE_VGPU = 1
    NVML_GRID_LICENSE_FEATURE_CODE_VWORKSTATION = 2
end

const nvmlVgpuTypeId_t = UInt32
const nvmlVgpuInstance_t = UInt32

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
    pid::UInt32
    processName::NTuple{64, UInt8}
    timeStamp::Culonglong
    smUtil::UInt32
    memUtil::UInt32
    encUtil::UInt32
    decUtil::UInt32
end

const nvmlVgpuProcessUtilizationSample_t = nvmlVgpuProcessUtilizationSample_st

struct nvmlProcessUtilizationSample_st
    pid::UInt32
    timeStamp::Culonglong
    smUtil::UInt32
    memUtil::UInt32
    encUtil::UInt32
    decUtil::UInt32
end

const nvmlProcessUtilizationSample_t = nvmlProcessUtilizationSample_st

struct nvmlGridLicensableFeature_st
    featureCode::nvmlGridLicenseFeatureCode_t
    featureState::UInt32
    licenseInfo::NTuple{128, UInt8}
    productName::NTuple{128, UInt8}
    featureEnabled::UInt32
end

const nvmlGridLicensableFeature_t = nvmlGridLicensableFeature_st

struct nvmlGridLicensableFeatures_st
    isGridLicenseSupported::Cint
    licensableFeaturesCount::UInt32
    gridLicensableFeatures::NTuple{3, nvmlGridLicensableFeature_t}
end

const nvmlGridLicensableFeatures_t = nvmlGridLicensableFeatures_st
const nvmlDeviceArchitecture_t = UInt32

struct nvmlFieldValue_st
    fieldId::UInt32
    scopeId::UInt32
    timestamp::Clonglong
    latencyUsec::Clonglong
    valueType::nvmlValueType_t
    nvmlReturn::nvmlReturn_t
    value::nvmlValue_t
end

const nvmlFieldValue_t = nvmlFieldValue_st
const nvmlUnit_st = Cvoid
const nvmlUnit_t = Ptr{nvmlUnit_st}

struct nvmlHwbcEntry_st
    hwbcId::UInt32
    firmwareVersion::NTuple{32, UInt8}
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
    cause::NTuple{256, UInt8}
    color::nvmlLedColor_t
end

const nvmlLedState_t = nvmlLedState_st

struct nvmlUnitInfo_st
    name::NTuple{96, UInt8}
    id::NTuple{96, UInt8}
    serial::NTuple{96, UInt8}
    firmwareVersion::NTuple{96, UInt8}
end

const nvmlUnitInfo_t = nvmlUnitInfo_st

struct nvmlPSUInfo_st
    state::NTuple{256, UInt8}
    current::UInt32
    voltage::UInt32
    power::UInt32
end

const nvmlPSUInfo_t = nvmlPSUInfo_st

struct nvmlUnitFanInfo_st
    speed::UInt32
    state::nvmlFanState_t
end

const nvmlUnitFanInfo_t = nvmlUnitFanInfo_st

struct nvmlUnitFanSpeeds_st
    fans::NTuple{24, nvmlUnitFanInfo_t}
    count::UInt32
end

const nvmlUnitFanSpeeds_t = nvmlUnitFanSpeeds_st
const nvmlEventSet_st = Cvoid
const nvmlEventSet_t = Ptr{nvmlEventSet_st}

struct nvmlEventData_st
    device::nvmlDevice_t
    eventType::Culonglong
    eventData::Culonglong
    gpuInstanceId::UInt32
    computeInstanceId::UInt32
end

const nvmlEventData_t = nvmlEventData_st

struct nvmlAccountingStats_st
    gpuUtilization::UInt32
    memoryUtilization::UInt32
    maxMemoryUsage::Culonglong
    time::Culonglong
    startTime::Culonglong
    isRunning::UInt32
    reserved::NTuple{5, UInt32}
end

const nvmlAccountingStats_t = nvmlAccountingStats_st

@cenum nvmlEncoderQueryType_enum::UInt32 begin
    NVML_ENCODER_QUERY_H264 = 0
    NVML_ENCODER_QUERY_HEVC = 1
end

const nvmlEncoderType_t = nvmlEncoderQueryType_enum

struct nvmlEncoderSessionInfo_st
    sessionId::UInt32
    pid::UInt32
    vgpuInstance::nvmlVgpuInstance_t
    codecType::nvmlEncoderType_t
    hResolution::UInt32
    vResolution::UInt32
    averageFps::UInt32
    averageLatency::UInt32
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
    sessionsCount::UInt32
    averageFPS::UInt32
    averageLatency::UInt32
end

const nvmlFBCStats_t = nvmlFBCStats_st

struct nvmlFBCSessionInfo_st
    sessionId::UInt32
    pid::UInt32
    vgpuInstance::nvmlVgpuInstance_t
    displayOrdinal::UInt32
    sessionType::nvmlFBCSessionType_t
    sessionFlags::UInt32
    hMaxResolution::UInt32
    vMaxResolution::UInt32
    hResolution::UInt32
    vResolution::UInt32
    averageFPS::UInt32
    averageLatency::UInt32
end

const nvmlFBCSessionInfo_t = nvmlFBCSessionInfo_st

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
const nvmlAffinityScope_t = UInt32

@cenum nvmlClockLimitId_enum::UInt32 begin
    NVML_CLOCK_LIMIT_ID_RANGE_START = 4294967040
    NVML_CLOCK_LIMIT_ID_TDP = 4294967041
    NVML_CLOCK_LIMIT_ID_UNLIMITED = 4294967042
end

const nvmlClockLimitId_t = nvmlClockLimitId_enum

struct nvmlVgpuVersion_st
    minVersion::UInt32
    maxVersion::UInt32
end

const nvmlVgpuVersion_t = nvmlVgpuVersion_st

struct nvmlVgpuMetadata_st
    version::UInt32
    revision::UInt32
    guestInfoState::nvmlVgpuGuestInfoState_t
    guestDriverVersion::NTuple{80, UInt8}
    hostDriverVersion::NTuple{80, UInt8}
    reserved::NTuple{6, UInt32}
    vgpuVirtualizationCaps::UInt32
    guestVgpuVersion::UInt32
    opaqueDataSize::UInt32
    opaqueData::NTuple{4, UInt8}
end

const nvmlVgpuMetadata_t = nvmlVgpuMetadata_st

struct nvmlVgpuPgpuMetadata_st
    version::UInt32
    revision::UInt32
    hostDriverVersion::NTuple{80, UInt8}
    pgpuVirtualizationCaps::UInt32
    reserved::NTuple{5, UInt32}
    hostSupportedVgpuRange::nvmlVgpuVersion_t
    opaqueDataSize::UInt32
    opaqueData::NTuple{4, UInt8}
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
    NVML_VGPU_COMPATIBILITY_LIMIT_OTHER = 2147483648
end

const nvmlVgpuPgpuCompatibilityLimitCode_t = nvmlVgpuPgpuCompatibilityLimitCode_enum

struct nvmlVgpuPgpuCompatibility_st
    vgpuVmCompatibility::nvmlVgpuVmCompatibility_t
    compatibilityLimitCode::nvmlVgpuPgpuCompatibilityLimitCode_t
end

const nvmlVgpuPgpuCompatibility_t = nvmlVgpuPgpuCompatibility_st

struct nvmlBlacklistDeviceInfo_st
    pciInfo::nvmlPciInfo_t
    uuid::NTuple{80, UInt8}
end

const nvmlBlacklistDeviceInfo_t = nvmlBlacklistDeviceInfo_st

struct nvmlGpuInstancePlacement_st
    start::UInt32
    size::UInt32
end

const nvmlGpuInstancePlacement_t = nvmlGpuInstancePlacement_st

struct nvmlGpuInstanceProfileInfo_st
    id::UInt32
    isP2pSupported::UInt32
    sliceCount::UInt32
    instanceCount::UInt32
    multiprocessorCount::UInt32
    copyEngineCount::UInt32
    decoderCount::UInt32
    encoderCount::UInt32
    jpegCount::UInt32
    ofaCount::UInt32
    memorySizeMB::Culonglong
end

const nvmlGpuInstanceProfileInfo_t = nvmlGpuInstanceProfileInfo_st

struct nvmlGpuInstanceInfo_st
    device::nvmlDevice_t
    id::UInt32
    profileId::UInt32
    placement::nvmlGpuInstancePlacement_t
end

const nvmlGpuInstanceInfo_t = nvmlGpuInstanceInfo_st
const nvmlGpuInstance_st = Cvoid
const nvmlGpuInstance_t = Ptr{nvmlGpuInstance_st}

struct nvmlComputeInstancePlacement_st
    start::UInt32
    size::UInt32
end

const nvmlComputeInstancePlacement_t = nvmlComputeInstancePlacement_st

struct nvmlComputeInstanceProfileInfo_st
    id::UInt32
    sliceCount::UInt32
    instanceCount::UInt32
    multiprocessorCount::UInt32
    sharedCopyEngineCount::UInt32
    sharedDecoderCount::UInt32
    sharedEncoderCount::UInt32
    sharedJpegCount::UInt32
    sharedOfaCount::UInt32
end

const nvmlComputeInstanceProfileInfo_t = nvmlComputeInstanceProfileInfo_st

struct nvmlComputeInstanceInfo_st
    device::nvmlDevice_t
    gpuInstance::nvmlGpuInstance_t
    id::UInt32
    profileId::UInt32
    placement::nvmlComputeInstancePlacement_t
end

const nvmlComputeInstanceInfo_t = nvmlComputeInstanceInfo_st
const nvmlComputeInstance_st = Cvoid
const nvmlComputeInstance_t = Ptr{nvmlComputeInstance_st}
