using CEnum

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(NVPAError(res))
end

macro check(ex, errs...)
    quote
        res = $(esc(ex))
        if res != NVPA_STATUS_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end

macro NVPA_STRUCT_SIZE(type, lastfield)
    type = esc(type)
    lastfield = QuoteNode(lastfield)
    quote
        $struct_size($type, $lastfield)
    end
end

struct NVPW_SetLibraryLoadPaths_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    numPaths::Csize_t
    ppPaths::Ptr{Cstring}
end

struct NVPW_SetLibraryLoadPathsW_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    numPaths::Csize_t
    ppwPaths::Ptr{Ptr{Cwchar_t}}
end

@cenum NVPA_Status::UInt32 begin
    NVPA_STATUS_SUCCESS = 0
    NVPA_STATUS_ERROR = 1
    NVPA_STATUS_INTERNAL_ERROR = 2
    NVPA_STATUS_NOT_INITIALIZED = 3
    NVPA_STATUS_NOT_LOADED = 4
    NVPA_STATUS_FUNCTION_NOT_FOUND = 5
    NVPA_STATUS_NOT_SUPPORTED = 6
    NVPA_STATUS_NOT_IMPLEMENTED = 7
    NVPA_STATUS_INVALID_ARGUMENT = 8
    NVPA_STATUS_INVALID_METRIC_ID = 9
    NVPA_STATUS_DRIVER_NOT_LOADED = 10
    NVPA_STATUS_OUT_OF_MEMORY = 11
    NVPA_STATUS_INVALID_THREAD_STATE = 12
    NVPA_STATUS_FAILED_CONTEXT_ALLOC = 13
    NVPA_STATUS_UNSUPPORTED_GPU = 14
    NVPA_STATUS_INSUFFICIENT_DRIVER_VERSION = 15
    NVPA_STATUS_OBJECT_NOT_REGISTERED = 16
    NVPA_STATUS_INSUFFICIENT_PRIVILEGE = 17
    NVPA_STATUS_INVALID_CONTEXT_STATE = 18
    NVPA_STATUS_INVALID_OBJECT_STATE = 19
    NVPA_STATUS_RESOURCE_UNAVAILABLE = 20
    NVPA_STATUS_DRIVER_LOADED_TOO_LATE = 21
    NVPA_STATUS_INSUFFICIENT_SPACE = 22
    NVPA_STATUS_OBJECT_MISMATCH = 23
    NVPA_STATUS_VIRTUALIZED_DEVICE_NOT_SUPPORTED = 24
    NVPA_STATUS_PROFILING_NOT_ALLOWED = 25
    NVPA_STATUS__COUNT = 26
end

@cenum NVPA_ActivityKind::UInt32 begin
    NVPA_ACTIVITY_KIND_INVALID = 0
    NVPA_ACTIVITY_KIND_PROFILER = 1
    NVPA_ACTIVITY_KIND_REALTIME_SAMPLED = 2
    NVPA_ACTIVITY_KIND_REALTIME_PROFILER = 3
    NVPA_ACTIVITY_KIND__COUNT = 4
end

const NVPA_Bool = UInt8

# typedef NVPA_Status ( * NVPA_GenericFn ) ( void )
const NVPA_GenericFn = Ptr{Cvoid}

function NVPA_GetProcAddress(pFunctionName)
    initialize_context()
    @ccall libnvperf_host.NVPA_GetProcAddress(pFunctionName::Cstring)::NVPA_GenericFn
end

@checked function NVPW_SetLibraryLoadPaths(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_SetLibraryLoadPaths(pParams::Ptr{NVPW_SetLibraryLoadPaths_Params})::NVPA_Status
end

@checked function NVPW_SetLibraryLoadPathsW(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_SetLibraryLoadPathsW(pParams::Ptr{NVPW_SetLibraryLoadPathsW_Params})::NVPA_Status
end

struct NVPW_InitializeHost_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
end

struct NVPW_CounterData_CalculateCounterDataImageCopySize_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pCounterDataPrefix::Ptr{UInt8}
    counterDataPrefixSize::Csize_t
    maxNumRanges::UInt32
    maxNumRangeTreeNodes::UInt32
    maxRangeNameLength::UInt32
    pCounterDataSrc::Ptr{UInt8}
    copyDataImageCounterSize::Csize_t
end

struct NVPW_CounterData_InitializeCounterDataImageCopy_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pCounterDataPrefix::Ptr{UInt8}
    counterDataPrefixSize::Csize_t
    maxNumRanges::UInt32
    maxNumRangeTreeNodes::UInt32
    maxRangeNameLength::UInt32
    pCounterDataSrc::Ptr{UInt8}
    pCounterDataDst::Ptr{UInt8}
end

mutable struct NVPA_CounterDataCombiner end

struct NVPW_CounterDataCombiner_Create_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pCounterDataDst::Ptr{UInt8}
    pCounterDataCombiner::Ptr{NVPA_CounterDataCombiner}
end

struct NVPW_CounterDataCombiner_Destroy_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pCounterDataCombiner::Ptr{NVPA_CounterDataCombiner}
end

struct NVPW_CounterDataCombiner_CreateRange_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pCounterDataCombiner::Ptr{NVPA_CounterDataCombiner}
    numDescriptions::Csize_t
    ppDescriptions::Ptr{Cstring}
    rangeIndexDst::Csize_t
end

struct NVPW_CounterDataCombiner_CopyIntoRange_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pCounterDataCombiner::Ptr{NVPA_CounterDataCombiner}
    rangeIndexDst::Csize_t
    pCounterDataSrc::Ptr{UInt8}
    rangeIndexSrc::Csize_t
end

struct NVPW_CounterDataCombiner_AccumulateIntoRange_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pCounterDataCombiner::Ptr{NVPA_CounterDataCombiner}
    rangeIndexDst::Csize_t
    dstMultiplier::UInt32
    pCounterDataSrc::Ptr{UInt8}
    rangeIndexSrc::Csize_t
    srcMultiplier::UInt32
end

struct NVPW_CounterDataCombiner_SumIntoRange_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pCounterDataCombiner::Ptr{NVPA_CounterDataCombiner}
    rangeIndexDst::Csize_t
    pCounterDataSrc::Ptr{UInt8}
    rangeIndexSrc::Csize_t
end

struct NVPW_CounterDataCombiner_WeightedSumIntoRange_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pCounterDataCombiner::Ptr{NVPA_CounterDataCombiner}
    rangeIndexDst::Csize_t
    dstMultiplier::Cdouble
    pCounterDataSrc::Ptr{UInt8}
    rangeIndexSrc::Csize_t
    srcMultiplier::Cdouble
end

struct NVPA_RawMetricRequest
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricName::Cstring
    isolated::NVPA_Bool
    keepInstances::NVPA_Bool
end

struct NVPW_GetSupportedChipNames_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    ppChipNames::Ptr{Cstring}
    numChipNames::Csize_t
end

mutable struct NVPA_RawMetricsConfig end

struct NVPW_RawMetricsConfig_Destroy_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pRawMetricsConfig::Ptr{NVPA_RawMetricsConfig}
end

struct NVPW_RawMetricsConfig_SetCounterAvailability_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pRawMetricsConfig::Ptr{NVPA_RawMetricsConfig}
    pCounterAvailabilityImage::Ptr{UInt8}
end

struct NVPW_RawMetricsConfig_BeginPassGroup_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pRawMetricsConfig::Ptr{NVPA_RawMetricsConfig}
    maxPassCount::Csize_t
end

struct NVPW_RawMetricsConfig_EndPassGroup_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pRawMetricsConfig::Ptr{NVPA_RawMetricsConfig}
end

struct NVPW_RawMetricsConfig_GetNumMetrics_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pRawMetricsConfig::Ptr{NVPA_RawMetricsConfig}
    numMetrics::Csize_t
end

struct NVPW_RawMetricsConfig_GetMetricProperties_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pRawMetricsConfig::Ptr{NVPA_RawMetricsConfig}
    metricIndex::Csize_t
    pMetricName::Cstring
    supportsPipelined::NVPA_Bool
    supportsIsolated::NVPA_Bool
end

struct NVPW_RawMetricsConfig_GetMetricProperties_V2_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pRawMetricsConfig::Ptr{NVPA_RawMetricsConfig}
    metricIndex::Csize_t
    pMetricName::Cstring
end

struct NVPW_RawMetricsConfig_AddMetrics_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pRawMetricsConfig::Ptr{NVPA_RawMetricsConfig}
    pRawMetricRequests::Ptr{NVPA_RawMetricRequest}
    numMetricRequests::Csize_t
end

struct NVPW_RawMetricsConfig_IsAddMetricsPossible_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pRawMetricsConfig::Ptr{NVPA_RawMetricsConfig}
    pRawMetricRequests::Ptr{NVPA_RawMetricRequest}
    numMetricRequests::Csize_t
    isPossible::NVPA_Bool
end

struct NVPW_RawMetricsConfig_GenerateConfigImage_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pRawMetricsConfig::Ptr{NVPA_RawMetricsConfig}
    mergeAllPassGroups::NVPA_Bool
end

struct NVPW_RawMetricsConfig_GetConfigImage_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pRawMetricsConfig::Ptr{NVPA_RawMetricsConfig}
    bytesAllocated::Csize_t
    pBuffer::Ptr{UInt8}
    bytesCopied::Csize_t
end

struct NVPW_RawMetricsConfig_GetNumPasses_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pRawMetricsConfig::Ptr{NVPA_RawMetricsConfig}
    numPipelinedPasses::Csize_t
    numIsolatedPasses::Csize_t
end

struct NVPW_RawMetricsConfig_GetNumPasses_V2_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pRawMetricsConfig::Ptr{NVPA_RawMetricsConfig}
    numPasses::Csize_t
end

struct NVPW_PeriodicSampler_Config_GetSocEstimatedSampleSize_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pConfig::Ptr{UInt8}
    configSize::Csize_t
    sampleSize::Csize_t
end

struct NVPW_PeriodicSampler_Config_GetGpuEstimatedSampleSize_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pConfig::Ptr{UInt8}
    configSize::Csize_t
    sampleSize::Csize_t
end

mutable struct NVPA_CounterDataBuilder end

struct NVPW_CounterDataBuilder_Create_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pCounterDataBuilder::Ptr{NVPA_CounterDataBuilder}
    pChipName::Cstring
end

struct NVPW_CounterDataBuilder_Destroy_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pCounterDataBuilder::Ptr{NVPA_CounterDataBuilder}
end

struct NVPW_CounterDataBuilder_AddMetrics_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pCounterDataBuilder::Ptr{NVPA_CounterDataBuilder}
    pRawMetricRequests::Ptr{NVPA_RawMetricRequest}
    numMetricRequests::Csize_t
end

struct NVPW_CounterDataBuilder_GetCounterDataPrefix_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pCounterDataBuilder::Ptr{NVPA_CounterDataBuilder}
    bytesAllocated::Csize_t
    pBuffer::Ptr{UInt8}
    bytesCopied::Csize_t
end

mutable struct NVPA_MetricsContext end

struct NVPW_MetricsContext_Destroy_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
end

struct NVPW_MetricsContext_RunScript_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
    printErrors::NVPA_Bool
    pSource::Cstring
    pFileName::Cstring
end

struct NVPW_MetricsContext_ExecScript_Begin_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
    isStatement::NVPA_Bool
    printErrors::NVPA_Bool
    pSource::Cstring
    pFileName::Cstring
    pResultStr::Cstring
end

struct NVPW_MetricsContext_ExecScript_End_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
end

struct NVPW_MetricsContext_GetCounterNames_Begin_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
    numCounters::Csize_t
    ppCounterNames::Ptr{Cstring}
end

struct NVPW_MetricsContext_GetCounterNames_End_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
end

struct NVPW_MetricsContext_GetThroughputNames_Begin_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
    numThroughputs::Csize_t
    ppThroughputNames::Ptr{Cstring}
end

struct NVPW_MetricsContext_GetThroughputNames_End_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
end

struct NVPW_MetricsContext_GetRatioNames_Begin_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
    numRatios::Csize_t
    ppRatioNames::Ptr{Cstring}
end

struct NVPW_MetricsContext_GetRatioNames_End_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
end

struct NVPW_MetricsContext_GetMetricNames_Begin_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
    numMetrics::Csize_t
    ppMetricNames::Ptr{Cstring}
    hidePeakSubMetrics::NVPA_Bool
    hidePerCycleSubMetrics::NVPA_Bool
    hidePctOfPeakSubMetrics::NVPA_Bool
    hidePctOfPeakSubMetricsOnThroughputs::NVPA_Bool
end

struct NVPW_MetricsContext_GetMetricNames_End_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
end

struct NVPW_MetricsContext_GetThroughputBreakdown_Begin_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
    pThroughputName::Cstring
    ppCounterNames::Ptr{Cstring}
    ppSubThroughputNames::Ptr{Cstring}
end

struct NVPW_MetricsContext_GetThroughputBreakdown_End_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
end

struct NVPW_MetricsContext_GetMetricProperties_Begin_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
    pMetricName::Cstring
    pDescription::Cstring
    pDimUnits::Cstring
    ppRawMetricDependencies::Ptr{Cstring}
    gpuBurstRate::Cdouble
    gpuSustainedRate::Cdouble
    ppOptionalRawMetricDependencies::Ptr{Cstring}
end

struct NVPW_MetricsContext_GetMetricProperties_End_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
end

struct NVPW_MetricsContext_SetCounterData_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
    pCounterDataImage::Ptr{UInt8}
    rangeIndex::Csize_t
    isolated::NVPA_Bool
end

struct NVPW_MetricsContext_SetUserData_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
    frameDuration::Cdouble
    regionDuration::Cdouble
end

struct NVPW_MetricsContext_EvaluateToGpuValues_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
    numMetrics::Csize_t
    ppMetricNames::Ptr{Cstring}
    pMetricValues::Ptr{Cdouble}
end

struct NVPW_MetricsContext_GetMetricSuffix_Begin_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
    pMetricName::Cstring
    numSuffixes::Csize_t
    ppSuffixes::Ptr{Cstring}
    hidePeakSubMetrics::NVPA_Bool
    hidePerCycleSubMetrics::NVPA_Bool
    hidePctOfPeakSubMetrics::NVPA_Bool
    hidePctOfPeakSubMetricsOnThroughputs::NVPA_Bool
end

struct NVPW_MetricsContext_GetMetricSuffix_End_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
end

struct NVPW_MetricsContext_GetMetricBaseNames_Begin_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
    numMetricBaseNames::Csize_t
    ppMetricBaseNames::Ptr{Cstring}
end

struct NVPW_MetricsContext_GetMetricBaseNames_End_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsContext::Ptr{NVPA_MetricsContext}
end

struct NVPW_MetricEvalRequest
    metricIndex::Csize_t
    metricType::UInt8
    rollupOp::UInt8
    submetric::UInt16
end

struct NVPW_DimUnitFactor
    dimUnit::UInt32
    exponent::Int8
end

mutable struct NVPW_MetricsEvaluator end

struct NVPW_MetricsEvaluator_Destroy_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsEvaluator::Ptr{NVPW_MetricsEvaluator}
end

struct NVPW_MetricsEvaluator_GetMetricNames_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsEvaluator::Ptr{NVPW_MetricsEvaluator}
    metricType::UInt8
    pMetricNames::Cstring
    pMetricNameBeginIndices::Ptr{Csize_t}
    numMetrics::Csize_t
end

struct NVPW_MetricsEvaluator_GetMetricTypeAndIndex_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsEvaluator::Ptr{NVPW_MetricsEvaluator}
    pMetricName::Cstring
    metricType::UInt8
    metricIndex::Csize_t
end

struct NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsEvaluator::Ptr{NVPW_MetricsEvaluator}
    pMetricName::Cstring
    pMetricEvalRequest::Ptr{NVPW_MetricEvalRequest}
    metricEvalRequestStructSize::Csize_t
end

struct NVPW_MetricsEvaluator_HwUnitToString_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsEvaluator::Ptr{NVPW_MetricsEvaluator}
    hwUnit::UInt32
    pHwUnitName::Cstring
end

struct NVPW_MetricsEvaluator_GetCounterProperties_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsEvaluator::Ptr{NVPW_MetricsEvaluator}
    counterIndex::Csize_t
    pDescription::Cstring
    hwUnit::UInt32
end

struct NVPW_MetricsEvaluator_GetRatioMetricProperties_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsEvaluator::Ptr{NVPW_MetricsEvaluator}
    ratioMetricIndex::Csize_t
    pDescription::Cstring
    hwUnit::UInt64
end

struct NVPW_MetricsEvaluator_GetThroughputMetricProperties_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsEvaluator::Ptr{NVPW_MetricsEvaluator}
    throughputMetricIndex::Csize_t
    pDescription::Cstring
    hwUnit::UInt32
    numCounters::Csize_t
    pCounterIndices::Ptr{Csize_t}
    numSubThroughputs::Csize_t
    pSubThroughputIndices::Ptr{Csize_t}
end

struct NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsEvaluator::Ptr{NVPW_MetricsEvaluator}
    metricType::UInt8
    pSupportedSubmetrics::Ptr{UInt16}
    numSupportedSubmetrics::Csize_t
end

struct NVPW_MetricsEvaluator_GetMetricRawDependencies_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsEvaluator::Ptr{NVPW_MetricsEvaluator}
    pMetricEvalRequests::Ptr{NVPW_MetricEvalRequest}
    numMetricEvalRequests::Csize_t
    metricEvalRequestStructSize::Csize_t
    metricEvalRequestStrideSize::Csize_t
    ppRawDependencies::Ptr{Cstring}
    numRawDependencies::Csize_t
    ppOptionalRawDependencies::Ptr{Cstring}
    numOptionalRawDependencies::Csize_t
end

struct NVPW_MetricsEvaluator_DimUnitToString_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsEvaluator::Ptr{NVPW_MetricsEvaluator}
    dimUnit::UInt32
    pSingularName::Cstring
    pPluralName::Cstring
end

struct NVPW_MetricsEvaluator_GetMetricDimUnits_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsEvaluator::Ptr{NVPW_MetricsEvaluator}
    pMetricEvalRequest::Ptr{NVPW_MetricEvalRequest}
    metricEvalRequestStructSize::Csize_t
    pDimUnits::Ptr{NVPW_DimUnitFactor}
    numDimUnits::Csize_t
    dimUnitFactorStructSize::Csize_t
end

struct NVPW_MetricsEvaluator_SetUserData_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsEvaluator::Ptr{NVPW_MetricsEvaluator}
    frameDuration::Cdouble
    regionDuration::Cdouble
    isolated::NVPA_Bool
end

struct NVPW_MetricsEvaluator_EvaluateToGpuValues_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsEvaluator::Ptr{NVPW_MetricsEvaluator}
    pMetricEvalRequests::Ptr{NVPW_MetricEvalRequest}
    numMetricEvalRequests::Csize_t
    metricEvalRequestStructSize::Csize_t
    metricEvalRequestStrideSize::Csize_t
    pCounterDataImage::Ptr{UInt8}
    counterDataImageSize::Csize_t
    rangeIndex::Csize_t
    isolated::NVPA_Bool
    pMetricValues::Ptr{Cdouble}
end

struct NVPW_MetricsEvaluator_SetDeviceAttributes_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pMetricsEvaluator::Ptr{NVPW_MetricsEvaluator}
    pCounterDataImage::Ptr{UInt8}
    counterDataImageSize::Csize_t
end

@checked function NVPW_InitializeHost(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_InitializeHost(pParams::Ptr{NVPW_InitializeHost_Params})::NVPA_Status
end

@checked function NVPW_CounterData_CalculateCounterDataImageCopySize(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_CounterData_CalculateCounterDataImageCopySize(pParams::Ptr{NVPW_CounterData_CalculateCounterDataImageCopySize_Params})::NVPA_Status
end

@checked function NVPW_CounterData_InitializeCounterDataImageCopy(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_CounterData_InitializeCounterDataImageCopy(pParams::Ptr{NVPW_CounterData_InitializeCounterDataImageCopy_Params})::NVPA_Status
end

@checked function NVPW_CounterDataCombiner_Create(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_CounterDataCombiner_Create(pParams::Ptr{NVPW_CounterDataCombiner_Create_Params})::NVPA_Status
end

@checked function NVPW_CounterDataCombiner_Destroy(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_CounterDataCombiner_Destroy(pParams::Ptr{NVPW_CounterDataCombiner_Destroy_Params})::NVPA_Status
end

@checked function NVPW_CounterDataCombiner_CreateRange(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_CounterDataCombiner_CreateRange(pParams::Ptr{NVPW_CounterDataCombiner_CreateRange_Params})::NVPA_Status
end

@checked function NVPW_CounterDataCombiner_CopyIntoRange(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_CounterDataCombiner_CopyIntoRange(pParams::Ptr{NVPW_CounterDataCombiner_CopyIntoRange_Params})::NVPA_Status
end

@checked function NVPW_CounterDataCombiner_AccumulateIntoRange(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_CounterDataCombiner_AccumulateIntoRange(pParams::Ptr{NVPW_CounterDataCombiner_AccumulateIntoRange_Params})::NVPA_Status
end

@checked function NVPW_CounterDataCombiner_SumIntoRange(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_CounterDataCombiner_SumIntoRange(pParams::Ptr{NVPW_CounterDataCombiner_SumIntoRange_Params})::NVPA_Status
end

@checked function NVPW_CounterDataCombiner_WeightedSumIntoRange(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_CounterDataCombiner_WeightedSumIntoRange(pParams::Ptr{NVPW_CounterDataCombiner_WeightedSumIntoRange_Params})::NVPA_Status
end

@checked function NVPW_GetSupportedChipNames(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_GetSupportedChipNames(pParams::Ptr{NVPW_GetSupportedChipNames_Params})::NVPA_Status
end

@checked function NVPW_RawMetricsConfig_Destroy(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_RawMetricsConfig_Destroy(pParams::Ptr{NVPW_RawMetricsConfig_Destroy_Params})::NVPA_Status
end

@checked function NVPW_RawMetricsConfig_SetCounterAvailability(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_RawMetricsConfig_SetCounterAvailability(pParams::Ptr{NVPW_RawMetricsConfig_SetCounterAvailability_Params})::NVPA_Status
end

@checked function NVPW_RawMetricsConfig_BeginPassGroup(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_RawMetricsConfig_BeginPassGroup(pParams::Ptr{NVPW_RawMetricsConfig_BeginPassGroup_Params})::NVPA_Status
end

@checked function NVPW_RawMetricsConfig_EndPassGroup(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_RawMetricsConfig_EndPassGroup(pParams::Ptr{NVPW_RawMetricsConfig_EndPassGroup_Params})::NVPA_Status
end

@checked function NVPW_RawMetricsConfig_GetNumMetrics(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_RawMetricsConfig_GetNumMetrics(pParams::Ptr{NVPW_RawMetricsConfig_GetNumMetrics_Params})::NVPA_Status
end

@checked function NVPW_RawMetricsConfig_GetMetricProperties(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_RawMetricsConfig_GetMetricProperties(pParams::Ptr{NVPW_RawMetricsConfig_GetMetricProperties_Params})::NVPA_Status
end

@checked function NVPW_RawMetricsConfig_GetMetricProperties_V2(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_RawMetricsConfig_GetMetricProperties_V2(pParams::Ptr{NVPW_RawMetricsConfig_GetMetricProperties_V2_Params})::NVPA_Status
end

@checked function NVPW_RawMetricsConfig_AddMetrics(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_RawMetricsConfig_AddMetrics(pParams::Ptr{NVPW_RawMetricsConfig_AddMetrics_Params})::NVPA_Status
end

@checked function NVPW_RawMetricsConfig_IsAddMetricsPossible(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_RawMetricsConfig_IsAddMetricsPossible(pParams::Ptr{NVPW_RawMetricsConfig_IsAddMetricsPossible_Params})::NVPA_Status
end

@checked function NVPW_RawMetricsConfig_GenerateConfigImage(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_RawMetricsConfig_GenerateConfigImage(pParams::Ptr{NVPW_RawMetricsConfig_GenerateConfigImage_Params})::NVPA_Status
end

@checked function NVPW_RawMetricsConfig_GetConfigImage(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_RawMetricsConfig_GetConfigImage(pParams::Ptr{NVPW_RawMetricsConfig_GetConfigImage_Params})::NVPA_Status
end

@checked function NVPW_RawMetricsConfig_GetNumPasses(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_RawMetricsConfig_GetNumPasses(pParams::Ptr{NVPW_RawMetricsConfig_GetNumPasses_Params})::NVPA_Status
end

@checked function NVPW_RawMetricsConfig_GetNumPasses_V2(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_RawMetricsConfig_GetNumPasses_V2(pParams::Ptr{NVPW_RawMetricsConfig_GetNumPasses_V2_Params})::NVPA_Status
end

@checked function NVPW_PeriodicSampler_Config_GetSocEstimatedSampleSize(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_PeriodicSampler_Config_GetSocEstimatedSampleSize(pParams::Ptr{NVPW_PeriodicSampler_Config_GetSocEstimatedSampleSize_Params})::NVPA_Status
end

@checked function NVPW_PeriodicSampler_Config_GetGpuEstimatedSampleSize(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_PeriodicSampler_Config_GetGpuEstimatedSampleSize(pParams::Ptr{NVPW_PeriodicSampler_Config_GetGpuEstimatedSampleSize_Params})::NVPA_Status
end

@checked function NVPW_CounterDataBuilder_Create(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_CounterDataBuilder_Create(pParams::Ptr{NVPW_CounterDataBuilder_Create_Params})::NVPA_Status
end

@checked function NVPW_CounterDataBuilder_Destroy(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_CounterDataBuilder_Destroy(pParams::Ptr{NVPW_CounterDataBuilder_Destroy_Params})::NVPA_Status
end

@checked function NVPW_CounterDataBuilder_AddMetrics(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_CounterDataBuilder_AddMetrics(pParams::Ptr{NVPW_CounterDataBuilder_AddMetrics_Params})::NVPA_Status
end

@checked function NVPW_CounterDataBuilder_GetCounterDataPrefix(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_CounterDataBuilder_GetCounterDataPrefix(pParams::Ptr{NVPW_CounterDataBuilder_GetCounterDataPrefix_Params})::NVPA_Status
end

@cenum NVPA_MetricDetailLevel::UInt32 begin
    NVPA_METRIC_DETAIL_LEVEL_INVALID = 0
    NVPA_METRIC_DETAIL_LEVEL_GPU = 1
    NVPA_METRIC_DETAIL_LEVEL_ALL = 2
    NVPA_METRIC_DETAIL_LEVEL_GPU_AND_LEAF_INSTANCES = 3
    NVPA_METRIC_DETAIL_LEVEL__COUNT = 4
end

@checked function NVPW_MetricsContext_Destroy(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_Destroy(pParams::Ptr{NVPW_MetricsContext_Destroy_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_RunScript(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_RunScript(pParams::Ptr{NVPW_MetricsContext_RunScript_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_ExecScript_Begin(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_ExecScript_Begin(pParams::Ptr{NVPW_MetricsContext_ExecScript_Begin_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_ExecScript_End(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_ExecScript_End(pParams::Ptr{NVPW_MetricsContext_ExecScript_End_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_GetCounterNames_Begin(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_GetCounterNames_Begin(pParams::Ptr{NVPW_MetricsContext_GetCounterNames_Begin_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_GetCounterNames_End(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_GetCounterNames_End(pParams::Ptr{NVPW_MetricsContext_GetCounterNames_End_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_GetThroughputNames_Begin(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_GetThroughputNames_Begin(pParams::Ptr{NVPW_MetricsContext_GetThroughputNames_Begin_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_GetThroughputNames_End(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_GetThroughputNames_End(pParams::Ptr{NVPW_MetricsContext_GetThroughputNames_End_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_GetRatioNames_Begin(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_GetRatioNames_Begin(pParams::Ptr{NVPW_MetricsContext_GetRatioNames_Begin_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_GetRatioNames_End(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_GetRatioNames_End(pParams::Ptr{NVPW_MetricsContext_GetRatioNames_End_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_GetMetricNames_Begin(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_GetMetricNames_Begin(pParams::Ptr{NVPW_MetricsContext_GetMetricNames_Begin_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_GetMetricNames_End(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_GetMetricNames_End(pParams::Ptr{NVPW_MetricsContext_GetMetricNames_End_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_GetThroughputBreakdown_Begin(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_GetThroughputBreakdown_Begin(pParams::Ptr{NVPW_MetricsContext_GetThroughputBreakdown_Begin_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_GetThroughputBreakdown_End(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_GetThroughputBreakdown_End(pParams::Ptr{NVPW_MetricsContext_GetThroughputBreakdown_End_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_GetMetricProperties_Begin(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_GetMetricProperties_Begin(pParams::Ptr{NVPW_MetricsContext_GetMetricProperties_Begin_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_GetMetricProperties_End(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_GetMetricProperties_End(pParams::Ptr{NVPW_MetricsContext_GetMetricProperties_End_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_SetCounterData(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_SetCounterData(pParams::Ptr{NVPW_MetricsContext_SetCounterData_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_SetUserData(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_SetUserData(pParams::Ptr{NVPW_MetricsContext_SetUserData_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_EvaluateToGpuValues(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_EvaluateToGpuValues(pParams::Ptr{NVPW_MetricsContext_EvaluateToGpuValues_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_GetMetricSuffix_Begin(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_GetMetricSuffix_Begin(pParams::Ptr{NVPW_MetricsContext_GetMetricSuffix_Begin_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_GetMetricSuffix_End(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_GetMetricSuffix_End(pParams::Ptr{NVPW_MetricsContext_GetMetricSuffix_End_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_GetMetricBaseNames_Begin(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_GetMetricBaseNames_Begin(pParams::Ptr{NVPW_MetricsContext_GetMetricBaseNames_Begin_Params})::NVPA_Status
end

@checked function NVPW_MetricsContext_GetMetricBaseNames_End(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsContext_GetMetricBaseNames_End(pParams::Ptr{NVPW_MetricsContext_GetMetricBaseNames_End_Params})::NVPA_Status
end

@cenum NVPW_DimUnitName::UInt32 begin
    NVPW_DIM_UNIT_INVALID = 0x00000000d1b4fc15
    NVPW_DIM_UNIT_UNITLESS = 2126137902
    NVPW_DIM_UNIT_ATTRIBUTES = 0x00000000e1165b29
    NVPW_DIM_UNIT_BYTES = 0x00000000e25e984f
    NVPW_DIM_UNIT_CTAS = 1960564139
    NVPW_DIM_UNIT_DRAM_CYCLES = 0x000000009e02c3cf
    NVPW_DIM_UNIT_FBP_CYCLES = 1785238957
    NVPW_DIM_UNIT_FE_OPS = 0x00000000adfed52b
    NVPW_DIM_UNIT_GPC_CYCLES = 1222631184
    NVPW_DIM_UNIT_IDC_REQUESTS = 2012649669
    NVPW_DIM_UNIT_INSTRUCTIONS = 1418625543
    NVPW_DIM_UNIT_KILOBYTES = 1335980302
    NVPW_DIM_UNIT_L1DATA_BANK_ACCESSES = 1479493682
    NVPW_DIM_UNIT_L1DATA_BANK_CONFLICTS = 0x00000000cca20763
    NVPW_DIM_UNIT_L1TEX_REQUESTS = 1306473767
    NVPW_DIM_UNIT_L1TEX_TAGS = 26573010
    NVPW_DIM_UNIT_L1TEX_WAVEFRONTS = 129373765
    NVPW_DIM_UNIT_L2_REQUESTS = 1143695106
    NVPW_DIM_UNIT_L2_SECTORS = 0x00000000cc17a4bc
    NVPW_DIM_UNIT_L2_TAGS = 0x00000000dfda1a6d
    NVPW_DIM_UNIT_NANOSECONDS = 0x00000000b5a52b80
    NVPW_DIM_UNIT_NVLRX_CYCLES = 0x00000000f1fdb0d2
    NVPW_DIM_UNIT_NVLTX_CYCLES = 1814350488
    NVPW_DIM_UNIT_PCIE_CYCLES = 1230450943
    NVPW_DIM_UNIT_PERCENT = 1284354694
    NVPW_DIM_UNIT_PIXELS = 0x00000000fbfc4f97
    NVPW_DIM_UNIT_PIXEL_SHADER_BARRIERS = 0x00000000dcdd7b36
    NVPW_DIM_UNIT_PRIMITIVES = 0x000000008d726362
    NVPW_DIM_UNIT_QUADS = 1539753497
    NVPW_DIM_UNIT_REGISTERS = 0x00000000a91d2a93
    NVPW_DIM_UNIT_SAMPLES = 746046551
    NVPW_DIM_UNIT_SECONDS = 1164825258
    NVPW_DIM_UNIT_SYS_CYCLES = 0x00000000c5572138
    NVPW_DIM_UNIT_TEXELS = 1293214069
    NVPW_DIM_UNIT_THREADS = 164261907
    NVPW_DIM_UNIT_VERTICES = 1873662209
    NVPW_DIM_UNIT_WARPS = 97951949
    NVPW_DIM_UNIT_WORKLOADS = 1728142656
end

@cenum NVPW_HwUnit::UInt32 begin
    NVPW_HW_UNIT_INVALID = 0x00000000d07fc9f5
    NVPW_HW_UNIT_CROP = 0x00000000ab315876
    NVPW_HW_UNIT_DRAM = 1662616918
    NVPW_HW_UNIT_DRAMC = 1401232876
    NVPW_HW_UNIT_FBP = 0x00000000afaa9dc2
    NVPW_HW_UNIT_FBPA = 690045803
    NVPW_HW_UNIT_FE = 0x00000000836c79a1
    NVPW_HW_UNIT_GPC = 1911735839
    NVPW_HW_UNIT_GPU = 1014363534
    NVPW_HW_UNIT_GR = 0x00000000aedb7755
    NVPW_HW_UNIT_IDC = 842765289
    NVPW_HW_UNIT_L1TEX = 893940957
    NVPW_HW_UNIT_LTS = 0x000000008b12d309
    NVPW_HW_UNIT_NVLRX = 0x00000000b8475e25
    NVPW_HW_UNIT_NVLTX = 869679659
    NVPW_HW_UNIT_PCIE = 0x00000000cca3742e
    NVPW_HW_UNIT_PDA = 345193251
    NVPW_HW_UNIT_PES = 804128425
    NVPW_HW_UNIT_PROP = 0x00000000c708fed3
    NVPW_HW_UNIT_RASTER = 187932504
    NVPW_HW_UNIT_SM = 724224710
    NVPW_HW_UNIT_SMSP = 0x00000000a9229915
    NVPW_HW_UNIT_SYS = 768990063
    NVPW_HW_UNIT_TPC = 1889024613
    NVPW_HW_UNIT_VAF = 753670509
    NVPW_HW_UNIT_VPC = 275561583
    NVPW_HW_UNIT_ZROP = 979500456
end

@cenum NVPW_RollupOp::UInt32 begin
    NVPW_ROLLUP_OP_AVG = 0
    NVPW_ROLLUP_OP_MAX = 1
    NVPW_ROLLUP_OP_MIN = 2
    NVPW_ROLLUP_OP_SUM = 3
    NVPW_ROLLUP_OP__COUNT = 4
end

@cenum NVPW_MetricType::UInt32 begin
    NVPW_METRIC_TYPE_COUNTER = 0
    NVPW_METRIC_TYPE_RATIO = 1
    NVPW_METRIC_TYPE_THROUGHPUT = 2
    NVPW_METRIC_TYPE__COUNT = 3
end

@cenum NVPW_Submetric::UInt32 begin
    NVPW_SUBMETRIC_NONE = 0
    NVPW_SUBMETRIC_PEAK_SUSTAINED = 1
    NVPW_SUBMETRIC_PEAK_SUSTAINED_ACTIVE = 2
    NVPW_SUBMETRIC_PEAK_SUSTAINED_ACTIVE_PER_SECOND = 3
    NVPW_SUBMETRIC_PEAK_SUSTAINED_ELAPSED = 4
    NVPW_SUBMETRIC_PEAK_SUSTAINED_ELAPSED_PER_SECOND = 5
    NVPW_SUBMETRIC_PEAK_SUSTAINED_FRAME = 6
    NVPW_SUBMETRIC_PEAK_SUSTAINED_FRAME_PER_SECOND = 7
    NVPW_SUBMETRIC_PEAK_SUSTAINED_REGION = 8
    NVPW_SUBMETRIC_PEAK_SUSTAINED_REGION_PER_SECOND = 9
    NVPW_SUBMETRIC_PER_CYCLE_ACTIVE = 10
    NVPW_SUBMETRIC_PER_CYCLE_ELAPSED = 11
    NVPW_SUBMETRIC_PER_CYCLE_IN_FRAME = 12
    NVPW_SUBMETRIC_PER_CYCLE_IN_REGION = 13
    NVPW_SUBMETRIC_PER_SECOND = 14
    NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_ACTIVE = 15
    NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_ELAPSED = 16
    NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_FRAME = 17
    NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_REGION = 18
    NVPW_SUBMETRIC_MAX_RATE = 19
    NVPW_SUBMETRIC_PCT = 20
    NVPW_SUBMETRIC_RATIO = 21
    NVPW_SUBMETRIC__COUNT = 22
end

@checked function NVPW_MetricsEvaluator_Destroy(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsEvaluator_Destroy(pParams::Ptr{NVPW_MetricsEvaluator_Destroy_Params})::NVPA_Status
end

@checked function NVPW_MetricsEvaluator_GetMetricNames(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsEvaluator_GetMetricNames(pParams::Ptr{NVPW_MetricsEvaluator_GetMetricNames_Params})::NVPA_Status
end

@checked function NVPW_MetricsEvaluator_GetMetricTypeAndIndex(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsEvaluator_GetMetricTypeAndIndex(pParams::Ptr{NVPW_MetricsEvaluator_GetMetricTypeAndIndex_Params})::NVPA_Status
end

@checked function NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(pParams::Ptr{NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params})::NVPA_Status
end

@checked function NVPW_MetricsEvaluator_HwUnitToString(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsEvaluator_HwUnitToString(pParams::Ptr{NVPW_MetricsEvaluator_HwUnitToString_Params})::NVPA_Status
end

@checked function NVPW_MetricsEvaluator_GetCounterProperties(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsEvaluator_GetCounterProperties(pParams::Ptr{NVPW_MetricsEvaluator_GetCounterProperties_Params})::NVPA_Status
end

@checked function NVPW_MetricsEvaluator_GetRatioMetricProperties(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsEvaluator_GetRatioMetricProperties(pParams::Ptr{NVPW_MetricsEvaluator_GetRatioMetricProperties_Params})::NVPA_Status
end

@checked function NVPW_MetricsEvaluator_GetThroughputMetricProperties(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsEvaluator_GetThroughputMetricProperties(pParams::Ptr{NVPW_MetricsEvaluator_GetThroughputMetricProperties_Params})::NVPA_Status
end

@checked function NVPW_MetricsEvaluator_GetSupportedSubmetrics(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsEvaluator_GetSupportedSubmetrics(pParams::Ptr{NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params})::NVPA_Status
end

@checked function NVPW_MetricsEvaluator_GetMetricRawDependencies(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsEvaluator_GetMetricRawDependencies(pParams::Ptr{NVPW_MetricsEvaluator_GetMetricRawDependencies_Params})::NVPA_Status
end

@checked function NVPW_MetricsEvaluator_DimUnitToString(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsEvaluator_DimUnitToString(pParams::Ptr{NVPW_MetricsEvaluator_DimUnitToString_Params})::NVPA_Status
end

@checked function NVPW_MetricsEvaluator_GetMetricDimUnits(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsEvaluator_GetMetricDimUnits(pParams::Ptr{NVPW_MetricsEvaluator_GetMetricDimUnits_Params})::NVPA_Status
end

@checked function NVPW_MetricsEvaluator_SetUserData(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsEvaluator_SetUserData(pParams::Ptr{NVPW_MetricsEvaluator_SetUserData_Params})::NVPA_Status
end

@checked function NVPW_MetricsEvaluator_EvaluateToGpuValues(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsEvaluator_EvaluateToGpuValues(pParams::Ptr{NVPW_MetricsEvaluator_EvaluateToGpuValues_Params})::NVPA_Status
end

@checked function NVPW_MetricsEvaluator_SetDeviceAttributes(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_MetricsEvaluator_SetDeviceAttributes(pParams::Ptr{NVPW_MetricsEvaluator_SetDeviceAttributes_Params})::NVPA_Status
end

struct NVPW_CUDA_MetricsContext_Create_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pChipName::Cstring
    pMetricsContext::Ptr{NVPA_MetricsContext}
end

struct NVPW_CUDA_RawMetricsConfig_Create_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    activityKind::NVPA_ActivityKind
    pChipName::Cstring
    pRawMetricsConfig::Ptr{NVPA_RawMetricsConfig}
end

struct NVPW_CUDA_RawMetricsConfig_Create_V2_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    activityKind::NVPA_ActivityKind
    pChipName::Cstring
    pCounterAvailabilityImage::Ptr{UInt8}
    pRawMetricsConfig::Ptr{NVPA_RawMetricsConfig}
end

struct NVPW_CUDA_CounterDataBuilder_Create_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pChipName::Cstring
    pCounterAvailabilityImage::Ptr{UInt8}
    pCounterDataBuilder::Ptr{NVPA_CounterDataBuilder}
end

struct NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pChipName::Cstring
    pCounterAvailabilityImage::Ptr{UInt8}
    scratchBufferSize::Csize_t
end

struct NVPW_CUDA_MetricsEvaluator_Initialize_Params
    structSize::Csize_t
    pPriv::Ptr{Cvoid}
    pScratchBuffer::Ptr{UInt8}
    scratchBufferSize::Csize_t
    pChipName::Cstring
    pCounterAvailabilityImage::Ptr{UInt8}
    pCounterDataImage::Ptr{UInt8}
    counterDataImageSize::Csize_t
    pMetricsEvaluator::Ptr{NVPW_MetricsEvaluator}
end

@checked function NVPW_CUDA_MetricsContext_Create(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_CUDA_MetricsContext_Create(pParams::Ptr{NVPW_CUDA_MetricsContext_Create_Params})::NVPA_Status
end

@checked function NVPW_CUDA_RawMetricsConfig_Create(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_CUDA_RawMetricsConfig_Create(pParams::Ptr{NVPW_CUDA_RawMetricsConfig_Create_Params})::NVPA_Status
end

@checked function NVPW_CUDA_RawMetricsConfig_Create_V2(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_CUDA_RawMetricsConfig_Create_V2(pParams::Ptr{NVPW_CUDA_RawMetricsConfig_Create_V2_Params})::NVPA_Status
end

@checked function NVPW_CUDA_CounterDataBuilder_Create(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_CUDA_CounterDataBuilder_Create(pParams::Ptr{NVPW_CUDA_CounterDataBuilder_Create_Params})::NVPA_Status
end

@checked function NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(pParams::Ptr{NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params})::NVPA_Status
end

@checked function NVPW_CUDA_MetricsEvaluator_Initialize(pParams)
    initialize_context()
    @ccall libnvperf_host.NVPW_CUDA_MetricsEvaluator_Initialize(pParams::Ptr{NVPW_CUDA_MetricsEvaluator_Initialize_Params})::NVPA_Status
end

const NVPW_SetLibraryLoadPaths_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_SetLibraryLoadPaths_Params,
                                                                      ppPaths)

const NVPW_SetLibraryLoadPathsW_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_SetLibraryLoadPathsW_Params,
                                                                       ppwPaths)

const NVPW_InitializeHost_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_InitializeHost_Params,
                                                                 pPriv)

const NVPW_CounterData_CalculateCounterDataImageCopySize_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_CounterData_CalculateCounterDataImageCopySize_Params,
                                                                                                copyDataImageCounterSize)

const NVPW_CounterData_InitializeCounterDataImageCopy_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_CounterData_InitializeCounterDataImageCopy_Params,
                                                                                             pCounterDataDst)

const NVPW_CounterDataCombiner_Create_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_CounterDataCombiner_Create_Params,
                                                                             pCounterDataCombiner)

const NVPW_CounterDataCombiner_Destroy_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_CounterDataCombiner_Destroy_Params,
                                                                              pCounterDataCombiner)

const NVPW_CounterDataCombiner_CreateRange_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_CounterDataCombiner_CreateRange_Params,
                                                                                  rangeIndexDst)

const NVPW_CounterDataCombiner_CopyIntoRange_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_CounterDataCombiner_CopyIntoRange_Params,
                                                                                    rangeIndexSrc)

const NVPW_CounterDataCombiner_AccumulateIntoRange_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_CounterDataCombiner_AccumulateIntoRange_Params,
                                                                                          srcMultiplier)

const NVPW_CounterDataCombiner_SumIntoRange_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_CounterDataCombiner_SumIntoRange_Params,
                                                                                   rangeIndexSrc)

const NVPW_CounterDataCombiner_WeightedSumIntoRange_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_CounterDataCombiner_WeightedSumIntoRange_Params,
                                                                                           srcMultiplier)

const NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPA_RawMetricRequest,
                                                              keepInstances)

const NVPW_GetSupportedChipNames_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_GetSupportedChipNames_Params,
                                                                        numChipNames)

const NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_RawMetricsConfig_Destroy_Params,
                                                                           pRawMetricsConfig)

const NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_RawMetricsConfig_SetCounterAvailability_Params,
                                                                                          pCounterAvailabilityImage)

const NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_RawMetricsConfig_BeginPassGroup_Params,
                                                                                  maxPassCount)

const NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_RawMetricsConfig_EndPassGroup_Params,
                                                                                pRawMetricsConfig)

const NVPW_RawMetricsConfig_GetNumMetrics_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_RawMetricsConfig_GetNumMetrics_Params,
                                                                                 numMetrics)

const NVPW_RawMetricsConfig_GetMetricProperties_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_RawMetricsConfig_GetMetricProperties_Params,
                                                                                       supportsIsolated)

const NVPW_RawMetricsConfig_GetMetricProperties_V2_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_RawMetricsConfig_GetMetricProperties_V2_Params,
                                                                                          pMetricName)

const NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_RawMetricsConfig_AddMetrics_Params,
                                                                              numMetricRequests)

const NVPW_RawMetricsConfig_IsAddMetricsPossible_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_RawMetricsConfig_IsAddMetricsPossible_Params,
                                                                                        isPossible)

const NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_RawMetricsConfig_GenerateConfigImage_Params,
                                                                                       mergeAllPassGroups)

const NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_RawMetricsConfig_GetConfigImage_Params,
                                                                                  bytesCopied)

const NVPW_RawMetricsConfig_GetNumPasses_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_RawMetricsConfig_GetNumPasses_Params,
                                                                                numIsolatedPasses)

const NVPW_RawMetricsConfig_GetNumPasses_V2_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_RawMetricsConfig_GetNumPasses_V2_Params,
                                                                                   numPasses)

const NVPW_PeriodicSampler_Config_GetSocEstimatedSampleSize_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_PeriodicSampler_Config_GetSocEstimatedSampleSize_Params,
                                                                                                   sampleSize)

const NVPW_PeriodicSampler_Config_GetGpuEstimatedSampleSize_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_PeriodicSampler_Config_GetGpuEstimatedSampleSize_Params,
                                                                                                   sampleSize)

const NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_CounterDataBuilder_Create_Params,
                                                                            pChipName)

const NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_CounterDataBuilder_Destroy_Params,
                                                                             pCounterDataBuilder)

const NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_CounterDataBuilder_AddMetrics_Params,
                                                                                numMetricRequests)

const NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_CounterDataBuilder_GetCounterDataPrefix_Params,
                                                                                          bytesCopied)

const NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_Destroy_Params,
                                                                         pMetricsContext)

const NVPW_MetricsContext_RunScript_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_RunScript_Params,
                                                                           pFileName)

const NVPW_MetricsContext_ExecScript_Begin_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_ExecScript_Begin_Params,
                                                                                  pResultStr)

const NVPW_MetricsContext_ExecScript_End_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_ExecScript_End_Params,
                                                                                pMetricsContext)

const NVPW_MetricsContext_GetCounterNames_Begin_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetCounterNames_Begin_Params,
                                                                                       ppCounterNames)

const NVPW_MetricsContext_GetCounterNames_End_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetCounterNames_End_Params,
                                                                                     pMetricsContext)

const NVPW_MetricsContext_GetThroughputNames_Begin_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetThroughputNames_Begin_Params,
                                                                                          ppThroughputNames)

const NVPW_MetricsContext_GetThroughputNames_End_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetThroughputNames_End_Params,
                                                                                        pMetricsContext)

const NVPW_MetricsContext_GetRatioNames_Begin_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetRatioNames_Begin_Params,
                                                                                     ppRatioNames)

const NVPW_MetricsContext_GetRatioNames_End_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetRatioNames_End_Params,
                                                                                   pMetricsContext)

const NVPW_MetricsContext_GetMetricNames_Begin_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetMetricNames_Begin_Params,
                                                                                      hidePctOfPeakSubMetricsOnThroughputs)

const NVPW_MetricsContext_GetMetricNames_End_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetMetricNames_End_Params,
                                                                                    pMetricsContext)

const NVPW_MetricsContext_GetThroughputBreakdown_Begin_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetThroughputBreakdown_Begin_Params,
                                                                                              ppSubThroughputNames)

const NVPW_MetricsContext_GetThroughputBreakdown_End_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetThroughputBreakdown_End_Params,
                                                                                            pMetricsContext)

const NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetMetricProperties_Begin_Params,
                                                                                           ppOptionalRawMetricDependencies)

const NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetMetricProperties_End_Params,
                                                                                         pMetricsContext)

const NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_SetCounterData_Params,
                                                                                isolated)

const NVPW_MetricsContext_SetUserData_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_SetUserData_Params,
                                                                             regionDuration)

const NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_EvaluateToGpuValues_Params,
                                                                                     pMetricValues)

const NVPW_MetricsContext_GetMetricSuffix_Begin_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetMetricSuffix_Begin_Params,
                                                                                       hidePctOfPeakSubMetricsOnThroughputs)

const NVPW_MetricsContext_GetMetricSuffix_End_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetMetricSuffix_End_Params,
                                                                                     pMetricsContext)

const NVPW_MetricsContext_GetMetricBaseNames_Begin_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetMetricBaseNames_Begin_Params,
                                                                                          ppMetricBaseNames)

const NVPW_MetricsContext_GetMetricBaseNames_End_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetMetricBaseNames_End_Params,
                                                                                        pMetricsContext)

const NVPW_MetricEvalRequest_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricEvalRequest,
                                                             submetric)

const NVPW_DimUnitFactor_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_DimUnitFactor, exponent)

const NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsEvaluator_Destroy_Params,
                                                                           pMetricsEvaluator)

const NVPW_MetricsEvaluator_GetMetricNames_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsEvaluator_GetMetricNames_Params,
                                                                                  numMetrics)

const NVPW_MetricsEvaluator_GetMetricTypeAndIndex_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsEvaluator_GetMetricTypeAndIndex_Params,
                                                                                         metricIndex)

const NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params,
                                                                                                        metricEvalRequestStructSize)

const NVPW_MetricsEvaluator_HwUnitToString_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsEvaluator_HwUnitToString_Params,
                                                                                  pHwUnitName)

const NVPW_MetricsEvaluator_GetCounterProperties_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsEvaluator_GetCounterProperties_Params,
                                                                                        hwUnit)

const NVPW_MetricsEvaluator_GetRatioMetricProperties_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsEvaluator_GetRatioMetricProperties_Params,
                                                                                            hwUnit)

const NVPW_MetricsEvaluator_GetThroughputMetricProperties_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsEvaluator_GetThroughputMetricProperties_Params,
                                                                                                 pSubThroughputIndices)

const NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params,
                                                                                          numSupportedSubmetrics)

const NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsEvaluator_GetMetricRawDependencies_Params,
                                                                                            numOptionalRawDependencies)

const NVPW_MetricsEvaluator_DimUnitToString_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsEvaluator_DimUnitToString_Params,
                                                                                   pPluralName)

const NVPW_MetricsEvaluator_GetMetricDimUnits_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsEvaluator_GetMetricDimUnits_Params,
                                                                                     dimUnitFactorStructSize)

const NVPW_MetricsEvaluator_SetUserData_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsEvaluator_SetUserData_Params,
                                                                               isolated)

const NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsEvaluator_EvaluateToGpuValues_Params,
                                                                                       pMetricValues)

const NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_MetricsEvaluator_SetDeviceAttributes_Params,
                                                                                       counterDataImageSize)

const NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_CUDA_MetricsContext_Create_Params,
                                                                             pMetricsContext)

const NVPW_CUDA_RawMetricsConfig_Create_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_CUDA_RawMetricsConfig_Create_Params,
                                                                               pRawMetricsConfig)

const NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_CUDA_RawMetricsConfig_Create_V2_Params,
                                                                                  pRawMetricsConfig)

const NVPW_CUDA_CounterDataBuilder_Create_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_CUDA_CounterDataBuilder_Create_Params,
                                                                                 pCounterDataBuilder)

const NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params,
                                                                                                   scratchBufferSize)

const NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE = @NVPA_STRUCT_SIZE(NVPW_CUDA_MetricsEvaluator_Initialize_Params,
                                                                                   pMetricsEvaluator)
