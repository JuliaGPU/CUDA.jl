# Julia wrapper for header: nvperf_host.h
# Automatically generated using Clang.jl


function NVPA_GetProcAddress(pFunctionName)
    @runtime_ccall((:NVPA_GetProcAddress, libnvperf[]), NVPA_GenericFn,
          (Cstring,),
          pFunctionName)
end

function NVPW_SetLibraryLoadPaths(pParams)
    @check @runtime_ccall((:NVPW_SetLibraryLoadPaths, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_SetLibraryLoadPaths_Params},),
                 pParams)
end

function NVPW_SetLibraryLoadPathsW(pParams)
    @check @runtime_ccall((:NVPW_SetLibraryLoadPathsW, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_SetLibraryLoadPathsW_Params},),
                 pParams)
end

function NVPA_InitializeHost()
    @check @runtime_ccall((:NVPA_InitializeHost, libnvperf[]), NVPA_Status, ())
end

function NVPW_InitializeHost(pParams)
    @check @runtime_ccall((:NVPW_InitializeHost, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_InitializeHost_Params},),
                 pParams)
end

function NVPA_CounterData_CalculateCounterDataImageCopySize(pCounterDataImageCopyOptions,
                                                            pCounterDataSrc,
                                                            pCopyDataImageCounterSize)
    @check @runtime_ccall((:NVPA_CounterData_CalculateCounterDataImageCopySize, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_CounterDataImageCopyOptions}, Ptr{UInt8}, Ptr{Csize_t}),
                 pCounterDataImageCopyOptions, pCounterDataSrc, pCopyDataImageCounterSize)
end

function NVPW_CounterData_CalculateCounterDataImageCopySize(pParams)
    @check @runtime_ccall((:NVPW_CounterData_CalculateCounterDataImageCopySize, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_CounterData_CalculateCounterDataImageCopySize_Params},),
                 pParams)
end

function NVPA_CounterData_InitializeCounterDataImageCopy(pCounterDataImageCopyOptions,
                                                         pCounterDataSrc, pCounterDataDst)
    @check @runtime_ccall((:NVPA_CounterData_InitializeCounterDataImageCopy, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_CounterDataImageCopyOptions}, Ptr{UInt8}, Ptr{UInt8}),
                 pCounterDataImageCopyOptions, pCounterDataSrc, pCounterDataDst)
end

function NVPW_CounterData_InitializeCounterDataImageCopy(pParams)
    @check @runtime_ccall((:NVPW_CounterData_InitializeCounterDataImageCopy, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_CounterData_InitializeCounterDataImageCopy_Params},),
                 pParams)
end

function NVPA_CounterDataCombiner_Create(pCounterDataCombinerOptions, ppCounterDataCombiner)
    @check @runtime_ccall((:NVPA_CounterDataCombiner_Create, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_CounterDataCombinerOptions}, Ptr{Ptr{NVPA_CounterDataCombiner}}),
                 pCounterDataCombinerOptions, ppCounterDataCombiner)
end

function NVPW_CounterDataCombiner_Create(pParams)
    @check @runtime_ccall((:NVPW_CounterDataCombiner_Create, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_CounterDataCombiner_Create_Params},),
                 pParams)
end

function NVPA_CounterDataCombiner_Destroy(pCounterDataCombiner)
    @check @runtime_ccall((:NVPA_CounterDataCombiner_Destroy, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_CounterDataCombiner},),
                 pCounterDataCombiner)
end

function NVPW_CounterDataCombiner_Destroy(pParams)
    @check @runtime_ccall((:NVPW_CounterDataCombiner_Destroy, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_CounterDataCombiner_Destroy_Params},),
                 pParams)
end

function NVPA_CounterDataCombiner_CreateRange(pCounterDataCombiner, numDescriptions,
                                              ppDescriptions, pRangeIndexDst)
    @check @runtime_ccall((:NVPA_CounterDataCombiner_CreateRange, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_CounterDataCombiner}, Csize_t, Ptr{Cstring}, Ptr{Csize_t}),
                 pCounterDataCombiner, numDescriptions, ppDescriptions, pRangeIndexDst)
end

function NVPW_CounterDataCombiner_CreateRange(pParams)
    @check @runtime_ccall((:NVPW_CounterDataCombiner_CreateRange, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_CounterDataCombiner_CreateRange_Params},),
                 pParams)
end

function NVPA_CounterDataCombiner_AccumulateIntoRange(pCounterDataCombiner, rangeIndexDst,
                                                      dstMultiplier, pCounterDataSrc,
                                                      rangeIndexSrc, srcMultiplier)
    @check @runtime_ccall((:NVPA_CounterDataCombiner_AccumulateIntoRange, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_CounterDataCombiner}, Csize_t, UInt32, Ptr{UInt8}, Csize_t,
                  UInt32),
                 pCounterDataCombiner, rangeIndexDst, dstMultiplier, pCounterDataSrc,
                 rangeIndexSrc, srcMultiplier)
end

function NVPW_CounterDataCombiner_AccumulateIntoRange(pParams)
    @check @runtime_ccall((:NVPW_CounterDataCombiner_AccumulateIntoRange, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_CounterDataCombiner_AccumulateIntoRange_Params},),
                 pParams)
end

function NVPW_CounterDataCombiner_SumIntoRange(pParams)
    @check @runtime_ccall((:NVPW_CounterDataCombiner_SumIntoRange, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_CounterDataCombiner_SumIntoRange_Params},),
                 pParams)
end

function NVPW_CounterDataCombiner_WeightedSumIntoRange(pParams)
    @check @runtime_ccall((:NVPW_CounterDataCombiner_WeightedSumIntoRange, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_CounterDataCombiner_WeightedSumIntoRange_Params},),
                 pParams)
end

function NVPA_GetSupportedChipNames(pSupportedChipNames)
    @check @runtime_ccall((:NVPA_GetSupportedChipNames, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_SupportedChipNames},),
                 pSupportedChipNames)
end

function NVPW_GetSupportedChipNames(pParams)
    @check @runtime_ccall((:NVPW_GetSupportedChipNames, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_GetSupportedChipNames_Params},),
                 pParams)
end

function NVPA_RawMetricsConfig_Create(pMetricsConfigOptions, ppRawMetricsConfig)
    @check @runtime_ccall((:NVPA_RawMetricsConfig_Create, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_RawMetricsConfigOptions}, Ptr{Ptr{NVPA_RawMetricsConfig}}),
                 pMetricsConfigOptions, ppRawMetricsConfig)
end

function NVPA_RawMetricsConfig_Destroy(pRawMetricsConfig)
    @check @runtime_ccall((:NVPA_RawMetricsConfig_Destroy, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_RawMetricsConfig},),
                 pRawMetricsConfig)
end

function NVPW_RawMetricsConfig_Destroy(pParams)
    @check @runtime_ccall((:NVPW_RawMetricsConfig_Destroy, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_RawMetricsConfig_Destroy_Params},),
                 pParams)
end

function NVPA_RawMetricsConfig_BeginPassGroup(pRawMetricsConfig, pRawMetricsPassGroupOptions)
    @check @runtime_ccall((:NVPA_RawMetricsConfig_BeginPassGroup, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_RawMetricsConfig}, Ptr{NVPA_RawMetricsPassGroupOptions}),
                 pRawMetricsConfig, pRawMetricsPassGroupOptions)
end

function NVPW_RawMetricsConfig_BeginPassGroup(pParams)
    @check @runtime_ccall((:NVPW_RawMetricsConfig_BeginPassGroup, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_RawMetricsConfig_BeginPassGroup_Params},),
                 pParams)
end

function NVPA_RawMetricsConfig_EndPassGroup(pRawMetricsConfig)
    @check @runtime_ccall((:NVPA_RawMetricsConfig_EndPassGroup, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_RawMetricsConfig},),
                 pRawMetricsConfig)
end

function NVPW_RawMetricsConfig_EndPassGroup(pParams)
    @check @runtime_ccall((:NVPW_RawMetricsConfig_EndPassGroup, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_RawMetricsConfig_EndPassGroup_Params},),
                 pParams)
end

function NVPA_RawMetricsConfig_GetNumMetrics(pRawMetricsConfig, pNumMetrics)
    @check @runtime_ccall((:NVPA_RawMetricsConfig_GetNumMetrics, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_RawMetricsConfig}, Ptr{Csize_t}),
                 pRawMetricsConfig, pNumMetrics)
end

function NVPW_RawMetricsConfig_GetNumMetrics(pParams)
    @check @runtime_ccall((:NVPW_RawMetricsConfig_GetNumMetrics, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_RawMetricsConfig_GetNumMetrics_Params},),
                 pParams)
end

function NVPA_RawMetricsConfig_GetMetricProperties(pRawMetricsConfig, metricIndex,
                                                   pRawMetricProperties)
    @check @runtime_ccall((:NVPA_RawMetricsConfig_GetMetricProperties, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_RawMetricsConfig}, Csize_t, Ptr{NVPA_RawMetricProperties}),
                 pRawMetricsConfig, metricIndex, pRawMetricProperties)
end

function NVPW_RawMetricsConfig_GetMetricProperties(pParams)
    @check @runtime_ccall((:NVPW_RawMetricsConfig_GetMetricProperties, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_RawMetricsConfig_GetMetricProperties_Params},),
                 pParams)
end

function NVPA_RawMetricsConfig_AddMetrics(pRawMetricsConfig, pRawMetricRequests,
                                          numMetricRequests)
    @check @runtime_ccall((:NVPA_RawMetricsConfig_AddMetrics, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_RawMetricsConfig}, Ptr{NVPA_RawMetricRequest}, Csize_t),
                 pRawMetricsConfig, pRawMetricRequests, numMetricRequests)
end

function NVPW_RawMetricsConfig_AddMetrics(pParams)
    @check @runtime_ccall((:NVPW_RawMetricsConfig_AddMetrics, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_RawMetricsConfig_AddMetrics_Params},),
                 pParams)
end

function NVPA_RawMetricsConfig_IsAddMetricsPossible(pRawMetricsConfig, pRawMetricRequests,
                                                    numMetricRequests, pIsPossible)
    @check @runtime_ccall((:NVPA_RawMetricsConfig_IsAddMetricsPossible, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_RawMetricsConfig}, Ptr{NVPA_RawMetricRequest}, Csize_t,
                  Ptr{NVPA_Bool}),
                 pRawMetricsConfig, pRawMetricRequests, numMetricRequests, pIsPossible)
end

function NVPW_RawMetricsConfig_IsAddMetricsPossible(pParams)
    @check @runtime_ccall((:NVPW_RawMetricsConfig_IsAddMetricsPossible, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_RawMetricsConfig_IsAddMetricsPossible_Params},),
                 pParams)
end

function NVPA_RawMetricsConfig_GenerateConfigImage(pRawMetricsConfig)
    @check @runtime_ccall((:NVPA_RawMetricsConfig_GenerateConfigImage, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_RawMetricsConfig},),
                 pRawMetricsConfig)
end

function NVPW_RawMetricsConfig_GenerateConfigImage(pParams)
    @check @runtime_ccall((:NVPW_RawMetricsConfig_GenerateConfigImage, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_RawMetricsConfig_GenerateConfigImage_Params},),
                 pParams)
end

function NVPA_RawMetricsConfig_GetConfigImage(pRawMetricsConfig, bufferSize, pBuffer,
                                              pBufferSize)
    @check @runtime_ccall((:NVPA_RawMetricsConfig_GetConfigImage, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_RawMetricsConfig}, Csize_t, Ptr{UInt8}, Ptr{Csize_t}),
                 pRawMetricsConfig, bufferSize, pBuffer, pBufferSize)
end

function NVPW_RawMetricsConfig_GetConfigImage(pParams)
    @check @runtime_ccall((:NVPW_RawMetricsConfig_GetConfigImage, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_RawMetricsConfig_GetConfigImage_Params},),
                 pParams)
end

function NVPA_RawMetricsConfig_GetNumPasses(pRawMetricsConfig, pNumPipelinedPasses,
                                            pNumIsolatedPasses)
    @check @runtime_ccall((:NVPA_RawMetricsConfig_GetNumPasses, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_RawMetricsConfig}, Ptr{Csize_t}, Ptr{Csize_t}),
                 pRawMetricsConfig, pNumPipelinedPasses, pNumIsolatedPasses)
end

function NVPW_RawMetricsConfig_GetNumPasses(pParams)
    @check @runtime_ccall((:NVPW_RawMetricsConfig_GetNumPasses, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_RawMetricsConfig_GetNumPasses_Params},),
                 pParams)
end

function NVPA_CounterDataBuilder_Create(pOptions, ppCounterDataBuilder)
    @check @runtime_ccall((:NVPA_CounterDataBuilder_Create, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_CounterDataBuilderOptions}, Ptr{Ptr{NVPA_CounterDataBuilder}}),
                 pOptions, ppCounterDataBuilder)
end

function NVPW_CounterDataBuilder_Create(pParams)
    @check @runtime_ccall((:NVPW_CounterDataBuilder_Create, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_CounterDataBuilder_Create_Params},),
                 pParams)
end

function NVPA_CounterDataBuilder_Destroy(pCounterDataBuilder)
    @check @runtime_ccall((:NVPA_CounterDataBuilder_Destroy, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_CounterDataBuilder},),
                 pCounterDataBuilder)
end

function NVPW_CounterDataBuilder_Destroy(pParams)
    @check @runtime_ccall((:NVPW_CounterDataBuilder_Destroy, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_CounterDataBuilder_Destroy_Params},),
                 pParams)
end

function NVPA_CounterDataBuilder_AddMetrics(pCounterDataBuilder, pRawMetricRequests,
                                            numMetricRequests)
    @check @runtime_ccall((:NVPA_CounterDataBuilder_AddMetrics, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_CounterDataBuilder}, Ptr{NVPA_RawMetricRequest}, Csize_t),
                 pCounterDataBuilder, pRawMetricRequests, numMetricRequests)
end

function NVPW_CounterDataBuilder_AddMetrics(pParams)
    @check @runtime_ccall((:NVPW_CounterDataBuilder_AddMetrics, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_CounterDataBuilder_AddMetrics_Params},),
                 pParams)
end

function NVPA_CounterDataBuilder_GetCounterDataPrefix(pCounterDataBuilder, bufferSize,
                                                      pBuffer, pBufferSize)
    @check @runtime_ccall((:NVPA_CounterDataBuilder_GetCounterDataPrefix, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_CounterDataBuilder}, Csize_t, Ptr{UInt8}, Ptr{Csize_t}),
                 pCounterDataBuilder, bufferSize, pBuffer, pBufferSize)
end

function NVPW_CounterDataBuilder_GetCounterDataPrefix(pParams)
    @check @runtime_ccall((:NVPW_CounterDataBuilder_GetCounterDataPrefix, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_CounterDataBuilder_GetCounterDataPrefix_Params},),
                 pParams)
end

function NVPA_MetricsContext_Create(pMetricsContextOptions, ppMetricsContext)
    @check @runtime_ccall((:NVPA_MetricsContext_Create, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_MetricsContextOptions}, Ptr{Ptr{NVPA_MetricsContext}}),
                 pMetricsContextOptions, ppMetricsContext)
end

function NVPA_MetricsContext_Destroy(pMetricsContext)
    @check @runtime_ccall((:NVPA_MetricsContext_Destroy, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_MetricsContext},),
                 pMetricsContext)
end

function NVPW_MetricsContext_Destroy(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_Destroy, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_Destroy_Params},),
                 pParams)
end

function NVPA_MetricsContext_RunScript(pMetricsContext, pOptions)
    @check @runtime_ccall((:NVPA_MetricsContext_RunScript, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_MetricsContext}, Ptr{NVPA_MetricsScriptOptions}),
                 pMetricsContext, pOptions)
end

function NVPW_MetricsContext_RunScript(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_RunScript, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_RunScript_Params},),
                 pParams)
end

function NVPA_MetricsContext_ExecScript_Begin(pMetricsContext, pOptions)
    @check @runtime_ccall((:NVPA_MetricsContext_ExecScript_Begin, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_MetricsContext}, Ptr{NVPA_MetricsExecOptions}),
                 pMetricsContext, pOptions)
end

function NVPW_MetricsContext_ExecScript_Begin(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_ExecScript_Begin, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_ExecScript_Begin_Params},),
                 pParams)
end

function NVPA_MetricsContext_ExecScript_End(pMetricsContext)
    @check @runtime_ccall((:NVPA_MetricsContext_ExecScript_End, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_MetricsContext},),
                 pMetricsContext)
end

function NVPW_MetricsContext_ExecScript_End(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_ExecScript_End, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_ExecScript_End_Params},),
                 pParams)
end

function NVPA_MetricsContext_GetCounterNames_Begin(pMetricsContext, pNumCounters,
                                                   pppCounterNames)
    @check @runtime_ccall((:NVPA_MetricsContext_GetCounterNames_Begin, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_MetricsContext}, Ptr{Csize_t}, Ptr{Ptr{Cstring}}),
                 pMetricsContext, pNumCounters, pppCounterNames)
end

function NVPW_MetricsContext_GetCounterNames_Begin(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_GetCounterNames_Begin, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_GetCounterNames_Begin_Params},),
                 pParams)
end

function NVPA_MetricsContext_GetCounterNames_End(pMetricsContext)
    @check @runtime_ccall((:NVPA_MetricsContext_GetCounterNames_End, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_MetricsContext},),
                 pMetricsContext)
end

function NVPW_MetricsContext_GetCounterNames_End(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_GetCounterNames_End, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_GetCounterNames_End_Params},),
                 pParams)
end

function NVPA_MetricsContext_GetThroughputNames_Begin(pMetricsContext, pNumThroughputs,
                                                      pppThroughputName)
    @check @runtime_ccall((:NVPA_MetricsContext_GetThroughputNames_Begin, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_MetricsContext}, Ptr{Csize_t}, Ptr{Ptr{Cstring}}),
                 pMetricsContext, pNumThroughputs, pppThroughputName)
end

function NVPW_MetricsContext_GetThroughputNames_Begin(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_GetThroughputNames_Begin, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_GetThroughputNames_Begin_Params},),
                 pParams)
end

function NVPA_MetricsContext_GetThroughputNames_End(pMetricsContext)
    @check @runtime_ccall((:NVPA_MetricsContext_GetThroughputNames_End, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_MetricsContext},),
                 pMetricsContext)
end

function NVPW_MetricsContext_GetThroughputNames_End(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_GetThroughputNames_End, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_GetThroughputNames_End_Params},),
                 pParams)
end

function NVPW_MetricsContext_GetRatioNames_Begin(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_GetRatioNames_Begin, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_GetRatioNames_Begin_Params},),
                 pParams)
end

function NVPW_MetricsContext_GetRatioNames_End(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_GetRatioNames_End, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_GetRatioNames_End_Params},),
                 pParams)
end

function NVPA_MetricsContext_GetMetricNames_Begin(pMetricsContext, pOptions)
    @check @runtime_ccall((:NVPA_MetricsContext_GetMetricNames_Begin, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_MetricsContext}, Ptr{NVPA_MetricsEnumerationOptions}),
                 pMetricsContext, pOptions)
end

function NVPW_MetricsContext_GetMetricNames_Begin(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_GetMetricNames_Begin, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_GetMetricNames_Begin_Params},),
                 pParams)
end

function NVPA_MetricsContext_GetMetricNames_End(pMetricsContext)
    @check @runtime_ccall((:NVPA_MetricsContext_GetMetricNames_End, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_MetricsContext},),
                 pMetricsContext)
end

function NVPW_MetricsContext_GetMetricNames_End(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_GetMetricNames_End, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_GetMetricNames_End_Params},),
                 pParams)
end

function NVPA_MetricsContext_GetThroughputBreakdown_Begin(pMetricsContext, pThroughputName,
                                                          pppCounterNames,
                                                          pppSubThroughputNames)
    @check @runtime_ccall((:NVPA_MetricsContext_GetThroughputBreakdown_Begin, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_MetricsContext}, Cstring, Ptr{Ptr{Cstring}}, Ptr{Ptr{Cstring}}),
                 pMetricsContext, pThroughputName, pppCounterNames, pppSubThroughputNames)
end

function NVPW_MetricsContext_GetThroughputBreakdown_Begin(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_GetThroughputBreakdown_Begin, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_GetThroughputBreakdown_Begin_Params},),
                 pParams)
end

function NVPA_MetricsContext_GetThroughputBreakdown_End(pMetricsContext)
    @check @runtime_ccall((:NVPA_MetricsContext_GetThroughputBreakdown_End, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_MetricsContext},),
                 pMetricsContext)
end

function NVPW_MetricsContext_GetThroughputBreakdown_End(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_GetThroughputBreakdown_End, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_GetThroughputBreakdown_End_Params},),
                 pParams)
end

function NVPA_MetricsContext_GetMetricProperties_Begin(pMetricsContext, pMetricName,
                                                       pMetricProperties)
    @check @runtime_ccall((:NVPA_MetricsContext_GetMetricProperties_Begin, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_MetricsContext}, Cstring, Ptr{NVPA_MetricProperties}),
                 pMetricsContext, pMetricName, pMetricProperties)
end

function NVPW_MetricsContext_GetMetricProperties_Begin(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_GetMetricProperties_Begin, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_GetMetricProperties_Begin_Params},),
                 pParams)
end

function NVPA_MetricsContext_GetMetricProperties_End(pMetricsContext)
    @check @runtime_ccall((:NVPA_MetricsContext_GetMetricProperties_End, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_MetricsContext},),
                 pMetricsContext)
end

function NVPW_MetricsContext_GetMetricProperties_End(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_GetMetricProperties_End, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_GetMetricProperties_End_Params},),
                 pParams)
end

function NVPA_MetricsContext_SetCounterData(pMetricsContext, pCounterDataImage, rangeIndex,
                                            isolated)
    @check @runtime_ccall((:NVPA_MetricsContext_SetCounterData, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_MetricsContext}, Ptr{UInt8}, Csize_t, NVPA_Bool),
                 pMetricsContext, pCounterDataImage, rangeIndex, isolated)
end

function NVPW_MetricsContext_SetCounterData(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_SetCounterData, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_SetCounterData_Params},),
                 pParams)
end

function NVPA_MetricsContext_SetUserData(pMetricsContext, pMetricUserData)
    @check @runtime_ccall((:NVPA_MetricsContext_SetUserData, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_MetricsContext}, Ptr{NVPA_MetricUserData}),
                 pMetricsContext, pMetricUserData)
end

function NVPW_MetricsContext_SetUserData(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_SetUserData, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_SetUserData_Params},),
                 pParams)
end

function NVPA_MetricsContext_EvaluateToGpuValues(pMetricsContext, numMetrics,
                                                 ppMetricNames, pMetricValues)
    @check @runtime_ccall((:NVPA_MetricsContext_EvaluateToGpuValues, libnvperf[]), NVPA_Status,
                 (Ptr{NVPA_MetricsContext}, Csize_t, Ptr{Cstring}, Ptr{Cdouble}),
                 pMetricsContext, numMetrics, ppMetricNames, pMetricValues)
end

function NVPW_MetricsContext_EvaluateToGpuValues(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_EvaluateToGpuValues, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_EvaluateToGpuValues_Params},),
                 pParams)
end

function NVPW_MetricsContext_GetMetricSuffix_Begin(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_GetMetricSuffix_Begin, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_GetMetricSuffix_Begin_Params},),
                 pParams)
end

function NVPW_MetricsContext_GetMetricSuffix_End(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_GetMetricSuffix_End, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_GetMetricSuffix_End_Params},),
                 pParams)
end

function NVPW_MetricsContext_GetMetricBaseNames_Begin(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_GetMetricBaseNames_Begin, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_GetMetricBaseNames_Begin_Params},),
                 pParams)
end

function NVPW_MetricsContext_GetMetricBaseNames_End(pParams)
    @check @runtime_ccall((:NVPW_MetricsContext_GetMetricBaseNames_End, libnvperf[]), NVPA_Status,
                 (Ptr{NVPW_MetricsContext_GetMetricBaseNames_End_Params},),
                 pParams)
end
