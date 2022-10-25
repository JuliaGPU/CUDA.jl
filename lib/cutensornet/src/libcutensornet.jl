using CEnum

# CUTENSORNET uses CUDA runtime objects, which are compatible with our driver usage
const cudaStream_t = CUstream

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == CUTENSORNET_STATUS_ALLOC_FAILED
        throw(OutOfGPUMemoryError())
    else
        throw(CUTENSORNETError(res))
    end
end

macro check(ex, errs...)
    check = :(isequal(err, CUTENSORNET_STATUS_ALLOC_FAILED))
    for err in errs
        check = :($check || isequal(err, $(esc(err))))
    end

    quote
        res = @retry_reclaim err -> $check $(esc(ex))
        if res != CUTENSORNET_STATUS_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end

@cenum cutensornetStatus_t::UInt32 begin
    CUTENSORNET_STATUS_SUCCESS = 0
    CUTENSORNET_STATUS_NOT_INITIALIZED = 1
    CUTENSORNET_STATUS_ALLOC_FAILED = 3
    CUTENSORNET_STATUS_INVALID_VALUE = 7
    CUTENSORNET_STATUS_ARCH_MISMATCH = 8
    CUTENSORNET_STATUS_MAPPING_ERROR = 11
    CUTENSORNET_STATUS_EXECUTION_FAILED = 13
    CUTENSORNET_STATUS_INTERNAL_ERROR = 14
    CUTENSORNET_STATUS_NOT_SUPPORTED = 15
    CUTENSORNET_STATUS_LICENSE_ERROR = 16
    CUTENSORNET_STATUS_CUBLAS_ERROR = 17
    CUTENSORNET_STATUS_CUDA_ERROR = 18
    CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE = 19
    CUTENSORNET_STATUS_INSUFFICIENT_DRIVER = 20
    CUTENSORNET_STATUS_IO_ERROR = 21
    CUTENSORNET_STATUS_CUTENSOR_VERSION_MISMATCH = 22
    CUTENSORNET_STATUS_NO_DEVICE_ALLOCATOR = 23
    CUTENSORNET_STATUS_ALL_HYPER_SAMPLES_FAILED = 24
end

@cenum cutensornetComputeType_t::UInt32 begin
    CUTENSORNET_COMPUTE_16F = 1
    CUTENSORNET_COMPUTE_16BF = 1024
    CUTENSORNET_COMPUTE_TF32 = 4096
    CUTENSORNET_COMPUTE_32F = 4
    CUTENSORNET_COMPUTE_64F = 16
    CUTENSORNET_COMPUTE_8U = 64
    CUTENSORNET_COMPUTE_8I = 256
    CUTENSORNET_COMPUTE_32U = 128
    CUTENSORNET_COMPUTE_32I = 512
end

@cenum cutensornetGraphAlgo_t::UInt32 begin
    CUTENSORNET_GRAPH_ALGO_RB = 0
    CUTENSORNET_GRAPH_ALGO_KWAY = 1
end

@cenum cutensornetMemoryModel_t::UInt32 begin
    CUTENSORNET_MEMORY_MODEL_HEURISTIC = 0
    CUTENSORNET_MEMORY_MODEL_CUTENSOR = 1
end

@cenum cutensornetOptimizerCost_t::UInt32 begin
    CUTENSORNET_OPTIMIZER_COST_FLOPS = 0
    CUTENSORNET_OPTIMIZER_COST_TIME = 1
    CUTENSORNET_OPTIMIZER_COST_TIME_TUNED = 2
end

@cenum cutensornetContractionOptimizerConfigAttributes_t::UInt32 begin
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_PARTITIONS = 0
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_CUTOFF_SIZE = 1
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM = 2
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_IMBALANCE_FACTOR = 3
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_ITERATIONS = 4
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_CUTS = 5
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS = 6
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_LEAVES = 7
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING = 8
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL = 9
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_FACTOR = 10
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES = 11
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_SLICE_FACTOR = 12
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES = 13
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_THREADS = 16
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SIMPLIFICATION_DISABLE_DR = 14
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED = 15
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE = 18
end

@cenum cutensornetContractionOptimizerInfoAttributes_t::UInt32 begin
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES = 0
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES = 1
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_MODE = 2
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_EXTENT = 3
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH = 4
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PHASE1_FLOP_COUNT = 5
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT = 6
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_LARGEST_TENSOR = 7
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_OVERHEAD = 8
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_INTERMEDIATE_MODES = 9
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_INTERMEDIATE_MODES = 10
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_EFFECTIVE_FLOPS_EST = 11
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_RUNTIME_EST = 12
end

@cenum cutensornetContractionAutotunePreferenceAttributes_t::UInt32 begin
    CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS = 0
    CUTENSORNET_CONTRACTION_AUTOTUNE_INTERMEDIATE_MODES = 1
end

const cutensornetNetworkDescriptor_t = Ptr{Cvoid}

const cutensornetContractionPlan_t = Ptr{Cvoid}

const cutensornetHandle_t = Ptr{Cvoid}

const cutensornetWorkspaceDescriptor_t = Ptr{Cvoid}

@cenum cutensornetWorksizePref_t::UInt32 begin
    CUTENSORNET_WORKSIZE_PREF_MIN = 0
    CUTENSORNET_WORKSIZE_PREF_RECOMMENDED = 1
    CUTENSORNET_WORKSIZE_PREF_MAX = 2
end

@cenum cutensornetMemspace_t::UInt32 begin
    CUTENSORNET_MEMSPACE_DEVICE = 0
end

struct cutensornetNodePair_t
    first::Int32
    second::Int32
end

struct cutensornetContractionPath_t
    numContractions::Int32
    data::Ptr{cutensornetNodePair_t}
end

const cutensornetContractionOptimizerConfig_t = Ptr{Cvoid}

const cutensornetContractionOptimizerInfo_t = Ptr{Cvoid}

const cutensornetContractionAutotunePreference_t = Ptr{Cvoid}

const cutensornetSliceGroup_t = Ptr{Cvoid}

struct cutensornetDeviceMemHandler_t
    ctx::Ptr{Cvoid}
    device_alloc::Ptr{Cvoid}
    device_free::Ptr{Cvoid}
    name::NTuple{64,Cchar}
end

# typedef void ( * cutensornetLoggerCallback_t ) ( int32_t logLevel , const char * functionName , const char * message )
const cutensornetLoggerCallback_t = Ptr{Cvoid}

# typedef void ( * cutensornetLoggerCallbackData_t ) ( int32_t logLevel , const char * functionName , const char * message , void * userData )
const cutensornetLoggerCallbackData_t = Ptr{Cvoid}

@checked function cutensornetCreate(handle)
    initialize_context()
    ccall((:cutensornetCreate, libcutensornet), cutensornetStatus_t,
          (Ptr{cutensornetHandle_t},), handle)
end

@checked function cutensornetDestroy(handle)
    initialize_context()
    ccall((:cutensornetDestroy, libcutensornet), cutensornetStatus_t,
          (cutensornetHandle_t,), handle)
end

@checked function cutensornetCreateNetworkDescriptor(handle, numInputs, numModesIn,
                                                     extentsIn, stridesIn, modesIn,
                                                     alignmentRequirementsIn, numModesOut,
                                                     extentsOut, stridesOut, modesOut,
                                                     alignmentRequirementsOut, dataType,
                                                     computeType, descNet)
    initialize_context()
    ccall((:cutensornetCreateNetworkDescriptor, libcutensornet), cutensornetStatus_t,
          (cutensornetHandle_t, Int32, Ptr{Int32}, Ptr{Ptr{Int64}}, Ptr{Ptr{Int64}},
           Ptr{Ptr{Int32}}, Ptr{UInt32}, Int32, Ptr{Int64}, Ptr{Int64}, Ptr{Int32}, UInt32,
           cudaDataType_t, cutensornetComputeType_t, Ptr{cutensornetNetworkDescriptor_t}),
          handle, numInputs, numModesIn, extentsIn, stridesIn, modesIn,
          alignmentRequirementsIn, numModesOut, extentsOut, stridesOut, modesOut,
          alignmentRequirementsOut, dataType, computeType, descNet)
end

@checked function cutensornetDestroyNetworkDescriptor(desc)
    initialize_context()
    ccall((:cutensornetDestroyNetworkDescriptor, libcutensornet), cutensornetStatus_t,
          (cutensornetNetworkDescriptor_t,), desc)
end

@checked function cutensornetGetOutputTensorDetails(handle, descNet, numModesOut,
                                                    dataSizeOut, modeLabelsOut, extentsOut,
                                                    stridesOut)
    initialize_context()
    ccall((:cutensornetGetOutputTensorDetails, libcutensornet), cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetNetworkDescriptor_t, Ptr{Int32}, Ptr{Csize_t},
           Ptr{Int32}, Ptr{Int64}, Ptr{Int64}), handle, descNet, numModesOut, dataSizeOut,
          modeLabelsOut, extentsOut, stridesOut)
end

@checked function cutensornetCreateWorkspaceDescriptor(handle, workDesc)
    initialize_context()
    ccall((:cutensornetCreateWorkspaceDescriptor, libcutensornet), cutensornetStatus_t,
          (cutensornetHandle_t, Ptr{cutensornetWorkspaceDescriptor_t}), handle, workDesc)
end

@checked function cutensornetWorkspaceComputeSizes(handle, descNet, optimizerInfo, workDesc)
    initialize_context()
    ccall((:cutensornetWorkspaceComputeSizes, libcutensornet), cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetNetworkDescriptor_t,
           cutensornetContractionOptimizerInfo_t, cutensornetWorkspaceDescriptor_t), handle,
          descNet, optimizerInfo, workDesc)
end

@checked function cutensornetWorkspaceGetSize(handle, workDesc, workPref, memSpace,
                                              workspaceSize)
    initialize_context()
    ccall((:cutensornetWorkspaceGetSize, libcutensornet), cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetWorkspaceDescriptor_t, cutensornetWorksizePref_t,
           cutensornetMemspace_t, Ptr{UInt64}), handle, workDesc, workPref, memSpace,
          workspaceSize)
end

@checked function cutensornetWorkspaceSet(handle, workDesc, memSpace, workspacePtr,
                                          workspaceSize)
    initialize_context()
    ccall((:cutensornetWorkspaceSet, libcutensornet), cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetWorkspaceDescriptor_t, cutensornetMemspace_t,
           PtrOrCuPtr{Cvoid}, UInt64), handle, workDesc, memSpace, workspacePtr,
          workspaceSize)
end

@checked function cutensornetWorkspaceGet(handle, workDesc, memSpace, workspacePtr,
                                          workspaceSize)
    initialize_context()
    ccall((:cutensornetWorkspaceGet, libcutensornet), cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetWorkspaceDescriptor_t, cutensornetMemspace_t,
           Ptr{Ptr{Cvoid}}, Ptr{UInt64}), handle, workDesc, memSpace, workspacePtr,
          workspaceSize)
end

@checked function cutensornetDestroyWorkspaceDescriptor(desc)
    initialize_context()
    ccall((:cutensornetDestroyWorkspaceDescriptor, libcutensornet), cutensornetStatus_t,
          (cutensornetWorkspaceDescriptor_t,), desc)
end

@checked function cutensornetCreateContractionOptimizerConfig(handle, optimizerConfig)
    initialize_context()
    ccall((:cutensornetCreateContractionOptimizerConfig, libcutensornet),
          cutensornetStatus_t,
          (cutensornetHandle_t, Ptr{cutensornetContractionOptimizerConfig_t}), handle,
          optimizerConfig)
end

@checked function cutensornetDestroyContractionOptimizerConfig(optimizerConfig)
    initialize_context()
    ccall((:cutensornetDestroyContractionOptimizerConfig, libcutensornet),
          cutensornetStatus_t, (cutensornetContractionOptimizerConfig_t,), optimizerConfig)
end

@checked function cutensornetContractionOptimizerConfigGetAttribute(handle, optimizerConfig,
                                                                    attr, buf, sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionOptimizerConfigGetAttribute, libcutensornet),
          cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetContractionOptimizerConfig_t,
           cutensornetContractionOptimizerConfigAttributes_t, Ptr{Cvoid}, Csize_t), handle,
          optimizerConfig, attr, buf, sizeInBytes)
end

@checked function cutensornetContractionOptimizerConfigSetAttribute(handle, optimizerConfig,
                                                                    attr, buf, sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionOptimizerConfigSetAttribute, libcutensornet),
          cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetContractionOptimizerConfig_t,
           cutensornetContractionOptimizerConfigAttributes_t, Ptr{Cvoid}, Csize_t), handle,
          optimizerConfig, attr, buf, sizeInBytes)
end

@checked function cutensornetDestroyContractionOptimizerInfo(optimizerInfo)
    initialize_context()
    ccall((:cutensornetDestroyContractionOptimizerInfo, libcutensornet),
          cutensornetStatus_t, (cutensornetContractionOptimizerInfo_t,), optimizerInfo)
end

@checked function cutensornetCreateContractionOptimizerInfo(handle, descNet, optimizerInfo)
    initialize_context()
    ccall((:cutensornetCreateContractionOptimizerInfo, libcutensornet), cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetNetworkDescriptor_t,
           Ptr{cutensornetContractionOptimizerInfo_t}), handle, descNet, optimizerInfo)
end

@checked function cutensornetContractionOptimize(handle, descNet, optimizerConfig,
                                                 workspaceSizeConstraint, optimizerInfo)
    initialize_context()
    ccall((:cutensornetContractionOptimize, libcutensornet), cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetNetworkDescriptor_t,
           cutensornetContractionOptimizerConfig_t, UInt64,
           cutensornetContractionOptimizerInfo_t), handle, descNet, optimizerConfig,
          workspaceSizeConstraint, optimizerInfo)
end

@checked function cutensornetContractionOptimizerInfoGetAttribute(handle, optimizerInfo,
                                                                  attr, buf, sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionOptimizerInfoGetAttribute, libcutensornet),
          cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetContractionOptimizerInfo_t,
           cutensornetContractionOptimizerInfoAttributes_t, Ptr{Cvoid}, Csize_t), handle,
          optimizerInfo, attr, buf, sizeInBytes)
end

@checked function cutensornetContractionOptimizerInfoSetAttribute(handle, optimizerInfo,
                                                                  attr, buf, sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionOptimizerInfoSetAttribute, libcutensornet),
          cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetContractionOptimizerInfo_t,
           cutensornetContractionOptimizerInfoAttributes_t, Ptr{Cvoid}, Csize_t), handle,
          optimizerInfo, attr, buf, sizeInBytes)
end

@checked function cutensornetContractionOptimizerInfoGetPackedSize(handle, optimizerInfo,
                                                                   sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionOptimizerInfoGetPackedSize, libcutensornet),
          cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetContractionOptimizerInfo_t, Ptr{Csize_t}),
          handle, optimizerInfo, sizeInBytes)
end

@checked function cutensornetContractionOptimizerInfoPackData(handle, optimizerInfo, buffer,
                                                              sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionOptimizerInfoPackData, libcutensornet),
          cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetContractionOptimizerInfo_t, Ptr{Cvoid}, Csize_t),
          handle, optimizerInfo, buffer, sizeInBytes)
end

@checked function cutensornetCreateContractionOptimizerInfoFromPackedData(handle, descNet,
                                                                          buffer,
                                                                          sizeInBytes,
                                                                          optimizerInfo)
    initialize_context()
    ccall((:cutensornetCreateContractionOptimizerInfoFromPackedData, libcutensornet),
          cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetNetworkDescriptor_t, Ptr{Cvoid}, Csize_t,
           Ptr{cutensornetContractionOptimizerInfo_t}), handle, descNet, buffer,
          sizeInBytes, optimizerInfo)
end

@checked function cutensornetUpdateContractionOptimizerInfoFromPackedData(handle, buffer,
                                                                          sizeInBytes,
                                                                          optimizerInfo)
    initialize_context()
    ccall((:cutensornetUpdateContractionOptimizerInfoFromPackedData, libcutensornet),
          cutensornetStatus_t,
          (cutensornetHandle_t, Ptr{Cvoid}, Csize_t, cutensornetContractionOptimizerInfo_t),
          handle, buffer, sizeInBytes, optimizerInfo)
end

@checked function cutensornetCreateContractionPlan(handle, descNet, optimizerInfo, workDesc,
                                                   plan)
    initialize_context()
    ccall((:cutensornetCreateContractionPlan, libcutensornet), cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetNetworkDescriptor_t,
           cutensornetContractionOptimizerInfo_t, cutensornetWorkspaceDescriptor_t,
           Ptr{cutensornetContractionPlan_t}), handle, descNet, optimizerInfo, workDesc,
          plan)
end

@checked function cutensornetDestroyContractionPlan(plan)
    initialize_context()
    ccall((:cutensornetDestroyContractionPlan, libcutensornet), cutensornetStatus_t,
          (cutensornetContractionPlan_t,), plan)
end

@checked function cutensornetContractionAutotune(handle, plan, rawDataIn, rawDataOut,
                                                 workDesc, pref, stream)
    initialize_context()
    ccall((:cutensornetContractionAutotune, libcutensornet), cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetContractionPlan_t, Ptr{CuPtr{Cvoid}},
           CuPtr{Cvoid}, cutensornetWorkspaceDescriptor_t,
           cutensornetContractionAutotunePreference_t, cudaStream_t), handle, plan,
          rawDataIn, rawDataOut, workDesc, pref, stream)
end

@checked function cutensornetCreateContractionAutotunePreference(handle, autotunePreference)
    initialize_context()
    ccall((:cutensornetCreateContractionAutotunePreference, libcutensornet),
          cutensornetStatus_t,
          (cutensornetHandle_t, Ptr{cutensornetContractionAutotunePreference_t}), handle,
          autotunePreference)
end

@checked function cutensornetContractionAutotunePreferenceGetAttribute(handle,
                                                                       autotunePreference,
                                                                       attr, buf,
                                                                       sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionAutotunePreferenceGetAttribute, libcutensornet),
          cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetContractionAutotunePreference_t,
           cutensornetContractionAutotunePreferenceAttributes_t, Ptr{Cvoid}, Csize_t),
          handle, autotunePreference, attr, buf, sizeInBytes)
end

@checked function cutensornetContractionAutotunePreferenceSetAttribute(handle,
                                                                       autotunePreference,
                                                                       attr, buf,
                                                                       sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionAutotunePreferenceSetAttribute, libcutensornet),
          cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetContractionAutotunePreference_t,
           cutensornetContractionAutotunePreferenceAttributes_t, Ptr{Cvoid}, Csize_t),
          handle, autotunePreference, attr, buf, sizeInBytes)
end

@checked function cutensornetDestroyContractionAutotunePreference(autotunePreference)
    initialize_context()
    ccall((:cutensornetDestroyContractionAutotunePreference, libcutensornet),
          cutensornetStatus_t, (cutensornetContractionAutotunePreference_t,),
          autotunePreference)
end

@checked function cutensornetContraction(handle, plan, rawDataIn, rawDataOut, workDesc,
                                         sliceId, stream)
    initialize_context()
    ccall((:cutensornetContraction, libcutensornet), cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetContractionPlan_t, Ptr{CuPtr{Cvoid}},
           CuPtr{Cvoid}, cutensornetWorkspaceDescriptor_t, Int64, cudaStream_t), handle,
          plan, rawDataIn, rawDataOut, workDesc, sliceId, stream)
end

@checked function cutensornetCreateSliceGroupFromIDRange(handle, sliceIdStart, sliceIdStop,
                                                         sliceIdStep, sliceGroup)
    initialize_context()
    ccall((:cutensornetCreateSliceGroupFromIDRange, libcutensornet), cutensornetStatus_t,
          (cutensornetHandle_t, Int64, Int64, Int64, Ptr{cutensornetSliceGroup_t}), handle,
          sliceIdStart, sliceIdStop, sliceIdStep, sliceGroup)
end

@checked function cutensornetCreateSliceGroupFromIDs(handle, beginIDSequence, endIDSequence,
                                                     sliceGroup)
    initialize_context()
    ccall((:cutensornetCreateSliceGroupFromIDs, libcutensornet), cutensornetStatus_t,
          (cutensornetHandle_t, Ptr{Int64}, Ptr{Int64}, Ptr{cutensornetSliceGroup_t}),
          handle, beginIDSequence, endIDSequence, sliceGroup)
end

@checked function cutensornetDestroySliceGroup(sliceGroup)
    initialize_context()
    ccall((:cutensornetDestroySliceGroup, libcutensornet), cutensornetStatus_t,
          (cutensornetSliceGroup_t,), sliceGroup)
end

@checked function cutensornetContractSlices(handle, plan, rawDataIn, rawDataOut,
                                            accumulateOutput, workDesc, sliceGroup, stream)
    initialize_context()
    ccall((:cutensornetContractSlices, libcutensornet), cutensornetStatus_t,
          (cutensornetHandle_t, cutensornetContractionPlan_t, Ptr{Ptr{Cvoid}}, Ptr{Cvoid},
           Int32, cutensornetWorkspaceDescriptor_t, cutensornetSliceGroup_t, cudaStream_t),
          handle, plan, rawDataIn, rawDataOut, accumulateOutput, workDesc, sliceGroup,
          stream)
end

@checked function cutensornetGetDeviceMemHandler(handle, devMemHandler)
    initialize_context()
    ccall((:cutensornetGetDeviceMemHandler, libcutensornet), cutensornetStatus_t,
          (cutensornetHandle_t, Ptr{cutensornetDeviceMemHandler_t}), handle, devMemHandler)
end

@checked function cutensornetSetDeviceMemHandler(handle, devMemHandler)
    initialize_context()
    ccall((:cutensornetSetDeviceMemHandler, libcutensornet), cutensornetStatus_t,
          (cutensornetHandle_t, Ptr{cutensornetDeviceMemHandler_t}), handle, devMemHandler)
end

@checked function cutensornetLoggerSetCallback(callback)
    ccall((:cutensornetLoggerSetCallback, libcutensornet), cutensornetStatus_t,
          (cutensornetLoggerCallback_t,), callback)
end

@checked function cutensornetLoggerSetCallbackData(callback, userData)
    initialize_context()
    ccall((:cutensornetLoggerSetCallbackData, libcutensornet), cutensornetStatus_t,
          (cutensornetLoggerCallbackData_t, Ptr{Cvoid}), callback, userData)
end

@checked function cutensornetLoggerSetFile(file)
    ccall((:cutensornetLoggerSetFile, libcutensornet), cutensornetStatus_t,
          (Ptr{Libc.FILE},), file)
end

@checked function cutensornetLoggerOpenFile(logFile)
    ccall((:cutensornetLoggerOpenFile, libcutensornet), cutensornetStatus_t, (Cstring,),
          logFile)
end

@checked function cutensornetLoggerSetLevel(level)
    initialize_context()
    ccall((:cutensornetLoggerSetLevel, libcutensornet), cutensornetStatus_t, (Int32,),
          level)
end

@checked function cutensornetLoggerSetMask(mask)
    ccall((:cutensornetLoggerSetMask, libcutensornet), cutensornetStatus_t, (Int32,), mask)
end

# no prototype is found for this function at cutensornet.h:750:21, please use with caution
@checked function cutensornetLoggerForceDisable()
    ccall((:cutensornetLoggerForceDisable, libcutensornet), cutensornetStatus_t, ())
end

# no prototype is found for this function at cutensornet.h:755:8, please use with caution
function cutensornetGetVersion()
    ccall((:cutensornetGetVersion, libcutensornet), Csize_t, ())
end

# no prototype is found for this function at cutensornet.h:761:8, please use with caution
function cutensornetGetCudartVersion()
    ccall((:cutensornetGetCudartVersion, libcutensornet), Csize_t, ())
end

function cutensornetGetErrorString(error)
    ccall((:cutensornetGetErrorString, libcutensornet), Cstring, (cutensornetStatus_t,),
          error)
end

const CUTENSORNET_ALLOCATOR_NAME_LEN = 64
