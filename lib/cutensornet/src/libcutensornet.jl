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
    CUTENSORNET_STATUS_CUSOLVER_ERROR = 25
    CUTENSORNET_STATUS_DEVICE_ALLOCATOR_ERROR = 26
    CUTENSORNET_STATUS_DISTRIBUTED_FAILURE = 27
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
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS = 10
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_LEAVES = 11
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING = 20
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL = 21
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_FACTOR = 22
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES = 23
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_SLICE_FACTOR = 24
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES = 30
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_THREADS = 31
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SIMPLIFICATION_DISABLE_DR = 40
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED = 60
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE = 61
end

@cenum cutensornetContractionOptimizerInfoAttributes_t::UInt32 begin
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH = 0
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES = 10
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES = 11
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_MODE = 12
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_EXTENT = 13
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_CONFIG = 14
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_OVERHEAD = 15
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PHASE1_FLOP_COUNT = 20
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT = 21
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_EFFECTIVE_FLOPS_EST = 22
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_RUNTIME_EST = 23
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_LARGEST_TENSOR = 24
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_INTERMEDIATE_MODES = 30
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_INTERMEDIATE_MODES = 31
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
    CUTENSORNET_MEMSPACE_HOST = 1
end

struct cutensornetNodePair_t
    first::Int32
    second::Int32
end

struct cutensornetContractionPath_t
    numContractions::Int32
    data::Ptr{cutensornetNodePair_t}
end

struct cutensornetSliceInfoPair_t
    slicedMode::Int32
    slicedExtent::Int64
end

struct cutensornetSlicingConfig_t
    numSlicedModes::UInt32
    data::Ptr{cutensornetSliceInfoPair_t}
end

const cutensornetContractionOptimizerConfig_t = Ptr{Cvoid}

const cutensornetContractionOptimizerInfo_t = Ptr{Cvoid}

const cutensornetContractionAutotunePreference_t = Ptr{Cvoid}

const cutensornetSliceGroup_t = Ptr{Cvoid}

struct cutensornetTensorQualifiers_t
    isConjugate::Int32
end

const cutensornetTensorDescriptor_t = Ptr{Cvoid}

const cutensornetTensorSVDConfig_t = Ptr{Cvoid}

@cenum cutensornetTensorSVDConfigAttributes_t::UInt32 begin
    CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF = 0
    CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF = 1
    CUTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION = 2
    CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION = 3
end

@cenum cutensornetTensorSVDPartition_t::UInt32 begin
    CUTENSORNET_TENSOR_SVD_PARTITION_NONE = 0
    CUTENSORNET_TENSOR_SVD_PARTITION_US = 1
    CUTENSORNET_TENSOR_SVD_PARTITION_SV = 2
    CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL = 3
end

@cenum cutensornetTensorSVDNormalization_t::UInt32 begin
    CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE = 0
    CUTENSORNET_TENSOR_SVD_NORMALIZATION_L1 = 1
    CUTENSORNET_TENSOR_SVD_NORMALIZATION_L2 = 2
    CUTENSORNET_TENSOR_SVD_NORMALIZATION_LINF = 3
end

const cutensornetTensorSVDInfo_t = Ptr{Cvoid}

@cenum cutensornetTensorSVDInfoAttributes_t::UInt32 begin
    CUTENSORNET_TENSOR_SVD_INFO_FULL_EXTENT = 0
    CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT = 1
    CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT = 2
end

@cenum cutensornetGateSplitAlgo_t::UInt32 begin
    CUTENSORNET_GATE_SPLIT_ALGO_DIRECT = 0
    CUTENSORNET_GATE_SPLIT_ALGO_REDUCED = 1
end

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
    @ccall libcutensornet.cutensornetCreate(handle::Ptr{cutensornetHandle_t})::cutensornetStatus_t
end

@checked function cutensornetDestroy(handle)
    initialize_context()
    @ccall libcutensornet.cutensornetDestroy(handle::cutensornetHandle_t)::cutensornetStatus_t
end

@checked function cutensornetCreateNetworkDescriptor(handle, numInputs, numModesIn,
                                                     extentsIn, stridesIn, modesIn,
                                                     qualifiersIn, numModesOut, extentsOut,
                                                     stridesOut, modesOut, dataType,
                                                     computeType, descNet)
    initialize_context()
    @ccall libcutensornet.cutensornetCreateNetworkDescriptor(handle::cutensornetHandle_t,
                                                             numInputs::Int32,
                                                             numModesIn::Ptr{Int32},
                                                             extentsIn::Ptr{Ptr{Int64}},
                                                             stridesIn::Ptr{Ptr{Int64}},
                                                             modesIn::Ptr{Ptr{Int32}},
                                                             qualifiersIn::Ptr{cutensornetTensorQualifiers_t},
                                                             numModesOut::Int32,
                                                             extentsOut::Ptr{Int64},
                                                             stridesOut::Ptr{Int64},
                                                             modesOut::Ptr{Int32},
                                                             dataType::cudaDataType_t,
                                                             computeType::cutensornetComputeType_t,
                                                             descNet::Ptr{cutensornetNetworkDescriptor_t})::cutensornetStatus_t
end

@checked function cutensornetDestroyNetworkDescriptor(desc)
    initialize_context()
    @ccall libcutensornet.cutensornetDestroyNetworkDescriptor(desc::cutensornetNetworkDescriptor_t)::cutensornetStatus_t
end

@checked function cutensornetGetOutputTensorDetails(handle, descNet, numModes, dataSize,
                                                    modeLabels, extents, strides)
    initialize_context()
    @ccall libcutensornet.cutensornetGetOutputTensorDetails(handle::cutensornetHandle_t,
                                                            descNet::cutensornetNetworkDescriptor_t,
                                                            numModes::Ptr{Int32},
                                                            dataSize::Ptr{Csize_t},
                                                            modeLabels::Ptr{Int32},
                                                            extents::Ptr{Int64},
                                                            strides::Ptr{Int64})::cutensornetStatus_t
end

@checked function cutensornetGetOutputTensorDescriptor(handle, descNet, outputTensorDesc)
    initialize_context()
    @ccall libcutensornet.cutensornetGetOutputTensorDescriptor(handle::cutensornetHandle_t,
                                                               descNet::cutensornetNetworkDescriptor_t,
                                                               outputTensorDesc::Ptr{cutensornetTensorDescriptor_t})::cutensornetStatus_t
end

@checked function cutensornetGetTensorDetails(handle, tensorDesc, numModes, dataSize,
                                              modeLabels, extents, strides)
    initialize_context()
    @ccall libcutensornet.cutensornetGetTensorDetails(handle::cutensornetHandle_t,
                                                      tensorDesc::cutensornetTensorDescriptor_t,
                                                      numModes::Ptr{Int32},
                                                      dataSize::Ptr{Csize_t},
                                                      modeLabels::Ptr{Int32},
                                                      extents::Ptr{Int64},
                                                      strides::Ptr{Int64})::cutensornetStatus_t
end

@checked function cutensornetCreateWorkspaceDescriptor(handle, workDesc)
    initialize_context()
    @ccall libcutensornet.cutensornetCreateWorkspaceDescriptor(handle::cutensornetHandle_t,
                                                               workDesc::Ptr{cutensornetWorkspaceDescriptor_t})::cutensornetStatus_t
end

@checked function cutensornetWorkspaceComputeSizes(handle, descNet, optimizerInfo, workDesc)
    initialize_context()
    @ccall libcutensornet.cutensornetWorkspaceComputeSizes(handle::cutensornetHandle_t,
                                                           descNet::cutensornetNetworkDescriptor_t,
                                                           optimizerInfo::cutensornetContractionOptimizerInfo_t,
                                                           workDesc::cutensornetWorkspaceDescriptor_t)::cutensornetStatus_t
end

@checked function cutensornetWorkspaceComputeContractionSizes(handle, descNet,
                                                              optimizerInfo, workDesc)
    initialize_context()
    @ccall libcutensornet.cutensornetWorkspaceComputeContractionSizes(handle::cutensornetHandle_t,
                                                                      descNet::cutensornetNetworkDescriptor_t,
                                                                      optimizerInfo::cutensornetContractionOptimizerInfo_t,
                                                                      workDesc::cutensornetWorkspaceDescriptor_t)::cutensornetStatus_t
end

@checked function cutensornetWorkspaceGetSize(handle, workDesc, workPref, memSpace,
                                              workspaceSize)
    initialize_context()
    @ccall libcutensornet.cutensornetWorkspaceGetSize(handle::cutensornetHandle_t,
                                                      workDesc::cutensornetWorkspaceDescriptor_t,
                                                      workPref::cutensornetWorksizePref_t,
                                                      memSpace::cutensornetMemspace_t,
                                                      workspaceSize::Ptr{UInt64})::cutensornetStatus_t
end

@checked function cutensornetWorkspaceSet(handle, workDesc, memSpace, workspacePtr,
                                          workspaceSize)
    initialize_context()
    @ccall libcutensornet.cutensornetWorkspaceSet(handle::cutensornetHandle_t,
                                                  workDesc::cutensornetWorkspaceDescriptor_t,
                                                  memSpace::cutensornetMemspace_t,
                                                  workspacePtr::PtrOrCuPtr{Cvoid},
                                                  workspaceSize::UInt64)::cutensornetStatus_t
end

@checked function cutensornetWorkspaceGet(handle, workDesc, memSpace, workspacePtr,
                                          workspaceSize)
    initialize_context()
    @ccall libcutensornet.cutensornetWorkspaceGet(handle::cutensornetHandle_t,
                                                  workDesc::cutensornetWorkspaceDescriptor_t,
                                                  memSpace::cutensornetMemspace_t,
                                                  workspacePtr::Ptr{Ptr{Cvoid}},
                                                  workspaceSize::Ptr{UInt64})::cutensornetStatus_t
end

@checked function cutensornetDestroyWorkspaceDescriptor(desc)
    initialize_context()
    @ccall libcutensornet.cutensornetDestroyWorkspaceDescriptor(desc::cutensornetWorkspaceDescriptor_t)::cutensornetStatus_t
end

@checked function cutensornetCreateContractionOptimizerConfig(handle, optimizerConfig)
    initialize_context()
    @ccall libcutensornet.cutensornetCreateContractionOptimizerConfig(handle::cutensornetHandle_t,
                                                                      optimizerConfig::Ptr{cutensornetContractionOptimizerConfig_t})::cutensornetStatus_t
end

@checked function cutensornetDestroyContractionOptimizerConfig(optimizerConfig)
    initialize_context()
    @ccall libcutensornet.cutensornetDestroyContractionOptimizerConfig(optimizerConfig::cutensornetContractionOptimizerConfig_t)::cutensornetStatus_t
end

@checked function cutensornetContractionOptimizerConfigGetAttribute(handle, optimizerConfig,
                                                                    attr, buf, sizeInBytes)
    initialize_context()
    @ccall libcutensornet.cutensornetContractionOptimizerConfigGetAttribute(handle::cutensornetHandle_t,
                                                                            optimizerConfig::cutensornetContractionOptimizerConfig_t,
                                                                            attr::cutensornetContractionOptimizerConfigAttributes_t,
                                                                            buf::Ptr{Cvoid},
                                                                            sizeInBytes::Csize_t)::cutensornetStatus_t
end

@checked function cutensornetContractionOptimizerConfigSetAttribute(handle, optimizerConfig,
                                                                    attr, buf, sizeInBytes)
    initialize_context()
    @ccall libcutensornet.cutensornetContractionOptimizerConfigSetAttribute(handle::cutensornetHandle_t,
                                                                            optimizerConfig::cutensornetContractionOptimizerConfig_t,
                                                                            attr::cutensornetContractionOptimizerConfigAttributes_t,
                                                                            buf::Ptr{Cvoid},
                                                                            sizeInBytes::Csize_t)::cutensornetStatus_t
end

@checked function cutensornetDestroyContractionOptimizerInfo(optimizerInfo)
    initialize_context()
    @ccall libcutensornet.cutensornetDestroyContractionOptimizerInfo(optimizerInfo::cutensornetContractionOptimizerInfo_t)::cutensornetStatus_t
end

@checked function cutensornetCreateContractionOptimizerInfo(handle, descNet, optimizerInfo)
    initialize_context()
    @ccall libcutensornet.cutensornetCreateContractionOptimizerInfo(handle::cutensornetHandle_t,
                                                                    descNet::cutensornetNetworkDescriptor_t,
                                                                    optimizerInfo::Ptr{cutensornetContractionOptimizerInfo_t})::cutensornetStatus_t
end

@checked function cutensornetContractionOptimize(handle, descNet, optimizerConfig,
                                                 workspaceSizeConstraint, optimizerInfo)
    initialize_context()
    @ccall libcutensornet.cutensornetContractionOptimize(handle::cutensornetHandle_t,
                                                         descNet::cutensornetNetworkDescriptor_t,
                                                         optimizerConfig::cutensornetContractionOptimizerConfig_t,
                                                         workspaceSizeConstraint::UInt64,
                                                         optimizerInfo::cutensornetContractionOptimizerInfo_t)::cutensornetStatus_t
end

@checked function cutensornetContractionOptimizerInfoGetAttribute(handle, optimizerInfo,
                                                                  attr, buf, sizeInBytes)
    initialize_context()
    @ccall libcutensornet.cutensornetContractionOptimizerInfoGetAttribute(handle::cutensornetHandle_t,
                                                                          optimizerInfo::cutensornetContractionOptimizerInfo_t,
                                                                          attr::cutensornetContractionOptimizerInfoAttributes_t,
                                                                          buf::Ptr{Cvoid},
                                                                          sizeInBytes::Csize_t)::cutensornetStatus_t
end

@checked function cutensornetContractionOptimizerInfoSetAttribute(handle, optimizerInfo,
                                                                  attr, buf, sizeInBytes)
    initialize_context()
    @ccall libcutensornet.cutensornetContractionOptimizerInfoSetAttribute(handle::cutensornetHandle_t,
                                                                          optimizerInfo::cutensornetContractionOptimizerInfo_t,
                                                                          attr::cutensornetContractionOptimizerInfoAttributes_t,
                                                                          buf::Ptr{Cvoid},
                                                                          sizeInBytes::Csize_t)::cutensornetStatus_t
end

@checked function cutensornetContractionOptimizerInfoGetPackedSize(handle, optimizerInfo,
                                                                   sizeInBytes)
    initialize_context()
    @ccall libcutensornet.cutensornetContractionOptimizerInfoGetPackedSize(handle::cutensornetHandle_t,
                                                                           optimizerInfo::cutensornetContractionOptimizerInfo_t,
                                                                           sizeInBytes::Ptr{Csize_t})::cutensornetStatus_t
end

@checked function cutensornetContractionOptimizerInfoPackData(handle, optimizerInfo, buffer,
                                                              sizeInBytes)
    initialize_context()
    @ccall libcutensornet.cutensornetContractionOptimizerInfoPackData(handle::cutensornetHandle_t,
                                                                      optimizerInfo::cutensornetContractionOptimizerInfo_t,
                                                                      buffer::Ptr{Cvoid},
                                                                      sizeInBytes::Csize_t)::cutensornetStatus_t
end

@checked function cutensornetCreateContractionOptimizerInfoFromPackedData(handle, descNet,
                                                                          buffer,
                                                                          sizeInBytes,
                                                                          optimizerInfo)
    initialize_context()
    @ccall libcutensornet.cutensornetCreateContractionOptimizerInfoFromPackedData(handle::cutensornetHandle_t,
                                                                                  descNet::cutensornetNetworkDescriptor_t,
                                                                                  buffer::Ptr{Cvoid},
                                                                                  sizeInBytes::Csize_t,
                                                                                  optimizerInfo::Ptr{cutensornetContractionOptimizerInfo_t})::cutensornetStatus_t
end

@checked function cutensornetUpdateContractionOptimizerInfoFromPackedData(handle, buffer,
                                                                          sizeInBytes,
                                                                          optimizerInfo)
    initialize_context()
    @ccall libcutensornet.cutensornetUpdateContractionOptimizerInfoFromPackedData(handle::cutensornetHandle_t,
                                                                                  buffer::Ptr{Cvoid},
                                                                                  sizeInBytes::Csize_t,
                                                                                  optimizerInfo::cutensornetContractionOptimizerInfo_t)::cutensornetStatus_t
end

@checked function cutensornetCreateContractionPlan(handle, descNet, optimizerInfo, workDesc,
                                                   plan)
    initialize_context()
    @ccall libcutensornet.cutensornetCreateContractionPlan(handle::cutensornetHandle_t,
                                                           descNet::cutensornetNetworkDescriptor_t,
                                                           optimizerInfo::cutensornetContractionOptimizerInfo_t,
                                                           workDesc::cutensornetWorkspaceDescriptor_t,
                                                           plan::Ptr{cutensornetContractionPlan_t})::cutensornetStatus_t
end

@checked function cutensornetDestroyContractionPlan(plan)
    initialize_context()
    @ccall libcutensornet.cutensornetDestroyContractionPlan(plan::cutensornetContractionPlan_t)::cutensornetStatus_t
end

@checked function cutensornetContractionAutotune(handle, plan, rawDataIn, rawDataOut,
                                                 workDesc, pref, stream)
    initialize_context()
    @ccall libcutensornet.cutensornetContractionAutotune(handle::cutensornetHandle_t,
                                                         plan::cutensornetContractionPlan_t,
                                                         rawDataIn::Ptr{CuPtr{Cvoid}},
                                                         rawDataOut::CuPtr{Cvoid},
                                                         workDesc::cutensornetWorkspaceDescriptor_t,
                                                         pref::cutensornetContractionAutotunePreference_t,
                                                         stream::cudaStream_t)::cutensornetStatus_t
end

@checked function cutensornetCreateContractionAutotunePreference(handle, autotunePreference)
    initialize_context()
    @ccall libcutensornet.cutensornetCreateContractionAutotunePreference(handle::cutensornetHandle_t,
                                                                         autotunePreference::Ptr{cutensornetContractionAutotunePreference_t})::cutensornetStatus_t
end

@checked function cutensornetContractionAutotunePreferenceGetAttribute(handle,
                                                                       autotunePreference,
                                                                       attr, buf,
                                                                       sizeInBytes)
    initialize_context()
    @ccall libcutensornet.cutensornetContractionAutotunePreferenceGetAttribute(handle::cutensornetHandle_t,
                                                                               autotunePreference::cutensornetContractionAutotunePreference_t,
                                                                               attr::cutensornetContractionAutotunePreferenceAttributes_t,
                                                                               buf::Ptr{Cvoid},
                                                                               sizeInBytes::Csize_t)::cutensornetStatus_t
end

@checked function cutensornetContractionAutotunePreferenceSetAttribute(handle,
                                                                       autotunePreference,
                                                                       attr, buf,
                                                                       sizeInBytes)
    initialize_context()
    @ccall libcutensornet.cutensornetContractionAutotunePreferenceSetAttribute(handle::cutensornetHandle_t,
                                                                               autotunePreference::cutensornetContractionAutotunePreference_t,
                                                                               attr::cutensornetContractionAutotunePreferenceAttributes_t,
                                                                               buf::Ptr{Cvoid},
                                                                               sizeInBytes::Csize_t)::cutensornetStatus_t
end

@checked function cutensornetDestroyContractionAutotunePreference(autotunePreference)
    initialize_context()
    @ccall libcutensornet.cutensornetDestroyContractionAutotunePreference(autotunePreference::cutensornetContractionAutotunePreference_t)::cutensornetStatus_t
end

@checked function cutensornetContraction(handle, plan, rawDataIn, rawDataOut, workDesc,
                                         sliceId, stream)
    initialize_context()
    @ccall libcutensornet.cutensornetContraction(handle::cutensornetHandle_t,
                                                 plan::cutensornetContractionPlan_t,
                                                 rawDataIn::Ptr{CuPtr{Cvoid}},
                                                 rawDataOut::CuPtr{Cvoid},
                                                 workDesc::cutensornetWorkspaceDescriptor_t,
                                                 sliceId::Int64,
                                                 stream::cudaStream_t)::cutensornetStatus_t
end

@checked function cutensornetCreateSliceGroupFromIDRange(handle, sliceIdStart, sliceIdStop,
                                                         sliceIdStep, sliceGroup)
    initialize_context()
    @ccall libcutensornet.cutensornetCreateSliceGroupFromIDRange(handle::cutensornetHandle_t,
                                                                 sliceIdStart::Int64,
                                                                 sliceIdStop::Int64,
                                                                 sliceIdStep::Int64,
                                                                 sliceGroup::Ptr{cutensornetSliceGroup_t})::cutensornetStatus_t
end

@checked function cutensornetCreateSliceGroupFromIDs(handle, beginIDSequence, endIDSequence,
                                                     sliceGroup)
    initialize_context()
    @ccall libcutensornet.cutensornetCreateSliceGroupFromIDs(handle::cutensornetHandle_t,
                                                             beginIDSequence::Ptr{Int64},
                                                             endIDSequence::Ptr{Int64},
                                                             sliceGroup::Ptr{cutensornetSliceGroup_t})::cutensornetStatus_t
end

@checked function cutensornetDestroySliceGroup(sliceGroup)
    initialize_context()
    @ccall libcutensornet.cutensornetDestroySliceGroup(sliceGroup::cutensornetSliceGroup_t)::cutensornetStatus_t
end

@checked function cutensornetContractSlices(handle, plan, rawDataIn, rawDataOut,
                                            accumulateOutput, workDesc, sliceGroup, stream)
    initialize_context()
    @ccall libcutensornet.cutensornetContractSlices(handle::cutensornetHandle_t,
                                                    plan::cutensornetContractionPlan_t,
                                                    rawDataIn::Ptr{CuPtr{Cvoid}},
                                                    rawDataOut::CuPtr{Cvoid},
                                                    accumulateOutput::Int32,
                                                    workDesc::cutensornetWorkspaceDescriptor_t,
                                                    sliceGroup::cutensornetSliceGroup_t,
                                                    stream::cudaStream_t)::cutensornetStatus_t
end

@checked function cutensornetCreateTensorDescriptor(handle, numModes, extents, strides,
                                                    modes, dataType, descTensor)
    initialize_context()
    @ccall libcutensornet.cutensornetCreateTensorDescriptor(handle::cutensornetHandle_t,
                                                            numModes::Int32,
                                                            extents::Ptr{Int64},
                                                            strides::Ptr{Int64},
                                                            modes::Ptr{Int32},
                                                            dataType::cudaDataType_t,
                                                            descTensor::Ptr{cutensornetTensorDescriptor_t})::cutensornetStatus_t
end

@checked function cutensornetDestroyTensorDescriptor(descTensor)
    initialize_context()
    @ccall libcutensornet.cutensornetDestroyTensorDescriptor(descTensor::cutensornetTensorDescriptor_t)::cutensornetStatus_t
end

@checked function cutensornetCreateTensorSVDConfig(handle, svdConfig)
    initialize_context()
    @ccall libcutensornet.cutensornetCreateTensorSVDConfig(handle::cutensornetHandle_t,
                                                           svdConfig::Ptr{cutensornetTensorSVDConfig_t})::cutensornetStatus_t
end

@checked function cutensornetDestroyTensorSVDConfig(svdConfig)
    initialize_context()
    @ccall libcutensornet.cutensornetDestroyTensorSVDConfig(svdConfig::cutensornetTensorSVDConfig_t)::cutensornetStatus_t
end

@checked function cutensornetTensorSVDConfigGetAttribute(handle, svdConfig, attr, buf,
                                                         sizeInBytes)
    initialize_context()
    @ccall libcutensornet.cutensornetTensorSVDConfigGetAttribute(handle::cutensornetHandle_t,
                                                                 svdConfig::cutensornetTensorSVDConfig_t,
                                                                 attr::cutensornetTensorSVDConfigAttributes_t,
                                                                 buf::Ptr{Cvoid},
                                                                 sizeInBytes::Csize_t)::cutensornetStatus_t
end

@checked function cutensornetTensorSVDConfigSetAttribute(handle, svdConfig, attr, buf,
                                                         sizeInBytes)
    initialize_context()
    @ccall libcutensornet.cutensornetTensorSVDConfigSetAttribute(handle::cutensornetHandle_t,
                                                                 svdConfig::cutensornetTensorSVDConfig_t,
                                                                 attr::cutensornetTensorSVDConfigAttributes_t,
                                                                 buf::Ptr{Cvoid},
                                                                 sizeInBytes::Csize_t)::cutensornetStatus_t
end

@checked function cutensornetWorkspaceComputeSVDSizes(handle, descTensorIn, descTensorU,
                                                      descTensorV, svdConfig, workDesc)
    initialize_context()
    @ccall libcutensornet.cutensornetWorkspaceComputeSVDSizes(handle::cutensornetHandle_t,
                                                              descTensorIn::cutensornetTensorDescriptor_t,
                                                              descTensorU::cutensornetTensorDescriptor_t,
                                                              descTensorV::cutensornetTensorDescriptor_t,
                                                              svdConfig::cutensornetTensorSVDConfig_t,
                                                              workDesc::cutensornetWorkspaceDescriptor_t)::cutensornetStatus_t
end

@checked function cutensornetWorkspaceComputeQRSizes(handle, descTensorIn, descTensorQ,
                                                     descTensorR, workDesc)
    initialize_context()
    @ccall libcutensornet.cutensornetWorkspaceComputeQRSizes(handle::cutensornetHandle_t,
                                                             descTensorIn::cutensornetTensorDescriptor_t,
                                                             descTensorQ::cutensornetTensorDescriptor_t,
                                                             descTensorR::cutensornetTensorDescriptor_t,
                                                             workDesc::cutensornetWorkspaceDescriptor_t)::cutensornetStatus_t
end

@checked function cutensornetCreateTensorSVDInfo(handle, svdInfo)
    initialize_context()
    @ccall libcutensornet.cutensornetCreateTensorSVDInfo(handle::cutensornetHandle_t,
                                                         svdInfo::Ptr{cutensornetTensorSVDInfo_t})::cutensornetStatus_t
end

@checked function cutensornetTensorSVDInfoGetAttribute(handle, svdInfo, attr, buf,
                                                       sizeInBytes)
    initialize_context()
    @ccall libcutensornet.cutensornetTensorSVDInfoGetAttribute(handle::cutensornetHandle_t,
                                                               svdInfo::cutensornetTensorSVDInfo_t,
                                                               attr::cutensornetTensorSVDInfoAttributes_t,
                                                               buf::Ptr{Cvoid},
                                                               sizeInBytes::Csize_t)::cutensornetStatus_t
end

@checked function cutensornetDestroyTensorSVDInfo(svdInfo)
    initialize_context()
    @ccall libcutensornet.cutensornetDestroyTensorSVDInfo(svdInfo::cutensornetTensorSVDInfo_t)::cutensornetStatus_t
end

@checked function cutensornetTensorSVD(handle, descTensorIn, rawDataIn, descTensorU, u, s,
                                       descTensorV, v, svdConfig, svdInfo, workDesc, stream)
    initialize_context()
    @ccall libcutensornet.cutensornetTensorSVD(handle::cutensornetHandle_t,
                                               descTensorIn::cutensornetTensorDescriptor_t,
                                               rawDataIn::Ptr{Cvoid},
                                               descTensorU::cutensornetTensorDescriptor_t,
                                               u::Ptr{Cvoid}, s::Ptr{Cvoid},
                                               descTensorV::cutensornetTensorDescriptor_t,
                                               v::Ptr{Cvoid},
                                               svdConfig::cutensornetTensorSVDConfig_t,
                                               svdInfo::cutensornetTensorSVDInfo_t,
                                               workDesc::cutensornetWorkspaceDescriptor_t,
                                               stream::cudaStream_t)::cutensornetStatus_t
end

@checked function cutensornetTensorQR(handle, descTensorIn, rawDataIn, descTensorQ, q,
                                      descTensorR, r, workDesc, stream)
    initialize_context()
    @ccall libcutensornet.cutensornetTensorQR(handle::cutensornetHandle_t,
                                              descTensorIn::cutensornetTensorDescriptor_t,
                                              rawDataIn::Ptr{Cvoid},
                                              descTensorQ::cutensornetTensorDescriptor_t,
                                              q::Ptr{Cvoid},
                                              descTensorR::cutensornetTensorDescriptor_t,
                                              r::Ptr{Cvoid},
                                              workDesc::cutensornetWorkspaceDescriptor_t,
                                              stream::cudaStream_t)::cutensornetStatus_t
end

@checked function cutensornetWorkspaceComputeGateSplitSizes(handle, descTensorInA,
                                                            descTensorInB, descTensorInG,
                                                            descTensorU, descTensorV,
                                                            gateAlgo, svdConfig,
                                                            computeType, workDesc)
    initialize_context()
    @ccall libcutensornet.cutensornetWorkspaceComputeGateSplitSizes(handle::cutensornetHandle_t,
                                                                    descTensorInA::cutensornetTensorDescriptor_t,
                                                                    descTensorInB::cutensornetTensorDescriptor_t,
                                                                    descTensorInG::cutensornetTensorDescriptor_t,
                                                                    descTensorU::cutensornetTensorDescriptor_t,
                                                                    descTensorV::cutensornetTensorDescriptor_t,
                                                                    gateAlgo::cutensornetGateSplitAlgo_t,
                                                                    svdConfig::cutensornetTensorSVDConfig_t,
                                                                    computeType::cutensornetComputeType_t,
                                                                    workDesc::cutensornetWorkspaceDescriptor_t)::cutensornetStatus_t
end

@checked function cutensornetGateSplit(handle, descTensorInA, rawDataInA, descTensorInB,
                                       rawDataInB, descTensorInG, rawDataInG, descTensorU,
                                       u, s, descTensorV, v, gateAlgo, svdConfig,
                                       computeType, svdInfo, workDesc, stream)
    initialize_context()
    @ccall libcutensornet.cutensornetGateSplit(handle::cutensornetHandle_t,
                                               descTensorInA::cutensornetTensorDescriptor_t,
                                               rawDataInA::Ptr{Cvoid},
                                               descTensorInB::cutensornetTensorDescriptor_t,
                                               rawDataInB::Ptr{Cvoid},
                                               descTensorInG::cutensornetTensorDescriptor_t,
                                               rawDataInG::Ptr{Cvoid},
                                               descTensorU::cutensornetTensorDescriptor_t,
                                               u::Ptr{Cvoid}, s::Ptr{Cvoid},
                                               descTensorV::cutensornetTensorDescriptor_t,
                                               v::Ptr{Cvoid},
                                               gateAlgo::cutensornetGateSplitAlgo_t,
                                               svdConfig::cutensornetTensorSVDConfig_t,
                                               computeType::cutensornetComputeType_t,
                                               svdInfo::cutensornetTensorSVDInfo_t,
                                               workDesc::cutensornetWorkspaceDescriptor_t,
                                               stream::cudaStream_t)::cutensornetStatus_t
end

@checked function cutensornetGetDeviceMemHandler(handle, devMemHandler)
    initialize_context()
    @ccall libcutensornet.cutensornetGetDeviceMemHandler(handle::cutensornetHandle_t,
                                                         devMemHandler::Ptr{cutensornetDeviceMemHandler_t})::cutensornetStatus_t
end

@checked function cutensornetSetDeviceMemHandler(handle, devMemHandler)
    initialize_context()
    @ccall libcutensornet.cutensornetSetDeviceMemHandler(handle::cutensornetHandle_t,
                                                         devMemHandler::Ptr{cutensornetDeviceMemHandler_t})::cutensornetStatus_t
end

@checked function cutensornetLoggerSetCallback(callback)
    @ccall libcutensornet.cutensornetLoggerSetCallback(callback::cutensornetLoggerCallback_t)::cutensornetStatus_t
end

@checked function cutensornetLoggerSetCallbackData(callback, userData)
    initialize_context()
    @ccall libcutensornet.cutensornetLoggerSetCallbackData(callback::cutensornetLoggerCallbackData_t,
                                                           userData::Ptr{Cvoid})::cutensornetStatus_t
end

@checked function cutensornetLoggerSetFile(file)
    @ccall libcutensornet.cutensornetLoggerSetFile(file::Ptr{Libc.FILE})::cutensornetStatus_t
end

@checked function cutensornetLoggerOpenFile(logFile)
    @ccall libcutensornet.cutensornetLoggerOpenFile(logFile::Cstring)::cutensornetStatus_t
end

@checked function cutensornetLoggerSetLevel(level)
    initialize_context()
    @ccall libcutensornet.cutensornetLoggerSetLevel(level::Int32)::cutensornetStatus_t
end

@checked function cutensornetLoggerSetMask(mask)
    @ccall libcutensornet.cutensornetLoggerSetMask(mask::Int32)::cutensornetStatus_t
end

# no prototype is found for this function at cutensornet.h:1110:21, please use with caution
@checked function cutensornetLoggerForceDisable()
    @ccall libcutensornet.cutensornetLoggerForceDisable()::cutensornetStatus_t
end

# no prototype is found for this function at cutensornet.h:1115:8, please use with caution
function cutensornetGetVersion()
    @ccall libcutensornet.cutensornetGetVersion()::Csize_t
end

# no prototype is found for this function at cutensornet.h:1121:8, please use with caution
function cutensornetGetCudartVersion()
    @ccall libcutensornet.cutensornetGetCudartVersion()::Csize_t
end

function cutensornetGetErrorString(error)
    @ccall libcutensornet.cutensornetGetErrorString(error::cutensornetStatus_t)::Cstring
end

@checked function cutensornetDistributedResetConfiguration(handle, commPtr, commSize)
    initialize_context()
    @ccall libcutensornet.cutensornetDistributedResetConfiguration(handle::cutensornetHandle_t,
                                                                   commPtr::Ptr{Cvoid},
                                                                   commSize::Csize_t)::cutensornetStatus_t
end

@checked function cutensornetDistributedGetNumRanks(handle, numRanks)
    initialize_context()
    @ccall libcutensornet.cutensornetDistributedGetNumRanks(handle::cutensornetHandle_t,
                                                            numRanks::Ptr{Int32})::cutensornetStatus_t
end

@checked function cutensornetDistributedGetProcRank(handle, procRank)
    initialize_context()
    @ccall libcutensornet.cutensornetDistributedGetProcRank(handle::cutensornetHandle_t,
                                                            procRank::Ptr{Int32})::cutensornetStatus_t
end

@checked function cutensornetDistributedSynchronize(handle)
    initialize_context()
    @ccall libcutensornet.cutensornetDistributedSynchronize(handle::cutensornetHandle_t)::cutensornetStatus_t
end

const CUTENSORNET_ALLOCATOR_NAME_LEN = 64
