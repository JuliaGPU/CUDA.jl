# Automatically generated using Clang.jl


const CUDNN_MAJOR = 7
const CUDNN_MINOR = 6
const CUDNN_PATCHLEVEL = 4
const CUDNN_VERSION = CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL
const CUDNN_DIM_MAX = 8
const CUDNN_LRN_MIN_N = 1
const CUDNN_LRN_MAX_N = 16
const CUDNN_LRN_MIN_K = 1.0e-5
const CUDNN_LRN_MIN_BETA = 0.01
const CUDNN_BN_MIN_EPSILON = 0.0
const CUDNN_SEQDATA_DIM_COUNT = 4
const CUDNN_ATTN_QUERYMAP_ALL_TO_ONE = 0
const CUDNN_ATTN_QUERYMAP_ONE_TO_ONE = UInt32(1) << 0
const CUDNN_ATTN_DISABLE_PROJ_BIASES = 0
const CUDNN_ATTN_ENABLE_PROJ_BIASES = UInt32(1) << 1
const CUDNN_ATTN_WKIND_COUNT = 8

@cenum cudnnSeverity_t::UInt32 begin
    CUDNN_SEV_FATAL = 0
    CUDNN_SEV_ERROR = 1
    CUDNN_SEV_WARNING = 2
    CUDNN_SEV_INFO = 3
end


const cudnnContext = Cvoid
const cudnnHandle_t = Ptr{cudnnContext}

@cenum cudnnStatus_t::UInt32 begin
    CUDNN_STATUS_SUCCESS = 0
    CUDNN_STATUS_NOT_INITIALIZED = 1
    CUDNN_STATUS_ALLOC_FAILED = 2
    CUDNN_STATUS_BAD_PARAM = 3
    CUDNN_STATUS_INTERNAL_ERROR = 4
    CUDNN_STATUS_INVALID_VALUE = 5
    CUDNN_STATUS_ARCH_MISMATCH = 6
    CUDNN_STATUS_MAPPING_ERROR = 7
    CUDNN_STATUS_EXECUTION_FAILED = 8
    CUDNN_STATUS_NOT_SUPPORTED = 9
    CUDNN_STATUS_LICENSE_ERROR = 10
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11
    CUDNN_STATUS_RUNTIME_IN_PROGRESS = 12
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 13
end


const cudnnRuntimeTag_t = Cvoid

@cenum cudnnErrQueryMode_t::UInt32 begin
    CUDNN_ERRQUERY_RAWCODE = 0
    CUDNN_ERRQUERY_NONBLOCKING = 1
    CUDNN_ERRQUERY_BLOCKING = 2
end


const cudnnTensorStruct = Cvoid
const cudnnTensorDescriptor_t = Ptr{cudnnTensorStruct}
const cudnnConvolutionStruct = Cvoid
const cudnnConvolutionDescriptor_t = Ptr{cudnnConvolutionStruct}
const cudnnPoolingStruct = Cvoid
const cudnnPoolingDescriptor_t = Ptr{cudnnPoolingStruct}
const cudnnFilterStruct = Cvoid
const cudnnFilterDescriptor_t = Ptr{cudnnFilterStruct}
const cudnnLRNStruct = Cvoid
const cudnnLRNDescriptor_t = Ptr{cudnnLRNStruct}
const cudnnActivationStruct = Cvoid
const cudnnActivationDescriptor_t = Ptr{cudnnActivationStruct}
const cudnnSpatialTransformerStruct = Cvoid
const cudnnSpatialTransformerDescriptor_t = Ptr{cudnnSpatialTransformerStruct}
const cudnnOpTensorStruct = Cvoid
const cudnnOpTensorDescriptor_t = Ptr{cudnnOpTensorStruct}
const cudnnReduceTensorStruct = Cvoid
const cudnnReduceTensorDescriptor_t = Ptr{cudnnReduceTensorStruct}
const cudnnCTCLossStruct = Cvoid
const cudnnCTCLossDescriptor_t = Ptr{cudnnCTCLossStruct}
const cudnnTensorTransformStruct = Cvoid
const cudnnTensorTransformDescriptor_t = Ptr{cudnnTensorTransformStruct}

@cenum cudnnDataType_t::UInt32 begin
    CUDNN_DATA_FLOAT = 0
    CUDNN_DATA_DOUBLE = 1
    CUDNN_DATA_HALF = 2
    CUDNN_DATA_INT8 = 3
    CUDNN_DATA_INT32 = 4
    CUDNN_DATA_INT8x4 = 5
    CUDNN_DATA_UINT8 = 6
    CUDNN_DATA_UINT8x4 = 7
    CUDNN_DATA_INT8x32 = 8
end

@cenum cudnnMathType_t::UInt32 begin
    CUDNN_DEFAULT_MATH = 0
    CUDNN_TENSOR_OP_MATH = 1
    CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = 2
end

@cenum cudnnNanPropagation_t::UInt32 begin
    CUDNN_NOT_PROPAGATE_NAN = 0
    CUDNN_PROPAGATE_NAN = 1
end

@cenum cudnnDeterminism_t::UInt32 begin
    CUDNN_NON_DETERMINISTIC = 0
    CUDNN_DETERMINISTIC = 1
end

@cenum cudnnReorderType_t::UInt32 begin
    CUDNN_DEFAULT_REORDER = 0
    CUDNN_NO_REORDER = 1
end

@cenum cudnnTensorFormat_t::UInt32 begin
    CUDNN_TENSOR_NCHW = 0
    CUDNN_TENSOR_NHWC = 1
    CUDNN_TENSOR_NCHW_VECT_C = 2
end

@cenum cudnnFoldingDirection_t::UInt32 begin
    CUDNN_TRANSFORM_FOLD = 0
    CUDNN_TRANSFORM_UNFOLD = 1
end

@cenum cudnnOpTensorOp_t::UInt32 begin
    CUDNN_OP_TENSOR_ADD = 0
    CUDNN_OP_TENSOR_MUL = 1
    CUDNN_OP_TENSOR_MIN = 2
    CUDNN_OP_TENSOR_MAX = 3
    CUDNN_OP_TENSOR_SQRT = 4
    CUDNN_OP_TENSOR_NOT = 5
end

@cenum cudnnReduceTensorOp_t::UInt32 begin
    CUDNN_REDUCE_TENSOR_ADD = 0
    CUDNN_REDUCE_TENSOR_MUL = 1
    CUDNN_REDUCE_TENSOR_MIN = 2
    CUDNN_REDUCE_TENSOR_MAX = 3
    CUDNN_REDUCE_TENSOR_AMAX = 4
    CUDNN_REDUCE_TENSOR_AVG = 5
    CUDNN_REDUCE_TENSOR_NORM1 = 6
    CUDNN_REDUCE_TENSOR_NORM2 = 7
    CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = 8
end

@cenum cudnnReduceTensorIndices_t::UInt32 begin
    CUDNN_REDUCE_TENSOR_NO_INDICES = 0
    CUDNN_REDUCE_TENSOR_FLATTENED_INDICES = 1
end

@cenum cudnnIndicesType_t::UInt32 begin
    CUDNN_32BIT_INDICES = 0
    CUDNN_64BIT_INDICES = 1
    CUDNN_16BIT_INDICES = 2
    CUDNN_8BIT_INDICES = 3
end

@cenum cudnnConvolutionMode_t::UInt32 begin
    CUDNN_CONVOLUTION = 0
    CUDNN_CROSS_CORRELATION = 1
end

@cenum cudnnConvolutionFwdPreference_t::UInt32 begin
    CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = 0
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2
end

@cenum cudnnConvolutionFwdAlgo_t::UInt32 begin
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3
    CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7
    CUDNN_CONVOLUTION_FWD_ALGO_COUNT = 8
end


struct cudnnConvolutionFwdAlgoPerf_t
    algo::cudnnConvolutionFwdAlgo_t
    status::cudnnStatus_t
    time::Cfloat
    memory::Csize_t
    determinism::cudnnDeterminism_t
    mathType::cudnnMathType_t
    reserved::NTuple{3, Cint}
end

@cenum cudnnConvolutionBwdFilterPreference_t::UInt32 begin
    CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = 1
    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2
end

@cenum cudnnConvolutionBwdFilterAlgo_t::UInt32 begin
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = 4
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = 6
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT = 7
end


struct cudnnConvolutionBwdFilterAlgoPerf_t
    algo::cudnnConvolutionBwdFilterAlgo_t
    status::cudnnStatus_t
    time::Cfloat
    memory::Csize_t
    determinism::cudnnDeterminism_t
    mathType::cudnnMathType_t
    reserved::NTuple{3, Cint}
end

@cenum cudnnConvolutionBwdDataPreference_t::UInt32 begin
    CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = 0
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = 1
    CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2
end

@cenum cudnnConvolutionBwdDataAlgo_t::UInt32 begin
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = 6
end


struct cudnnConvolutionBwdDataAlgoPerf_t
    algo::cudnnConvolutionBwdDataAlgo_t
    status::cudnnStatus_t
    time::Cfloat
    memory::Csize_t
    determinism::cudnnDeterminism_t
    mathType::cudnnMathType_t
    reserved::NTuple{3, Cint}
end

@cenum cudnnSoftmaxAlgorithm_t::UInt32 begin
    CUDNN_SOFTMAX_FAST = 0
    CUDNN_SOFTMAX_ACCURATE = 1
    CUDNN_SOFTMAX_LOG = 2
end

@cenum cudnnSoftmaxMode_t::UInt32 begin
    CUDNN_SOFTMAX_MODE_INSTANCE = 0
    CUDNN_SOFTMAX_MODE_CHANNEL = 1
end

@cenum cudnnPoolingMode_t::UInt32 begin
    CUDNN_POOLING_MAX = 0
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2
    CUDNN_POOLING_MAX_DETERMINISTIC = 3
end

@cenum cudnnActivationMode_t::UInt32 begin
    CUDNN_ACTIVATION_SIGMOID = 0
    CUDNN_ACTIVATION_RELU = 1
    CUDNN_ACTIVATION_TANH = 2
    CUDNN_ACTIVATION_CLIPPED_RELU = 3
    CUDNN_ACTIVATION_ELU = 4
    CUDNN_ACTIVATION_IDENTITY = 5
end

@cenum cudnnLRNMode_t::UInt32 begin
    CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0
end

@cenum cudnnDivNormMode_t::UInt32 begin
    CUDNN_DIVNORM_PRECOMPUTED_MEANS = 0
end

@cenum cudnnBatchNormMode_t::UInt32 begin
    CUDNN_BATCHNORM_PER_ACTIVATION = 0
    CUDNN_BATCHNORM_SPATIAL = 1
    CUDNN_BATCHNORM_SPATIAL_PERSISTENT = 2
end

@cenum cudnnBatchNormOps_t::UInt32 begin
    CUDNN_BATCHNORM_OPS_BN = 0
    CUDNN_BATCHNORM_OPS_BN_ACTIVATION = 1
    CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION = 2
end

@cenum cudnnSamplerType_t::UInt32 begin
    CUDNN_SAMPLER_BILINEAR = 0
end


const cudnnDropoutStruct = Cvoid
const cudnnDropoutDescriptor_t = Ptr{cudnnDropoutStruct}

@cenum cudnnRNNAlgo_t::UInt32 begin
    CUDNN_RNN_ALGO_STANDARD = 0
    CUDNN_RNN_ALGO_PERSIST_STATIC = 1
    CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2
    CUDNN_RNN_ALGO_COUNT = 3
end

@cenum cudnnRNNMode_t::UInt32 begin
    CUDNN_RNN_RELU = 0
    CUDNN_RNN_TANH = 1
    CUDNN_LSTM = 2
    CUDNN_GRU = 3
end

@cenum cudnnRNNBiasMode_t::UInt32 begin
    CUDNN_RNN_NO_BIAS = 0
    CUDNN_RNN_SINGLE_INP_BIAS = 1
    CUDNN_RNN_DOUBLE_BIAS = 2
    CUDNN_RNN_SINGLE_REC_BIAS = 3
end

@cenum cudnnDirectionMode_t::UInt32 begin
    CUDNN_UNIDIRECTIONAL = 0
    CUDNN_BIDIRECTIONAL = 1
end

@cenum cudnnRNNInputMode_t::UInt32 begin
    CUDNN_LINEAR_INPUT = 0
    CUDNN_SKIP_INPUT = 1
end

@cenum cudnnRNNClipMode_t::UInt32 begin
    CUDNN_RNN_CLIP_NONE = 0
    CUDNN_RNN_CLIP_MINMAX = 1
end

@cenum cudnnRNNDataLayout_t::UInt32 begin
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED = 0
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED = 1
    CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED = 2
end

@cenum cudnnRNNPaddingMode_t::UInt32 begin
    CUDNN_RNN_PADDED_IO_DISABLED = 0
    CUDNN_RNN_PADDED_IO_ENABLED = 1
end


const cudnnRNNStruct = Cvoid
const cudnnRNNDescriptor_t = Ptr{cudnnRNNStruct}
const cudnnPersistentRNNPlan = Cvoid
const cudnnPersistentRNNPlan_t = Ptr{cudnnPersistentRNNPlan}
const cudnnRNNDataStruct = Cvoid
const cudnnRNNDataDescriptor_t = Ptr{cudnnRNNDataStruct}
const cudnnAlgorithmStruct = Cvoid
const cudnnAlgorithmDescriptor_t = Ptr{cudnnAlgorithmStruct}
const cudnnAlgorithmPerformanceStruct = Cvoid
const cudnnAlgorithmPerformance_t = Ptr{cudnnAlgorithmPerformanceStruct}

@cenum cudnnSeqDataAxis_t::UInt32 begin
    CUDNN_SEQDATA_TIME_DIM = 0
    CUDNN_SEQDATA_BATCH_DIM = 1
    CUDNN_SEQDATA_BEAM_DIM = 2
    CUDNN_SEQDATA_VECT_DIM = 3
end


const cudnnSeqDataStruct = Cvoid
const cudnnSeqDataDescriptor_t = Ptr{cudnnSeqDataStruct}
const cudnnAttnQueryMap_t = UInt32
const cudnnAttnStruct = Cvoid
const cudnnAttnDescriptor_t = Ptr{cudnnAttnStruct}

@cenum cudnnMultiHeadAttnWeightKind_t::UInt32 begin
    CUDNN_MH_ATTN_Q_WEIGHTS = 0
    CUDNN_MH_ATTN_K_WEIGHTS = 1
    CUDNN_MH_ATTN_V_WEIGHTS = 2
    CUDNN_MH_ATTN_O_WEIGHTS = 3
    CUDNN_MH_ATTN_Q_BIASES = 4
    CUDNN_MH_ATTN_K_BIASES = 5
    CUDNN_MH_ATTN_V_BIASES = 6
    CUDNN_MH_ATTN_O_BIASES = 7
end

@cenum cudnnWgradMode_t::UInt32 begin
    CUDNN_WGRAD_MODE_ADD = 0
    CUDNN_WGRAD_MODE_SET = 1
end

@cenum cudnnCTCLossAlgo_t::UInt32 begin
    CUDNN_CTC_LOSS_ALGO_DETERMINISTIC = 0
    CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC = 1
end

@cenum cudnnLossNormalizationMode_t::UInt32 begin
    CUDNN_LOSS_NORMALIZATION_NONE = 0
    CUDNN_LOSS_NORMALIZATION_SOFTMAX = 1
end


# FIXME: can't use such a union as the type in a ccall expression
#Algorithm = Union{cudnnConvolutionFwdAlgo_t, cudnnConvolutionBwdFilterAlgo_t, cudnnConvolutionBwdDataAlgo_t, cudnnRNNAlgo_t, cudnnCTCLossAlgo_t}
#struct cudnnAlgorithm_t
#    algo::Algorithm
#end
cudnnAlgorithm_t = Cint

struct cudnnDebug_t
    cudnn_version::UInt32
    cudnnStatus::cudnnStatus_t
    time_sec::UInt32
    time_usec::UInt32
    time_delta::UInt32
    handle::cudnnHandle_t
    stream::CUstream
    pid::Culonglong
    tid::Culonglong
    cudaDeviceId::Cint
    reserved::NTuple{15, Cint}
end

const cudnnCallback_t = Ptr{Cvoid}
const cudnnFusedOpsConstParamStruct = Cvoid
const cudnnFusedOpsConstParamPack_t = Ptr{cudnnFusedOpsConstParamStruct}
const cudnnFusedOpsVariantParamStruct = Cvoid
const cudnnFusedOpsVariantParamPack_t = Ptr{cudnnFusedOpsVariantParamStruct}
const cudnnFusedOpsPlanStruct = Cvoid
const cudnnFusedOpsPlan_t = Ptr{cudnnFusedOpsPlanStruct}

@cenum cudnnFusedOps_t::UInt32 begin
    CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS = 0
    CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD = 1
    CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING = 2
    CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE = 3
    CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION = 4
    CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK = 5
    CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM = 6
end

@cenum cudnnFusedOpsConstParamLabel_t::UInt32 begin
    CUDNN_PARAM_XDESC = 0
    CUDNN_PARAM_XDATA_PLACEHOLDER = 1
    CUDNN_PARAM_BN_MODE = 2
    CUDNN_PARAM_BN_EQSCALEBIAS_DESC = 3
    CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER = 4
    CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER = 5
    CUDNN_PARAM_ACTIVATION_DESC = 6
    CUDNN_PARAM_CONV_DESC = 7
    CUDNN_PARAM_WDESC = 8
    CUDNN_PARAM_WDATA_PLACEHOLDER = 9
    CUDNN_PARAM_DWDESC = 10
    CUDNN_PARAM_DWDATA_PLACEHOLDER = 11
    CUDNN_PARAM_YDESC = 12
    CUDNN_PARAM_YDATA_PLACEHOLDER = 13
    CUDNN_PARAM_DYDESC = 14
    CUDNN_PARAM_DYDATA_PLACEHOLDER = 15
    CUDNN_PARAM_YSTATS_DESC = 16
    CUDNN_PARAM_YSUM_PLACEHOLDER = 17
    CUDNN_PARAM_YSQSUM_PLACEHOLDER = 18
    CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC = 19
    CUDNN_PARAM_BN_SCALE_PLACEHOLDER = 20
    CUDNN_PARAM_BN_BIAS_PLACEHOLDER = 21
    CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER = 22
    CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER = 23
    CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER = 24
    CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER = 25
    CUDNN_PARAM_ZDESC = 26
    CUDNN_PARAM_ZDATA_PLACEHOLDER = 27
    CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC = 28
    CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER = 29
    CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER = 30
    CUDNN_PARAM_ACTIVATION_BITMASK_DESC = 31
    CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER = 32
    CUDNN_PARAM_DXDESC = 33
    CUDNN_PARAM_DXDATA_PLACEHOLDER = 34
    CUDNN_PARAM_DZDESC = 35
    CUDNN_PARAM_DZDATA_PLACEHOLDER = 36
    CUDNN_PARAM_BN_DSCALE_PLACEHOLDER = 37
    CUDNN_PARAM_BN_DBIAS_PLACEHOLDER = 38
end

@cenum cudnnFusedOpsPointerPlaceHolder_t::UInt32 begin
    CUDNN_PTR_NULL = 0
    CUDNN_PTR_ELEM_ALIGNED = 1
    CUDNN_PTR_16B_ALIGNED = 2
end

@cenum cudnnFusedOpsVariantParamLabel_t::UInt32 begin
    CUDNN_PTR_XDATA = 0
    CUDNN_PTR_BN_EQSCALE = 1
    CUDNN_PTR_BN_EQBIAS = 2
    CUDNN_PTR_WDATA = 3
    CUDNN_PTR_DWDATA = 4
    CUDNN_PTR_YDATA = 5
    CUDNN_PTR_DYDATA = 6
    CUDNN_PTR_YSUM = 7
    CUDNN_PTR_YSQSUM = 8
    CUDNN_PTR_WORKSPACE = 9
    CUDNN_PTR_BN_SCALE = 10
    CUDNN_PTR_BN_BIAS = 11
    CUDNN_PTR_BN_SAVED_MEAN = 12
    CUDNN_PTR_BN_SAVED_INVSTD = 13
    CUDNN_PTR_BN_RUNNING_MEAN = 14
    CUDNN_PTR_BN_RUNNING_VAR = 15
    CUDNN_PTR_ZDATA = 16
    CUDNN_PTR_BN_Z_EQSCALE = 17
    CUDNN_PTR_BN_Z_EQBIAS = 18
    CUDNN_PTR_ACTIVATION_BITMASK = 19
    CUDNN_PTR_DXDATA = 20
    CUDNN_PTR_DZDATA = 21
    CUDNN_PTR_BN_DSCALE = 22
    CUDNN_PTR_BN_DBIAS = 23
    CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES = 100
    CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT = 101
    CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR = 102
    CUDNN_SCALAR_DOUBLE_BN_EPSILON = 103
end

