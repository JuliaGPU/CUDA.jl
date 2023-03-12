using CEnum

# cuDNN uses CUDA runtime objects, which are compatible with our driver usage
const cudaStream_t = CUstream

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == CUDNN_STATUS_ALLOC_FAILED
        throw(OutOfGPUMemoryError())
    else
        throw(CUDNNError(res))
    end
end

macro check(ex, errs...)
    check = :(isequal(err, CUDNN_STATUS_ALLOC_FAILED))
    for err in errs
        check = :($check || isequal(err, $(esc(err))))
    end

    quote
        res = @retry_reclaim err -> $check $(esc(ex))
        if res != CUDNN_STATUS_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end

mutable struct cudnnContext end

const cudnnHandle_t = Ptr{cudnnContext}

function cudnnGetVersion()
    @ccall libcudnn.cudnnGetVersion()::Csize_t
end

function cudnnGetMaxDeviceVersion()
    @ccall libcudnn.cudnnGetMaxDeviceVersion()::Csize_t
end

function cudnnGetCudartVersion()
    @ccall libcudnn.cudnnGetCudartVersion()::Csize_t
end

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
    CUDNN_STATUS_VERSION_MISMATCH = 14
end

function cudnnGetErrorString(status)
    @ccall libcudnn.cudnnGetErrorString(status::cudnnStatus_t)::Cstring
end

mutable struct cudnnRuntimeTag_t end

@cenum cudnnErrQueryMode_t::UInt32 begin
    CUDNN_ERRQUERY_RAWCODE = 0
    CUDNN_ERRQUERY_NONBLOCKING = 1
    CUDNN_ERRQUERY_BLOCKING = 2
end

@checked function cudnnQueryRuntimeError(handle, rstatus, mode, tag)
    initialize_context()
    @ccall libcudnn.cudnnQueryRuntimeError(handle::cudnnHandle_t,
                                           rstatus::Ptr{cudnnStatus_t},
                                           mode::cudnnErrQueryMode_t,
                                           tag::Ptr{cudnnRuntimeTag_t})::cudnnStatus_t
end

@checked function cudnnGetProperty(type, value)
    @ccall libcudnn.cudnnGetProperty(type::libraryPropertyType,
                                     value::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnCreate(handle)
    initialize_context()
    @ccall libcudnn.cudnnCreate(handle::Ptr{cudnnHandle_t})::cudnnStatus_t
end

@checked function cudnnDestroy(handle)
    initialize_context()
    @ccall libcudnn.cudnnDestroy(handle::cudnnHandle_t)::cudnnStatus_t
end

@checked function cudnnSetStream(handle, streamId)
    initialize_context()
    @ccall libcudnn.cudnnSetStream(handle::cudnnHandle_t,
                                   streamId::cudaStream_t)::cudnnStatus_t
end

@checked function cudnnGetStream(handle, streamId)
    initialize_context()
    @ccall libcudnn.cudnnGetStream(handle::cudnnHandle_t,
                                   streamId::Ptr{cudaStream_t})::cudnnStatus_t
end

mutable struct cudnnTensorStruct end

const cudnnTensorDescriptor_t = Ptr{cudnnTensorStruct}

mutable struct cudnnPoolingStruct end

const cudnnPoolingDescriptor_t = Ptr{cudnnPoolingStruct}

mutable struct cudnnFilterStruct end

const cudnnFilterDescriptor_t = Ptr{cudnnFilterStruct}

mutable struct cudnnLRNStruct end

const cudnnLRNDescriptor_t = Ptr{cudnnLRNStruct}

mutable struct cudnnActivationStruct end

const cudnnActivationDescriptor_t = Ptr{cudnnActivationStruct}

mutable struct cudnnSpatialTransformerStruct end

const cudnnSpatialTransformerDescriptor_t = Ptr{cudnnSpatialTransformerStruct}

mutable struct cudnnOpTensorStruct end

const cudnnOpTensorDescriptor_t = Ptr{cudnnOpTensorStruct}

mutable struct cudnnReduceTensorStruct end

const cudnnReduceTensorDescriptor_t = Ptr{cudnnReduceTensorStruct}

mutable struct cudnnCTCLossStruct end

const cudnnCTCLossDescriptor_t = Ptr{cudnnCTCLossStruct}

mutable struct cudnnTensorTransformStruct end

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
    CUDNN_DATA_BFLOAT16 = 9
    CUDNN_DATA_INT64 = 10
    CUDNN_DATA_BOOLEAN = 11
    CUDNN_DATA_FP8_E4M3 = 12
    CUDNN_DATA_FP8_E5M2 = 13
    CUDNN_DATA_FAST_FLOAT_FOR_FP8 = 14
end

@cenum cudnnMathType_t::UInt32 begin
    CUDNN_DEFAULT_MATH = 0
    CUDNN_TENSOR_OP_MATH = 1
    CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = 2
    CUDNN_FMA_MATH = 3
end

@cenum cudnnNanPropagation_t::UInt32 begin
    CUDNN_NOT_PROPAGATE_NAN = 0
    CUDNN_PROPAGATE_NAN = 1
end

@cenum cudnnDeterminism_t::UInt32 begin
    CUDNN_NON_DETERMINISTIC = 0
    CUDNN_DETERMINISTIC = 1
end

@checked function cudnnCreateTensorDescriptor(tensorDesc)
    initialize_context()
    @ccall libcudnn.cudnnCreateTensorDescriptor(tensorDesc::Ptr{cudnnTensorDescriptor_t})::cudnnStatus_t
end

@cenum cudnnTensorFormat_t::UInt32 begin
    CUDNN_TENSOR_NCHW = 0
    CUDNN_TENSOR_NHWC = 1
    CUDNN_TENSOR_NCHW_VECT_C = 2
end

@checked function cudnnSetTensor4dDescriptor(tensorDesc, format, dataType, n, c, h, w)
    initialize_context()
    @ccall libcudnn.cudnnSetTensor4dDescriptor(tensorDesc::cudnnTensorDescriptor_t,
                                               format::cudnnTensorFormat_t,
                                               dataType::cudnnDataType_t, n::Cint, c::Cint,
                                               h::Cint, w::Cint)::cudnnStatus_t
end

@checked function cudnnSetTensor4dDescriptorEx(tensorDesc, dataType, n, c, h, w, nStride,
                                               cStride, hStride, wStride)
    initialize_context()
    @ccall libcudnn.cudnnSetTensor4dDescriptorEx(tensorDesc::cudnnTensorDescriptor_t,
                                                 dataType::cudnnDataType_t, n::Cint,
                                                 c::Cint, h::Cint, w::Cint, nStride::Cint,
                                                 cStride::Cint, hStride::Cint,
                                                 wStride::Cint)::cudnnStatus_t
end

@checked function cudnnGetTensor4dDescriptor(tensorDesc, dataType, n, c, h, w, nStride,
                                             cStride, hStride, wStride)
    initialize_context()
    @ccall libcudnn.cudnnGetTensor4dDescriptor(tensorDesc::cudnnTensorDescriptor_t,
                                               dataType::Ptr{cudnnDataType_t}, n::Ptr{Cint},
                                               c::Ptr{Cint}, h::Ptr{Cint}, w::Ptr{Cint},
                                               nStride::Ptr{Cint}, cStride::Ptr{Cint},
                                               hStride::Ptr{Cint},
                                               wStride::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnSetTensorNdDescriptor(tensorDesc, dataType, nbDims, dimA, strideA)
    initialize_context()
    @ccall libcudnn.cudnnSetTensorNdDescriptor(tensorDesc::cudnnTensorDescriptor_t,
                                               dataType::cudnnDataType_t, nbDims::Cint,
                                               dimA::Ptr{Cint},
                                               strideA::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnSetTensorNdDescriptorEx(tensorDesc, format, dataType, nbDims, dimA)
    initialize_context()
    @ccall libcudnn.cudnnSetTensorNdDescriptorEx(tensorDesc::cudnnTensorDescriptor_t,
                                                 format::cudnnTensorFormat_t,
                                                 dataType::cudnnDataType_t, nbDims::Cint,
                                                 dimA::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnGetTensorNdDescriptor(tensorDesc, nbDimsRequested, dataType, nbDims,
                                             dimA, strideA)
    initialize_context()
    @ccall libcudnn.cudnnGetTensorNdDescriptor(tensorDesc::cudnnTensorDescriptor_t,
                                               nbDimsRequested::Cint,
                                               dataType::Ptr{cudnnDataType_t},
                                               nbDims::Ptr{Cint}, dimA::Ptr{Cint},
                                               strideA::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnGetTensorSizeInBytes(tensorDesc, size)
    initialize_context()
    @ccall libcudnn.cudnnGetTensorSizeInBytes(tensorDesc::cudnnTensorDescriptor_t,
                                              size::Ptr{Csize_t})::cudnnStatus_t
end

@checked function cudnnDestroyTensorDescriptor(tensorDesc)
    initialize_context()
    @ccall libcudnn.cudnnDestroyTensorDescriptor(tensorDesc::cudnnTensorDescriptor_t)::cudnnStatus_t
end

@cenum cudnnFoldingDirection_t::UInt32 begin
    CUDNN_TRANSFORM_FOLD = 0
    CUDNN_TRANSFORM_UNFOLD = 1
end

@checked function cudnnInitTransformDest(transformDesc, srcDesc, destDesc, destSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnInitTransformDest(transformDesc::cudnnTensorTransformDescriptor_t,
                                           srcDesc::cudnnTensorDescriptor_t,
                                           destDesc::cudnnTensorDescriptor_t,
                                           destSizeInBytes::Ptr{Csize_t})::cudnnStatus_t
end

@checked function cudnnCreateTensorTransformDescriptor(transformDesc)
    initialize_context()
    @ccall libcudnn.cudnnCreateTensorTransformDescriptor(transformDesc::Ptr{cudnnTensorTransformDescriptor_t})::cudnnStatus_t
end

@checked function cudnnSetTensorTransformDescriptor(transformDesc, nbDims, destFormat,
                                                    padBeforeA, padAfterA, foldA, direction)
    initialize_context()
    @ccall libcudnn.cudnnSetTensorTransformDescriptor(transformDesc::cudnnTensorTransformDescriptor_t,
                                                      nbDims::UInt32,
                                                      destFormat::cudnnTensorFormat_t,
                                                      padBeforeA::Ptr{Int32},
                                                      padAfterA::Ptr{Int32},
                                                      foldA::Ptr{UInt32},
                                                      direction::cudnnFoldingDirection_t)::cudnnStatus_t
end

@checked function cudnnGetTensorTransformDescriptor(transformDesc, nbDimsRequested,
                                                    destFormat, padBeforeA, padAfterA,
                                                    foldA, direction)
    initialize_context()
    @ccall libcudnn.cudnnGetTensorTransformDescriptor(transformDesc::cudnnTensorTransformDescriptor_t,
                                                      nbDimsRequested::UInt32,
                                                      destFormat::Ptr{cudnnTensorFormat_t},
                                                      padBeforeA::Ptr{Int32},
                                                      padAfterA::Ptr{Int32},
                                                      foldA::Ptr{UInt32},
                                                      direction::Ptr{cudnnFoldingDirection_t})::cudnnStatus_t
end

@checked function cudnnDestroyTensorTransformDescriptor(transformDesc)
    initialize_context()
    @ccall libcudnn.cudnnDestroyTensorTransformDescriptor(transformDesc::cudnnTensorTransformDescriptor_t)::cudnnStatus_t
end

@checked function cudnnTransformTensor(handle, alpha, xDesc, x, beta, yDesc, y)
    initialize_context()
    @ccall libcudnn.cudnnTransformTensor(handle::cudnnHandle_t, alpha::Ptr{Cvoid},
                                         xDesc::cudnnTensorDescriptor_t, x::Ptr{Cvoid},
                                         beta::Ptr{Cvoid}, yDesc::cudnnTensorDescriptor_t,
                                         y::Ptr{Cvoid})::cudnnStatus_t
end

@checked function cudnnTransformTensorEx(handle, transDesc, alpha, srcDesc, srcData, beta,
                                         destDesc, destData)
    initialize_context()
    @ccall libcudnn.cudnnTransformTensorEx(handle::cudnnHandle_t,
                                           transDesc::cudnnTensorTransformDescriptor_t,
                                           alpha::Ptr{Cvoid},
                                           srcDesc::cudnnTensorDescriptor_t,
                                           srcData::Ptr{Cvoid}, beta::Ptr{Cvoid},
                                           destDesc::cudnnTensorDescriptor_t,
                                           destData::Ptr{Cvoid})::cudnnStatus_t
end

@checked function cudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C)
    initialize_context()
    @ccall libcudnn.cudnnAddTensor(handle::cudnnHandle_t, alpha::Ptr{Cvoid},
                                   aDesc::cudnnTensorDescriptor_t, A::CuPtr{Cvoid},
                                   beta::Ptr{Cvoid}, cDesc::cudnnTensorDescriptor_t,
                                   C::CuPtr{Cvoid})::cudnnStatus_t
end

@cenum cudnnOpTensorOp_t::UInt32 begin
    CUDNN_OP_TENSOR_ADD = 0
    CUDNN_OP_TENSOR_MUL = 1
    CUDNN_OP_TENSOR_MIN = 2
    CUDNN_OP_TENSOR_MAX = 3
    CUDNN_OP_TENSOR_SQRT = 4
    CUDNN_OP_TENSOR_NOT = 5
end

@checked function cudnnCreateOpTensorDescriptor(opTensorDesc)
    initialize_context()
    @ccall libcudnn.cudnnCreateOpTensorDescriptor(opTensorDesc::Ptr{cudnnOpTensorDescriptor_t})::cudnnStatus_t
end

@checked function cudnnSetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType,
                                             opTensorNanOpt)
    initialize_context()
    @ccall libcudnn.cudnnSetOpTensorDescriptor(opTensorDesc::cudnnOpTensorDescriptor_t,
                                               opTensorOp::cudnnOpTensorOp_t,
                                               opTensorCompType::cudnnDataType_t,
                                               opTensorNanOpt::cudnnNanPropagation_t)::cudnnStatus_t
end

@checked function cudnnGetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType,
                                             opTensorNanOpt)
    initialize_context()
    @ccall libcudnn.cudnnGetOpTensorDescriptor(opTensorDesc::cudnnOpTensorDescriptor_t,
                                               opTensorOp::Ptr{cudnnOpTensorOp_t},
                                               opTensorCompType::Ptr{cudnnDataType_t},
                                               opTensorNanOpt::Ptr{cudnnNanPropagation_t})::cudnnStatus_t
end

@checked function cudnnDestroyOpTensorDescriptor(opTensorDesc)
    initialize_context()
    @ccall libcudnn.cudnnDestroyOpTensorDescriptor(opTensorDesc::cudnnOpTensorDescriptor_t)::cudnnStatus_t
end

@checked function cudnnOpTensor(handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B,
                                beta, cDesc, C)
    initialize_context()
    @ccall libcudnn.cudnnOpTensor(handle::cudnnHandle_t,
                                  opTensorDesc::cudnnOpTensorDescriptor_t,
                                  alpha1::Ptr{Cvoid}, aDesc::cudnnTensorDescriptor_t,
                                  A::CuPtr{Cvoid}, alpha2::Ptr{Cvoid},
                                  bDesc::cudnnTensorDescriptor_t, B::CuPtr{Cvoid},
                                  beta::Ptr{Cvoid}, cDesc::cudnnTensorDescriptor_t,
                                  C::CuPtr{Cvoid})::cudnnStatus_t
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

@checked function cudnnCreateReduceTensorDescriptor(reduceTensorDesc)
    initialize_context()
    @ccall libcudnn.cudnnCreateReduceTensorDescriptor(reduceTensorDesc::Ptr{cudnnReduceTensorDescriptor_t})::cudnnStatus_t
end

@checked function cudnnSetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp,
                                                 reduceTensorCompType, reduceTensorNanOpt,
                                                 reduceTensorIndices,
                                                 reduceTensorIndicesType)
    initialize_context()
    @ccall libcudnn.cudnnSetReduceTensorDescriptor(reduceTensorDesc::cudnnReduceTensorDescriptor_t,
                                                   reduceTensorOp::cudnnReduceTensorOp_t,
                                                   reduceTensorCompType::cudnnDataType_t,
                                                   reduceTensorNanOpt::cudnnNanPropagation_t,
                                                   reduceTensorIndices::cudnnReduceTensorIndices_t,
                                                   reduceTensorIndicesType::cudnnIndicesType_t)::cudnnStatus_t
end

@checked function cudnnGetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp,
                                                 reduceTensorCompType, reduceTensorNanOpt,
                                                 reduceTensorIndices,
                                                 reduceTensorIndicesType)
    initialize_context()
    @ccall libcudnn.cudnnGetReduceTensorDescriptor(reduceTensorDesc::cudnnReduceTensorDescriptor_t,
                                                   reduceTensorOp::Ptr{cudnnReduceTensorOp_t},
                                                   reduceTensorCompType::Ptr{cudnnDataType_t},
                                                   reduceTensorNanOpt::Ptr{cudnnNanPropagation_t},
                                                   reduceTensorIndices::Ptr{cudnnReduceTensorIndices_t},
                                                   reduceTensorIndicesType::Ptr{cudnnIndicesType_t})::cudnnStatus_t
end

@checked function cudnnDestroyReduceTensorDescriptor(reduceTensorDesc)
    initialize_context()
    @ccall libcudnn.cudnnDestroyReduceTensorDescriptor(reduceTensorDesc::cudnnReduceTensorDescriptor_t)::cudnnStatus_t
end

@checked function cudnnGetReductionIndicesSize(handle, reduceTensorDesc, aDesc, cDesc,
                                               sizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnGetReductionIndicesSize(handle::cudnnHandle_t,
                                                 reduceTensorDesc::cudnnReduceTensorDescriptor_t,
                                                 aDesc::cudnnTensorDescriptor_t,
                                                 cDesc::cudnnTensorDescriptor_t,
                                                 sizeInBytes::Ptr{Csize_t})::cudnnStatus_t
end

@checked function cudnnGetReductionWorkspaceSize(handle, reduceTensorDesc, aDesc, cDesc,
                                                 sizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnGetReductionWorkspaceSize(handle::cudnnHandle_t,
                                                   reduceTensorDesc::cudnnReduceTensorDescriptor_t,
                                                   aDesc::cudnnTensorDescriptor_t,
                                                   cDesc::cudnnTensorDescriptor_t,
                                                   sizeInBytes::Ref{Csize_t})::cudnnStatus_t
end

@checked function cudnnReduceTensor(handle, reduceTensorDesc, indices, indicesSizeInBytes,
                                    workspace, workspaceSizeInBytes, alpha, aDesc, A, beta,
                                    cDesc, C)
    initialize_context()
    @ccall libcudnn.cudnnReduceTensor(handle::cudnnHandle_t,
                                      reduceTensorDesc::cudnnReduceTensorDescriptor_t,
                                      indices::Ptr{Cvoid}, indicesSizeInBytes::Csize_t,
                                      workspace::CuPtr{Cvoid},
                                      workspaceSizeInBytes::Csize_t, alpha::Ptr{Cvoid},
                                      aDesc::cudnnTensorDescriptor_t, A::CuPtr{Cvoid},
                                      beta::Ptr{Cvoid}, cDesc::cudnnTensorDescriptor_t,
                                      C::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnSetTensor(handle, yDesc, y, valuePtr)
    initialize_context()
    @ccall libcudnn.cudnnSetTensor(handle::cudnnHandle_t, yDesc::cudnnTensorDescriptor_t,
                                   y::CuPtr{Cvoid}, valuePtr::Ptr{Cvoid})::cudnnStatus_t
end

@checked function cudnnScaleTensor(handle, yDesc, y, alpha)
    initialize_context()
    @ccall libcudnn.cudnnScaleTensor(handle::cudnnHandle_t, yDesc::cudnnTensorDescriptor_t,
                                     y::CuPtr{Cvoid}, alpha::Ptr{Cvoid})::cudnnStatus_t
end

@checked function cudnnCreateFilterDescriptor(filterDesc)
    initialize_context()
    @ccall libcudnn.cudnnCreateFilterDescriptor(filterDesc::Ptr{cudnnFilterDescriptor_t})::cudnnStatus_t
end

@checked function cudnnSetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w)
    initialize_context()
    @ccall libcudnn.cudnnSetFilter4dDescriptor(filterDesc::cudnnFilterDescriptor_t,
                                               dataType::cudnnDataType_t,
                                               format::cudnnTensorFormat_t, k::Cint,
                                               c::Cint, h::Cint, w::Cint)::cudnnStatus_t
end

@checked function cudnnGetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w)
    initialize_context()
    @ccall libcudnn.cudnnGetFilter4dDescriptor(filterDesc::cudnnFilterDescriptor_t,
                                               dataType::Ptr{cudnnDataType_t},
                                               format::Ptr{cudnnTensorFormat_t},
                                               k::Ptr{Cint}, c::Ptr{Cint}, h::Ptr{Cint},
                                               w::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnSetFilterNdDescriptor(filterDesc, dataType, format, nbDims,
                                             filterDimA)
    initialize_context()
    @ccall libcudnn.cudnnSetFilterNdDescriptor(filterDesc::cudnnFilterDescriptor_t,
                                               dataType::cudnnDataType_t,
                                               format::cudnnTensorFormat_t, nbDims::Cint,
                                               filterDimA::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnGetFilterNdDescriptor(filterDesc, nbDimsRequested, dataType, format,
                                             nbDims, filterDimA)
    initialize_context()
    @ccall libcudnn.cudnnGetFilterNdDescriptor(filterDesc::cudnnFilterDescriptor_t,
                                               nbDimsRequested::Cint,
                                               dataType::Ptr{cudnnDataType_t},
                                               format::Ptr{cudnnTensorFormat_t},
                                               nbDims::Ptr{Cint},
                                               filterDimA::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnGetFilterSizeInBytes(filterDesc, size)
    initialize_context()
    @ccall libcudnn.cudnnGetFilterSizeInBytes(filterDesc::cudnnFilterDescriptor_t,
                                              size::Ptr{Csize_t})::cudnnStatus_t
end

@checked function cudnnTransformFilter(handle, transDesc, alpha, srcDesc, srcData, beta,
                                       destDesc, destData)
    initialize_context()
    @ccall libcudnn.cudnnTransformFilter(handle::cudnnHandle_t,
                                         transDesc::cudnnTensorTransformDescriptor_t,
                                         alpha::Ptr{Cvoid},
                                         srcDesc::cudnnFilterDescriptor_t,
                                         srcData::CuPtr{Cvoid}, beta::Ptr{Cvoid},
                                         destDesc::cudnnFilterDescriptor_t,
                                         destData::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnDestroyFilterDescriptor(filterDesc)
    initialize_context()
    @ccall libcudnn.cudnnDestroyFilterDescriptor(filterDesc::cudnnFilterDescriptor_t)::cudnnStatus_t
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

@checked function cudnnSoftmaxForward(handle, algo, mode, alpha, xDesc, x, beta, yDesc, y)
    initialize_context()
    @ccall libcudnn.cudnnSoftmaxForward(handle::cudnnHandle_t,
                                        algo::cudnnSoftmaxAlgorithm_t,
                                        mode::cudnnSoftmaxMode_t, alpha::Ptr{Cvoid},
                                        xDesc::cudnnTensorDescriptor_t, x::CuPtr{Cvoid},
                                        beta::Ptr{Cvoid}, yDesc::cudnnTensorDescriptor_t,
                                        y::CuPtr{Cvoid})::cudnnStatus_t
end

@cenum cudnnPoolingMode_t::UInt32 begin
    CUDNN_POOLING_MAX = 0
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2
    CUDNN_POOLING_MAX_DETERMINISTIC = 3
end

@checked function cudnnCreatePoolingDescriptor(poolingDesc)
    initialize_context()
    @ccall libcudnn.cudnnCreatePoolingDescriptor(poolingDesc::Ptr{cudnnPoolingDescriptor_t})::cudnnStatus_t
end

@checked function cudnnSetPooling2dDescriptor(poolingDesc, mode, maxpoolingNanOpt,
                                              windowHeight, windowWidth, verticalPadding,
                                              horizontalPadding, verticalStride,
                                              horizontalStride)
    initialize_context()
    @ccall libcudnn.cudnnSetPooling2dDescriptor(poolingDesc::cudnnPoolingDescriptor_t,
                                                mode::cudnnPoolingMode_t,
                                                maxpoolingNanOpt::cudnnNanPropagation_t,
                                                windowHeight::Cint, windowWidth::Cint,
                                                verticalPadding::Cint,
                                                horizontalPadding::Cint,
                                                verticalStride::Cint,
                                                horizontalStride::Cint)::cudnnStatus_t
end

@checked function cudnnGetPooling2dDescriptor(poolingDesc, mode, maxpoolingNanOpt,
                                              windowHeight, windowWidth, verticalPadding,
                                              horizontalPadding, verticalStride,
                                              horizontalStride)
    initialize_context()
    @ccall libcudnn.cudnnGetPooling2dDescriptor(poolingDesc::cudnnPoolingDescriptor_t,
                                                mode::Ptr{cudnnPoolingMode_t},
                                                maxpoolingNanOpt::Ptr{cudnnNanPropagation_t},
                                                windowHeight::Ptr{Cint},
                                                windowWidth::Ptr{Cint},
                                                verticalPadding::Ptr{Cint},
                                                horizontalPadding::Ptr{Cint},
                                                verticalStride::Ptr{Cint},
                                                horizontalStride::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnSetPoolingNdDescriptor(poolingDesc, mode, maxpoolingNanOpt, nbDims,
                                              windowDimA, paddingA, strideA)
    initialize_context()
    @ccall libcudnn.cudnnSetPoolingNdDescriptor(poolingDesc::cudnnPoolingDescriptor_t,
                                                mode::cudnnPoolingMode_t,
                                                maxpoolingNanOpt::cudnnNanPropagation_t,
                                                nbDims::Cint, windowDimA::Ptr{Cint},
                                                paddingA::Ptr{Cint},
                                                strideA::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnGetPoolingNdDescriptor(poolingDesc, nbDimsRequested, mode,
                                              maxpoolingNanOpt, nbDims, windowDimA,
                                              paddingA, strideA)
    initialize_context()
    @ccall libcudnn.cudnnGetPoolingNdDescriptor(poolingDesc::cudnnPoolingDescriptor_t,
                                                nbDimsRequested::Cint,
                                                mode::Ptr{cudnnPoolingMode_t},
                                                maxpoolingNanOpt::Ptr{cudnnNanPropagation_t},
                                                nbDims::Ptr{Cint}, windowDimA::Ptr{Cint},
                                                paddingA::Ptr{Cint},
                                                strideA::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnGetPoolingNdForwardOutputDim(poolingDesc, inputTensorDesc, nbDims,
                                                    outputTensorDimA)
    initialize_context()
    @ccall libcudnn.cudnnGetPoolingNdForwardOutputDim(poolingDesc::cudnnPoolingDescriptor_t,
                                                      inputTensorDesc::cudnnTensorDescriptor_t,
                                                      nbDims::Cint,
                                                      outputTensorDimA::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnGetPooling2dForwardOutputDim(poolingDesc, inputTensorDesc, n, c, h,
                                                    w)
    initialize_context()
    @ccall libcudnn.cudnnGetPooling2dForwardOutputDim(poolingDesc::cudnnPoolingDescriptor_t,
                                                      inputTensorDesc::cudnnTensorDescriptor_t,
                                                      n::Ptr{Cint}, c::Ptr{Cint},
                                                      h::Ptr{Cint},
                                                      w::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnDestroyPoolingDescriptor(poolingDesc)
    initialize_context()
    @ccall libcudnn.cudnnDestroyPoolingDescriptor(poolingDesc::cudnnPoolingDescriptor_t)::cudnnStatus_t
end

@checked function cudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y)
    initialize_context()
    @ccall libcudnn.cudnnPoolingForward(handle::cudnnHandle_t,
                                        poolingDesc::cudnnPoolingDescriptor_t,
                                        alpha::Ptr{Cvoid}, xDesc::cudnnTensorDescriptor_t,
                                        x::CuPtr{Cvoid}, beta::Ptr{Cvoid},
                                        yDesc::cudnnTensorDescriptor_t,
                                        y::CuPtr{Cvoid})::cudnnStatus_t
end

@cenum cudnnActivationMode_t::UInt32 begin
    CUDNN_ACTIVATION_SIGMOID = 0
    CUDNN_ACTIVATION_RELU = 1
    CUDNN_ACTIVATION_TANH = 2
    CUDNN_ACTIVATION_CLIPPED_RELU = 3
    CUDNN_ACTIVATION_ELU = 4
    CUDNN_ACTIVATION_IDENTITY = 5
    CUDNN_ACTIVATION_SWISH = 6
end

@checked function cudnnCreateActivationDescriptor(activationDesc)
    initialize_context()
    @ccall libcudnn.cudnnCreateActivationDescriptor(activationDesc::Ptr{cudnnActivationDescriptor_t})::cudnnStatus_t
end

@checked function cudnnSetActivationDescriptor(activationDesc, mode, reluNanOpt, coef)
    initialize_context()
    @ccall libcudnn.cudnnSetActivationDescriptor(activationDesc::cudnnActivationDescriptor_t,
                                                 mode::cudnnActivationMode_t,
                                                 reluNanOpt::cudnnNanPropagation_t,
                                                 coef::Cdouble)::cudnnStatus_t
end

@checked function cudnnGetActivationDescriptor(activationDesc, mode, reluNanOpt, coef)
    initialize_context()
    @ccall libcudnn.cudnnGetActivationDescriptor(activationDesc::cudnnActivationDescriptor_t,
                                                 mode::Ptr{cudnnActivationMode_t},
                                                 reluNanOpt::Ptr{cudnnNanPropagation_t},
                                                 coef::Ptr{Cdouble})::cudnnStatus_t
end

@checked function cudnnSetActivationDescriptorSwishBeta(activationDesc, swish_beta)
    initialize_context()
    @ccall libcudnn.cudnnSetActivationDescriptorSwishBeta(activationDesc::cudnnActivationDescriptor_t,
                                                          swish_beta::Cdouble)::cudnnStatus_t
end

@checked function cudnnGetActivationDescriptorSwishBeta(activationDesc, swish_beta)
    initialize_context()
    @ccall libcudnn.cudnnGetActivationDescriptorSwishBeta(activationDesc::cudnnActivationDescriptor_t,
                                                          swish_beta::Ptr{Cdouble})::cudnnStatus_t
end

@checked function cudnnDestroyActivationDescriptor(activationDesc)
    initialize_context()
    @ccall libcudnn.cudnnDestroyActivationDescriptor(activationDesc::cudnnActivationDescriptor_t)::cudnnStatus_t
end

@checked function cudnnActivationForward(handle, activationDesc, alpha, xDesc, x, beta,
                                         yDesc, y)
    initialize_context()
    @ccall libcudnn.cudnnActivationForward(handle::cudnnHandle_t,
                                           activationDesc::cudnnActivationDescriptor_t,
                                           alpha::Ptr{Cvoid},
                                           xDesc::cudnnTensorDescriptor_t, x::CuPtr{Cvoid},
                                           beta::Ptr{Cvoid}, yDesc::cudnnTensorDescriptor_t,
                                           y::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnCreateLRNDescriptor(normDesc)
    initialize_context()
    @ccall libcudnn.cudnnCreateLRNDescriptor(normDesc::Ptr{cudnnLRNDescriptor_t})::cudnnStatus_t
end

@cenum cudnnLRNMode_t::UInt32 begin
    CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0
end

@checked function cudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
    initialize_context()
    @ccall libcudnn.cudnnSetLRNDescriptor(normDesc::cudnnLRNDescriptor_t, lrnN::Cuint,
                                          lrnAlpha::Cdouble, lrnBeta::Cdouble,
                                          lrnK::Cdouble)::cudnnStatus_t
end

@checked function cudnnGetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
    initialize_context()
    @ccall libcudnn.cudnnGetLRNDescriptor(normDesc::cudnnLRNDescriptor_t, lrnN::Ptr{Cuint},
                                          lrnAlpha::Ptr{Cdouble}, lrnBeta::Ptr{Cdouble},
                                          lrnK::Ptr{Cdouble})::cudnnStatus_t
end

@checked function cudnnDestroyLRNDescriptor(lrnDesc)
    initialize_context()
    @ccall libcudnn.cudnnDestroyLRNDescriptor(lrnDesc::cudnnLRNDescriptor_t)::cudnnStatus_t
end

@checked function cudnnLRNCrossChannelForward(handle, normDesc, lrnMode, alpha, xDesc, x,
                                              beta, yDesc, y)
    initialize_context()
    @ccall libcudnn.cudnnLRNCrossChannelForward(handle::cudnnHandle_t,
                                                normDesc::cudnnLRNDescriptor_t,
                                                lrnMode::cudnnLRNMode_t, alpha::Ptr{Cvoid},
                                                xDesc::cudnnTensorDescriptor_t,
                                                x::CuPtr{Cvoid}, beta::Ptr{Cvoid},
                                                yDesc::cudnnTensorDescriptor_t,
                                                y::CuPtr{Cvoid})::cudnnStatus_t
end

@cenum cudnnDivNormMode_t::UInt32 begin
    CUDNN_DIVNORM_PRECOMPUTED_MEANS = 0
end

@checked function cudnnDivisiveNormalizationForward(handle, normDesc, mode, alpha, xDesc, x,
                                                    means, temp, temp2, beta, yDesc, y)
    initialize_context()
    @ccall libcudnn.cudnnDivisiveNormalizationForward(handle::cudnnHandle_t,
                                                      normDesc::cudnnLRNDescriptor_t,
                                                      mode::cudnnDivNormMode_t,
                                                      alpha::Ptr{Cvoid},
                                                      xDesc::cudnnTensorDescriptor_t,
                                                      x::CuPtr{Cvoid}, means::CuPtr{Cvoid},
                                                      temp::CuPtr{Cvoid},
                                                      temp2::CuPtr{Cvoid}, beta::Ptr{Cvoid},
                                                      yDesc::cudnnTensorDescriptor_t,
                                                      y::CuPtr{Cvoid})::cudnnStatus_t
end

@cenum cudnnBatchNormMode_t::UInt32 begin
    CUDNN_BATCHNORM_PER_ACTIVATION = 0
    CUDNN_BATCHNORM_SPATIAL = 1
    CUDNN_BATCHNORM_SPATIAL_PERSISTENT = 2
end

@checked function cudnnDeriveBNTensorDescriptor(derivedBnDesc, xDesc, mode)
    initialize_context()
    @ccall libcudnn.cudnnDeriveBNTensorDescriptor(derivedBnDesc::cudnnTensorDescriptor_t,
                                                  xDesc::cudnnTensorDescriptor_t,
                                                  mode::cudnnBatchNormMode_t)::cudnnStatus_t
end

@cenum cudnnBatchNormOps_t::UInt32 begin
    CUDNN_BATCHNORM_OPS_BN = 0
    CUDNN_BATCHNORM_OPS_BN_ACTIVATION = 1
    CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION = 2
end

@checked function cudnnBatchNormalizationForwardInference(handle, mode, alpha, beta, xDesc,
                                                          x, yDesc, y,
                                                          bnScaleBiasMeanVarDesc, bnScale,
                                                          bnBias, estimatedMean,
                                                          estimatedVariance, epsilon)
    initialize_context()
    @ccall libcudnn.cudnnBatchNormalizationForwardInference(handle::cudnnHandle_t,
                                                            mode::cudnnBatchNormMode_t,
                                                            alpha::Ptr{Cvoid},
                                                            beta::Ptr{Cvoid},
                                                            xDesc::cudnnTensorDescriptor_t,
                                                            x::CuPtr{Cvoid},
                                                            yDesc::cudnnTensorDescriptor_t,
                                                            y::CuPtr{Cvoid},
                                                            bnScaleBiasMeanVarDesc::cudnnTensorDescriptor_t,
                                                            bnScale::CuPtr{Cvoid},
                                                            bnBias::CuPtr{Cvoid},
                                                            estimatedMean::CuPtr{Cvoid},
                                                            estimatedVariance::CuPtr{Cvoid},
                                                            epsilon::Cdouble)::cudnnStatus_t
end

@cenum cudnnNormMode_t::UInt32 begin
    CUDNN_NORM_PER_ACTIVATION = 0
    CUDNN_NORM_PER_CHANNEL = 1
end

@cenum cudnnNormAlgo_t::UInt32 begin
    CUDNN_NORM_ALGO_STANDARD = 0
    CUDNN_NORM_ALGO_PERSIST = 1
end

@checked function cudnnDeriveNormTensorDescriptor(derivedNormScaleBiasDesc,
                                                  derivedNormMeanVarDesc, xDesc, mode,
                                                  groupCnt)
    initialize_context()
    @ccall libcudnn.cudnnDeriveNormTensorDescriptor(derivedNormScaleBiasDesc::cudnnTensorDescriptor_t,
                                                    derivedNormMeanVarDesc::cudnnTensorDescriptor_t,
                                                    xDesc::cudnnTensorDescriptor_t,
                                                    mode::cudnnNormMode_t,
                                                    groupCnt::Cint)::cudnnStatus_t
end

@cenum cudnnNormOps_t::UInt32 begin
    CUDNN_NORM_OPS_NORM = 0
    CUDNN_NORM_OPS_NORM_ACTIVATION = 1
    CUDNN_NORM_OPS_NORM_ADD_ACTIVATION = 2
end

@checked function cudnnNormalizationForwardInference(handle, mode, normOps, algo, alpha,
                                                     beta, xDesc, x, normScaleBiasDesc,
                                                     normScale, normBias, normMeanVarDesc,
                                                     estimatedMean, estimatedVariance,
                                                     zDesc, z, activationDesc, yDesc, y,
                                                     epsilon, groupCnt)
    initialize_context()
    @ccall libcudnn.cudnnNormalizationForwardInference(handle::cudnnHandle_t,
                                                       mode::cudnnNormMode_t,
                                                       normOps::cudnnNormOps_t,
                                                       algo::cudnnNormAlgo_t,
                                                       alpha::Ptr{Cvoid}, beta::Ptr{Cvoid},
                                                       xDesc::cudnnTensorDescriptor_t,
                                                       x::CuPtr{Cvoid},
                                                       normScaleBiasDesc::cudnnTensorDescriptor_t,
                                                       normScale::CuPtr{Cvoid},
                                                       normBias::CuPtr{Cvoid},
                                                       normMeanVarDesc::cudnnTensorDescriptor_t,
                                                       estimatedMean::CuPtr{Cvoid},
                                                       estimatedVariance::CuPtr{Cvoid},
                                                       zDesc::cudnnTensorDescriptor_t,
                                                       z::CuPtr{Cvoid},
                                                       activationDesc::cudnnActivationDescriptor_t,
                                                       yDesc::cudnnTensorDescriptor_t,
                                                       y::CuPtr{Cvoid}, epsilon::Cdouble,
                                                       groupCnt::Cint)::cudnnStatus_t
end

@cenum cudnnSamplerType_t::UInt32 begin
    CUDNN_SAMPLER_BILINEAR = 0
end

@checked function cudnnCreateSpatialTransformerDescriptor(stDesc)
    initialize_context()
    @ccall libcudnn.cudnnCreateSpatialTransformerDescriptor(stDesc::Ptr{cudnnSpatialTransformerDescriptor_t})::cudnnStatus_t
end

@checked function cudnnSetSpatialTransformerNdDescriptor(stDesc, samplerType, dataType,
                                                         nbDims, dimA)
    initialize_context()
    @ccall libcudnn.cudnnSetSpatialTransformerNdDescriptor(stDesc::cudnnSpatialTransformerDescriptor_t,
                                                           samplerType::cudnnSamplerType_t,
                                                           dataType::cudnnDataType_t,
                                                           nbDims::Cint,
                                                           dimA::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnDestroySpatialTransformerDescriptor(stDesc)
    initialize_context()
    @ccall libcudnn.cudnnDestroySpatialTransformerDescriptor(stDesc::cudnnSpatialTransformerDescriptor_t)::cudnnStatus_t
end

@checked function cudnnSpatialTfGridGeneratorForward(handle, stDesc, theta, grid)
    initialize_context()
    @ccall libcudnn.cudnnSpatialTfGridGeneratorForward(handle::cudnnHandle_t,
                                                       stDesc::cudnnSpatialTransformerDescriptor_t,
                                                       theta::CuPtr{Cvoid},
                                                       grid::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnSpatialTfSamplerForward(handle, stDesc, alpha, xDesc, x, grid, beta,
                                               yDesc, y)
    initialize_context()
    @ccall libcudnn.cudnnSpatialTfSamplerForward(handle::cudnnHandle_t,
                                                 stDesc::cudnnSpatialTransformerDescriptor_t,
                                                 alpha::Ptr{Cvoid},
                                                 xDesc::cudnnTensorDescriptor_t,
                                                 x::CuPtr{Cvoid}, grid::CuPtr{Cvoid},
                                                 beta::Ptr{Cvoid},
                                                 yDesc::cudnnTensorDescriptor_t,
                                                 y::CuPtr{Cvoid})::cudnnStatus_t
end

mutable struct cudnnDropoutStruct end

const cudnnDropoutDescriptor_t = Ptr{cudnnDropoutStruct}

@checked function cudnnCreateDropoutDescriptor(dropoutDesc)
    initialize_context()
    @ccall libcudnn.cudnnCreateDropoutDescriptor(dropoutDesc::Ptr{cudnnDropoutDescriptor_t})::cudnnStatus_t
end

@checked function cudnnDestroyDropoutDescriptor(dropoutDesc)
    initialize_context()
    @ccall libcudnn.cudnnDestroyDropoutDescriptor(dropoutDesc::cudnnDropoutDescriptor_t)::cudnnStatus_t
end

@checked function cudnnDropoutGetStatesSize(handle, sizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnDropoutGetStatesSize(handle::cudnnHandle_t,
                                              sizeInBytes::Ptr{Csize_t})::cudnnStatus_t
end

@checked function cudnnDropoutGetReserveSpaceSize(xdesc, sizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnDropoutGetReserveSpaceSize(xdesc::cudnnTensorDescriptor_t,
                                                    sizeInBytes::Ref{Csize_t})::cudnnStatus_t
end

@checked function cudnnSetDropoutDescriptor(dropoutDesc, handle, dropout, states,
                                            stateSizeInBytes, seed)
    initialize_context()
    @ccall libcudnn.cudnnSetDropoutDescriptor(dropoutDesc::cudnnDropoutDescriptor_t,
                                              handle::cudnnHandle_t, dropout::Cfloat,
                                              states::CuPtr{Cvoid},
                                              stateSizeInBytes::Csize_t,
                                              seed::Culonglong)::cudnnStatus_t
end

@checked function cudnnRestoreDropoutDescriptor(dropoutDesc, handle, dropout, states,
                                                stateSizeInBytes, seed)
    initialize_context()
    @ccall libcudnn.cudnnRestoreDropoutDescriptor(dropoutDesc::cudnnDropoutDescriptor_t,
                                                  handle::cudnnHandle_t, dropout::Cfloat,
                                                  states::CuPtr{Cvoid},
                                                  stateSizeInBytes::Csize_t,
                                                  seed::Culonglong)::cudnnStatus_t
end

@checked function cudnnGetDropoutDescriptor(dropoutDesc, handle, dropout, states, seed)
    initialize_context()
    @ccall libcudnn.cudnnGetDropoutDescriptor(dropoutDesc::cudnnDropoutDescriptor_t,
                                              handle::cudnnHandle_t, dropout::Ptr{Cfloat},
                                              states::Ptr{CuPtr{Cvoid}},
                                              seed::Ptr{Culonglong})::cudnnStatus_t
end

@checked function cudnnDropoutForward(handle, dropoutDesc, xdesc, x, ydesc, y, reserveSpace,
                                      reserveSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnDropoutForward(handle::cudnnHandle_t,
                                        dropoutDesc::cudnnDropoutDescriptor_t,
                                        xdesc::cudnnTensorDescriptor_t, x::CuPtr{Cvoid},
                                        ydesc::cudnnTensorDescriptor_t, y::CuPtr{Cvoid},
                                        reserveSpace::CuPtr{Cvoid},
                                        reserveSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

mutable struct cudnnAlgorithmStruct end

const cudnnAlgorithmDescriptor_t = Ptr{cudnnAlgorithmStruct}

mutable struct cudnnAlgorithmPerformanceStruct end

const cudnnAlgorithmPerformance_t = Ptr{cudnnAlgorithmPerformanceStruct}

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

@cenum cudnnConvolutionBwdDataAlgo_t::UInt32 begin
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = 6
end

@cenum cudnnRNNAlgo_t::UInt32 begin
    CUDNN_RNN_ALGO_STANDARD = 0
    CUDNN_RNN_ALGO_PERSIST_STATIC = 1
    CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2
    CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H = 3
    CUDNN_RNN_ALGO_COUNT = 4
end

@cenum cudnnCTCLossAlgo_t::UInt32 begin
    CUDNN_CTC_LOSS_ALGO_DETERMINISTIC = 0
    CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC = 1
end

struct Algorithm
    data::NTuple{4,UInt8}
end

function Base.getproperty(x::Ptr{Algorithm}, f::Symbol)
    f === :convFwdAlgo && return Ptr{cudnnConvolutionFwdAlgo_t}(x + 0)
    f === :convBwdFilterAlgo && return Ptr{cudnnConvolutionBwdFilterAlgo_t}(x + 0)
    f === :convBwdDataAlgo && return Ptr{cudnnConvolutionBwdDataAlgo_t}(x + 0)
    f === :RNNAlgo && return Ptr{cudnnRNNAlgo_t}(x + 0)
    f === :CTCLossAlgo && return Ptr{cudnnCTCLossAlgo_t}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::Algorithm, f::Symbol)
    r = Ref{Algorithm}(x)
    ptr = Base.unsafe_convert(Ptr{Algorithm}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{Algorithm}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct cudnnAlgorithmUnionStruct
    data::NTuple{4,UInt8}
end

function Base.getproperty(x::Ptr{cudnnAlgorithmUnionStruct}, f::Symbol)
    f === :algo && return Ptr{Algorithm}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::cudnnAlgorithmUnionStruct, f::Symbol)
    r = Ref{cudnnAlgorithmUnionStruct}(x)
    ptr = Base.unsafe_convert(Ptr{cudnnAlgorithmUnionStruct}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{cudnnAlgorithmUnionStruct}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

const cudnnAlgorithm_t = cudnnAlgorithmUnionStruct

@checked function cudnnCreateAlgorithmDescriptor(algoDesc)
    initialize_context()
    @ccall libcudnn.cudnnCreateAlgorithmDescriptor(algoDesc::Ptr{cudnnAlgorithmDescriptor_t})::cudnnStatus_t
end

@checked function cudnnSetAlgorithmDescriptor(algoDesc, algorithm)
    initialize_context()
    @ccall libcudnn.cudnnSetAlgorithmDescriptor(algoDesc::cudnnAlgorithmDescriptor_t,
                                                algorithm::cudnnAlgorithm_t)::cudnnStatus_t
end

@checked function cudnnGetAlgorithmDescriptor(algoDesc, algorithm)
    initialize_context()
    @ccall libcudnn.cudnnGetAlgorithmDescriptor(algoDesc::cudnnAlgorithmDescriptor_t,
                                                algorithm::Ptr{cudnnAlgorithm_t})::cudnnStatus_t
end

@checked function cudnnCopyAlgorithmDescriptor(src, dest)
    initialize_context()
    @ccall libcudnn.cudnnCopyAlgorithmDescriptor(src::cudnnAlgorithmDescriptor_t,
                                                 dest::cudnnAlgorithmDescriptor_t)::cudnnStatus_t
end

@checked function cudnnDestroyAlgorithmDescriptor(algoDesc)
    initialize_context()
    @ccall libcudnn.cudnnDestroyAlgorithmDescriptor(algoDesc::cudnnAlgorithmDescriptor_t)::cudnnStatus_t
end

@checked function cudnnCreateAlgorithmPerformance(algoPerf, numberToCreate)
    initialize_context()
    @ccall libcudnn.cudnnCreateAlgorithmPerformance(algoPerf::Ptr{cudnnAlgorithmPerformance_t},
                                                    numberToCreate::Cint)::cudnnStatus_t
end

@checked function cudnnSetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory)
    initialize_context()
    @ccall libcudnn.cudnnSetAlgorithmPerformance(algoPerf::cudnnAlgorithmPerformance_t,
                                                 algoDesc::cudnnAlgorithmDescriptor_t,
                                                 status::cudnnStatus_t, time::Cfloat,
                                                 memory::Csize_t)::cudnnStatus_t
end

@checked function cudnnGetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory)
    initialize_context()
    @ccall libcudnn.cudnnGetAlgorithmPerformance(algoPerf::cudnnAlgorithmPerformance_t,
                                                 algoDesc::Ptr{cudnnAlgorithmDescriptor_t},
                                                 status::Ptr{cudnnStatus_t},
                                                 time::Ptr{Cfloat},
                                                 memory::Ptr{Csize_t})::cudnnStatus_t
end

@checked function cudnnDestroyAlgorithmPerformance(algoPerf, numberToDestroy)
    initialize_context()
    @ccall libcudnn.cudnnDestroyAlgorithmPerformance(algoPerf::Ptr{cudnnAlgorithmPerformance_t},
                                                     numberToDestroy::Cint)::cudnnStatus_t
end

@checked function cudnnGetAlgorithmSpaceSize(handle, algoDesc, algoSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnGetAlgorithmSpaceSize(handle::cudnnHandle_t,
                                               algoDesc::cudnnAlgorithmDescriptor_t,
                                               algoSpaceSizeInBytes::Ptr{Csize_t})::cudnnStatus_t
end

@checked function cudnnSaveAlgorithm(handle, algoDesc, algoSpace, algoSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnSaveAlgorithm(handle::cudnnHandle_t,
                                       algoDesc::cudnnAlgorithmDescriptor_t,
                                       algoSpace::Ptr{Cvoid},
                                       algoSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@checked function cudnnRestoreAlgorithm(handle, algoSpace, algoSpaceSizeInBytes, algoDesc)
    initialize_context()
    @ccall libcudnn.cudnnRestoreAlgorithm(handle::cudnnHandle_t, algoSpace::Ptr{Cvoid},
                                          algoSpaceSizeInBytes::Csize_t,
                                          algoDesc::cudnnAlgorithmDescriptor_t)::cudnnStatus_t
end

@cenum cudnnSeverity_t::UInt32 begin
    CUDNN_SEV_FATAL = 0
    CUDNN_SEV_ERROR = 1
    CUDNN_SEV_WARNING = 2
    CUDNN_SEV_INFO = 3
end

struct cudnnDebugStruct
    cudnn_version::Cuint
    cudnnStatus::cudnnStatus_t
    time_sec::Cuint
    time_usec::Cuint
    time_delta::Cuint
    handle::cudnnHandle_t
    stream::cudaStream_t
    pid::Culonglong
    tid::Culonglong
    cudaDeviceId::Cint
    reserved::NTuple{15,Cint}
end

const cudnnDebug_t = cudnnDebugStruct

# typedef void ( * cudnnCallback_t ) ( cudnnSeverity_t sev , void * udata , const cudnnDebug_t * dbg , const char * msg )
const cudnnCallback_t = Ptr{Cvoid}

@checked function cudnnSetCallback(mask, udata, fptr)
    @ccall libcudnn.cudnnSetCallback(mask::Cuint, udata::Ptr{Cvoid},
                                     fptr::cudnnCallback_t)::cudnnStatus_t
end

@checked function cudnnGetCallback(mask, udata, fptr)
    @ccall libcudnn.cudnnGetCallback(mask::Ptr{Cuint}, udata::Ptr{Ptr{Cvoid}},
                                     fptr::Ptr{cudnnCallback_t})::cudnnStatus_t
end

@checked function cudnnOpsInferVersionCheck()
    initialize_context()
    @ccall libcudnn.cudnnOpsInferVersionCheck()::cudnnStatus_t
end

@checked function cudnnSoftmaxBackward(handle, algo, mode, alpha, yDesc, y, dyDesc, dy,
                                       beta, dxDesc, dx)
    initialize_context()
    @ccall libcudnn.cudnnSoftmaxBackward(handle::cudnnHandle_t,
                                         algo::cudnnSoftmaxAlgorithm_t,
                                         mode::cudnnSoftmaxMode_t, alpha::Ptr{Cvoid},
                                         yDesc::cudnnTensorDescriptor_t, y::CuPtr{Cvoid},
                                         dyDesc::cudnnTensorDescriptor_t, dy::CuPtr{Cvoid},
                                         beta::Ptr{Cvoid}, dxDesc::cudnnTensorDescriptor_t,
                                         dx::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnPoolingBackward(handle, poolingDesc, alpha, yDesc, y, dyDesc, dy,
                                       xDesc, x, beta, dxDesc, dx)
    initialize_context()
    @ccall libcudnn.cudnnPoolingBackward(handle::cudnnHandle_t,
                                         poolingDesc::cudnnPoolingDescriptor_t,
                                         alpha::Ptr{Cvoid}, yDesc::cudnnTensorDescriptor_t,
                                         y::CuPtr{Cvoid}, dyDesc::cudnnTensorDescriptor_t,
                                         dy::CuPtr{Cvoid}, xDesc::cudnnTensorDescriptor_t,
                                         x::CuPtr{Cvoid}, beta::Ptr{Cvoid},
                                         dxDesc::cudnnTensorDescriptor_t,
                                         dx::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnActivationBackward(handle, activationDesc, alpha, yDesc, y, dyDesc,
                                          dy, xDesc, x, beta, dxDesc, dx)
    initialize_context()
    @ccall libcudnn.cudnnActivationBackward(handle::cudnnHandle_t,
                                            activationDesc::cudnnActivationDescriptor_t,
                                            alpha::Ptr{Cvoid},
                                            yDesc::cudnnTensorDescriptor_t, y::CuPtr{Cvoid},
                                            dyDesc::cudnnTensorDescriptor_t,
                                            dy::CuPtr{Cvoid},
                                            xDesc::cudnnTensorDescriptor_t, x::CuPtr{Cvoid},
                                            beta::Ptr{Cvoid},
                                            dxDesc::cudnnTensorDescriptor_t,
                                            dx::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnLRNCrossChannelBackward(handle, normDesc, lrnMode, alpha, yDesc, y,
                                               dyDesc, dy, xDesc, x, beta, dxDesc, dx)
    initialize_context()
    @ccall libcudnn.cudnnLRNCrossChannelBackward(handle::cudnnHandle_t,
                                                 normDesc::cudnnLRNDescriptor_t,
                                                 lrnMode::cudnnLRNMode_t, alpha::Ptr{Cvoid},
                                                 yDesc::cudnnTensorDescriptor_t,
                                                 y::CuPtr{Cvoid},
                                                 dyDesc::cudnnTensorDescriptor_t,
                                                 dy::CuPtr{Cvoid},
                                                 xDesc::cudnnTensorDescriptor_t,
                                                 x::CuPtr{Cvoid}, beta::Ptr{Cvoid},
                                                 dxDesc::cudnnTensorDescriptor_t,
                                                 dx::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnDivisiveNormalizationBackward(handle, normDesc, mode, alpha, xDesc,
                                                     x, means, dy, temp, temp2, beta,
                                                     dXdMeansDesc, dx, dMeans)
    initialize_context()
    @ccall libcudnn.cudnnDivisiveNormalizationBackward(handle::cudnnHandle_t,
                                                       normDesc::cudnnLRNDescriptor_t,
                                                       mode::cudnnDivNormMode_t,
                                                       alpha::Ptr{Cvoid},
                                                       xDesc::cudnnTensorDescriptor_t,
                                                       x::CuPtr{Cvoid}, means::CuPtr{Cvoid},
                                                       dy::CuPtr{Cvoid}, temp::CuPtr{Cvoid},
                                                       temp2::CuPtr{Cvoid},
                                                       beta::Ptr{Cvoid},
                                                       dXdMeansDesc::cudnnTensorDescriptor_t,
                                                       dx::CuPtr{Cvoid},
                                                       dMeans::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle, mode,
                                                                           bnOps, xDesc,
                                                                           zDesc, yDesc,
                                                                           bnScaleBiasMeanVarDesc,
                                                                           activationDesc,
                                                                           sizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle::cudnnHandle_t,
                                                                             mode::cudnnBatchNormMode_t,
                                                                             bnOps::cudnnBatchNormOps_t,
                                                                             xDesc::cudnnTensorDescriptor_t,
                                                                             zDesc::cudnnTensorDescriptor_t,
                                                                             yDesc::cudnnTensorDescriptor_t,
                                                                             bnScaleBiasMeanVarDesc::cudnnTensorDescriptor_t,
                                                                             activationDesc::cudnnActivationDescriptor_t,
                                                                             sizeInBytes::Ref{Csize_t})::cudnnStatus_t
end

@checked function cudnnGetBatchNormalizationBackwardExWorkspaceSize(handle, mode, bnOps,
                                                                    xDesc, yDesc, dyDesc,
                                                                    dzDesc, dxDesc,
                                                                    dBnScaleBiasDesc,
                                                                    activationDesc,
                                                                    sizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnGetBatchNormalizationBackwardExWorkspaceSize(handle::cudnnHandle_t,
                                                                      mode::cudnnBatchNormMode_t,
                                                                      bnOps::cudnnBatchNormOps_t,
                                                                      xDesc::cudnnTensorDescriptor_t,
                                                                      yDesc::cudnnTensorDescriptor_t,
                                                                      dyDesc::cudnnTensorDescriptor_t,
                                                                      dzDesc::cudnnTensorDescriptor_t,
                                                                      dxDesc::cudnnTensorDescriptor_t,
                                                                      dBnScaleBiasDesc::cudnnTensorDescriptor_t,
                                                                      activationDesc::cudnnActivationDescriptor_t,
                                                                      sizeInBytes::Ref{Csize_t})::cudnnStatus_t
end

@checked function cudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle, mode, bnOps,
                                                                       activationDesc,
                                                                       xDesc, sizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle::cudnnHandle_t,
                                                                         mode::cudnnBatchNormMode_t,
                                                                         bnOps::cudnnBatchNormOps_t,
                                                                         activationDesc::cudnnActivationDescriptor_t,
                                                                         xDesc::cudnnTensorDescriptor_t,
                                                                         sizeInBytes::Ref{Csize_t})::cudnnStatus_t
end

@checked function cudnnBatchNormalizationForwardTraining(handle, mode, alpha, beta, xDesc,
                                                         x, yDesc, y,
                                                         bnScaleBiasMeanVarDesc, bnScale,
                                                         bnBias, exponentialAverageFactor,
                                                         resultRunningMean,
                                                         resultRunningVariance, epsilon,
                                                         resultSaveMean,
                                                         resultSaveInvVariance)
    initialize_context()
    @ccall libcudnn.cudnnBatchNormalizationForwardTraining(handle::cudnnHandle_t,
                                                           mode::cudnnBatchNormMode_t,
                                                           alpha::Ptr{Cvoid},
                                                           beta::Ptr{Cvoid},
                                                           xDesc::cudnnTensorDescriptor_t,
                                                           x::CuPtr{Cvoid},
                                                           yDesc::cudnnTensorDescriptor_t,
                                                           y::CuPtr{Cvoid},
                                                           bnScaleBiasMeanVarDesc::cudnnTensorDescriptor_t,
                                                           bnScale::CuPtr{Cvoid},
                                                           bnBias::CuPtr{Cvoid},
                                                           exponentialAverageFactor::Cdouble,
                                                           resultRunningMean::CuPtr{Cvoid},
                                                           resultRunningVariance::CuPtr{Cvoid},
                                                           epsilon::Cdouble,
                                                           resultSaveMean::CuPtr{Cvoid},
                                                           resultSaveInvVariance::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnBatchNormalizationForwardTrainingEx(handle, mode, bnOps, alpha, beta,
                                                           xDesc, xData, zDesc, zData,
                                                           yDesc, yData,
                                                           bnScaleBiasMeanVarDesc, bnScale,
                                                           bnBias, exponentialAverageFactor,
                                                           resultRunningMean,
                                                           resultRunningVariance, epsilon,
                                                           resultSaveMean,
                                                           resultSaveInvVariance,
                                                           activationDesc, workspace,
                                                           workSpaceSizeInBytes,
                                                           reserveSpace,
                                                           reserveSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnBatchNormalizationForwardTrainingEx(handle::cudnnHandle_t,
                                                             mode::cudnnBatchNormMode_t,
                                                             bnOps::cudnnBatchNormOps_t,
                                                             alpha::Ptr{Cvoid},
                                                             beta::Ptr{Cvoid},
                                                             xDesc::cudnnTensorDescriptor_t,
                                                             xData::CuPtr{Cvoid},
                                                             zDesc::cudnnTensorDescriptor_t,
                                                             zData::CuPtr{Cvoid},
                                                             yDesc::cudnnTensorDescriptor_t,
                                                             yData::CuPtr{Cvoid},
                                                             bnScaleBiasMeanVarDesc::cudnnTensorDescriptor_t,
                                                             bnScale::CuPtr{Cvoid},
                                                             bnBias::CuPtr{Cvoid},
                                                             exponentialAverageFactor::Cdouble,
                                                             resultRunningMean::CuPtr{Cvoid},
                                                             resultRunningVariance::CuPtr{Cvoid},
                                                             epsilon::Cdouble,
                                                             resultSaveMean::CuPtr{Cvoid},
                                                             resultSaveInvVariance::CuPtr{Cvoid},
                                                             activationDesc::cudnnActivationDescriptor_t,
                                                             workspace::CuPtr{Cvoid},
                                                             workSpaceSizeInBytes::Csize_t,
                                                             reserveSpace::CuPtr{Cvoid},
                                                             reserveSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@checked function cudnnBatchNormalizationBackward(handle, mode, alphaDataDiff, betaDataDiff,
                                                  alphaParamDiff, betaParamDiff, xDesc, x,
                                                  dyDesc, dy, dxDesc, dx, dBnScaleBiasDesc,
                                                  bnScale, dBnScaleResult, dBnBiasResult,
                                                  epsilon, savedMean, savedInvVariance)
    initialize_context()
    @ccall libcudnn.cudnnBatchNormalizationBackward(handle::cudnnHandle_t,
                                                    mode::cudnnBatchNormMode_t,
                                                    alphaDataDiff::Ptr{Cvoid},
                                                    betaDataDiff::Ptr{Cvoid},
                                                    alphaParamDiff::Ptr{Cvoid},
                                                    betaParamDiff::Ptr{Cvoid},
                                                    xDesc::cudnnTensorDescriptor_t,
                                                    x::CuPtr{Cvoid},
                                                    dyDesc::cudnnTensorDescriptor_t,
                                                    dy::CuPtr{Cvoid},
                                                    dxDesc::cudnnTensorDescriptor_t,
                                                    dx::CuPtr{Cvoid},
                                                    dBnScaleBiasDesc::cudnnTensorDescriptor_t,
                                                    bnScale::CuPtr{Cvoid},
                                                    dBnScaleResult::CuPtr{Cvoid},
                                                    dBnBiasResult::CuPtr{Cvoid},
                                                    epsilon::Cdouble,
                                                    savedMean::CuPtr{Cvoid},
                                                    savedInvVariance::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnBatchNormalizationBackwardEx(handle, mode, bnOps, alphaDataDiff,
                                                    betaDataDiff, alphaParamDiff,
                                                    betaParamDiff, xDesc, xData, yDesc,
                                                    yData, dyDesc, dyData, dzDesc, dzData,
                                                    dxDesc, dxData, dBnScaleBiasDesc,
                                                    bnScaleData, bnBiasData, dBnScaleData,
                                                    dBnBiasData, epsilon, savedMean,
                                                    savedInvVariance, activationDesc,
                                                    workSpace, workSpaceSizeInBytes,
                                                    reserveSpace, reserveSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnBatchNormalizationBackwardEx(handle::cudnnHandle_t,
                                                      mode::cudnnBatchNormMode_t,
                                                      bnOps::cudnnBatchNormOps_t,
                                                      alphaDataDiff::Ptr{Cvoid},
                                                      betaDataDiff::Ptr{Cvoid},
                                                      alphaParamDiff::Ptr{Cvoid},
                                                      betaParamDiff::Ptr{Cvoid},
                                                      xDesc::cudnnTensorDescriptor_t,
                                                      xData::CuPtr{Cvoid},
                                                      yDesc::cudnnTensorDescriptor_t,
                                                      yData::CuPtr{Cvoid},
                                                      dyDesc::cudnnTensorDescriptor_t,
                                                      dyData::CuPtr{Cvoid},
                                                      dzDesc::cudnnTensorDescriptor_t,
                                                      dzData::CuPtr{Cvoid},
                                                      dxDesc::cudnnTensorDescriptor_t,
                                                      dxData::CuPtr{Cvoid},
                                                      dBnScaleBiasDesc::cudnnTensorDescriptor_t,
                                                      bnScaleData::CuPtr{Cvoid},
                                                      bnBiasData::CuPtr{Cvoid},
                                                      dBnScaleData::CuPtr{Cvoid},
                                                      dBnBiasData::CuPtr{Cvoid},
                                                      epsilon::Cdouble,
                                                      savedMean::CuPtr{Cvoid},
                                                      savedInvVariance::CuPtr{Cvoid},
                                                      activationDesc::cudnnActivationDescriptor_t,
                                                      workSpace::CuPtr{Cvoid},
                                                      workSpaceSizeInBytes::Csize_t,
                                                      reserveSpace::CuPtr{Cvoid},
                                                      reserveSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@checked function cudnnGetNormalizationForwardTrainingWorkspaceSize(handle, mode, normOps,
                                                                    algo, xDesc, zDesc,
                                                                    yDesc,
                                                                    normScaleBiasDesc,
                                                                    activationDesc,
                                                                    normMeanVarDesc,
                                                                    sizeInBytes, groupCnt)
    initialize_context()
    @ccall libcudnn.cudnnGetNormalizationForwardTrainingWorkspaceSize(handle::cudnnHandle_t,
                                                                      mode::cudnnNormMode_t,
                                                                      normOps::cudnnNormOps_t,
                                                                      algo::cudnnNormAlgo_t,
                                                                      xDesc::cudnnTensorDescriptor_t,
                                                                      zDesc::cudnnTensorDescriptor_t,
                                                                      yDesc::cudnnTensorDescriptor_t,
                                                                      normScaleBiasDesc::cudnnTensorDescriptor_t,
                                                                      activationDesc::cudnnActivationDescriptor_t,
                                                                      normMeanVarDesc::cudnnTensorDescriptor_t,
                                                                      sizeInBytes::Ref{Csize_t},
                                                                      groupCnt::Cint)::cudnnStatus_t
end

@checked function cudnnGetNormalizationBackwardWorkspaceSize(handle, mode, normOps, algo,
                                                             xDesc, yDesc, dyDesc, dzDesc,
                                                             dxDesc, dNormScaleBiasDesc,
                                                             activationDesc,
                                                             normMeanVarDesc, sizeInBytes,
                                                             groupCnt)
    initialize_context()
    @ccall libcudnn.cudnnGetNormalizationBackwardWorkspaceSize(handle::cudnnHandle_t,
                                                               mode::cudnnNormMode_t,
                                                               normOps::cudnnNormOps_t,
                                                               algo::cudnnNormAlgo_t,
                                                               xDesc::cudnnTensorDescriptor_t,
                                                               yDesc::cudnnTensorDescriptor_t,
                                                               dyDesc::cudnnTensorDescriptor_t,
                                                               dzDesc::cudnnTensorDescriptor_t,
                                                               dxDesc::cudnnTensorDescriptor_t,
                                                               dNormScaleBiasDesc::cudnnTensorDescriptor_t,
                                                               activationDesc::cudnnActivationDescriptor_t,
                                                               normMeanVarDesc::cudnnTensorDescriptor_t,
                                                               sizeInBytes::Ref{Csize_t},
                                                               groupCnt::Cint)::cudnnStatus_t
end

@checked function cudnnGetNormalizationTrainingReserveSpaceSize(handle, mode, normOps, algo,
                                                                activationDesc, xDesc,
                                                                sizeInBytes, groupCnt)
    initialize_context()
    @ccall libcudnn.cudnnGetNormalizationTrainingReserveSpaceSize(handle::cudnnHandle_t,
                                                                  mode::cudnnNormMode_t,
                                                                  normOps::cudnnNormOps_t,
                                                                  algo::cudnnNormAlgo_t,
                                                                  activationDesc::cudnnActivationDescriptor_t,
                                                                  xDesc::cudnnTensorDescriptor_t,
                                                                  sizeInBytes::Ref{Csize_t},
                                                                  groupCnt::Cint)::cudnnStatus_t
end

@checked function cudnnNormalizationForwardTraining(handle, mode, normOps, algo, alpha,
                                                    beta, xDesc, xData, normScaleBiasDesc,
                                                    normScale, normBias,
                                                    exponentialAverageFactor,
                                                    normMeanVarDesc, resultRunningMean,
                                                    resultRunningVariance, epsilon,
                                                    resultSaveMean, resultSaveInvVariance,
                                                    activationDesc, zDesc, zData, yDesc,
                                                    yData, workspace, workSpaceSizeInBytes,
                                                    reserveSpace, reserveSpaceSizeInBytes,
                                                    groupCnt)
    initialize_context()
    @ccall libcudnn.cudnnNormalizationForwardTraining(handle::cudnnHandle_t,
                                                      mode::cudnnNormMode_t,
                                                      normOps::cudnnNormOps_t,
                                                      algo::cudnnNormAlgo_t,
                                                      alpha::Ptr{Cvoid}, beta::Ptr{Cvoid},
                                                      xDesc::cudnnTensorDescriptor_t,
                                                      xData::CuPtr{Cvoid},
                                                      normScaleBiasDesc::cudnnTensorDescriptor_t,
                                                      normScale::CuPtr{Cvoid},
                                                      normBias::CuPtr{Cvoid},
                                                      exponentialAverageFactor::Cdouble,
                                                      normMeanVarDesc::cudnnTensorDescriptor_t,
                                                      resultRunningMean::CuPtr{Cvoid},
                                                      resultRunningVariance::CuPtr{Cvoid},
                                                      epsilon::Cdouble,
                                                      resultSaveMean::CuPtr{Cvoid},
                                                      resultSaveInvVariance::CuPtr{Cvoid},
                                                      activationDesc::cudnnActivationDescriptor_t,
                                                      zDesc::cudnnTensorDescriptor_t,
                                                      zData::CuPtr{Cvoid},
                                                      yDesc::cudnnTensorDescriptor_t,
                                                      yData::CuPtr{Cvoid},
                                                      workspace::CuPtr{Cvoid},
                                                      workSpaceSizeInBytes::Csize_t,
                                                      reserveSpace::CuPtr{Cvoid},
                                                      reserveSpaceSizeInBytes::Csize_t,
                                                      groupCnt::Cint)::cudnnStatus_t
end

@checked function cudnnNormalizationBackward(handle, mode, normOps, algo, alphaDataDiff,
                                             betaDataDiff, alphaParamDiff, betaParamDiff,
                                             xDesc, xData, yDesc, yData, dyDesc, dyData,
                                             dzDesc, dzData, dxDesc, dxData,
                                             dNormScaleBiasDesc, normScaleData,
                                             normBiasData, dNormScaleData, dNormBiasData,
                                             epsilon, normMeanVarDesc, savedMean,
                                             savedInvVariance, activationDesc, workSpace,
                                             workSpaceSizeInBytes, reserveSpace,
                                             reserveSpaceSizeInBytes, groupCnt)
    initialize_context()
    @ccall libcudnn.cudnnNormalizationBackward(handle::cudnnHandle_t, mode::cudnnNormMode_t,
                                               normOps::cudnnNormOps_t,
                                               algo::cudnnNormAlgo_t,
                                               alphaDataDiff::Ptr{Cvoid},
                                               betaDataDiff::Ptr{Cvoid},
                                               alphaParamDiff::Ptr{Cvoid},
                                               betaParamDiff::Ptr{Cvoid},
                                               xDesc::cudnnTensorDescriptor_t,
                                               xData::CuPtr{Cvoid},
                                               yDesc::cudnnTensorDescriptor_t,
                                               yData::CuPtr{Cvoid},
                                               dyDesc::cudnnTensorDescriptor_t,
                                               dyData::CuPtr{Cvoid},
                                               dzDesc::cudnnTensorDescriptor_t,
                                               dzData::CuPtr{Cvoid},
                                               dxDesc::cudnnTensorDescriptor_t,
                                               dxData::CuPtr{Cvoid},
                                               dNormScaleBiasDesc::cudnnTensorDescriptor_t,
                                               normScaleData::CuPtr{Cvoid},
                                               normBiasData::CuPtr{Cvoid},
                                               dNormScaleData::CuPtr{Cvoid},
                                               dNormBiasData::CuPtr{Cvoid},
                                               epsilon::Cdouble,
                                               normMeanVarDesc::cudnnTensorDescriptor_t,
                                               savedMean::CuPtr{Cvoid},
                                               savedInvVariance::CuPtr{Cvoid},
                                               activationDesc::cudnnActivationDescriptor_t,
                                               workSpace::CuPtr{Cvoid},
                                               workSpaceSizeInBytes::Csize_t,
                                               reserveSpace::CuPtr{Cvoid},
                                               reserveSpaceSizeInBytes::Csize_t,
                                               groupCnt::Cint)::cudnnStatus_t
end

@checked function cudnnSpatialTfGridGeneratorBackward(handle, stDesc, dgrid, dtheta)
    initialize_context()
    @ccall libcudnn.cudnnSpatialTfGridGeneratorBackward(handle::cudnnHandle_t,
                                                        stDesc::cudnnSpatialTransformerDescriptor_t,
                                                        dgrid::CuPtr{Cvoid},
                                                        dtheta::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnSpatialTfSamplerBackward(handle, stDesc, alpha, xDesc, x, beta,
                                                dxDesc, dx, alphaDgrid, dyDesc, dy, grid,
                                                betaDgrid, dgrid)
    initialize_context()
    @ccall libcudnn.cudnnSpatialTfSamplerBackward(handle::cudnnHandle_t,
                                                  stDesc::cudnnSpatialTransformerDescriptor_t,
                                                  alpha::Ptr{Cvoid},
                                                  xDesc::cudnnTensorDescriptor_t,
                                                  x::CuPtr{Cvoid}, beta::Ptr{Cvoid},
                                                  dxDesc::cudnnTensorDescriptor_t,
                                                  dx::CuPtr{Cvoid}, alphaDgrid::Ptr{Cvoid},
                                                  dyDesc::cudnnTensorDescriptor_t,
                                                  dy::CuPtr{Cvoid}, grid::CuPtr{Cvoid},
                                                  betaDgrid::Ptr{Cvoid},
                                                  dgrid::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnDropoutBackward(handle, dropoutDesc, dydesc, dy, dxdesc, dx,
                                       reserveSpace, reserveSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnDropoutBackward(handle::cudnnHandle_t,
                                         dropoutDesc::cudnnDropoutDescriptor_t,
                                         dydesc::cudnnTensorDescriptor_t, dy::CuPtr{Cvoid},
                                         dxdesc::cudnnTensorDescriptor_t, dx::CuPtr{Cvoid},
                                         reserveSpace::CuPtr{Cvoid},
                                         reserveSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@checked function cudnnOpsTrainVersionCheck()
    initialize_context()
    @ccall libcudnn.cudnnOpsTrainVersionCheck()::cudnnStatus_t
end

@cenum cudnnForwardMode_t::UInt32 begin
    CUDNN_FWD_MODE_INFERENCE = 0
    CUDNN_FWD_MODE_TRAINING = 1
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

const cudnnRNNPaddingMode_t = Cuint

mutable struct cudnnRNNStruct end

const cudnnRNNDescriptor_t = Ptr{cudnnRNNStruct}

mutable struct cudnnPersistentRNNPlan end

const cudnnPersistentRNNPlan_t = Ptr{cudnnPersistentRNNPlan}

mutable struct cudnnRNNDataStruct end

const cudnnRNNDataDescriptor_t = Ptr{cudnnRNNDataStruct}

@checked function cudnnCreateRNNDescriptor(rnnDesc)
    initialize_context()
    @ccall libcudnn.cudnnCreateRNNDescriptor(rnnDesc::Ptr{cudnnRNNDescriptor_t})::cudnnStatus_t
end

@checked function cudnnDestroyRNNDescriptor(rnnDesc)
    initialize_context()
    @ccall libcudnn.cudnnDestroyRNNDescriptor(rnnDesc::cudnnRNNDescriptor_t)::cudnnStatus_t
end

@checked function cudnnSetRNNDescriptor_v8(rnnDesc, algo, cellMode, biasMode, dirMode,
                                           inputMode, dataType, mathPrec, mathType,
                                           inputSize, hiddenSize, projSize, numLayers,
                                           dropoutDesc, auxFlags)
    initialize_context()
    @ccall libcudnn.cudnnSetRNNDescriptor_v8(rnnDesc::cudnnRNNDescriptor_t,
                                             algo::cudnnRNNAlgo_t, cellMode::cudnnRNNMode_t,
                                             biasMode::cudnnRNNBiasMode_t,
                                             dirMode::cudnnDirectionMode_t,
                                             inputMode::cudnnRNNInputMode_t,
                                             dataType::cudnnDataType_t,
                                             mathPrec::cudnnDataType_t,
                                             mathType::cudnnMathType_t, inputSize::Int32,
                                             hiddenSize::Int32, projSize::Int32,
                                             numLayers::Int32,
                                             dropoutDesc::cudnnDropoutDescriptor_t,
                                             auxFlags::UInt32)::cudnnStatus_t
end

@checked function cudnnGetRNNDescriptor_v8(rnnDesc, algo, cellMode, biasMode, dirMode,
                                           inputMode, dataType, mathPrec, mathType,
                                           inputSize, hiddenSize, projSize, numLayers,
                                           dropoutDesc, auxFlags)
    initialize_context()
    @ccall libcudnn.cudnnGetRNNDescriptor_v8(rnnDesc::cudnnRNNDescriptor_t,
                                             algo::Ref{cudnnRNNAlgo_t},
                                             cellMode::Ref{cudnnRNNMode_t},
                                             biasMode::Ref{cudnnRNNBiasMode_t},
                                             dirMode::Ref{cudnnDirectionMode_t},
                                             inputMode::Ref{cudnnRNNInputMode_t},
                                             dataType::Ref{cudnnDataType_t},
                                             mathPrec::Ref{cudnnDataType_t},
                                             mathType::Ref{cudnnMathType_t},
                                             inputSize::Ref{Int32}, hiddenSize::Ref{Int32},
                                             projSize::Ref{Int32}, numLayers::Ref{Int32},
                                             dropoutDesc::Ref{cudnnDropoutDescriptor_t},
                                             auxFlags::Ref{UInt32})::cudnnStatus_t
end

@checked function cudnnSetRNNDescriptor_v6(handle, rnnDesc, hiddenSize, numLayers,
                                           dropoutDesc, inputMode, direction, cellMode,
                                           algo, mathPrec)
    initialize_context()
    @ccall libcudnn.cudnnSetRNNDescriptor_v6(handle::cudnnHandle_t,
                                             rnnDesc::cudnnRNNDescriptor_t,
                                             hiddenSize::Cint, numLayers::Cint,
                                             dropoutDesc::cudnnDropoutDescriptor_t,
                                             inputMode::cudnnRNNInputMode_t,
                                             direction::cudnnDirectionMode_t,
                                             cellMode::cudnnRNNMode_t, algo::cudnnRNNAlgo_t,
                                             mathPrec::cudnnDataType_t)::cudnnStatus_t
end

@checked function cudnnGetRNNDescriptor_v6(handle, rnnDesc, hiddenSize, numLayers,
                                           dropoutDesc, inputMode, direction, cellMode,
                                           algo, mathPrec)
    initialize_context()
    @ccall libcudnn.cudnnGetRNNDescriptor_v6(handle::cudnnHandle_t,
                                             rnnDesc::cudnnRNNDescriptor_t,
                                             hiddenSize::Ref{Cint}, numLayers::Ref{Cint},
                                             dropoutDesc::Ref{cudnnDropoutDescriptor_t},
                                             inputMode::Ref{cudnnRNNInputMode_t},
                                             direction::Ref{cudnnDirectionMode_t},
                                             cellMode::Ref{cudnnRNNMode_t},
                                             algo::Ref{cudnnRNNAlgo_t},
                                             mathPrec::Ref{cudnnDataType_t})::cudnnStatus_t
end

@checked function cudnnSetRNNMatrixMathType(rnnDesc, mType)
    initialize_context()
    @ccall libcudnn.cudnnSetRNNMatrixMathType(rnnDesc::cudnnRNNDescriptor_t,
                                              mType::cudnnMathType_t)::cudnnStatus_t
end

@checked function cudnnGetRNNMatrixMathType(rnnDesc, mType)
    initialize_context()
    @ccall libcudnn.cudnnGetRNNMatrixMathType(rnnDesc::cudnnRNNDescriptor_t,
                                              mType::Ptr{cudnnMathType_t})::cudnnStatus_t
end

@checked function cudnnSetRNNBiasMode(rnnDesc, biasMode)
    initialize_context()
    @ccall libcudnn.cudnnSetRNNBiasMode(rnnDesc::cudnnRNNDescriptor_t,
                                        biasMode::cudnnRNNBiasMode_t)::cudnnStatus_t
end

@checked function cudnnGetRNNBiasMode(rnnDesc, biasMode)
    initialize_context()
    @ccall libcudnn.cudnnGetRNNBiasMode(rnnDesc::cudnnRNNDescriptor_t,
                                        biasMode::Ptr{cudnnRNNBiasMode_t})::cudnnStatus_t
end

@checked function cudnnRNNSetClip_v8(rnnDesc, clipMode, clipNanOpt, lclip, rclip)
    initialize_context()
    @ccall libcudnn.cudnnRNNSetClip_v8(rnnDesc::cudnnRNNDescriptor_t,
                                       clipMode::cudnnRNNClipMode_t,
                                       clipNanOpt::cudnnNanPropagation_t, lclip::Cdouble,
                                       rclip::Cdouble)::cudnnStatus_t
end

@checked function cudnnRNNGetClip_v8(rnnDesc, clipMode, clipNanOpt, lclip, rclip)
    initialize_context()
    @ccall libcudnn.cudnnRNNGetClip_v8(rnnDesc::cudnnRNNDescriptor_t,
                                       clipMode::Ref{cudnnRNNClipMode_t},
                                       clipNanOpt::Ref{cudnnNanPropagation_t},
                                       lclip::Ref{Cdouble},
                                       rclip::Ref{Cdouble})::cudnnStatus_t
end

@checked function cudnnRNNSetClip(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip)
    initialize_context()
    @ccall libcudnn.cudnnRNNSetClip(handle::cudnnHandle_t, rnnDesc::cudnnRNNDescriptor_t,
                                    clipMode::cudnnRNNClipMode_t,
                                    clipNanOpt::cudnnNanPropagation_t, lclip::Cdouble,
                                    rclip::Cdouble)::cudnnStatus_t
end

@checked function cudnnRNNGetClip(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip)
    initialize_context()
    @ccall libcudnn.cudnnRNNGetClip(handle::cudnnHandle_t, rnnDesc::cudnnRNNDescriptor_t,
                                    clipMode::Ptr{cudnnRNNClipMode_t},
                                    clipNanOpt::Ptr{cudnnNanPropagation_t},
                                    lclip::Ptr{Cdouble}, rclip::Ptr{Cdouble})::cudnnStatus_t
end

@checked function cudnnSetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize)
    initialize_context()
    @ccall libcudnn.cudnnSetRNNProjectionLayers(handle::cudnnHandle_t,
                                                rnnDesc::cudnnRNNDescriptor_t,
                                                recProjSize::Cint,
                                                outProjSize::Cint)::cudnnStatus_t
end

@checked function cudnnGetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize)
    initialize_context()
    @ccall libcudnn.cudnnGetRNNProjectionLayers(handle::cudnnHandle_t,
                                                rnnDesc::cudnnRNNDescriptor_t,
                                                recProjSize::Ptr{Cint},
                                                outProjSize::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnCreatePersistentRNNPlan(rnnDesc, minibatch, dataType, plan)
    initialize_context()
    @ccall libcudnn.cudnnCreatePersistentRNNPlan(rnnDesc::cudnnRNNDescriptor_t,
                                                 minibatch::Cint, dataType::cudnnDataType_t,
                                                 plan::Ptr{cudnnPersistentRNNPlan_t})::cudnnStatus_t
end

@checked function cudnnDestroyPersistentRNNPlan(plan)
    initialize_context()
    @ccall libcudnn.cudnnDestroyPersistentRNNPlan(plan::cudnnPersistentRNNPlan_t)::cudnnStatus_t
end

@checked function cudnnSetPersistentRNNPlan(rnnDesc, plan)
    initialize_context()
    @ccall libcudnn.cudnnSetPersistentRNNPlan(rnnDesc::cudnnRNNDescriptor_t,
                                              plan::cudnnPersistentRNNPlan_t)::cudnnStatus_t
end

@checked function cudnnBuildRNNDynamic(handle, rnnDesc, miniBatch)
    initialize_context()
    @ccall libcudnn.cudnnBuildRNNDynamic(handle::cudnnHandle_t,
                                         rnnDesc::cudnnRNNDescriptor_t,
                                         miniBatch::Cint)::cudnnStatus_t
end

@checked function cudnnGetRNNWorkspaceSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnGetRNNWorkspaceSize(handle::cudnnHandle_t,
                                             rnnDesc::cudnnRNNDescriptor_t, seqLength::Cint,
                                             xDesc::Ptr{cudnnTensorDescriptor_t},
                                             sizeInBytes::Ref{Csize_t})::cudnnStatus_t
end

@checked function cudnnGetRNNTrainingReserveSize(handle, rnnDesc, seqLength, xDesc,
                                                 sizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnGetRNNTrainingReserveSize(handle::cudnnHandle_t,
                                                   rnnDesc::cudnnRNNDescriptor_t,
                                                   seqLength::Cint,
                                                   xDesc::Ptr{cudnnTensorDescriptor_t},
                                                   sizeInBytes::Ref{Csize_t})::cudnnStatus_t
end

@checked function cudnnGetRNNTempSpaceSizes(handle, rnnDesc, fMode, xDesc, workSpaceSize,
                                            reserveSpaceSize)
    initialize_context()
    @ccall libcudnn.cudnnGetRNNTempSpaceSizes(handle::cudnnHandle_t,
                                              rnnDesc::cudnnRNNDescriptor_t,
                                              fMode::cudnnForwardMode_t,
                                              xDesc::cudnnRNNDataDescriptor_t,
                                              workSpaceSize::Ref{Csize_t},
                                              reserveSpaceSize::Ref{Csize_t})::cudnnStatus_t
end

@checked function cudnnGetRNNParamsSize(handle, rnnDesc, xDesc, sizeInBytes, dataType)
    initialize_context()
    @ccall libcudnn.cudnnGetRNNParamsSize(handle::cudnnHandle_t,
                                          rnnDesc::cudnnRNNDescriptor_t,
                                          xDesc::cudnnTensorDescriptor_t,
                                          sizeInBytes::Ref{Csize_t},
                                          dataType::cudnnDataType_t)::cudnnStatus_t
end

@checked function cudnnGetRNNWeightSpaceSize(handle, rnnDesc, weightSpaceSize)
    initialize_context()
    @ccall libcudnn.cudnnGetRNNWeightSpaceSize(handle::cudnnHandle_t,
                                               rnnDesc::cudnnRNNDescriptor_t,
                                               weightSpaceSize::Ref{Csize_t})::cudnnStatus_t
end

@checked function cudnnGetRNNLinLayerMatrixParams(handle, rnnDesc, pseudoLayer, xDesc,
                                                  wDesc, w, linLayerID, linLayerMatDesc,
                                                  linLayerMat)
    initialize_context()
    @ccall libcudnn.cudnnGetRNNLinLayerMatrixParams(handle::cudnnHandle_t,
                                                    rnnDesc::cudnnRNNDescriptor_t,
                                                    pseudoLayer::Cint,
                                                    xDesc::cudnnTensorDescriptor_t,
                                                    wDesc::cudnnFilterDescriptor_t,
                                                    w::CuPtr{Cvoid}, linLayerID::Cint,
                                                    linLayerMatDesc::cudnnFilterDescriptor_t,
                                                    linLayerMat::Ptr{Ptr{Cvoid}})::cudnnStatus_t
end

@checked function cudnnGetRNNLinLayerBiasParams(handle, rnnDesc, pseudoLayer, xDesc, wDesc,
                                                w, linLayerID, linLayerBiasDesc,
                                                linLayerBias)
    initialize_context()
    @ccall libcudnn.cudnnGetRNNLinLayerBiasParams(handle::cudnnHandle_t,
                                                  rnnDesc::cudnnRNNDescriptor_t,
                                                  pseudoLayer::Cint,
                                                  xDesc::cudnnTensorDescriptor_t,
                                                  wDesc::cudnnFilterDescriptor_t,
                                                  w::CuPtr{Cvoid}, linLayerID::Cint,
                                                  linLayerBiasDesc::cudnnFilterDescriptor_t,
                                                  linLayerBias::Ptr{Ptr{Cvoid}})::cudnnStatus_t
end

@checked function cudnnGetRNNWeightParams(handle, rnnDesc, pseudoLayer, weightSpaceSize,
                                          weightSpace, linLayerID, mDesc, mAddr, bDesc,
                                          bAddr)
    initialize_context()
    @ccall libcudnn.cudnnGetRNNWeightParams(handle::cudnnHandle_t,
                                            rnnDesc::cudnnRNNDescriptor_t,
                                            pseudoLayer::Int32, weightSpaceSize::Csize_t,
                                            weightSpace::CuPtr{Cvoid}, linLayerID::Int32,
                                            mDesc::cudnnTensorDescriptor_t,
                                            mAddr::Ptr{CuPtr{Cvoid}},
                                            bDesc::cudnnTensorDescriptor_t,
                                            bAddr::Ptr{CuPtr{Cvoid}})::cudnnStatus_t
end

@checked function cudnnRNNForwardInference(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx,
                                           cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy,
                                           cyDesc, cy, workSpace, workSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnRNNForwardInference(handle::cudnnHandle_t,
                                             rnnDesc::cudnnRNNDescriptor_t, seqLength::Cint,
                                             xDesc::Ptr{cudnnTensorDescriptor_t},
                                             x::CuPtr{Cvoid},
                                             hxDesc::cudnnTensorDescriptor_t,
                                             hx::CuPtr{Cvoid},
                                             cxDesc::cudnnTensorDescriptor_t,
                                             cx::CuPtr{Cvoid},
                                             wDesc::cudnnFilterDescriptor_t,
                                             w::CuPtr{Cvoid},
                                             yDesc::Ptr{cudnnTensorDescriptor_t},
                                             y::CuPtr{Cvoid},
                                             hyDesc::cudnnTensorDescriptor_t,
                                             hy::CuPtr{Cvoid},
                                             cyDesc::cudnnTensorDescriptor_t,
                                             cy::CuPtr{Cvoid}, workSpace::CuPtr{Cvoid},
                                             workSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@checked function cudnnSetRNNPaddingMode(rnnDesc, paddingMode)
    initialize_context()
    @ccall libcudnn.cudnnSetRNNPaddingMode(rnnDesc::cudnnRNNDescriptor_t,
                                           paddingMode::Cuint)::cudnnStatus_t
end

@checked function cudnnGetRNNPaddingMode(rnnDesc, paddingMode)
    initialize_context()
    @ccall libcudnn.cudnnGetRNNPaddingMode(rnnDesc::cudnnRNNDescriptor_t,
                                           paddingMode::Ptr{Cuint})::cudnnStatus_t
end

@checked function cudnnCreateRNNDataDescriptor(rnnDataDesc)
    initialize_context()
    @ccall libcudnn.cudnnCreateRNNDataDescriptor(rnnDataDesc::Ptr{cudnnRNNDataDescriptor_t})::cudnnStatus_t
end

@checked function cudnnDestroyRNNDataDescriptor(rnnDataDesc)
    initialize_context()
    @ccall libcudnn.cudnnDestroyRNNDataDescriptor(rnnDataDesc::cudnnRNNDataDescriptor_t)::cudnnStatus_t
end

@checked function cudnnSetRNNDataDescriptor(rnnDataDesc, dataType, layout, maxSeqLength,
                                            batchSize, vectorSize, seqLengthArray,
                                            paddingFill)
    initialize_context()
    @ccall libcudnn.cudnnSetRNNDataDescriptor(rnnDataDesc::cudnnRNNDataDescriptor_t,
                                              dataType::cudnnDataType_t,
                                              layout::cudnnRNNDataLayout_t,
                                              maxSeqLength::Cint, batchSize::Cint,
                                              vectorSize::Cint, seqLengthArray::Ptr{Cint},
                                              paddingFill::Ptr{Cvoid})::cudnnStatus_t
end

@checked function cudnnGetRNNDataDescriptor(rnnDataDesc, dataType, layout, maxSeqLength,
                                            batchSize, vectorSize, arrayLengthRequested,
                                            seqLengthArray, paddingFill)
    initialize_context()
    @ccall libcudnn.cudnnGetRNNDataDescriptor(rnnDataDesc::cudnnRNNDataDescriptor_t,
                                              dataType::Ptr{cudnnDataType_t},
                                              layout::Ptr{cudnnRNNDataLayout_t},
                                              maxSeqLength::Ptr{Cint}, batchSize::Ptr{Cint},
                                              vectorSize::Ptr{Cint},
                                              arrayLengthRequested::Cint,
                                              seqLengthArray::Ptr{Cint},
                                              paddingFill::Ptr{Cvoid})::cudnnStatus_t
end

@checked function cudnnRNNForwardInferenceEx(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc,
                                             cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy,
                                             kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc,
                                             queries, workSpace, workSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnRNNForwardInferenceEx(handle::cudnnHandle_t,
                                               rnnDesc::cudnnRNNDescriptor_t,
                                               xDesc::cudnnRNNDataDescriptor_t,
                                               x::CuPtr{Cvoid},
                                               hxDesc::cudnnTensorDescriptor_t,
                                               hx::CuPtr{Cvoid},
                                               cxDesc::cudnnTensorDescriptor_t,
                                               cx::CuPtr{Cvoid},
                                               wDesc::cudnnFilterDescriptor_t,
                                               w::CuPtr{Cvoid},
                                               yDesc::cudnnRNNDataDescriptor_t,
                                               y::CuPtr{Cvoid},
                                               hyDesc::cudnnTensorDescriptor_t,
                                               hy::CuPtr{Cvoid},
                                               cyDesc::cudnnTensorDescriptor_t,
                                               cy::CuPtr{Cvoid},
                                               kDesc::cudnnRNNDataDescriptor_t,
                                               keys::Ptr{Cvoid},
                                               cDesc::cudnnRNNDataDescriptor_t,
                                               cAttn::Ptr{Cvoid},
                                               iDesc::cudnnRNNDataDescriptor_t,
                                               iAttn::Ptr{Cvoid},
                                               qDesc::cudnnRNNDataDescriptor_t,
                                               queries::CuPtr{Cvoid},
                                               workSpace::CuPtr{Cvoid},
                                               workSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@checked function cudnnRNNForward(handle, rnnDesc, fwdMode, devSeqLengths, xDesc, x, yDesc,
                                  y, hDesc, hx, hy, cDesc, cx, cy, weightSpaceSize,
                                  weightSpace, workSpaceSize, workSpace, reserveSpaceSize,
                                  reserveSpace)
    initialize_context()
    @ccall libcudnn.cudnnRNNForward(handle::cudnnHandle_t, rnnDesc::cudnnRNNDescriptor_t,
                                    fwdMode::cudnnForwardMode_t,
                                    devSeqLengths::CuPtr{Int32},
                                    xDesc::cudnnRNNDataDescriptor_t, x::CuPtr{Cvoid},
                                    yDesc::cudnnRNNDataDescriptor_t, y::CuPtr{Cvoid},
                                    hDesc::cudnnTensorDescriptor_t, hx::CuPtr{Cvoid},
                                    hy::CuPtr{Cvoid}, cDesc::cudnnTensorDescriptor_t,
                                    cx::CuPtr{Cvoid}, cy::CuPtr{Cvoid},
                                    weightSpaceSize::Csize_t, weightSpace::CuPtr{Cvoid},
                                    workSpaceSize::Csize_t, workSpace::CuPtr{Cvoid},
                                    reserveSpaceSize::Csize_t,
                                    reserveSpace::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnSetRNNAlgorithmDescriptor(handle, rnnDesc, algoDesc)
    initialize_context()
    @ccall libcudnn.cudnnSetRNNAlgorithmDescriptor(handle::cudnnHandle_t,
                                                   rnnDesc::cudnnRNNDescriptor_t,
                                                   algoDesc::cudnnAlgorithmDescriptor_t)::cudnnStatus_t
end

@checked function cudnnGetRNNForwardInferenceAlgorithmMaxCount(handle, rnnDesc, count)
    initialize_context()
    @ccall libcudnn.cudnnGetRNNForwardInferenceAlgorithmMaxCount(handle::cudnnHandle_t,
                                                                 rnnDesc::cudnnRNNDescriptor_t,
                                                                 count::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnFindRNNForwardInferenceAlgorithmEx(handle, rnnDesc, seqLength, xDesc,
                                                          x, hxDesc, hx, cxDesc, cx, wDesc,
                                                          w, yDesc, y, hyDesc, hy, cyDesc,
                                                          cy, findIntensity,
                                                          requestedAlgoCount,
                                                          returnedAlgoCount, perfResults,
                                                          workspace, workSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnFindRNNForwardInferenceAlgorithmEx(handle::cudnnHandle_t,
                                                            rnnDesc::cudnnRNNDescriptor_t,
                                                            seqLength::Cint,
                                                            xDesc::Ptr{cudnnTensorDescriptor_t},
                                                            x::CuPtr{Cvoid},
                                                            hxDesc::cudnnTensorDescriptor_t,
                                                            hx::CuPtr{Cvoid},
                                                            cxDesc::cudnnTensorDescriptor_t,
                                                            cx::CuPtr{Cvoid},
                                                            wDesc::cudnnFilterDescriptor_t,
                                                            w::CuPtr{Cvoid},
                                                            yDesc::Ptr{cudnnTensorDescriptor_t},
                                                            y::CuPtr{Cvoid},
                                                            hyDesc::cudnnTensorDescriptor_t,
                                                            hy::CuPtr{Cvoid},
                                                            cyDesc::cudnnTensorDescriptor_t,
                                                            cy::CuPtr{Cvoid},
                                                            findIntensity::Cfloat,
                                                            requestedAlgoCount::Cint,
                                                            returnedAlgoCount::Ptr{Cint},
                                                            perfResults::Ptr{cudnnAlgorithmPerformance_t},
                                                            workspace::CuPtr{Cvoid},
                                                            workSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@cenum cudnnSeqDataAxis_t::UInt32 begin
    CUDNN_SEQDATA_TIME_DIM = 0
    CUDNN_SEQDATA_BATCH_DIM = 1
    CUDNN_SEQDATA_BEAM_DIM = 2
    CUDNN_SEQDATA_VECT_DIM = 3
end

mutable struct cudnnSeqDataStruct end

const cudnnSeqDataDescriptor_t = Ptr{cudnnSeqDataStruct}

@checked function cudnnCreateSeqDataDescriptor(seqDataDesc)
    initialize_context()
    @ccall libcudnn.cudnnCreateSeqDataDescriptor(seqDataDesc::Ptr{cudnnSeqDataDescriptor_t})::cudnnStatus_t
end

@checked function cudnnDestroySeqDataDescriptor(seqDataDesc)
    initialize_context()
    @ccall libcudnn.cudnnDestroySeqDataDescriptor(seqDataDesc::cudnnSeqDataDescriptor_t)::cudnnStatus_t
end

@checked function cudnnSetSeqDataDescriptor(seqDataDesc, dataType, nbDims, dimA, axes,
                                            seqLengthArraySize, seqLengthArray, paddingFill)
    initialize_context()
    @ccall libcudnn.cudnnSetSeqDataDescriptor(seqDataDesc::cudnnSeqDataDescriptor_t,
                                              dataType::cudnnDataType_t, nbDims::Cint,
                                              dimA::Ptr{Cint},
                                              axes::Ptr{cudnnSeqDataAxis_t},
                                              seqLengthArraySize::Csize_t,
                                              seqLengthArray::Ptr{Cint},
                                              paddingFill::Ptr{Cvoid})::cudnnStatus_t
end

@checked function cudnnGetSeqDataDescriptor(seqDataDesc, dataType, nbDims, nbDimsRequested,
                                            dimA, axes, seqLengthArraySize,
                                            seqLengthSizeRequested, seqLengthArray,
                                            paddingFill)
    initialize_context()
    @ccall libcudnn.cudnnGetSeqDataDescriptor(seqDataDesc::cudnnSeqDataDescriptor_t,
                                              dataType::Ptr{cudnnDataType_t},
                                              nbDims::Ptr{Cint}, nbDimsRequested::Cint,
                                              dimA::Ptr{Cint},
                                              axes::Ptr{cudnnSeqDataAxis_t},
                                              seqLengthArraySize::Ptr{Csize_t},
                                              seqLengthSizeRequested::Csize_t,
                                              seqLengthArray::Ptr{Cint},
                                              paddingFill::Ptr{Cvoid})::cudnnStatus_t
end

const cudnnAttnQueryMap_t = Cuint

mutable struct cudnnAttnStruct end

const cudnnAttnDescriptor_t = Ptr{cudnnAttnStruct}

@checked function cudnnCreateAttnDescriptor(attnDesc)
    initialize_context()
    @ccall libcudnn.cudnnCreateAttnDescriptor(attnDesc::Ptr{cudnnAttnDescriptor_t})::cudnnStatus_t
end

@checked function cudnnDestroyAttnDescriptor(attnDesc)
    initialize_context()
    @ccall libcudnn.cudnnDestroyAttnDescriptor(attnDesc::cudnnAttnDescriptor_t)::cudnnStatus_t
end

@checked function cudnnSetAttnDescriptor(attnDesc, attnMode, nHeads, smScaler, dataType,
                                         computePrec, mathType, attnDropoutDesc,
                                         postDropoutDesc, qSize, kSize, vSize, qProjSize,
                                         kProjSize, vProjSize, oProjSize, qoMaxSeqLength,
                                         kvMaxSeqLength, maxBatchSize, maxBeamSize)
    initialize_context()
    @ccall libcudnn.cudnnSetAttnDescriptor(attnDesc::cudnnAttnDescriptor_t, attnMode::Cuint,
                                           nHeads::Cint, smScaler::Cdouble,
                                           dataType::cudnnDataType_t,
                                           computePrec::cudnnDataType_t,
                                           mathType::cudnnMathType_t,
                                           attnDropoutDesc::cudnnDropoutDescriptor_t,
                                           postDropoutDesc::cudnnDropoutDescriptor_t,
                                           qSize::Cint, kSize::Cint, vSize::Cint,
                                           qProjSize::Cint, kProjSize::Cint,
                                           vProjSize::Cint, oProjSize::Cint,
                                           qoMaxSeqLength::Cint, kvMaxSeqLength::Cint,
                                           maxBatchSize::Cint,
                                           maxBeamSize::Cint)::cudnnStatus_t
end

@checked function cudnnGetAttnDescriptor(attnDesc, attnMode, nHeads, smScaler, dataType,
                                         computePrec, mathType, attnDropoutDesc,
                                         postDropoutDesc, qSize, kSize, vSize, qProjSize,
                                         kProjSize, vProjSize, oProjSize, qoMaxSeqLength,
                                         kvMaxSeqLength, maxBatchSize, maxBeamSize)
    initialize_context()
    @ccall libcudnn.cudnnGetAttnDescriptor(attnDesc::cudnnAttnDescriptor_t,
                                           attnMode::Ptr{Cuint}, nHeads::Ptr{Cint},
                                           smScaler::Ptr{Cdouble},
                                           dataType::Ptr{cudnnDataType_t},
                                           computePrec::Ptr{cudnnDataType_t},
                                           mathType::Ptr{cudnnMathType_t},
                                           attnDropoutDesc::Ptr{cudnnDropoutDescriptor_t},
                                           postDropoutDesc::Ptr{cudnnDropoutDescriptor_t},
                                           qSize::Ptr{Cint}, kSize::Ptr{Cint},
                                           vSize::Ptr{Cint}, qProjSize::Ptr{Cint},
                                           kProjSize::Ptr{Cint}, vProjSize::Ptr{Cint},
                                           oProjSize::Ptr{Cint}, qoMaxSeqLength::Ptr{Cint},
                                           kvMaxSeqLength::Ptr{Cint},
                                           maxBatchSize::Ptr{Cint},
                                           maxBeamSize::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnGetMultiHeadAttnBuffers(handle, attnDesc, weightSizeInBytes,
                                               workSpaceSizeInBytes,
                                               reserveSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnGetMultiHeadAttnBuffers(handle::cudnnHandle_t,
                                                 attnDesc::cudnnAttnDescriptor_t,
                                                 weightSizeInBytes::Ptr{Csize_t},
                                                 workSpaceSizeInBytes::Ptr{Csize_t},
                                                 reserveSpaceSizeInBytes::Ptr{Csize_t})::cudnnStatus_t
end

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

@checked function cudnnGetMultiHeadAttnWeights(handle, attnDesc, wKind, weightSizeInBytes,
                                               weights, wDesc, wAddr)
    initialize_context()
    @ccall libcudnn.cudnnGetMultiHeadAttnWeights(handle::cudnnHandle_t,
                                                 attnDesc::cudnnAttnDescriptor_t,
                                                 wKind::cudnnMultiHeadAttnWeightKind_t,
                                                 weightSizeInBytes::Csize_t,
                                                 weights::CuPtr{Cvoid},
                                                 wDesc::cudnnTensorDescriptor_t,
                                                 wAddr::CuPtr{Ptr{Cvoid}})::cudnnStatus_t
end

@checked function cudnnMultiHeadAttnForward(handle, attnDesc, currIdx, loWinIdx, hiWinIdx,
                                            devSeqLengthsQO, devSeqLengthsKV, qDesc,
                                            queries, residuals, kDesc, keys, vDesc, values,
                                            oDesc, out, weightSizeInBytes, weights,
                                            workSpaceSizeInBytes, workSpace,
                                            reserveSpaceSizeInBytes, reserveSpace)
    initialize_context()
    @ccall libcudnn.cudnnMultiHeadAttnForward(handle::cudnnHandle_t,
                                              attnDesc::cudnnAttnDescriptor_t,
                                              currIdx::Cint, loWinIdx::Ptr{Cint},
                                              hiWinIdx::Ptr{Cint},
                                              devSeqLengthsQO::CuPtr{Cint},
                                              devSeqLengthsKV::CuPtr{Cint},
                                              qDesc::cudnnSeqDataDescriptor_t,
                                              queries::CuPtr{Cvoid},
                                              residuals::CuPtr{Cvoid},
                                              kDesc::cudnnSeqDataDescriptor_t,
                                              keys::CuPtr{Cvoid},
                                              vDesc::cudnnSeqDataDescriptor_t,
                                              values::CuPtr{Cvoid},
                                              oDesc::cudnnSeqDataDescriptor_t,
                                              out::CuPtr{Cvoid}, weightSizeInBytes::Csize_t,
                                              weights::CuPtr{Cvoid},
                                              workSpaceSizeInBytes::Csize_t,
                                              workSpace::CuPtr{Cvoid},
                                              reserveSpaceSizeInBytes::Csize_t,
                                              reserveSpace::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnAdvInferVersionCheck()
    initialize_context()
    @ccall libcudnn.cudnnAdvInferVersionCheck()::cudnnStatus_t
end

@cenum cudnnWgradMode_t::UInt32 begin
    CUDNN_WGRAD_MODE_ADD = 0
    CUDNN_WGRAD_MODE_SET = 1
end

@checked function cudnnRNNForwardTraining(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx,
                                          cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy,
                                          cyDesc, cy, workSpace, workSpaceSizeInBytes,
                                          reserveSpace, reserveSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnRNNForwardTraining(handle::cudnnHandle_t,
                                            rnnDesc::cudnnRNNDescriptor_t, seqLength::Cint,
                                            xDesc::Ptr{cudnnTensorDescriptor_t},
                                            x::CuPtr{Cvoid},
                                            hxDesc::cudnnTensorDescriptor_t,
                                            hx::CuPtr{Cvoid},
                                            cxDesc::cudnnTensorDescriptor_t,
                                            cx::CuPtr{Cvoid},
                                            wDesc::cudnnFilterDescriptor_t, w::CuPtr{Cvoid},
                                            yDesc::Ptr{cudnnTensorDescriptor_t},
                                            y::CuPtr{Cvoid},
                                            hyDesc::cudnnTensorDescriptor_t,
                                            hy::CuPtr{Cvoid},
                                            cyDesc::cudnnTensorDescriptor_t,
                                            cy::CuPtr{Cvoid}, workSpace::CuPtr{Cvoid},
                                            workSpaceSizeInBytes::Csize_t,
                                            reserveSpace::CuPtr{Cvoid},
                                            reserveSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@checked function cudnnRNNBackwardData(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy,
                                       dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx,
                                       cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx,
                                       workSpace, workSpaceSizeInBytes, reserveSpace,
                                       reserveSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnRNNBackwardData(handle::cudnnHandle_t,
                                         rnnDesc::cudnnRNNDescriptor_t, seqLength::Cint,
                                         yDesc::Ptr{cudnnTensorDescriptor_t},
                                         y::CuPtr{Cvoid},
                                         dyDesc::Ptr{cudnnTensorDescriptor_t},
                                         dy::CuPtr{Cvoid}, dhyDesc::cudnnTensorDescriptor_t,
                                         dhy::CuPtr{Cvoid},
                                         dcyDesc::cudnnTensorDescriptor_t,
                                         dcy::CuPtr{Cvoid}, wDesc::cudnnFilterDescriptor_t,
                                         w::CuPtr{Cvoid}, hxDesc::cudnnTensorDescriptor_t,
                                         hx::CuPtr{Cvoid}, cxDesc::cudnnTensorDescriptor_t,
                                         cx::CuPtr{Cvoid},
                                         dxDesc::Ptr{cudnnTensorDescriptor_t},
                                         dx::CuPtr{Cvoid}, dhxDesc::cudnnTensorDescriptor_t,
                                         dhx::CuPtr{Cvoid},
                                         dcxDesc::cudnnTensorDescriptor_t,
                                         dcx::CuPtr{Cvoid}, workSpace::CuPtr{Cvoid},
                                         workSpaceSizeInBytes::Csize_t,
                                         reserveSpace::CuPtr{Cvoid},
                                         reserveSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@checked function cudnnRNNBackwardData_v8(handle, rnnDesc, devSeqLengths, yDesc, y, dy,
                                          xDesc, dx, hDesc, hx, dhy, dhx, cDesc, cx, dcy,
                                          dcx, weightSpaceSize, weightSpace, workSpaceSize,
                                          workSpace, reserveSpaceSize, reserveSpace)
    initialize_context()
    @ccall libcudnn.cudnnRNNBackwardData_v8(handle::cudnnHandle_t,
                                            rnnDesc::cudnnRNNDescriptor_t,
                                            devSeqLengths::CuPtr{Int32},
                                            yDesc::cudnnRNNDataDescriptor_t,
                                            y::CuPtr{Cvoid}, dy::CuPtr{Cvoid},
                                            xDesc::cudnnRNNDataDescriptor_t,
                                            dx::CuPtr{Cvoid},
                                            hDesc::cudnnTensorDescriptor_t,
                                            hx::CuPtr{Cvoid}, dhy::CuPtr{Cvoid},
                                            dhx::CuPtr{Cvoid},
                                            cDesc::cudnnTensorDescriptor_t,
                                            cx::CuPtr{Cvoid}, dcy::CuPtr{Cvoid},
                                            dcx::CuPtr{Cvoid}, weightSpaceSize::Csize_t,
                                            weightSpace::CuPtr{Cvoid},
                                            workSpaceSize::Csize_t, workSpace::CuPtr{Cvoid},
                                            reserveSpaceSize::Csize_t,
                                            reserveSpace::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnRNNBackwardWeights(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx,
                                          yDesc, y, workSpace, workSpaceSizeInBytes, dwDesc,
                                          dw, reserveSpace, reserveSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnRNNBackwardWeights(handle::cudnnHandle_t,
                                            rnnDesc::cudnnRNNDescriptor_t, seqLength::Cint,
                                            xDesc::Ptr{cudnnTensorDescriptor_t},
                                            x::CuPtr{Cvoid},
                                            hxDesc::cudnnTensorDescriptor_t,
                                            hx::CuPtr{Cvoid},
                                            yDesc::Ptr{cudnnTensorDescriptor_t},
                                            y::CuPtr{Cvoid}, workSpace::CuPtr{Cvoid},
                                            workSpaceSizeInBytes::Csize_t,
                                            dwDesc::cudnnFilterDescriptor_t,
                                            dw::CuPtr{Cvoid}, reserveSpace::CuPtr{Cvoid},
                                            reserveSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@checked function cudnnRNNBackwardWeights_v8(handle, rnnDesc, addGrad, devSeqLengths, xDesc,
                                             x, hDesc, hx, yDesc, y, weightSpaceSize,
                                             dweightSpace, workSpaceSize, workSpace,
                                             reserveSpaceSize, reserveSpace)
    initialize_context()
    @ccall libcudnn.cudnnRNNBackwardWeights_v8(handle::cudnnHandle_t,
                                               rnnDesc::cudnnRNNDescriptor_t,
                                               addGrad::cudnnWgradMode_t,
                                               devSeqLengths::CuPtr{Int32},
                                               xDesc::cudnnRNNDataDescriptor_t,
                                               x::CuPtr{Cvoid},
                                               hDesc::cudnnTensorDescriptor_t,
                                               hx::CuPtr{Cvoid},
                                               yDesc::cudnnRNNDataDescriptor_t,
                                               y::CuPtr{Cvoid}, weightSpaceSize::Csize_t,
                                               dweightSpace::CuPtr{Cvoid},
                                               workSpaceSize::Csize_t,
                                               workSpace::CuPtr{Cvoid},
                                               reserveSpaceSize::Csize_t,
                                               reserveSpace::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnRNNForwardTrainingEx(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc,
                                            cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy,
                                            kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc,
                                            queries, workSpace, workSpaceSizeInBytes,
                                            reserveSpace, reserveSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnRNNForwardTrainingEx(handle::cudnnHandle_t,
                                              rnnDesc::cudnnRNNDescriptor_t,
                                              xDesc::cudnnRNNDataDescriptor_t,
                                              x::CuPtr{Cvoid},
                                              hxDesc::cudnnTensorDescriptor_t,
                                              hx::CuPtr{Cvoid},
                                              cxDesc::cudnnTensorDescriptor_t,
                                              cx::CuPtr{Cvoid},
                                              wDesc::cudnnFilterDescriptor_t,
                                              w::CuPtr{Cvoid},
                                              yDesc::cudnnRNNDataDescriptor_t,
                                              y::CuPtr{Cvoid},
                                              hyDesc::cudnnTensorDescriptor_t,
                                              hy::CuPtr{Cvoid},
                                              cyDesc::cudnnTensorDescriptor_t,
                                              cy::CuPtr{Cvoid},
                                              kDesc::cudnnRNNDataDescriptor_t,
                                              keys::CuPtr{Cvoid},
                                              cDesc::cudnnRNNDataDescriptor_t,
                                              cAttn::CuPtr{Cvoid},
                                              iDesc::cudnnRNNDataDescriptor_t,
                                              iAttn::CuPtr{Cvoid},
                                              qDesc::cudnnRNNDataDescriptor_t,
                                              queries::CuPtr{Cvoid},
                                              workSpace::CuPtr{Cvoid},
                                              workSpaceSizeInBytes::Csize_t,
                                              reserveSpace::CuPtr{Cvoid},
                                              reserveSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@checked function cudnnRNNBackwardDataEx(handle, rnnDesc, yDesc, y, dyDesc, dy, dcDesc,
                                         dcAttn, dhyDesc, dhy, dcyDesc, dcy, wDesc, w,
                                         hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx,
                                         dcxDesc, dcx, dkDesc, dkeys, workSpace,
                                         workSpaceSizeInBytes, reserveSpace,
                                         reserveSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnRNNBackwardDataEx(handle::cudnnHandle_t,
                                           rnnDesc::cudnnRNNDescriptor_t,
                                           yDesc::cudnnRNNDataDescriptor_t, y::CuPtr{Cvoid},
                                           dyDesc::cudnnRNNDataDescriptor_t,
                                           dy::CuPtr{Cvoid},
                                           dcDesc::cudnnRNNDataDescriptor_t,
                                           dcAttn::CuPtr{Cvoid},
                                           dhyDesc::cudnnTensorDescriptor_t,
                                           dhy::CuPtr{Cvoid},
                                           dcyDesc::cudnnTensorDescriptor_t,
                                           dcy::CuPtr{Cvoid},
                                           wDesc::cudnnFilterDescriptor_t, w::CuPtr{Cvoid},
                                           hxDesc::cudnnTensorDescriptor_t,
                                           hx::CuPtr{Cvoid},
                                           cxDesc::cudnnTensorDescriptor_t,
                                           cx::CuPtr{Cvoid},
                                           dxDesc::cudnnRNNDataDescriptor_t,
                                           dx::CuPtr{Cvoid},
                                           dhxDesc::cudnnTensorDescriptor_t,
                                           dhx::Ptr{Cvoid},
                                           dcxDesc::cudnnTensorDescriptor_t,
                                           dcx::CuPtr{Cvoid},
                                           dkDesc::cudnnRNNDataDescriptor_t,
                                           dkeys::Ptr{Cvoid}, workSpace::CuPtr{Cvoid},
                                           workSpaceSizeInBytes::Csize_t,
                                           reserveSpace::CuPtr{Cvoid},
                                           reserveSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@checked function cudnnRNNBackwardWeightsEx(handle, rnnDesc, xDesc, x, hxDesc, hx, yDesc, y,
                                            workSpace, workSpaceSizeInBytes, dwDesc, dw,
                                            reserveSpace, reserveSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnRNNBackwardWeightsEx(handle::cudnnHandle_t,
                                              rnnDesc::cudnnRNNDescriptor_t,
                                              xDesc::cudnnRNNDataDescriptor_t,
                                              x::CuPtr{Cvoid},
                                              hxDesc::cudnnTensorDescriptor_t,
                                              hx::CuPtr{Cvoid},
                                              yDesc::cudnnRNNDataDescriptor_t,
                                              y::CuPtr{Cvoid}, workSpace::CuPtr{Cvoid},
                                              workSpaceSizeInBytes::Csize_t,
                                              dwDesc::cudnnFilterDescriptor_t,
                                              dw::CuPtr{Cvoid}, reserveSpace::CuPtr{Cvoid},
                                              reserveSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@checked function cudnnGetRNNForwardTrainingAlgorithmMaxCount(handle, rnnDesc, count)
    initialize_context()
    @ccall libcudnn.cudnnGetRNNForwardTrainingAlgorithmMaxCount(handle::cudnnHandle_t,
                                                                rnnDesc::cudnnRNNDescriptor_t,
                                                                count::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnFindRNNForwardTrainingAlgorithmEx(handle, rnnDesc, seqLength, xDesc,
                                                         x, hxDesc, hx, cxDesc, cx, wDesc,
                                                         w, yDesc, y, hyDesc, hy, cyDesc,
                                                         cy, findIntensity,
                                                         requestedAlgoCount,
                                                         returnedAlgoCount, perfResults,
                                                         workspace, workSpaceSizeInBytes,
                                                         reserveSpace,
                                                         reserveSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnFindRNNForwardTrainingAlgorithmEx(handle::cudnnHandle_t,
                                                           rnnDesc::cudnnRNNDescriptor_t,
                                                           seqLength::Cint,
                                                           xDesc::Ptr{cudnnTensorDescriptor_t},
                                                           x::CuPtr{Cvoid},
                                                           hxDesc::cudnnTensorDescriptor_t,
                                                           hx::CuPtr{Cvoid},
                                                           cxDesc::cudnnTensorDescriptor_t,
                                                           cx::CuPtr{Cvoid},
                                                           wDesc::cudnnFilterDescriptor_t,
                                                           w::CuPtr{Cvoid},
                                                           yDesc::Ptr{cudnnTensorDescriptor_t},
                                                           y::CuPtr{Cvoid},
                                                           hyDesc::cudnnTensorDescriptor_t,
                                                           hy::CuPtr{Cvoid},
                                                           cyDesc::cudnnTensorDescriptor_t,
                                                           cy::CuPtr{Cvoid},
                                                           findIntensity::Cfloat,
                                                           requestedAlgoCount::Cint,
                                                           returnedAlgoCount::Ptr{Cint},
                                                           perfResults::Ptr{cudnnAlgorithmPerformance_t},
                                                           workspace::CuPtr{Cvoid},
                                                           workSpaceSizeInBytes::Csize_t,
                                                           reserveSpace::CuPtr{Cvoid},
                                                           reserveSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@checked function cudnnGetRNNBackwardDataAlgorithmMaxCount(handle, rnnDesc, count)
    initialize_context()
    @ccall libcudnn.cudnnGetRNNBackwardDataAlgorithmMaxCount(handle::cudnnHandle_t,
                                                             rnnDesc::cudnnRNNDescriptor_t,
                                                             count::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnFindRNNBackwardDataAlgorithmEx(handle, rnnDesc, seqLength, yDesc, y,
                                                      dyDesc, dy, dhyDesc, dhy, dcyDesc,
                                                      dcy, wDesc, w, hxDesc, hx, cxDesc, cx,
                                                      dxDesc, dx, dhxDesc, dhx, dcxDesc,
                                                      dcx, findIntensity,
                                                      requestedAlgoCount, returnedAlgoCount,
                                                      perfResults, workspace,
                                                      workSpaceSizeInBytes, reserveSpace,
                                                      reserveSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnFindRNNBackwardDataAlgorithmEx(handle::cudnnHandle_t,
                                                        rnnDesc::cudnnRNNDescriptor_t,
                                                        seqLength::Cint,
                                                        yDesc::Ptr{cudnnTensorDescriptor_t},
                                                        y::CuPtr{Cvoid},
                                                        dyDesc::Ptr{cudnnTensorDescriptor_t},
                                                        dy::CuPtr{Cvoid},
                                                        dhyDesc::cudnnTensorDescriptor_t,
                                                        dhy::CuPtr{Cvoid},
                                                        dcyDesc::cudnnTensorDescriptor_t,
                                                        dcy::CuPtr{Cvoid},
                                                        wDesc::cudnnFilterDescriptor_t,
                                                        w::CuPtr{Cvoid},
                                                        hxDesc::cudnnTensorDescriptor_t,
                                                        hx::CuPtr{Cvoid},
                                                        cxDesc::cudnnTensorDescriptor_t,
                                                        cx::CuPtr{Cvoid},
                                                        dxDesc::Ptr{cudnnTensorDescriptor_t},
                                                        dx::CuPtr{Cvoid},
                                                        dhxDesc::cudnnTensorDescriptor_t,
                                                        dhx::CuPtr{Cvoid},
                                                        dcxDesc::cudnnTensorDescriptor_t,
                                                        dcx::CuPtr{Cvoid},
                                                        findIntensity::Cfloat,
                                                        requestedAlgoCount::Cint,
                                                        returnedAlgoCount::Ptr{Cint},
                                                        perfResults::Ptr{cudnnAlgorithmPerformance_t},
                                                        workspace::CuPtr{Cvoid},
                                                        workSpaceSizeInBytes::Csize_t,
                                                        reserveSpace::CuPtr{Cvoid},
                                                        reserveSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@checked function cudnnGetRNNBackwardWeightsAlgorithmMaxCount(handle, rnnDesc, count)
    initialize_context()
    @ccall libcudnn.cudnnGetRNNBackwardWeightsAlgorithmMaxCount(handle::cudnnHandle_t,
                                                                rnnDesc::cudnnRNNDescriptor_t,
                                                                count::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnFindRNNBackwardWeightsAlgorithmEx(handle, rnnDesc, seqLength, xDesc,
                                                         x, hxDesc, hx, yDesc, y,
                                                         findIntensity, requestedAlgoCount,
                                                         returnedAlgoCount, perfResults,
                                                         workspace, workSpaceSizeInBytes,
                                                         dwDesc, dw, reserveSpace,
                                                         reserveSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnFindRNNBackwardWeightsAlgorithmEx(handle::cudnnHandle_t,
                                                           rnnDesc::cudnnRNNDescriptor_t,
                                                           seqLength::Cint,
                                                           xDesc::Ptr{cudnnTensorDescriptor_t},
                                                           x::CuPtr{Cvoid},
                                                           hxDesc::cudnnTensorDescriptor_t,
                                                           hx::CuPtr{Cvoid},
                                                           yDesc::Ptr{cudnnTensorDescriptor_t},
                                                           y::CuPtr{Cvoid},
                                                           findIntensity::Cfloat,
                                                           requestedAlgoCount::Cint,
                                                           returnedAlgoCount::Ptr{Cint},
                                                           perfResults::Ptr{cudnnAlgorithmPerformance_t},
                                                           workspace::CuPtr{Cvoid},
                                                           workSpaceSizeInBytes::Csize_t,
                                                           dwDesc::cudnnFilterDescriptor_t,
                                                           dw::CuPtr{Cvoid},
                                                           reserveSpace::CuPtr{Cvoid},
                                                           reserveSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@checked function cudnnMultiHeadAttnBackwardData(handle, attnDesc, loWinIdx, hiWinIdx,
                                                 devSeqLengthsDQDO, devSeqLengthsDKDV,
                                                 doDesc, dout, dqDesc, dqueries, queries,
                                                 dkDesc, dkeys, keys, dvDesc, dvalues,
                                                 values, weightSizeInBytes, weights,
                                                 workSpaceSizeInBytes, workSpace,
                                                 reserveSpaceSizeInBytes, reserveSpace)
    initialize_context()
    @ccall libcudnn.cudnnMultiHeadAttnBackwardData(handle::cudnnHandle_t,
                                                   attnDesc::cudnnAttnDescriptor_t,
                                                   loWinIdx::Ptr{Cint}, hiWinIdx::Ptr{Cint},
                                                   devSeqLengthsDQDO::CuPtr{Cint},
                                                   devSeqLengthsDKDV::CuPtr{Cint},
                                                   doDesc::cudnnSeqDataDescriptor_t,
                                                   dout::CuPtr{Cvoid},
                                                   dqDesc::cudnnSeqDataDescriptor_t,
                                                   dqueries::CuPtr{Cvoid},
                                                   queries::CuPtr{Cvoid},
                                                   dkDesc::cudnnSeqDataDescriptor_t,
                                                   dkeys::CuPtr{Cvoid}, keys::CuPtr{Cvoid},
                                                   dvDesc::cudnnSeqDataDescriptor_t,
                                                   dvalues::CuPtr{Cvoid},
                                                   values::CuPtr{Cvoid},
                                                   weightSizeInBytes::Csize_t,
                                                   weights::CuPtr{Cvoid},
                                                   workSpaceSizeInBytes::Csize_t,
                                                   workSpace::CuPtr{Cvoid},
                                                   reserveSpaceSizeInBytes::Csize_t,
                                                   reserveSpace::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnMultiHeadAttnBackwardWeights(handle, attnDesc, addGrad, qDesc,
                                                    queries, kDesc, keys, vDesc, values,
                                                    doDesc, dout, weightSizeInBytes,
                                                    weights, dweights, workSpaceSizeInBytes,
                                                    workSpace, reserveSpaceSizeInBytes,
                                                    reserveSpace)
    initialize_context()
    @ccall libcudnn.cudnnMultiHeadAttnBackwardWeights(handle::cudnnHandle_t,
                                                      attnDesc::cudnnAttnDescriptor_t,
                                                      addGrad::cudnnWgradMode_t,
                                                      qDesc::cudnnSeqDataDescriptor_t,
                                                      queries::CuPtr{Cvoid},
                                                      kDesc::cudnnSeqDataDescriptor_t,
                                                      keys::CuPtr{Cvoid},
                                                      vDesc::cudnnSeqDataDescriptor_t,
                                                      values::CuPtr{Cvoid},
                                                      doDesc::cudnnSeqDataDescriptor_t,
                                                      dout::CuPtr{Cvoid},
                                                      weightSizeInBytes::Csize_t,
                                                      weights::CuPtr{Cvoid},
                                                      dweights::CuPtr{Cvoid},
                                                      workSpaceSizeInBytes::Csize_t,
                                                      workSpace::CuPtr{Cvoid},
                                                      reserveSpaceSizeInBytes::Csize_t,
                                                      reserveSpace::CuPtr{Cvoid})::cudnnStatus_t
end

@cenum cudnnLossNormalizationMode_t::UInt32 begin
    CUDNN_LOSS_NORMALIZATION_NONE = 0
    CUDNN_LOSS_NORMALIZATION_SOFTMAX = 1
end

@checked function cudnnCreateCTCLossDescriptor(ctcLossDesc)
    initialize_context()
    @ccall libcudnn.cudnnCreateCTCLossDescriptor(ctcLossDesc::Ptr{cudnnCTCLossDescriptor_t})::cudnnStatus_t
end

@checked function cudnnSetCTCLossDescriptor(ctcLossDesc, compType)
    initialize_context()
    @ccall libcudnn.cudnnSetCTCLossDescriptor(ctcLossDesc::cudnnCTCLossDescriptor_t,
                                              compType::cudnnDataType_t)::cudnnStatus_t
end

@checked function cudnnSetCTCLossDescriptorEx(ctcLossDesc, compType, normMode, gradMode)
    initialize_context()
    @ccall libcudnn.cudnnSetCTCLossDescriptorEx(ctcLossDesc::cudnnCTCLossDescriptor_t,
                                                compType::cudnnDataType_t,
                                                normMode::cudnnLossNormalizationMode_t,
                                                gradMode::cudnnNanPropagation_t)::cudnnStatus_t
end

@checked function cudnnSetCTCLossDescriptor_v8(ctcLossDesc, compType, normMode, gradMode,
                                               maxLabelLength)
    initialize_context()
    @ccall libcudnn.cudnnSetCTCLossDescriptor_v8(ctcLossDesc::cudnnCTCLossDescriptor_t,
                                                 compType::cudnnDataType_t,
                                                 normMode::cudnnLossNormalizationMode_t,
                                                 gradMode::cudnnNanPropagation_t,
                                                 maxLabelLength::Cint)::cudnnStatus_t
end

@checked function cudnnGetCTCLossDescriptor(ctcLossDesc, compType)
    initialize_context()
    @ccall libcudnn.cudnnGetCTCLossDescriptor(ctcLossDesc::cudnnCTCLossDescriptor_t,
                                              compType::Ptr{cudnnDataType_t})::cudnnStatus_t
end

@checked function cudnnGetCTCLossDescriptorEx(ctcLossDesc, compType, normMode, gradMode)
    initialize_context()
    @ccall libcudnn.cudnnGetCTCLossDescriptorEx(ctcLossDesc::cudnnCTCLossDescriptor_t,
                                                compType::Ptr{cudnnDataType_t},
                                                normMode::Ptr{cudnnLossNormalizationMode_t},
                                                gradMode::Ptr{cudnnNanPropagation_t})::cudnnStatus_t
end

@checked function cudnnGetCTCLossDescriptor_v8(ctcLossDesc, compType, normMode, gradMode,
                                               maxLabelLength)
    initialize_context()
    @ccall libcudnn.cudnnGetCTCLossDescriptor_v8(ctcLossDesc::cudnnCTCLossDescriptor_t,
                                                 compType::Ref{cudnnDataType_t},
                                                 normMode::Ref{cudnnLossNormalizationMode_t},
                                                 gradMode::Ref{cudnnNanPropagation_t},
                                                 maxLabelLength::Ref{Cint})::cudnnStatus_t
end

@checked function cudnnDestroyCTCLossDescriptor(ctcLossDesc)
    initialize_context()
    @ccall libcudnn.cudnnDestroyCTCLossDescriptor(ctcLossDesc::cudnnCTCLossDescriptor_t)::cudnnStatus_t
end

@checked function cudnnCTCLoss(handle, probsDesc, probs, hostLabels, hostLabelLengths,
                               hostInputLengths, costs, gradientsDesc, gradients, algo,
                               ctcLossDesc, workspace, workSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnCTCLoss(handle::cudnnHandle_t, probsDesc::cudnnTensorDescriptor_t,
                                 probs::CuPtr{Cvoid}, hostLabels::Ptr{Cint},
                                 hostLabelLengths::Ptr{Cint}, hostInputLengths::Ptr{Cint},
                                 costs::CuPtr{Cvoid},
                                 gradientsDesc::cudnnTensorDescriptor_t,
                                 gradients::CuPtr{Cvoid}, algo::cudnnCTCLossAlgo_t,
                                 ctcLossDesc::cudnnCTCLossDescriptor_t,
                                 workspace::CuPtr{Cvoid},
                                 workSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@checked function cudnnCTCLoss_v8(handle, algo, ctcLossDesc, probsDesc, probs, labels,
                                  labelLengths, inputLengths, costs, gradientsDesc,
                                  gradients, workSpaceSizeInBytes, workspace)
    initialize_context()
    @ccall libcudnn.cudnnCTCLoss_v8(handle::cudnnHandle_t, algo::cudnnCTCLossAlgo_t,
                                    ctcLossDesc::cudnnCTCLossDescriptor_t,
                                    probsDesc::cudnnTensorDescriptor_t, probs::Ptr{Cvoid},
                                    labels::CuPtr{Cint}, labelLengths::CuPtr{Cint},
                                    inputLengths::CuPtr{Cint}, costs::Ptr{Cvoid},
                                    gradientsDesc::cudnnTensorDescriptor_t,
                                    gradients::Ptr{Cvoid}, workSpaceSizeInBytes::Csize_t,
                                    workspace::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnGetCTCLossWorkspaceSize(handle, probsDesc, gradientsDesc, labels,
                                               labelLengths, inputLengths, algo,
                                               ctcLossDesc, sizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnGetCTCLossWorkspaceSize(handle::cudnnHandle_t,
                                                 probsDesc::cudnnTensorDescriptor_t,
                                                 gradientsDesc::cudnnTensorDescriptor_t,
                                                 labels::Ptr{Cint}, labelLengths::Ptr{Cint},
                                                 inputLengths::Ptr{Cint},
                                                 algo::cudnnCTCLossAlgo_t,
                                                 ctcLossDesc::cudnnCTCLossDescriptor_t,
                                                 sizeInBytes::Ref{Csize_t})::cudnnStatus_t
end

@checked function cudnnGetCTCLossWorkspaceSize_v8(handle, algo, ctcLossDesc, probsDesc,
                                                  gradientsDesc, sizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnGetCTCLossWorkspaceSize_v8(handle::cudnnHandle_t,
                                                    algo::cudnnCTCLossAlgo_t,
                                                    ctcLossDesc::cudnnCTCLossDescriptor_t,
                                                    probsDesc::cudnnTensorDescriptor_t,
                                                    gradientsDesc::cudnnTensorDescriptor_t,
                                                    sizeInBytes::Ptr{Csize_t})::cudnnStatus_t
end

@checked function cudnnAdvTrainVersionCheck()
    initialize_context()
    @ccall libcudnn.cudnnAdvTrainVersionCheck()::cudnnStatus_t
end

mutable struct cudnnConvolutionStruct end

const cudnnConvolutionDescriptor_t = Ptr{cudnnConvolutionStruct}

@cenum cudnnConvolutionMode_t::UInt32 begin
    CUDNN_CONVOLUTION = 0
    CUDNN_CROSS_CORRELATION = 1
end

@cenum cudnnReorderType_t::UInt32 begin
    CUDNN_DEFAULT_REORDER = 0
    CUDNN_NO_REORDER = 1
end

struct cudnnConvolutionFwdAlgoPerfStruct
    algo::cudnnConvolutionFwdAlgo_t
    status::cudnnStatus_t
    time::Cfloat
    memory::Csize_t
    determinism::cudnnDeterminism_t
    mathType::cudnnMathType_t
    reserved::NTuple{3,Cint}
end

const cudnnConvolutionFwdAlgoPerf_t = cudnnConvolutionFwdAlgoPerfStruct

@checked function cudnnCreateConvolutionDescriptor(convDesc)
    initialize_context()
    @ccall libcudnn.cudnnCreateConvolutionDescriptor(convDesc::Ptr{cudnnConvolutionDescriptor_t})::cudnnStatus_t
end

@checked function cudnnDestroyConvolutionDescriptor(convDesc)
    initialize_context()
    @ccall libcudnn.cudnnDestroyConvolutionDescriptor(convDesc::cudnnConvolutionDescriptor_t)::cudnnStatus_t
end

@checked function cudnnSetConvolutionMathType(convDesc, mathType)
    initialize_context()
    @ccall libcudnn.cudnnSetConvolutionMathType(convDesc::cudnnConvolutionDescriptor_t,
                                                mathType::cudnnMathType_t)::cudnnStatus_t
end

@checked function cudnnGetConvolutionMathType(convDesc, mathType)
    initialize_context()
    @ccall libcudnn.cudnnGetConvolutionMathType(convDesc::cudnnConvolutionDescriptor_t,
                                                mathType::Ptr{cudnnMathType_t})::cudnnStatus_t
end

@checked function cudnnSetConvolutionGroupCount(convDesc, groupCount)
    initialize_context()
    @ccall libcudnn.cudnnSetConvolutionGroupCount(convDesc::cudnnConvolutionDescriptor_t,
                                                  groupCount::Cint)::cudnnStatus_t
end

@checked function cudnnGetConvolutionGroupCount(convDesc, groupCount)
    initialize_context()
    @ccall libcudnn.cudnnGetConvolutionGroupCount(convDesc::cudnnConvolutionDescriptor_t,
                                                  groupCount::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnSetConvolutionReorderType(convDesc, reorderType)
    initialize_context()
    @ccall libcudnn.cudnnSetConvolutionReorderType(convDesc::cudnnConvolutionDescriptor_t,
                                                   reorderType::cudnnReorderType_t)::cudnnStatus_t
end

@checked function cudnnGetConvolutionReorderType(convDesc, reorderType)
    initialize_context()
    @ccall libcudnn.cudnnGetConvolutionReorderType(convDesc::cudnnConvolutionDescriptor_t,
                                                   reorderType::Ptr{cudnnReorderType_t})::cudnnStatus_t
end

@checked function cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, dilation_h,
                                                  dilation_w, mode, computeType)
    initialize_context()
    @ccall libcudnn.cudnnSetConvolution2dDescriptor(convDesc::cudnnConvolutionDescriptor_t,
                                                    pad_h::Cint, pad_w::Cint, u::Cint,
                                                    v::Cint, dilation_h::Cint,
                                                    dilation_w::Cint,
                                                    mode::cudnnConvolutionMode_t,
                                                    computeType::cudnnDataType_t)::cudnnStatus_t
end

@checked function cudnnGetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, dilation_h,
                                                  dilation_w, mode, computeType)
    initialize_context()
    @ccall libcudnn.cudnnGetConvolution2dDescriptor(convDesc::cudnnConvolutionDescriptor_t,
                                                    pad_h::Ptr{Cint}, pad_w::Ptr{Cint},
                                                    u::Ptr{Cint}, v::Ptr{Cint},
                                                    dilation_h::Ptr{Cint},
                                                    dilation_w::Ptr{Cint},
                                                    mode::Ptr{cudnnConvolutionMode_t},
                                                    computeType::Ptr{cudnnDataType_t})::cudnnStatus_t
end

@checked function cudnnSetConvolutionNdDescriptor(convDesc, arrayLength, padA,
                                                  filterStrideA, dilationA, mode,
                                                  computeType)
    initialize_context()
    @ccall libcudnn.cudnnSetConvolutionNdDescriptor(convDesc::cudnnConvolutionDescriptor_t,
                                                    arrayLength::Cint, padA::Ptr{Cint},
                                                    filterStrideA::Ptr{Cint},
                                                    dilationA::Ptr{Cint},
                                                    mode::cudnnConvolutionMode_t,
                                                    computeType::cudnnDataType_t)::cudnnStatus_t
end

@checked function cudnnGetConvolutionNdDescriptor(convDesc, arrayLengthRequested,
                                                  arrayLength, padA, strideA, dilationA,
                                                  mode, computeType)
    initialize_context()
    @ccall libcudnn.cudnnGetConvolutionNdDescriptor(convDesc::cudnnConvolutionDescriptor_t,
                                                    arrayLengthRequested::Cint,
                                                    arrayLength::Ptr{Cint}, padA::Ptr{Cint},
                                                    strideA::Ptr{Cint},
                                                    dilationA::Ptr{Cint},
                                                    mode::Ptr{cudnnConvolutionMode_t},
                                                    computeType::Ptr{cudnnDataType_t})::cudnnStatus_t
end

@checked function cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensorDesc,
                                                        filterDesc, n, c, h, w)
    initialize_context()
    @ccall libcudnn.cudnnGetConvolution2dForwardOutputDim(convDesc::cudnnConvolutionDescriptor_t,
                                                          inputTensorDesc::cudnnTensorDescriptor_t,
                                                          filterDesc::cudnnFilterDescriptor_t,
                                                          n::Ptr{Cint}, c::Ptr{Cint},
                                                          h::Ptr{Cint},
                                                          w::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnGetConvolutionNdForwardOutputDim(convDesc, inputTensorDesc,
                                                        filterDesc, nbDims, tensorOuputDimA)
    initialize_context()
    @ccall libcudnn.cudnnGetConvolutionNdForwardOutputDim(convDesc::cudnnConvolutionDescriptor_t,
                                                          inputTensorDesc::cudnnTensorDescriptor_t,
                                                          filterDesc::cudnnFilterDescriptor_t,
                                                          nbDims::Cint,
                                                          tensorOuputDimA::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnGetConvolutionForwardAlgorithmMaxCount(handle, count)
    initialize_context()
    @ccall libcudnn.cudnnGetConvolutionForwardAlgorithmMaxCount(handle::cudnnHandle_t,
                                                                count::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnGetConvolutionForwardAlgorithm_v7(handle, srcDesc, filterDesc,
                                                         convDesc, destDesc,
                                                         requestedAlgoCount,
                                                         returnedAlgoCount, perfResults)
    initialize_context()
    @ccall libcudnn.cudnnGetConvolutionForwardAlgorithm_v7(handle::cudnnHandle_t,
                                                           srcDesc::cudnnTensorDescriptor_t,
                                                           filterDesc::cudnnFilterDescriptor_t,
                                                           convDesc::cudnnConvolutionDescriptor_t,
                                                           destDesc::cudnnTensorDescriptor_t,
                                                           requestedAlgoCount::Cint,
                                                           returnedAlgoCount::Ptr{Cint},
                                                           perfResults::Ptr{cudnnConvolutionFwdAlgoPerf_t})::cudnnStatus_t
end

@checked function cudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc,
                                                       yDesc, requestedAlgoCount,
                                                       returnedAlgoCount, perfResults)
    initialize_context()
    @ccall libcudnn.cudnnFindConvolutionForwardAlgorithm(handle::cudnnHandle_t,
                                                         xDesc::cudnnTensorDescriptor_t,
                                                         wDesc::cudnnFilterDescriptor_t,
                                                         convDesc::cudnnConvolutionDescriptor_t,
                                                         yDesc::cudnnTensorDescriptor_t,
                                                         requestedAlgoCount::Cint,
                                                         returnedAlgoCount::Ptr{Cint},
                                                         perfResults::Ptr{cudnnConvolutionFwdAlgoPerf_t})::cudnnStatus_t
end

@checked function cudnnFindConvolutionForwardAlgorithmEx(handle, xDesc, x, wDesc, w,
                                                         convDesc, yDesc, y,
                                                         requestedAlgoCount,
                                                         returnedAlgoCount, perfResults,
                                                         workSpace, workSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnFindConvolutionForwardAlgorithmEx(handle::cudnnHandle_t,
                                                           xDesc::cudnnTensorDescriptor_t,
                                                           x::CuPtr{Cvoid},
                                                           wDesc::cudnnFilterDescriptor_t,
                                                           w::CuPtr{Cvoid},
                                                           convDesc::cudnnConvolutionDescriptor_t,
                                                           yDesc::cudnnTensorDescriptor_t,
                                                           y::CuPtr{Cvoid},
                                                           requestedAlgoCount::Cint,
                                                           returnedAlgoCount::Ptr{Cint},
                                                           perfResults::Ptr{cudnnConvolutionFwdAlgoPerf_t},
                                                           workSpace::CuPtr{Cvoid},
                                                           workSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@checked function cudnnIm2Col(handle, xDesc, x, wDesc, convDesc, colBuffer)
    initialize_context()
    @ccall libcudnn.cudnnIm2Col(handle::cudnnHandle_t, xDesc::cudnnTensorDescriptor_t,
                                x::CuPtr{Cvoid}, wDesc::cudnnFilterDescriptor_t,
                                convDesc::cudnnConvolutionDescriptor_t,
                                colBuffer::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnReorderFilterAndBias(handle, filterDesc, reorderType, filterData,
                                            reorderedFilterData, reorderBias, biasData,
                                            reorderedBiasData)
    initialize_context()
    @ccall libcudnn.cudnnReorderFilterAndBias(handle::cudnnHandle_t,
                                              filterDesc::cudnnFilterDescriptor_t,
                                              reorderType::cudnnReorderType_t,
                                              filterData::CuPtr{Cvoid},
                                              reorderedFilterData::CuPtr{Cvoid},
                                              reorderBias::Cint, biasData::CuPtr{Cvoid},
                                              reorderedBiasData::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc,
                                                          yDesc, algo, sizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnGetConvolutionForwardWorkspaceSize(handle::cudnnHandle_t,
                                                            xDesc::cudnnTensorDescriptor_t,
                                                            wDesc::cudnnFilterDescriptor_t,
                                                            convDesc::cudnnConvolutionDescriptor_t,
                                                            yDesc::cudnnTensorDescriptor_t,
                                                            algo::cudnnConvolutionFwdAlgo_t,
                                                            sizeInBytes::Ref{Csize_t})::cudnnStatus_t
end

@checked function cudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo,
                                          workSpace, workSpaceSizeInBytes, beta, yDesc, y)
    initialize_context()
    @ccall libcudnn.cudnnConvolutionForward(handle::cudnnHandle_t, alpha::Ptr{Cvoid},
                                            xDesc::cudnnTensorDescriptor_t, x::CuPtr{Cvoid},
                                            wDesc::cudnnFilterDescriptor_t, w::CuPtr{Cvoid},
                                            convDesc::cudnnConvolutionDescriptor_t,
                                            algo::cudnnConvolutionFwdAlgo_t,
                                            workSpace::CuPtr{Cvoid},
                                            workSpaceSizeInBytes::Csize_t, beta::Ptr{Cvoid},
                                            yDesc::cudnnTensorDescriptor_t,
                                            y::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnConvolutionBiasActivationForward(handle, alpha1, xDesc, x, wDesc, w,
                                                        convDesc, algo, workSpace,
                                                        workSpaceSizeInBytes, alpha2, zDesc,
                                                        z, biasDesc, bias, activationDesc,
                                                        yDesc, y)
    initialize_context()
    @ccall libcudnn.cudnnConvolutionBiasActivationForward(handle::cudnnHandle_t,
                                                          alpha1::Ptr{Cvoid},
                                                          xDesc::cudnnTensorDescriptor_t,
                                                          x::CuPtr{Cvoid},
                                                          wDesc::cudnnFilterDescriptor_t,
                                                          w::CuPtr{Cvoid},
                                                          convDesc::cudnnConvolutionDescriptor_t,
                                                          algo::cudnnConvolutionFwdAlgo_t,
                                                          workSpace::CuPtr{Cvoid},
                                                          workSpaceSizeInBytes::Csize_t,
                                                          alpha2::Ptr{Cvoid},
                                                          zDesc::cudnnTensorDescriptor_t,
                                                          z::CuPtr{Cvoid},
                                                          biasDesc::cudnnTensorDescriptor_t,
                                                          bias::CuPtr{Cvoid},
                                                          activationDesc::cudnnActivationDescriptor_t,
                                                          yDesc::cudnnTensorDescriptor_t,
                                                          y::CuPtr{Cvoid})::cudnnStatus_t
end

struct cudnnConvolutionBwdDataAlgoPerfStruct
    algo::cudnnConvolutionBwdDataAlgo_t
    status::cudnnStatus_t
    time::Cfloat
    memory::Csize_t
    determinism::cudnnDeterminism_t
    mathType::cudnnMathType_t
    reserved::NTuple{3,Cint}
end

const cudnnConvolutionBwdDataAlgoPerf_t = cudnnConvolutionBwdDataAlgoPerfStruct

@checked function cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, count)
    initialize_context()
    @ccall libcudnn.cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle::cudnnHandle_t,
                                                                     count::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnFindConvolutionBackwardDataAlgorithm(handle, wDesc, dyDesc, convDesc,
                                                            dxDesc, requestedAlgoCount,
                                                            returnedAlgoCount, perfResults)
    initialize_context()
    @ccall libcudnn.cudnnFindConvolutionBackwardDataAlgorithm(handle::cudnnHandle_t,
                                                              wDesc::cudnnFilterDescriptor_t,
                                                              dyDesc::cudnnTensorDescriptor_t,
                                                              convDesc::cudnnConvolutionDescriptor_t,
                                                              dxDesc::cudnnTensorDescriptor_t,
                                                              requestedAlgoCount::Cint,
                                                              returnedAlgoCount::Ptr{Cint},
                                                              perfResults::Ptr{cudnnConvolutionBwdDataAlgoPerf_t})::cudnnStatus_t
end

@checked function cudnnFindConvolutionBackwardDataAlgorithmEx(handle, wDesc, w, dyDesc, dy,
                                                              convDesc, dxDesc, dx,
                                                              requestedAlgoCount,
                                                              returnedAlgoCount,
                                                              perfResults, workSpace,
                                                              workSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnFindConvolutionBackwardDataAlgorithmEx(handle::cudnnHandle_t,
                                                                wDesc::cudnnFilterDescriptor_t,
                                                                w::CuPtr{Cvoid},
                                                                dyDesc::cudnnTensorDescriptor_t,
                                                                dy::CuPtr{Cvoid},
                                                                convDesc::cudnnConvolutionDescriptor_t,
                                                                dxDesc::cudnnTensorDescriptor_t,
                                                                dx::CuPtr{Cvoid},
                                                                requestedAlgoCount::Cint,
                                                                returnedAlgoCount::Ptr{Cint},
                                                                perfResults::Ptr{cudnnConvolutionBwdDataAlgoPerf_t},
                                                                workSpace::CuPtr{Cvoid},
                                                                workSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@checked function cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, filterDesc, diffDesc,
                                                              convDesc, gradDesc,
                                                              requestedAlgoCount,
                                                              returnedAlgoCount,
                                                              perfResults)
    initialize_context()
    @ccall libcudnn.cudnnGetConvolutionBackwardDataAlgorithm_v7(handle::cudnnHandle_t,
                                                                filterDesc::cudnnFilterDescriptor_t,
                                                                diffDesc::cudnnTensorDescriptor_t,
                                                                convDesc::cudnnConvolutionDescriptor_t,
                                                                gradDesc::cudnnTensorDescriptor_t,
                                                                requestedAlgoCount::Cint,
                                                                returnedAlgoCount::Ptr{Cint},
                                                                perfResults::Ptr{cudnnConvolutionBwdDataAlgoPerf_t})::cudnnStatus_t
end

@checked function cudnnGetConvolutionBackwardDataWorkspaceSize(handle, wDesc, dyDesc,
                                                               convDesc, dxDesc, algo,
                                                               sizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(handle::cudnnHandle_t,
                                                                 wDesc::cudnnFilterDescriptor_t,
                                                                 dyDesc::cudnnTensorDescriptor_t,
                                                                 convDesc::cudnnConvolutionDescriptor_t,
                                                                 dxDesc::cudnnTensorDescriptor_t,
                                                                 algo::cudnnConvolutionBwdDataAlgo_t,
                                                                 sizeInBytes::Ref{Csize_t})::cudnnStatus_t
end

@checked function cudnnConvolutionBackwardData(handle, alpha, wDesc, w, dyDesc, dy,
                                               convDesc, algo, workSpace,
                                               workSpaceSizeInBytes, beta, dxDesc, dx)
    initialize_context()
    @ccall libcudnn.cudnnConvolutionBackwardData(handle::cudnnHandle_t, alpha::Ptr{Cvoid},
                                                 wDesc::cudnnFilterDescriptor_t,
                                                 w::CuPtr{Cvoid},
                                                 dyDesc::cudnnTensorDescriptor_t,
                                                 dy::CuPtr{Cvoid},
                                                 convDesc::cudnnConvolutionDescriptor_t,
                                                 algo::cudnnConvolutionBwdDataAlgo_t,
                                                 workSpace::CuPtr{Cvoid},
                                                 workSpaceSizeInBytes::Csize_t,
                                                 beta::Ptr{Cvoid},
                                                 dxDesc::cudnnTensorDescriptor_t,
                                                 dx::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnGetFoldedConvBackwardDataDescriptors(handle, filterDesc, diffDesc,
                                                            convDesc, gradDesc,
                                                            transformFormat,
                                                            foldedFilterDesc,
                                                            paddedDiffDesc, foldedConvDesc,
                                                            foldedGradDesc,
                                                            filterFoldTransDesc,
                                                            diffPadTransDesc,
                                                            gradFoldTransDesc,
                                                            gradUnfoldTransDesc)
    initialize_context()
    @ccall libcudnn.cudnnGetFoldedConvBackwardDataDescriptors(handle::cudnnHandle_t,
                                                              filterDesc::cudnnFilterDescriptor_t,
                                                              diffDesc::cudnnTensorDescriptor_t,
                                                              convDesc::cudnnConvolutionDescriptor_t,
                                                              gradDesc::cudnnTensorDescriptor_t,
                                                              transformFormat::cudnnTensorFormat_t,
                                                              foldedFilterDesc::cudnnFilterDescriptor_t,
                                                              paddedDiffDesc::cudnnTensorDescriptor_t,
                                                              foldedConvDesc::cudnnConvolutionDescriptor_t,
                                                              foldedGradDesc::cudnnTensorDescriptor_t,
                                                              filterFoldTransDesc::cudnnTensorTransformDescriptor_t,
                                                              diffPadTransDesc::cudnnTensorTransformDescriptor_t,
                                                              gradFoldTransDesc::cudnnTensorTransformDescriptor_t,
                                                              gradUnfoldTransDesc::cudnnTensorTransformDescriptor_t)::cudnnStatus_t
end

mutable struct cudnnFusedOpsConstParamStruct end

const cudnnFusedOpsConstParamPack_t = Ptr{cudnnFusedOpsConstParamStruct}

mutable struct cudnnFusedOpsVariantParamStruct end

const cudnnFusedOpsVariantParamPack_t = Ptr{cudnnFusedOpsVariantParamStruct}

mutable struct cudnnFusedOpsPlanStruct end

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

@checked function cudnnCnnInferVersionCheck()
    initialize_context()
    @ccall libcudnn.cudnnCnnInferVersionCheck()::cudnnStatus_t
end

struct cudnnConvolutionBwdFilterAlgoPerfStruct
    algo::cudnnConvolutionBwdFilterAlgo_t
    status::cudnnStatus_t
    time::Cfloat
    memory::Csize_t
    determinism::cudnnDeterminism_t
    mathType::cudnnMathType_t
    reserved::NTuple{3,Cint}
end

const cudnnConvolutionBwdFilterAlgoPerf_t = cudnnConvolutionBwdFilterAlgoPerfStruct

@checked function cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, count)
    initialize_context()
    @ccall libcudnn.cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle::cudnnHandle_t,
                                                                       count::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnFindConvolutionBackwardFilterAlgorithm(handle, xDesc, dyDesc,
                                                              convDesc, dwDesc,
                                                              requestedAlgoCount,
                                                              returnedAlgoCount,
                                                              perfResults)
    initialize_context()
    @ccall libcudnn.cudnnFindConvolutionBackwardFilterAlgorithm(handle::cudnnHandle_t,
                                                                xDesc::cudnnTensorDescriptor_t,
                                                                dyDesc::cudnnTensorDescriptor_t,
                                                                convDesc::cudnnConvolutionDescriptor_t,
                                                                dwDesc::cudnnFilterDescriptor_t,
                                                                requestedAlgoCount::Cint,
                                                                returnedAlgoCount::Ptr{Cint},
                                                                perfResults::Ptr{cudnnConvolutionBwdFilterAlgoPerf_t})::cudnnStatus_t
end

@checked function cudnnFindConvolutionBackwardFilterAlgorithmEx(handle, xDesc, x, dyDesc, y,
                                                                convDesc, dwDesc, dw,
                                                                requestedAlgoCount,
                                                                returnedAlgoCount,
                                                                perfResults, workSpace,
                                                                workSpaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnFindConvolutionBackwardFilterAlgorithmEx(handle::cudnnHandle_t,
                                                                  xDesc::cudnnTensorDescriptor_t,
                                                                  x::CuPtr{Cvoid},
                                                                  dyDesc::cudnnTensorDescriptor_t,
                                                                  y::CuPtr{Cvoid},
                                                                  convDesc::cudnnConvolutionDescriptor_t,
                                                                  dwDesc::cudnnFilterDescriptor_t,
                                                                  dw::CuPtr{Cvoid},
                                                                  requestedAlgoCount::Cint,
                                                                  returnedAlgoCount::Ptr{Cint},
                                                                  perfResults::Ptr{cudnnConvolutionBwdFilterAlgoPerf_t},
                                                                  workSpace::CuPtr{Cvoid},
                                                                  workSpaceSizeInBytes::Csize_t)::cudnnStatus_t
end

@checked function cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, srcDesc, diffDesc,
                                                                convDesc, gradDesc,
                                                                requestedAlgoCount,
                                                                returnedAlgoCount,
                                                                perfResults)
    initialize_context()
    @ccall libcudnn.cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle::cudnnHandle_t,
                                                                  srcDesc::cudnnTensorDescriptor_t,
                                                                  diffDesc::cudnnTensorDescriptor_t,
                                                                  convDesc::cudnnConvolutionDescriptor_t,
                                                                  gradDesc::cudnnFilterDescriptor_t,
                                                                  requestedAlgoCount::Cint,
                                                                  returnedAlgoCount::Ptr{Cint},
                                                                  perfResults::Ptr{cudnnConvolutionBwdFilterAlgoPerf_t})::cudnnStatus_t
end

@checked function cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, xDesc, dyDesc,
                                                                 convDesc, gradDesc, algo,
                                                                 sizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(handle::cudnnHandle_t,
                                                                   xDesc::cudnnTensorDescriptor_t,
                                                                   dyDesc::cudnnTensorDescriptor_t,
                                                                   convDesc::cudnnConvolutionDescriptor_t,
                                                                   gradDesc::cudnnFilterDescriptor_t,
                                                                   algo::cudnnConvolutionBwdFilterAlgo_t,
                                                                   sizeInBytes::Ref{Csize_t})::cudnnStatus_t
end

@checked function cudnnConvolutionBackwardFilter(handle, alpha, xDesc, x, dyDesc, dy,
                                                 convDesc, algo, workSpace,
                                                 workSpaceSizeInBytes, beta, dwDesc, dw)
    initialize_context()
    @ccall libcudnn.cudnnConvolutionBackwardFilter(handle::cudnnHandle_t, alpha::Ptr{Cvoid},
                                                   xDesc::cudnnTensorDescriptor_t,
                                                   x::CuPtr{Cvoid},
                                                   dyDesc::cudnnTensorDescriptor_t,
                                                   dy::CuPtr{Cvoid},
                                                   convDesc::cudnnConvolutionDescriptor_t,
                                                   algo::cudnnConvolutionBwdFilterAlgo_t,
                                                   workSpace::CuPtr{Cvoid},
                                                   workSpaceSizeInBytes::Csize_t,
                                                   beta::Ptr{Cvoid},
                                                   dwDesc::cudnnFilterDescriptor_t,
                                                   dw::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnConvolutionBackwardBias(handle, alpha, dyDesc, dy, beta, dbDesc, db)
    initialize_context()
    @ccall libcudnn.cudnnConvolutionBackwardBias(handle::cudnnHandle_t, alpha::Ptr{Cvoid},
                                                 dyDesc::cudnnTensorDescriptor_t,
                                                 dy::CuPtr{Cvoid}, beta::Ptr{Cvoid},
                                                 dbDesc::cudnnTensorDescriptor_t,
                                                 db::CuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnCreateFusedOpsConstParamPack(constPack, ops)
    initialize_context()
    @ccall libcudnn.cudnnCreateFusedOpsConstParamPack(constPack::Ptr{cudnnFusedOpsConstParamPack_t},
                                                      ops::cudnnFusedOps_t)::cudnnStatus_t
end

@checked function cudnnDestroyFusedOpsConstParamPack(constPack)
    initialize_context()
    @ccall libcudnn.cudnnDestroyFusedOpsConstParamPack(constPack::cudnnFusedOpsConstParamPack_t)::cudnnStatus_t
end

@checked function cudnnSetFusedOpsConstParamPackAttribute(constPack, paramLabel, param)
    initialize_context()
    @ccall libcudnn.cudnnSetFusedOpsConstParamPackAttribute(constPack::cudnnFusedOpsConstParamPack_t,
                                                            paramLabel::cudnnFusedOpsConstParamLabel_t,
                                                            param::Ptr{Cvoid})::cudnnStatus_t
end

@checked function cudnnGetFusedOpsConstParamPackAttribute(constPack, paramLabel, param,
                                                          isNULL)
    initialize_context()
    @ccall libcudnn.cudnnGetFusedOpsConstParamPackAttribute(constPack::cudnnFusedOpsConstParamPack_t,
                                                            paramLabel::cudnnFusedOpsConstParamLabel_t,
                                                            param::Ptr{Cvoid},
                                                            isNULL::Ptr{Cint})::cudnnStatus_t
end

@checked function cudnnCreateFusedOpsVariantParamPack(varPack, ops)
    initialize_context()
    @ccall libcudnn.cudnnCreateFusedOpsVariantParamPack(varPack::Ptr{cudnnFusedOpsVariantParamPack_t},
                                                        ops::cudnnFusedOps_t)::cudnnStatus_t
end

@checked function cudnnDestroyFusedOpsVariantParamPack(varPack)
    initialize_context()
    @ccall libcudnn.cudnnDestroyFusedOpsVariantParamPack(varPack::cudnnFusedOpsVariantParamPack_t)::cudnnStatus_t
end

@checked function cudnnSetFusedOpsVariantParamPackAttribute(varPack, paramLabel, ptr)
    initialize_context()
    @ccall libcudnn.cudnnSetFusedOpsVariantParamPackAttribute(varPack::cudnnFusedOpsVariantParamPack_t,
                                                              paramLabel::cudnnFusedOpsVariantParamLabel_t,
                                                              ptr::PtrOrCuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnGetFusedOpsVariantParamPackAttribute(varPack, paramLabel, ptr)
    initialize_context()
    @ccall libcudnn.cudnnGetFusedOpsVariantParamPackAttribute(varPack::cudnnFusedOpsVariantParamPack_t,
                                                              paramLabel::cudnnFusedOpsVariantParamLabel_t,
                                                              ptr::PtrOrCuPtr{Cvoid})::cudnnStatus_t
end

@checked function cudnnCreateFusedOpsPlan(plan, ops)
    initialize_context()
    @ccall libcudnn.cudnnCreateFusedOpsPlan(plan::Ptr{cudnnFusedOpsPlan_t},
                                            ops::cudnnFusedOps_t)::cudnnStatus_t
end

@checked function cudnnDestroyFusedOpsPlan(plan)
    initialize_context()
    @ccall libcudnn.cudnnDestroyFusedOpsPlan(plan::cudnnFusedOpsPlan_t)::cudnnStatus_t
end

@checked function cudnnMakeFusedOpsPlan(handle, plan, constPack, workspaceSizeInBytes)
    initialize_context()
    @ccall libcudnn.cudnnMakeFusedOpsPlan(handle::cudnnHandle_t, plan::cudnnFusedOpsPlan_t,
                                          constPack::cudnnFusedOpsConstParamPack_t,
                                          workspaceSizeInBytes::Ptr{Csize_t})::cudnnStatus_t
end

@checked function cudnnFusedOpsExecute(handle, plan, varPack)
    initialize_context()
    @ccall libcudnn.cudnnFusedOpsExecute(handle::cudnnHandle_t, plan::cudnnFusedOpsPlan_t,
                                         varPack::cudnnFusedOpsVariantParamPack_t)::cudnnStatus_t
end

@checked function cudnnCnnTrainVersionCheck()
    initialize_context()
    @ccall libcudnn.cudnnCnnTrainVersionCheck()::cudnnStatus_t
end

const CUDNN_MAX_SM_MAJOR_NUMBER = 9

const CUDNN_MAX_SM_MINOR_NUMBER = 0

const CUDNN_SM_50 = 500

const CUDNN_SM_52 = 520

const CUDNN_SM_53 = 530

const CUDNN_SM_60 = 600

const CUDNN_SM_61 = 610

const CUDNN_SM_62 = 620

const CUDNN_SM_70 = 700

const CUDNN_SM_72 = 720

const CUDNN_SM_75 = 750

const CUDNN_SM_80 = 800

const CUDNN_SM_86 = 860

const CUDNN_SM_87 = 870

const CUDNN_SM_89 = 890

const CUDNN_SM_90 = 900

const CUDNN_DIM_MAX = 8

const CUDNN_LRN_MIN_N = 1

const CUDNN_LRN_MAX_N = 16

const CUDNN_LRN_MIN_K = 1.0e-5

const CUDNN_LRN_MIN_BETA = 0.01

const CUDNN_BN_MIN_EPSILON = 0.0

const CUDNN_RNN_PADDED_IO_DISABLED = 0

const CUDNN_RNN_PADDED_IO_ENABLED = Cuint(1) << 0

const CUDNN_SEQDATA_DIM_COUNT = 4

const CUDNN_ATTN_QUERYMAP_ALL_TO_ONE = 0

const CUDNN_ATTN_QUERYMAP_ONE_TO_ONE = Cuint(1) << 0

const CUDNN_ATTN_DISABLE_PROJ_BIASES = 0

const CUDNN_ATTN_ENABLE_PROJ_BIASES = Cuint(1) << 1

const CUDNN_ATTN_WKIND_COUNT = 8
