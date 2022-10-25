using CEnum

# CUDNN uses CUDA runtime objects, which are compatible with our driver usage
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
        res = @retry_reclaim err->$check $(esc(ex))
        if res != CUDNN_STATUS_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end


mutable struct cudnnContext end

const cudnnHandle_t = Ptr{cudnnContext}

function cudnnGetVersion()
    ccall((:cudnnGetVersion, libcudnn), Csize_t, ())
end

function cudnnGetMaxDeviceVersion()
    ccall((:cudnnGetMaxDeviceVersion, libcudnn), Csize_t, ())
end

function cudnnGetCudartVersion()
    ccall((:cudnnGetCudartVersion, libcudnn), Csize_t, ())
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
    ccall((:cudnnGetErrorString, libcudnn), Cstring, (cudnnStatus_t,), status)
end

mutable struct cudnnRuntimeTag_t end

@cenum cudnnErrQueryMode_t::UInt32 begin
    CUDNN_ERRQUERY_RAWCODE = 0
    CUDNN_ERRQUERY_NONBLOCKING = 1
    CUDNN_ERRQUERY_BLOCKING = 2
end

@checked function cudnnQueryRuntimeError(handle, rstatus, mode, tag)
        initialize_context()
        ccall((:cudnnQueryRuntimeError, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{cudnnStatus_t}, cudnnErrQueryMode_t, Ptr{cudnnRuntimeTag_t}), handle, rstatus, mode, tag)
    end

@checked function cudnnGetProperty(type, value)
        ccall((:cudnnGetProperty, libcudnn), cudnnStatus_t, (libraryPropertyType, Ptr{Cint}), type, value)
    end

@checked function cudnnCreate(handle)
        initialize_context()
        ccall((:cudnnCreate, libcudnn), cudnnStatus_t, (Ptr{cudnnHandle_t},), handle)
    end

@checked function cudnnDestroy(handle)
        initialize_context()
        ccall((:cudnnDestroy, libcudnn), cudnnStatus_t, (cudnnHandle_t,), handle)
    end

@checked function cudnnSetStream(handle, streamId)
        initialize_context()
        ccall((:cudnnSetStream, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudaStream_t), handle, streamId)
    end

@checked function cudnnGetStream(handle, streamId)
        initialize_context()
        ccall((:cudnnGetStream, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{cudaStream_t}), handle, streamId)
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
        ccall((:cudnnCreateTensorDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnTensorDescriptor_t},), tensorDesc)
    end

@cenum cudnnTensorFormat_t::UInt32 begin
    CUDNN_TENSOR_NCHW = 0
    CUDNN_TENSOR_NHWC = 1
    CUDNN_TENSOR_NCHW_VECT_C = 2
end

@checked function cudnnSetTensor4dDescriptor(tensorDesc, format, dataType, n, c, h, w)
        initialize_context()
        ccall((:cudnnSetTensor4dDescriptor, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, Cint, Cint, Cint, Cint), tensorDesc, format, dataType, n, c, h, w)
    end

@checked function cudnnSetTensor4dDescriptorEx(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride)
        initialize_context()
        ccall((:cudnnSetTensor4dDescriptorEx, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, cudnnDataType_t, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint), tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride)
    end

@checked function cudnnGetTensor4dDescriptor(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride)
        initialize_context()
        ccall((:cudnnGetTensor4dDescriptor, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, Ptr{cudnnDataType_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride)
    end

@checked function cudnnSetTensorNdDescriptor(tensorDesc, dataType, nbDims, dimA, strideA)
        initialize_context()
        ccall((:cudnnSetTensorNdDescriptor, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, cudnnDataType_t, Cint, Ptr{Cint}, Ptr{Cint}), tensorDesc, dataType, nbDims, dimA, strideA)
    end

@checked function cudnnSetTensorNdDescriptorEx(tensorDesc, format, dataType, nbDims, dimA)
        initialize_context()
        ccall((:cudnnSetTensorNdDescriptorEx, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, Cint, Ptr{Cint}), tensorDesc, format, dataType, nbDims, dimA)
    end

@checked function cudnnGetTensorNdDescriptor(tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA)
        initialize_context()
        ccall((:cudnnGetTensorNdDescriptor, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, Cint, Ptr{cudnnDataType_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA)
    end

@checked function cudnnGetTensorSizeInBytes(tensorDesc, size)
        initialize_context()
        ccall((:cudnnGetTensorSizeInBytes, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, Ptr{Csize_t}), tensorDesc, size)
    end

@checked function cudnnDestroyTensorDescriptor(tensorDesc)
        initialize_context()
        ccall((:cudnnDestroyTensorDescriptor, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t,), tensorDesc)
    end

@cenum cudnnFoldingDirection_t::UInt32 begin
    CUDNN_TRANSFORM_FOLD = 0
    CUDNN_TRANSFORM_UNFOLD = 1
end

@checked function cudnnInitTransformDest(transformDesc, srcDesc, destDesc, destSizeInBytes)
        initialize_context()
        ccall((:cudnnInitTransformDest, libcudnn), cudnnStatus_t, (cudnnTensorTransformDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, Ptr{Csize_t}), transformDesc, srcDesc, destDesc, destSizeInBytes)
    end

@checked function cudnnCreateTensorTransformDescriptor(transformDesc)
        initialize_context()
        ccall((:cudnnCreateTensorTransformDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnTensorTransformDescriptor_t},), transformDesc)
    end

@checked function cudnnSetTensorTransformDescriptor(transformDesc, nbDims, destFormat, padBeforeA, padAfterA, foldA, direction)
        initialize_context()
        ccall((:cudnnSetTensorTransformDescriptor, libcudnn), cudnnStatus_t, (cudnnTensorTransformDescriptor_t, UInt32, cudnnTensorFormat_t, Ptr{Int32}, Ptr{Int32}, Ptr{UInt32}, cudnnFoldingDirection_t), transformDesc, nbDims, destFormat, padBeforeA, padAfterA, foldA, direction)
    end

@checked function cudnnGetTensorTransformDescriptor(transformDesc, nbDimsRequested, destFormat, padBeforeA, padAfterA, foldA, direction)
        initialize_context()
        ccall((:cudnnGetTensorTransformDescriptor, libcudnn), cudnnStatus_t, (cudnnTensorTransformDescriptor_t, UInt32, Ptr{cudnnTensorFormat_t}, Ptr{Int32}, Ptr{Int32}, Ptr{UInt32}, Ptr{cudnnFoldingDirection_t}), transformDesc, nbDimsRequested, destFormat, padBeforeA, padAfterA, foldA, direction)
    end

@checked function cudnnDestroyTensorTransformDescriptor(transformDesc)
        initialize_context()
        ccall((:cudnnDestroyTensorTransformDescriptor, libcudnn), cudnnStatus_t, (cudnnTensorTransformDescriptor_t,), transformDesc)
    end

@checked function cudnnTransformTensor(handle, alpha, xDesc, x, beta, yDesc, y)
        initialize_context()
        ccall((:cudnnTransformTensor, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, Ptr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, Ptr{Cvoid}), handle, alpha, xDesc, x, beta, yDesc, y)
    end

@checked function cudnnTransformTensorEx(handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData)
        initialize_context()
        ccall((:cudnnTransformTensorEx, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorTransformDescriptor_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, Ptr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, Ptr{Cvoid}), handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData)
    end

@checked function cudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C)
        initialize_context()
        ccall((:cudnnAddTensor, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}), handle, alpha, aDesc, A, beta, cDesc, C)
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
        ccall((:cudnnCreateOpTensorDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnOpTensorDescriptor_t},), opTensorDesc)
    end

@checked function cudnnSetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)
        initialize_context()
        ccall((:cudnnSetOpTensorDescriptor, libcudnn), cudnnStatus_t, (cudnnOpTensorDescriptor_t, cudnnOpTensorOp_t, cudnnDataType_t, cudnnNanPropagation_t), opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)
    end

@checked function cudnnGetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)
        initialize_context()
        ccall((:cudnnGetOpTensorDescriptor, libcudnn), cudnnStatus_t, (cudnnOpTensorDescriptor_t, Ptr{cudnnOpTensorOp_t}, Ptr{cudnnDataType_t}, Ptr{cudnnNanPropagation_t}), opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)
    end

@checked function cudnnDestroyOpTensorDescriptor(opTensorDesc)
        initialize_context()
        ccall((:cudnnDestroyOpTensorDescriptor, libcudnn), cudnnStatus_t, (cudnnOpTensorDescriptor_t,), opTensorDesc)
    end

@checked function cudnnOpTensor(handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C)
        initialize_context()
        ccall((:cudnnOpTensor, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnOpTensorDescriptor_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}), handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C)
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
        ccall((:cudnnCreateReduceTensorDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnReduceTensorDescriptor_t},), reduceTensorDesc)
    end

@checked function cudnnSetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType)
        initialize_context()
        ccall((:cudnnSetReduceTensorDescriptor, libcudnn), cudnnStatus_t, (cudnnReduceTensorDescriptor_t, cudnnReduceTensorOp_t, cudnnDataType_t, cudnnNanPropagation_t, cudnnReduceTensorIndices_t, cudnnIndicesType_t), reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType)
    end

@checked function cudnnGetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType)
        initialize_context()
        ccall((:cudnnGetReduceTensorDescriptor, libcudnn), cudnnStatus_t, (cudnnReduceTensorDescriptor_t, Ptr{cudnnReduceTensorOp_t}, Ptr{cudnnDataType_t}, Ptr{cudnnNanPropagation_t}, Ptr{cudnnReduceTensorIndices_t}, Ptr{cudnnIndicesType_t}), reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType)
    end

@checked function cudnnDestroyReduceTensorDescriptor(reduceTensorDesc)
        initialize_context()
        ccall((:cudnnDestroyReduceTensorDescriptor, libcudnn), cudnnStatus_t, (cudnnReduceTensorDescriptor_t,), reduceTensorDesc)
    end

@checked function cudnnGetReductionIndicesSize(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes)
        initialize_context()
        ccall((:cudnnGetReductionIndicesSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnReduceTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, Ptr{Csize_t}), handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes)
    end

@checked function cudnnGetReductionWorkspaceSize(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes)
        initialize_context()
        ccall((:cudnnGetReductionWorkspaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnReduceTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, Ref{Csize_t}), handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes)
    end

@checked function cudnnReduceTensor(handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace, workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C)
        initialize_context()
        ccall((:cudnnReduceTensor, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnReduceTensorDescriptor_t, Ptr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}), handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace, workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C)
    end

@checked function cudnnSetTensor(handle, yDesc, y, valuePtr)
        initialize_context()
        ccall((:cudnnSetTensor, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}), handle, yDesc, y, valuePtr)
    end

@checked function cudnnScaleTensor(handle, yDesc, y, alpha)
        initialize_context()
        ccall((:cudnnScaleTensor, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}), handle, yDesc, y, alpha)
    end

@checked function cudnnCreateFilterDescriptor(filterDesc)
        initialize_context()
        ccall((:cudnnCreateFilterDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnFilterDescriptor_t},), filterDesc)
    end

@checked function cudnnSetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w)
        initialize_context()
        ccall((:cudnnSetFilter4dDescriptor, libcudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, cudnnDataType_t, cudnnTensorFormat_t, Cint, Cint, Cint, Cint), filterDesc, dataType, format, k, c, h, w)
    end

@checked function cudnnGetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w)
        initialize_context()
        ccall((:cudnnGetFilter4dDescriptor, libcudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, Ptr{cudnnDataType_t}, Ptr{cudnnTensorFormat_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), filterDesc, dataType, format, k, c, h, w)
    end

@checked function cudnnSetFilterNdDescriptor(filterDesc, dataType, format, nbDims, filterDimA)
        initialize_context()
        ccall((:cudnnSetFilterNdDescriptor, libcudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, cudnnDataType_t, cudnnTensorFormat_t, Cint, Ptr{Cint}), filterDesc, dataType, format, nbDims, filterDimA)
    end

@checked function cudnnGetFilterNdDescriptor(filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA)
        initialize_context()
        ccall((:cudnnGetFilterNdDescriptor, libcudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, Cint, Ptr{cudnnDataType_t}, Ptr{cudnnTensorFormat_t}, Ptr{Cint}, Ptr{Cint}), filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA)
    end

@checked function cudnnGetFilterSizeInBytes(filterDesc, size)
        initialize_context()
        ccall((:cudnnGetFilterSizeInBytes, libcudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, Ptr{Csize_t}), filterDesc, size)
    end

@checked function cudnnTransformFilter(handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData)
        initialize_context()
        ccall((:cudnnTransformFilter, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorTransformDescriptor_t, Ptr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}), handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData)
    end

@checked function cudnnDestroyFilterDescriptor(filterDesc)
        initialize_context()
        ccall((:cudnnDestroyFilterDescriptor, libcudnn), cudnnStatus_t, (cudnnFilterDescriptor_t,), filterDesc)
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
        ccall((:cudnnSoftmaxForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}), handle, algo, mode, alpha, xDesc, x, beta, yDesc, y)
    end

@cenum cudnnPoolingMode_t::UInt32 begin
    CUDNN_POOLING_MAX = 0
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2
    CUDNN_POOLING_MAX_DETERMINISTIC = 3
end

@checked function cudnnCreatePoolingDescriptor(poolingDesc)
        initialize_context()
        ccall((:cudnnCreatePoolingDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnPoolingDescriptor_t},), poolingDesc)
    end

@checked function cudnnSetPooling2dDescriptor(poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)
        initialize_context()
        ccall((:cudnnSetPooling2dDescriptor, libcudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnNanPropagation_t, Cint, Cint, Cint, Cint, Cint, Cint), poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)
    end

@checked function cudnnGetPooling2dDescriptor(poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)
        initialize_context()
        ccall((:cudnnGetPooling2dDescriptor, libcudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, Ptr{cudnnPoolingMode_t}, Ptr{cudnnNanPropagation_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)
    end

@checked function cudnnSetPoolingNdDescriptor(poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA)
        initialize_context()
        ccall((:cudnnSetPoolingNdDescriptor, libcudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnNanPropagation_t, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA)
    end

@checked function cudnnGetPoolingNdDescriptor(poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA)
        initialize_context()
        ccall((:cudnnGetPoolingNdDescriptor, libcudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, Cint, Ptr{cudnnPoolingMode_t}, Ptr{cudnnNanPropagation_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA)
    end

@checked function cudnnGetPoolingNdForwardOutputDim(poolingDesc, inputTensorDesc, nbDims, outputTensorDimA)
        initialize_context()
        ccall((:cudnnGetPoolingNdForwardOutputDim, libcudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint}), poolingDesc, inputTensorDesc, nbDims, outputTensorDimA)
    end

@checked function cudnnGetPooling2dForwardOutputDim(poolingDesc, inputTensorDesc, n, c, h, w)
        initialize_context()
        ccall((:cudnnGetPooling2dForwardOutputDim, libcudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, cudnnTensorDescriptor_t, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), poolingDesc, inputTensorDesc, n, c, h, w)
    end

@checked function cudnnDestroyPoolingDescriptor(poolingDesc)
        initialize_context()
        ccall((:cudnnDestroyPoolingDescriptor, libcudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t,), poolingDesc)
    end

@checked function cudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y)
        initialize_context()
        ccall((:cudnnPoolingForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnPoolingDescriptor_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}), handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y)
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
        ccall((:cudnnCreateActivationDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnActivationDescriptor_t},), activationDesc)
    end

@checked function cudnnSetActivationDescriptor(activationDesc, mode, reluNanOpt, coef)
        initialize_context()
        ccall((:cudnnSetActivationDescriptor, libcudnn), cudnnStatus_t, (cudnnActivationDescriptor_t, cudnnActivationMode_t, cudnnNanPropagation_t, Cdouble), activationDesc, mode, reluNanOpt, coef)
    end

@checked function cudnnGetActivationDescriptor(activationDesc, mode, reluNanOpt, coef)
        initialize_context()
        ccall((:cudnnGetActivationDescriptor, libcudnn), cudnnStatus_t, (cudnnActivationDescriptor_t, Ptr{cudnnActivationMode_t}, Ptr{cudnnNanPropagation_t}, Ptr{Cdouble}), activationDesc, mode, reluNanOpt, coef)
    end

@checked function cudnnSetActivationDescriptorSwishBeta(activationDesc, swish_beta)
        initialize_context()
        ccall((:cudnnSetActivationDescriptorSwishBeta, libcudnn), cudnnStatus_t, (cudnnActivationDescriptor_t, Cdouble), activationDesc, swish_beta)
    end

@checked function cudnnGetActivationDescriptorSwishBeta(activationDesc, swish_beta)
        initialize_context()
        ccall((:cudnnGetActivationDescriptorSwishBeta, libcudnn), cudnnStatus_t, (cudnnActivationDescriptor_t, Ptr{Cdouble}), activationDesc, swish_beta)
    end

@checked function cudnnDestroyActivationDescriptor(activationDesc)
        initialize_context()
        ccall((:cudnnDestroyActivationDescriptor, libcudnn), cudnnStatus_t, (cudnnActivationDescriptor_t,), activationDesc)
    end

@checked function cudnnActivationForward(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y)
        initialize_context()
        ccall((:cudnnActivationForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnActivationDescriptor_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}), handle, activationDesc, alpha, xDesc, x, beta, yDesc, y)
    end

@checked function cudnnCreateLRNDescriptor(normDesc)
        initialize_context()
        ccall((:cudnnCreateLRNDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnLRNDescriptor_t},), normDesc)
    end

@cenum cudnnLRNMode_t::UInt32 begin
    CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0
end

@checked function cudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
        initialize_context()
        ccall((:cudnnSetLRNDescriptor, libcudnn), cudnnStatus_t, (cudnnLRNDescriptor_t, Cuint, Cdouble, Cdouble, Cdouble), normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
    end

@checked function cudnnGetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
        initialize_context()
        ccall((:cudnnGetLRNDescriptor, libcudnn), cudnnStatus_t, (cudnnLRNDescriptor_t, Ptr{Cuint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
    end

@checked function cudnnDestroyLRNDescriptor(lrnDesc)
        initialize_context()
        ccall((:cudnnDestroyLRNDescriptor, libcudnn), cudnnStatus_t, (cudnnLRNDescriptor_t,), lrnDesc)
    end

@checked function cudnnLRNCrossChannelForward(handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y)
        initialize_context()
        ccall((:cudnnLRNCrossChannelForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnLRNMode_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}), handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y)
    end

@cenum cudnnDivNormMode_t::UInt32 begin
    CUDNN_DIVNORM_PRECOMPUTED_MEANS = 0
end

@checked function cudnnDivisiveNormalizationForward(handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y)
        initialize_context()
        ccall((:cudnnDivisiveNormalizationForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnDivNormMode_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}), handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y)
    end

@cenum cudnnBatchNormMode_t::UInt32 begin
    CUDNN_BATCHNORM_PER_ACTIVATION = 0
    CUDNN_BATCHNORM_SPATIAL = 1
    CUDNN_BATCHNORM_SPATIAL_PERSISTENT = 2
end

@checked function cudnnDeriveBNTensorDescriptor(derivedBnDesc, xDesc, mode)
        initialize_context()
        ccall((:cudnnDeriveBNTensorDescriptor, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnBatchNormMode_t), derivedBnDesc, xDesc, mode)
    end

@cenum cudnnBatchNormOps_t::UInt32 begin
    CUDNN_BATCHNORM_OPS_BN = 0
    CUDNN_BATCHNORM_OPS_BN_ACTIVATION = 1
    CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION = 2
end

@checked function cudnnBatchNormalizationForwardInference(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon)
        initialize_context()
        ccall((:cudnnBatchNormalizationForwardInference, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnBatchNormMode_t, Ptr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, Cdouble), handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon)
    end

@cenum cudnnNormMode_t::UInt32 begin
    CUDNN_NORM_PER_ACTIVATION = 0
    CUDNN_NORM_PER_CHANNEL = 1
end

@cenum cudnnNormAlgo_t::UInt32 begin
    CUDNN_NORM_ALGO_STANDARD = 0
    CUDNN_NORM_ALGO_PERSIST = 1
end

@checked function cudnnDeriveNormTensorDescriptor(derivedNormScaleBiasDesc, derivedNormMeanVarDesc, xDesc, mode, groupCnt)
        initialize_context()
        ccall((:cudnnDeriveNormTensorDescriptor, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnNormMode_t, Cint), derivedNormScaleBiasDesc, derivedNormMeanVarDesc, xDesc, mode, groupCnt)
    end

@cenum cudnnNormOps_t::UInt32 begin
    CUDNN_NORM_OPS_NORM = 0
    CUDNN_NORM_OPS_NORM_ACTIVATION = 1
    CUDNN_NORM_OPS_NORM_ADD_ACTIVATION = 2
end

@checked function cudnnNormalizationForwardInference(handle, mode, normOps, algo, alpha, beta, xDesc, x, normScaleBiasDesc, normScale, normBias, normMeanVarDesc, estimatedMean, estimatedVariance, zDesc, z, activationDesc, yDesc, y, epsilon, groupCnt)
        initialize_context()
        ccall((:cudnnNormalizationForwardInference, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnNormMode_t, cudnnNormOps_t, cudnnNormAlgo_t, Ptr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnActivationDescriptor_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Cdouble, Cint), handle, mode, normOps, algo, alpha, beta, xDesc, x, normScaleBiasDesc, normScale, normBias, normMeanVarDesc, estimatedMean, estimatedVariance, zDesc, z, activationDesc, yDesc, y, epsilon, groupCnt)
    end

@cenum cudnnSamplerType_t::UInt32 begin
    CUDNN_SAMPLER_BILINEAR = 0
end

@checked function cudnnCreateSpatialTransformerDescriptor(stDesc)
        initialize_context()
        ccall((:cudnnCreateSpatialTransformerDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnSpatialTransformerDescriptor_t},), stDesc)
    end

@checked function cudnnSetSpatialTransformerNdDescriptor(stDesc, samplerType, dataType, nbDims, dimA)
        initialize_context()
        ccall((:cudnnSetSpatialTransformerNdDescriptor, libcudnn), cudnnStatus_t, (cudnnSpatialTransformerDescriptor_t, cudnnSamplerType_t, cudnnDataType_t, Cint, Ptr{Cint}), stDesc, samplerType, dataType, nbDims, dimA)
    end

@checked function cudnnDestroySpatialTransformerDescriptor(stDesc)
        initialize_context()
        ccall((:cudnnDestroySpatialTransformerDescriptor, libcudnn), cudnnStatus_t, (cudnnSpatialTransformerDescriptor_t,), stDesc)
    end

@checked function cudnnSpatialTfGridGeneratorForward(handle, stDesc, theta, grid)
        initialize_context()
        ccall((:cudnnSpatialTfGridGeneratorForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}), handle, stDesc, theta, grid)
    end

@checked function cudnnSpatialTfSamplerForward(handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y)
        initialize_context()
        ccall((:cudnnSpatialTfSamplerForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}), handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y)
    end

mutable struct cudnnDropoutStruct end

const cudnnDropoutDescriptor_t = Ptr{cudnnDropoutStruct}

@checked function cudnnCreateDropoutDescriptor(dropoutDesc)
        initialize_context()
        ccall((:cudnnCreateDropoutDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnDropoutDescriptor_t},), dropoutDesc)
    end

@checked function cudnnDestroyDropoutDescriptor(dropoutDesc)
        initialize_context()
        ccall((:cudnnDestroyDropoutDescriptor, libcudnn), cudnnStatus_t, (cudnnDropoutDescriptor_t,), dropoutDesc)
    end

@checked function cudnnDropoutGetStatesSize(handle, sizeInBytes)
        initialize_context()
        ccall((:cudnnDropoutGetStatesSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Csize_t}), handle, sizeInBytes)
    end

@checked function cudnnDropoutGetReserveSpaceSize(xdesc, sizeInBytes)
        initialize_context()
        ccall((:cudnnDropoutGetReserveSpaceSize, libcudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, Ref{Csize_t}), xdesc, sizeInBytes)
    end

@checked function cudnnSetDropoutDescriptor(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed)
        initialize_context()
        ccall((:cudnnSetDropoutDescriptor, libcudnn), cudnnStatus_t, (cudnnDropoutDescriptor_t, cudnnHandle_t, Cfloat, CuPtr{Cvoid}, Csize_t, Culonglong), dropoutDesc, handle, dropout, states, stateSizeInBytes, seed)
    end

@checked function cudnnRestoreDropoutDescriptor(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed)
        initialize_context()
        ccall((:cudnnRestoreDropoutDescriptor, libcudnn), cudnnStatus_t, (cudnnDropoutDescriptor_t, cudnnHandle_t, Cfloat, CuPtr{Cvoid}, Csize_t, Culonglong), dropoutDesc, handle, dropout, states, stateSizeInBytes, seed)
    end

@checked function cudnnGetDropoutDescriptor(dropoutDesc, handle, dropout, states, seed)
        initialize_context()
        ccall((:cudnnGetDropoutDescriptor, libcudnn), cudnnStatus_t, (cudnnDropoutDescriptor_t, cudnnHandle_t, Ptr{Cfloat}, Ptr{CuPtr{Cvoid}}, Ptr{Culonglong}), dropoutDesc, handle, dropout, states, seed)
    end

@checked function cudnnDropoutForward(handle, dropoutDesc, xdesc, x, ydesc, y, reserveSpace, reserveSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnDropoutForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnDropoutDescriptor_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t), handle, dropoutDesc, xdesc, x, ydesc, y, reserveSpace, reserveSpaceSizeInBytes)
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
    data::NTuple{4, UInt8}
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
    unsafe_store!(getproperty(x, f), v)
end

struct cudnnAlgorithmUnionStruct
    data::NTuple{4, UInt8}
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
    unsafe_store!(getproperty(x, f), v)
end

const cudnnAlgorithm_t = cudnnAlgorithmUnionStruct

@checked function cudnnCreateAlgorithmDescriptor(algoDesc)
        initialize_context()
        ccall((:cudnnCreateAlgorithmDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnAlgorithmDescriptor_t},), algoDesc)
    end

@checked function cudnnSetAlgorithmDescriptor(algoDesc, algorithm)
        initialize_context()
        ccall((:cudnnSetAlgorithmDescriptor, libcudnn), cudnnStatus_t, (cudnnAlgorithmDescriptor_t, cudnnAlgorithm_t), algoDesc, algorithm)
    end

@checked function cudnnGetAlgorithmDescriptor(algoDesc, algorithm)
        initialize_context()
        ccall((:cudnnGetAlgorithmDescriptor, libcudnn), cudnnStatus_t, (cudnnAlgorithmDescriptor_t, Ptr{cudnnAlgorithm_t}), algoDesc, algorithm)
    end

@checked function cudnnCopyAlgorithmDescriptor(src, dest)
        initialize_context()
        ccall((:cudnnCopyAlgorithmDescriptor, libcudnn), cudnnStatus_t, (cudnnAlgorithmDescriptor_t, cudnnAlgorithmDescriptor_t), src, dest)
    end

@checked function cudnnDestroyAlgorithmDescriptor(algoDesc)
        initialize_context()
        ccall((:cudnnDestroyAlgorithmDescriptor, libcudnn), cudnnStatus_t, (cudnnAlgorithmDescriptor_t,), algoDesc)
    end

@checked function cudnnCreateAlgorithmPerformance(algoPerf, numberToCreate)
        initialize_context()
        ccall((:cudnnCreateAlgorithmPerformance, libcudnn), cudnnStatus_t, (Ptr{cudnnAlgorithmPerformance_t}, Cint), algoPerf, numberToCreate)
    end

@checked function cudnnSetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory)
        initialize_context()
        ccall((:cudnnSetAlgorithmPerformance, libcudnn), cudnnStatus_t, (cudnnAlgorithmPerformance_t, cudnnAlgorithmDescriptor_t, cudnnStatus_t, Cfloat, Csize_t), algoPerf, algoDesc, status, time, memory)
    end

@checked function cudnnGetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory)
        initialize_context()
        ccall((:cudnnGetAlgorithmPerformance, libcudnn), cudnnStatus_t, (cudnnAlgorithmPerformance_t, Ptr{cudnnAlgorithmDescriptor_t}, Ptr{cudnnStatus_t}, Ptr{Cfloat}, Ptr{Csize_t}), algoPerf, algoDesc, status, time, memory)
    end

@checked function cudnnDestroyAlgorithmPerformance(algoPerf, numberToDestroy)
        initialize_context()
        ccall((:cudnnDestroyAlgorithmPerformance, libcudnn), cudnnStatus_t, (Ptr{cudnnAlgorithmPerformance_t}, Cint), algoPerf, numberToDestroy)
    end

@checked function cudnnGetAlgorithmSpaceSize(handle, algoDesc, algoSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnGetAlgorithmSpaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnAlgorithmDescriptor_t, Ptr{Csize_t}), handle, algoDesc, algoSpaceSizeInBytes)
    end

@checked function cudnnSaveAlgorithm(handle, algoDesc, algoSpace, algoSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnSaveAlgorithm, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnAlgorithmDescriptor_t, Ptr{Cvoid}, Csize_t), handle, algoDesc, algoSpace, algoSpaceSizeInBytes)
    end

@checked function cudnnRestoreAlgorithm(handle, algoSpace, algoSpaceSizeInBytes, algoDesc)
        initialize_context()
        ccall((:cudnnRestoreAlgorithm, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Cvoid}, Csize_t, cudnnAlgorithmDescriptor_t), handle, algoSpace, algoSpaceSizeInBytes, algoDesc)
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
    reserved::NTuple{15, Cint}
end

const cudnnDebug_t = cudnnDebugStruct

# typedef void ( * cudnnCallback_t ) ( cudnnSeverity_t sev , void * udata , const cudnnDebug_t * dbg , const char * msg )
const cudnnCallback_t = Ptr{Cvoid}

@checked function cudnnSetCallback(mask, udata, fptr)
        ccall((:cudnnSetCallback, libcudnn), cudnnStatus_t, (Cuint, Ptr{Cvoid}, cudnnCallback_t), mask, udata, fptr)
    end

@checked function cudnnGetCallback(mask, udata, fptr)
        ccall((:cudnnGetCallback, libcudnn), cudnnStatus_t, (Ptr{Cuint}, Ptr{Ptr{Cvoid}}, Ptr{cudnnCallback_t}), mask, udata, fptr)
    end

@checked function cudnnOpsInferVersionCheck()
        initialize_context()
        ccall((:cudnnOpsInferVersionCheck, libcudnn), cudnnStatus_t, ())
    end

@checked function cudnnSoftmaxBackward(handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx)
        initialize_context()
        ccall((:cudnnSoftmaxBackward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}), handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx)
    end

@checked function cudnnPoolingBackward(handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
        initialize_context()
        ccall((:cudnnPoolingBackward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnPoolingDescriptor_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}), handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
    end

@checked function cudnnActivationBackward(handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
        initialize_context()
        ccall((:cudnnActivationBackward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnActivationDescriptor_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}), handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
    end

@checked function cudnnLRNCrossChannelBackward(handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
        initialize_context()
        ccall((:cudnnLRNCrossChannelBackward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnLRNMode_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}), handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
    end

@checked function cudnnDivisiveNormalizationBackward(handle, normDesc, mode, alpha, xDesc, x, means, dy, temp, temp2, beta, dXdMeansDesc, dx, dMeans)
        initialize_context()
        ccall((:cudnnDivisiveNormalizationBackward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnDivNormMode_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}), handle, normDesc, mode, alpha, xDesc, x, means, dy, temp, temp2, beta, dXdMeansDesc, dx, dMeans)
    end

@checked function cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle, mode, bnOps, xDesc, zDesc, yDesc, bnScaleBiasMeanVarDesc, activationDesc, sizeInBytes)
        initialize_context()
        ccall((:cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnActivationDescriptor_t, Ref{Csize_t}), handle, mode, bnOps, xDesc, zDesc, yDesc, bnScaleBiasMeanVarDesc, activationDesc, sizeInBytes)
    end

@checked function cudnnGetBatchNormalizationBackwardExWorkspaceSize(handle, mode, bnOps, xDesc, yDesc, dyDesc, dzDesc, dxDesc, dBnScaleBiasDesc, activationDesc, sizeInBytes)
        initialize_context()
        ccall((:cudnnGetBatchNormalizationBackwardExWorkspaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnActivationDescriptor_t, Ref{Csize_t}), handle, mode, bnOps, xDesc, yDesc, dyDesc, dzDesc, dxDesc, dBnScaleBiasDesc, activationDesc, sizeInBytes)
    end

@checked function cudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle, mode, bnOps, activationDesc, xDesc, sizeInBytes)
        initialize_context()
        ccall((:cudnnGetBatchNormalizationTrainingExReserveSpaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, cudnnActivationDescriptor_t, cudnnTensorDescriptor_t, Ref{Csize_t}), handle, mode, bnOps, activationDesc, xDesc, sizeInBytes)
    end

@checked function cudnnBatchNormalizationForwardTraining(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance)
        initialize_context()
        ccall((:cudnnBatchNormalizationForwardTraining, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnBatchNormMode_t, Ptr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Cdouble, CuPtr{Cvoid}, CuPtr{Cvoid}, Cdouble, CuPtr{Cvoid}, CuPtr{Cvoid}), handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance)
    end

@checked function cudnnBatchNormalizationForwardTrainingEx(handle, mode, bnOps, alpha, beta, xDesc, xData, zDesc, zData, yDesc, yData, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance, activationDesc, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnBatchNormalizationForwardTrainingEx, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, Ptr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Cdouble, CuPtr{Cvoid}, CuPtr{Cvoid}, Cdouble, CuPtr{Cvoid}, CuPtr{Cvoid}, cudnnActivationDescriptor_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t), handle, mode, bnOps, alpha, beta, xDesc, xData, zDesc, zData, yDesc, yData, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance, activationDesc, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
    end

@checked function cudnnBatchNormalizationBackward(handle, mode, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, x, dyDesc, dy, dxDesc, dx, dBnScaleBiasDesc, bnScale, dBnScaleResult, dBnBiasResult, epsilon, savedMean, savedInvVariance)
        initialize_context()
        ccall((:cudnnBatchNormalizationBackward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnBatchNormMode_t, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, Cdouble, CuPtr{Cvoid}, CuPtr{Cvoid}), handle, mode, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, x, dyDesc, dy, dxDesc, dx, dBnScaleBiasDesc, bnScale, dBnScaleResult, dBnBiasResult, epsilon, savedMean, savedInvVariance)
    end

@checked function cudnnBatchNormalizationBackwardEx(handle, mode, bnOps, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData, dxDesc, dxData, dBnScaleBiasDesc, bnScaleData, bnBiasData, dBnScaleData, dBnBiasData, epsilon, savedMean, savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnBatchNormalizationBackwardEx, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, Cdouble, CuPtr{Cvoid}, CuPtr{Cvoid}, cudnnActivationDescriptor_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t), handle, mode, bnOps, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData, dxDesc, dxData, dBnScaleBiasDesc, bnScaleData, bnBiasData, dBnScaleData, dBnBiasData, epsilon, savedMean, savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
    end

@checked function cudnnGetNormalizationForwardTrainingWorkspaceSize(handle, mode, normOps, algo, xDesc, zDesc, yDesc, normScaleBiasDesc, activationDesc, normMeanVarDesc, sizeInBytes, groupCnt)
        initialize_context()
        ccall((:cudnnGetNormalizationForwardTrainingWorkspaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnNormMode_t, cudnnNormOps_t, cudnnNormAlgo_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnActivationDescriptor_t, cudnnTensorDescriptor_t, Ref{Csize_t}, Cint), handle, mode, normOps, algo, xDesc, zDesc, yDesc, normScaleBiasDesc, activationDesc, normMeanVarDesc, sizeInBytes, groupCnt)
    end

@checked function cudnnGetNormalizationBackwardWorkspaceSize(handle, mode, normOps, algo, xDesc, yDesc, dyDesc, dzDesc, dxDesc, dNormScaleBiasDesc, activationDesc, normMeanVarDesc, sizeInBytes, groupCnt)
        initialize_context()
        ccall((:cudnnGetNormalizationBackwardWorkspaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnNormMode_t, cudnnNormOps_t, cudnnNormAlgo_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnActivationDescriptor_t, cudnnTensorDescriptor_t, Ref{Csize_t}, Cint), handle, mode, normOps, algo, xDesc, yDesc, dyDesc, dzDesc, dxDesc, dNormScaleBiasDesc, activationDesc, normMeanVarDesc, sizeInBytes, groupCnt)
    end

@checked function cudnnGetNormalizationTrainingReserveSpaceSize(handle, mode, normOps, algo, activationDesc, xDesc, sizeInBytes, groupCnt)
        initialize_context()
        ccall((:cudnnGetNormalizationTrainingReserveSpaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnNormMode_t, cudnnNormOps_t, cudnnNormAlgo_t, cudnnActivationDescriptor_t, cudnnTensorDescriptor_t, Ref{Csize_t}, Cint), handle, mode, normOps, algo, activationDesc, xDesc, sizeInBytes, groupCnt)
    end

@checked function cudnnNormalizationForwardTraining(handle, mode, normOps, algo, alpha, beta, xDesc, xData, normScaleBiasDesc, normScale, normBias, exponentialAverageFactor, normMeanVarDesc, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance, activationDesc, zDesc, zData, yDesc, yData, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes, groupCnt)
        initialize_context()
        ccall((:cudnnNormalizationForwardTraining, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnNormMode_t, cudnnNormOps_t, cudnnNormAlgo_t, Ptr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Cdouble, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Cdouble, CuPtr{Cvoid}, CuPtr{Cvoid}, cudnnActivationDescriptor_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, Cint), handle, mode, normOps, algo, alpha, beta, xDesc, xData, normScaleBiasDesc, normScale, normBias, exponentialAverageFactor, normMeanVarDesc, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance, activationDesc, zDesc, zData, yDesc, yData, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes, groupCnt)
    end

@checked function cudnnNormalizationBackward(handle, mode, normOps, algo, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData, dxDesc, dxData, dNormScaleBiasDesc, normScaleData, normBiasData, dNormScaleData, dNormBiasData, epsilon, normMeanVarDesc, savedMean, savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes, groupCnt)
        initialize_context()
        ccall((:cudnnNormalizationBackward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnNormMode_t, cudnnNormOps_t, cudnnNormAlgo_t, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, Cdouble, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, cudnnActivationDescriptor_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, Cint), handle, mode, normOps, algo, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData, dxDesc, dxData, dNormScaleBiasDesc, normScaleData, normBiasData, dNormScaleData, dNormBiasData, epsilon, normMeanVarDesc, savedMean, savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes, groupCnt)
    end

@checked function cudnnSpatialTfGridGeneratorBackward(handle, stDesc, dgrid, dtheta)
        initialize_context()
        ccall((:cudnnSpatialTfGridGeneratorBackward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}), handle, stDesc, dgrid, dtheta)
    end

@checked function cudnnSpatialTfSamplerBackward(handle, stDesc, alpha, xDesc, x, beta, dxDesc, dx, alphaDgrid, dyDesc, dy, grid, betaDgrid, dgrid)
        initialize_context()
        ccall((:cudnnSpatialTfSamplerBackward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Ptr{Cvoid}, CuPtr{Cvoid}), handle, stDesc, alpha, xDesc, x, beta, dxDesc, dx, alphaDgrid, dyDesc, dy, grid, betaDgrid, dgrid)
    end

@checked function cudnnDropoutBackward(handle, dropoutDesc, dydesc, dy, dxdesc, dx, reserveSpace, reserveSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnDropoutBackward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnDropoutDescriptor_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t), handle, dropoutDesc, dydesc, dy, dxdesc, dx, reserveSpace, reserveSpaceSizeInBytes)
    end

@checked function cudnnOpsTrainVersionCheck()
        initialize_context()
        ccall((:cudnnOpsTrainVersionCheck, libcudnn), cudnnStatus_t, ())
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
        ccall((:cudnnCreateRNNDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnRNNDescriptor_t},), rnnDesc)
    end

@checked function cudnnDestroyRNNDescriptor(rnnDesc)
        initialize_context()
        ccall((:cudnnDestroyRNNDescriptor, libcudnn), cudnnStatus_t, (cudnnRNNDescriptor_t,), rnnDesc)
    end

@checked function cudnnSetRNNDescriptor_v8(rnnDesc, algo, cellMode, biasMode, dirMode, inputMode, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropoutDesc, auxFlags)
        initialize_context()
        ccall((:cudnnSetRNNDescriptor_v8, libcudnn), cudnnStatus_t, (cudnnRNNDescriptor_t, cudnnRNNAlgo_t, cudnnRNNMode_t, cudnnRNNBiasMode_t, cudnnDirectionMode_t, cudnnRNNInputMode_t, cudnnDataType_t, cudnnDataType_t, cudnnMathType_t, Int32, Int32, Int32, Int32, cudnnDropoutDescriptor_t, UInt32), rnnDesc, algo, cellMode, biasMode, dirMode, inputMode, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropoutDesc, auxFlags)
    end

@checked function cudnnGetRNNDescriptor_v8(rnnDesc, algo, cellMode, biasMode, dirMode, inputMode, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropoutDesc, auxFlags)
        initialize_context()
        ccall((:cudnnGetRNNDescriptor_v8, libcudnn), cudnnStatus_t, (cudnnRNNDescriptor_t, Ref{cudnnRNNAlgo_t}, Ref{cudnnRNNMode_t}, Ref{cudnnRNNBiasMode_t}, Ref{cudnnDirectionMode_t}, Ref{cudnnRNNInputMode_t}, Ref{cudnnDataType_t}, Ref{cudnnDataType_t}, Ref{cudnnMathType_t}, Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{cudnnDropoutDescriptor_t}, Ref{UInt32}), rnnDesc, algo, cellMode, biasMode, dirMode, inputMode, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropoutDesc, auxFlags)
    end

@checked function cudnnSetRNNDescriptor_v6(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, cellMode, algo, mathPrec)
        initialize_context()
        ccall((:cudnnSetRNNDescriptor_v6, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Cint, cudnnDropoutDescriptor_t, cudnnRNNInputMode_t, cudnnDirectionMode_t, cudnnRNNMode_t, cudnnRNNAlgo_t, cudnnDataType_t), handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, cellMode, algo, mathPrec)
    end

@checked function cudnnGetRNNDescriptor_v6(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, cellMode, algo, mathPrec)
        initialize_context()
        ccall((:cudnnGetRNNDescriptor_v6, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Ref{Cint}, Ref{Cint}, Ref{cudnnDropoutDescriptor_t}, Ref{cudnnRNNInputMode_t}, Ref{cudnnDirectionMode_t}, Ref{cudnnRNNMode_t}, Ref{cudnnRNNAlgo_t}, Ref{cudnnDataType_t}), handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, cellMode, algo, mathPrec)
    end

@checked function cudnnSetRNNMatrixMathType(rnnDesc, mType)
        initialize_context()
        ccall((:cudnnSetRNNMatrixMathType, libcudnn), cudnnStatus_t, (cudnnRNNDescriptor_t, cudnnMathType_t), rnnDesc, mType)
    end

@checked function cudnnGetRNNMatrixMathType(rnnDesc, mType)
        initialize_context()
        ccall((:cudnnGetRNNMatrixMathType, libcudnn), cudnnStatus_t, (cudnnRNNDescriptor_t, Ptr{cudnnMathType_t}), rnnDesc, mType)
    end

@checked function cudnnSetRNNBiasMode(rnnDesc, biasMode)
        initialize_context()
        ccall((:cudnnSetRNNBiasMode, libcudnn), cudnnStatus_t, (cudnnRNNDescriptor_t, cudnnRNNBiasMode_t), rnnDesc, biasMode)
    end

@checked function cudnnGetRNNBiasMode(rnnDesc, biasMode)
        initialize_context()
        ccall((:cudnnGetRNNBiasMode, libcudnn), cudnnStatus_t, (cudnnRNNDescriptor_t, Ptr{cudnnRNNBiasMode_t}), rnnDesc, biasMode)
    end

@checked function cudnnRNNSetClip_v8(rnnDesc, clipMode, clipNanOpt, lclip, rclip)
        initialize_context()
        ccall((:cudnnRNNSetClip_v8, libcudnn), cudnnStatus_t, (cudnnRNNDescriptor_t, cudnnRNNClipMode_t, cudnnNanPropagation_t, Cdouble, Cdouble), rnnDesc, clipMode, clipNanOpt, lclip, rclip)
    end

@checked function cudnnRNNGetClip_v8(rnnDesc, clipMode, clipNanOpt, lclip, rclip)
        initialize_context()
        ccall((:cudnnRNNGetClip_v8, libcudnn), cudnnStatus_t, (cudnnRNNDescriptor_t, Ref{cudnnRNNClipMode_t}, Ref{cudnnNanPropagation_t}, Ref{Cdouble}, Ref{Cdouble}), rnnDesc, clipMode, clipNanOpt, lclip, rclip)
    end

@checked function cudnnRNNSetClip(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip)
        initialize_context()
        ccall((:cudnnRNNSetClip, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnRNNClipMode_t, cudnnNanPropagation_t, Cdouble, Cdouble), handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip)
    end

@checked function cudnnRNNGetClip(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip)
        initialize_context()
        ccall((:cudnnRNNGetClip, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{cudnnRNNClipMode_t}, Ptr{cudnnNanPropagation_t}, Ptr{Cdouble}, Ptr{Cdouble}), handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip)
    end

@checked function cudnnSetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize)
        initialize_context()
        ccall((:cudnnSetRNNProjectionLayers, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Cint), handle, rnnDesc, recProjSize, outProjSize)
    end

@checked function cudnnGetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize)
        initialize_context()
        ccall((:cudnnGetRNNProjectionLayers, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{Cint}, Ptr{Cint}), handle, rnnDesc, recProjSize, outProjSize)
    end

@checked function cudnnCreatePersistentRNNPlan(rnnDesc, minibatch, dataType, plan)
        initialize_context()
        ccall((:cudnnCreatePersistentRNNPlan, libcudnn), cudnnStatus_t, (cudnnRNNDescriptor_t, Cint, cudnnDataType_t, Ptr{cudnnPersistentRNNPlan_t}), rnnDesc, minibatch, dataType, plan)
    end

@checked function cudnnDestroyPersistentRNNPlan(plan)
        initialize_context()
        ccall((:cudnnDestroyPersistentRNNPlan, libcudnn), cudnnStatus_t, (cudnnPersistentRNNPlan_t,), plan)
    end

@checked function cudnnSetPersistentRNNPlan(rnnDesc, plan)
        initialize_context()
        ccall((:cudnnSetPersistentRNNPlan, libcudnn), cudnnStatus_t, (cudnnRNNDescriptor_t, cudnnPersistentRNNPlan_t), rnnDesc, plan)
    end

@checked function cudnnBuildRNNDynamic(handle, rnnDesc, miniBatch)
        initialize_context()
        ccall((:cudnnBuildRNNDynamic, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint), handle, rnnDesc, miniBatch)
    end

@checked function cudnnGetRNNWorkspaceSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes)
        initialize_context()
        ccall((:cudnnGetRNNWorkspaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, Ref{Csize_t}), handle, rnnDesc, seqLength, xDesc, sizeInBytes)
    end

@checked function cudnnGetRNNTrainingReserveSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes)
        initialize_context()
        ccall((:cudnnGetRNNTrainingReserveSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, Ref{Csize_t}), handle, rnnDesc, seqLength, xDesc, sizeInBytes)
    end

@checked function cudnnGetRNNTempSpaceSizes(handle, rnnDesc, fMode, xDesc, workSpaceSize, reserveSpaceSize)
        initialize_context()
        ccall((:cudnnGetRNNTempSpaceSizes, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnForwardMode_t, cudnnRNNDataDescriptor_t, Ref{Csize_t}, Ref{Csize_t}), handle, rnnDesc, fMode, xDesc, workSpaceSize, reserveSpaceSize)
    end

@checked function cudnnGetRNNParamsSize(handle, rnnDesc, xDesc, sizeInBytes, dataType)
        initialize_context()
        ccall((:cudnnGetRNNParamsSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnTensorDescriptor_t, Ref{Csize_t}, cudnnDataType_t), handle, rnnDesc, xDesc, sizeInBytes, dataType)
    end

@checked function cudnnGetRNNWeightSpaceSize(handle, rnnDesc, weightSpaceSize)
        initialize_context()
        ccall((:cudnnGetRNNWeightSpaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Ref{Csize_t}), handle, rnnDesc, weightSpaceSize)
    end

@checked function cudnnGetRNNLinLayerMatrixParams(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerMatDesc, linLayerMat)
        initialize_context()
        ccall((:cudnnGetRNNLinLayerMatrixParams, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, CuPtr{Cvoid}, Cint, cudnnFilterDescriptor_t, Ptr{Ptr{Cvoid}}), handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerMatDesc, linLayerMat)
    end

@checked function cudnnGetRNNLinLayerBiasParams(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerBiasDesc, linLayerBias)
        initialize_context()
        ccall((:cudnnGetRNNLinLayerBiasParams, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, CuPtr{Cvoid}, Cint, cudnnFilterDescriptor_t, Ptr{Ptr{Cvoid}}), handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerBiasDesc, linLayerBias)
    end

@checked function cudnnGetRNNWeightParams(handle, rnnDesc, pseudoLayer, weightSpaceSize, weightSpace, linLayerID, mDesc, mAddr, bDesc, bAddr)
        initialize_context()
        ccall((:cudnnGetRNNWeightParams, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Int32, Csize_t, CuPtr{Cvoid}, Int32, cudnnTensorDescriptor_t, Ptr{CuPtr{Cvoid}}, cudnnTensorDescriptor_t, Ptr{CuPtr{Cvoid}}), handle, rnnDesc, pseudoLayer, weightSpaceSize, weightSpace, linLayerID, mDesc, mAddr, bDesc, bAddr)
    end

@checked function cudnnRNNForwardInference(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workSpace, workSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnRNNForwardInference, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t), handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workSpace, workSpaceSizeInBytes)
    end

@checked function cudnnSetRNNPaddingMode(rnnDesc, paddingMode)
        initialize_context()
        ccall((:cudnnSetRNNPaddingMode, libcudnn), cudnnStatus_t, (cudnnRNNDescriptor_t, Cuint), rnnDesc, paddingMode)
    end

@checked function cudnnGetRNNPaddingMode(rnnDesc, paddingMode)
        initialize_context()
        ccall((:cudnnGetRNNPaddingMode, libcudnn), cudnnStatus_t, (cudnnRNNDescriptor_t, Ptr{Cuint}), rnnDesc, paddingMode)
    end

@checked function cudnnCreateRNNDataDescriptor(rnnDataDesc)
        initialize_context()
        ccall((:cudnnCreateRNNDataDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnRNNDataDescriptor_t},), rnnDataDesc)
    end

@checked function cudnnDestroyRNNDataDescriptor(rnnDataDesc)
        initialize_context()
        ccall((:cudnnDestroyRNNDataDescriptor, libcudnn), cudnnStatus_t, (cudnnRNNDataDescriptor_t,), rnnDataDesc)
    end

@checked function cudnnSetRNNDataDescriptor(rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, seqLengthArray, paddingFill)
        initialize_context()
        ccall((:cudnnSetRNNDataDescriptor, libcudnn), cudnnStatus_t, (cudnnRNNDataDescriptor_t, cudnnDataType_t, cudnnRNNDataLayout_t, Cint, Cint, Cint, Ptr{Cint}, Ptr{Cvoid}), rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, seqLengthArray, paddingFill)
    end

@checked function cudnnGetRNNDataDescriptor(rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, arrayLengthRequested, seqLengthArray, paddingFill)
        initialize_context()
        ccall((:cudnnGetRNNDataDescriptor, libcudnn), cudnnStatus_t, (cudnnRNNDataDescriptor_t, Ptr{cudnnDataType_t}, Ptr{cudnnRNNDataLayout_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cint}, Ptr{Cvoid}), rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, arrayLengthRequested, seqLengthArray, paddingFill)
    end

@checked function cudnnRNNForwardInferenceEx(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace, workSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnRNNForwardInferenceEx, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, Ptr{Cvoid}, cudnnRNNDataDescriptor_t, Ptr{Cvoid}, cudnnRNNDataDescriptor_t, Ptr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t), handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace, workSpaceSizeInBytes)
    end

@checked function cudnnRNNForward(handle, rnnDesc, fwdMode, devSeqLengths, xDesc, x, yDesc, y, hDesc, hx, hy, cDesc, cx, cy, weightSpaceSize, weightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace)
        initialize_context()
        ccall((:cudnnRNNForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnForwardMode_t, CuPtr{Int32}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}), handle, rnnDesc, fwdMode, devSeqLengths, xDesc, x, yDesc, y, hDesc, hx, hy, cDesc, cx, cy, weightSpaceSize, weightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace)
    end

@checked function cudnnSetRNNAlgorithmDescriptor(handle, rnnDesc, algoDesc)
        initialize_context()
        ccall((:cudnnSetRNNAlgorithmDescriptor, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnAlgorithmDescriptor_t), handle, rnnDesc, algoDesc)
    end

@checked function cudnnGetRNNForwardInferenceAlgorithmMaxCount(handle, rnnDesc, count)
        initialize_context()
        ccall((:cudnnGetRNNForwardInferenceAlgorithmMaxCount, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{Cint}), handle, rnnDesc, count)
    end

@checked function cudnnFindRNNForwardInferenceAlgorithmEx(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnFindRNNForwardInferenceAlgorithmEx, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Cfloat, Cint, Ptr{Cint}, Ptr{cudnnAlgorithmPerformance_t}, CuPtr{Cvoid}, Csize_t), handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes)
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
        ccall((:cudnnCreateSeqDataDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnSeqDataDescriptor_t},), seqDataDesc)
    end

@checked function cudnnDestroySeqDataDescriptor(seqDataDesc)
        initialize_context()
        ccall((:cudnnDestroySeqDataDescriptor, libcudnn), cudnnStatus_t, (cudnnSeqDataDescriptor_t,), seqDataDesc)
    end

@checked function cudnnSetSeqDataDescriptor(seqDataDesc, dataType, nbDims, dimA, axes, seqLengthArraySize, seqLengthArray, paddingFill)
        initialize_context()
        ccall((:cudnnSetSeqDataDescriptor, libcudnn), cudnnStatus_t, (cudnnSeqDataDescriptor_t, cudnnDataType_t, Cint, Ptr{Cint}, Ptr{cudnnSeqDataAxis_t}, Csize_t, Ptr{Cint}, Ptr{Cvoid}), seqDataDesc, dataType, nbDims, dimA, axes, seqLengthArraySize, seqLengthArray, paddingFill)
    end

@checked function cudnnGetSeqDataDescriptor(seqDataDesc, dataType, nbDims, nbDimsRequested, dimA, axes, seqLengthArraySize, seqLengthSizeRequested, seqLengthArray, paddingFill)
        initialize_context()
        ccall((:cudnnGetSeqDataDescriptor, libcudnn), cudnnStatus_t, (cudnnSeqDataDescriptor_t, Ptr{cudnnDataType_t}, Ptr{Cint}, Cint, Ptr{Cint}, Ptr{cudnnSeqDataAxis_t}, Ptr{Csize_t}, Csize_t, Ptr{Cint}, Ptr{Cvoid}), seqDataDesc, dataType, nbDims, nbDimsRequested, dimA, axes, seqLengthArraySize, seqLengthSizeRequested, seqLengthArray, paddingFill)
    end

const cudnnAttnQueryMap_t = Cuint

mutable struct cudnnAttnStruct end

const cudnnAttnDescriptor_t = Ptr{cudnnAttnStruct}

@checked function cudnnCreateAttnDescriptor(attnDesc)
        initialize_context()
        ccall((:cudnnCreateAttnDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnAttnDescriptor_t},), attnDesc)
    end

@checked function cudnnDestroyAttnDescriptor(attnDesc)
        initialize_context()
        ccall((:cudnnDestroyAttnDescriptor, libcudnn), cudnnStatus_t, (cudnnAttnDescriptor_t,), attnDesc)
    end

@checked function cudnnSetAttnDescriptor(attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize)
        initialize_context()
        ccall((:cudnnSetAttnDescriptor, libcudnn), cudnnStatus_t, (cudnnAttnDescriptor_t, Cuint, Cint, Cdouble, cudnnDataType_t, cudnnDataType_t, cudnnMathType_t, cudnnDropoutDescriptor_t, cudnnDropoutDescriptor_t, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint), attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize)
    end

@checked function cudnnGetAttnDescriptor(attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize)
        initialize_context()
        ccall((:cudnnGetAttnDescriptor, libcudnn), cudnnStatus_t, (cudnnAttnDescriptor_t, Ptr{Cuint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{cudnnDataType_t}, Ptr{cudnnDataType_t}, Ptr{cudnnMathType_t}, Ptr{cudnnDropoutDescriptor_t}, Ptr{cudnnDropoutDescriptor_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize)
    end

@checked function cudnnGetMultiHeadAttnBuffers(handle, attnDesc, weightSizeInBytes, workSpaceSizeInBytes, reserveSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnGetMultiHeadAttnBuffers, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnAttnDescriptor_t, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}), handle, attnDesc, weightSizeInBytes, workSpaceSizeInBytes, reserveSpaceSizeInBytes)
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

@checked function cudnnGetMultiHeadAttnWeights(handle, attnDesc, wKind, weightSizeInBytes, weights, wDesc, wAddr)
        initialize_context()
        ccall((:cudnnGetMultiHeadAttnWeights, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnAttnDescriptor_t, cudnnMultiHeadAttnWeightKind_t, Csize_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Ptr{Cvoid}}), handle, attnDesc, wKind, weightSizeInBytes, weights, wDesc, wAddr)
    end

@checked function cudnnMultiHeadAttnForward(handle, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, queries, residuals, kDesc, keys, vDesc, values, oDesc, out, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace)
        initialize_context()
        ccall((:cudnnMultiHeadAttnForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnAttnDescriptor_t, Cint, Ptr{Cint}, Ptr{Cint}, CuPtr{Cint}, CuPtr{Cint}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}), handle, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, queries, residuals, kDesc, keys, vDesc, values, oDesc, out, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace)
    end

@checked function cudnnAdvInferVersionCheck()
        initialize_context()
        ccall((:cudnnAdvInferVersionCheck, libcudnn), cudnnStatus_t, ())
    end

@cenum cudnnWgradMode_t::UInt32 begin
    CUDNN_WGRAD_MODE_ADD = 0
    CUDNN_WGRAD_MODE_SET = 1
end

@checked function cudnnRNNForwardTraining(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnRNNForwardTraining, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t), handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
    end

@checked function cudnnRNNBackwardData(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnRNNBackwardData, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t), handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
    end

@checked function cudnnRNNBackwardData_v8(handle, rnnDesc, devSeqLengths, yDesc, y, dy, xDesc, dx, hDesc, hx, dhy, dhx, cDesc, cx, dcy, dcx, weightSpaceSize, weightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace)
        initialize_context()
        ccall((:cudnnRNNBackwardData_v8, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, CuPtr{Int32}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}), handle, rnnDesc, devSeqLengths, yDesc, y, dy, xDesc, dx, hDesc, hx, dhy, dhx, cDesc, cx, dcy, dcx, weightSpaceSize, weightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace)
    end

@checked function cudnnRNNBackwardWeights(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, workSpace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnRNNBackwardWeights, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, cudnnFilterDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t), handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, workSpace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes)
    end

@checked function cudnnRNNBackwardWeights_v8(handle, rnnDesc, addGrad, devSeqLengths, xDesc, x, hDesc, hx, yDesc, y, weightSpaceSize, dweightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace)
        initialize_context()
        ccall((:cudnnRNNBackwardWeights_v8, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnWgradMode_t, CuPtr{Int32}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}), handle, rnnDesc, addGrad, devSeqLengths, xDesc, x, hDesc, hx, yDesc, y, weightSpaceSize, dweightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace)
    end

@checked function cudnnRNNForwardTrainingEx(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnRNNForwardTrainingEx, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t), handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
    end

@checked function cudnnRNNBackwardDataEx(handle, rnnDesc, yDesc, y, dyDesc, dy, dcDesc, dcAttn, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, dkDesc, dkeys, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnRNNBackwardDataEx, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, Ptr{Cvoid}, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t), handle, rnnDesc, yDesc, y, dyDesc, dy, dcDesc, dcAttn, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, dkDesc, dkeys, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
    end

@checked function cudnnRNNBackwardWeightsEx(handle, rnnDesc, xDesc, x, hxDesc, hx, yDesc, y, workSpace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnRNNBackwardWeightsEx, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, cudnnFilterDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t), handle, rnnDesc, xDesc, x, hxDesc, hx, yDesc, y, workSpace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes)
    end

@checked function cudnnGetRNNForwardTrainingAlgorithmMaxCount(handle, rnnDesc, count)
        initialize_context()
        ccall((:cudnnGetRNNForwardTrainingAlgorithmMaxCount, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{Cint}), handle, rnnDesc, count)
    end

@checked function cudnnFindRNNForwardTrainingAlgorithmEx(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnFindRNNForwardTrainingAlgorithmEx, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Cfloat, Cint, Ptr{Cint}, Ptr{cudnnAlgorithmPerformance_t}, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t), handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
    end

@checked function cudnnGetRNNBackwardDataAlgorithmMaxCount(handle, rnnDesc, count)
        initialize_context()
        ccall((:cudnnGetRNNBackwardDataAlgorithmMaxCount, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{Cint}), handle, rnnDesc, count)
    end

@checked function cudnnFindRNNBackwardDataAlgorithmEx(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnFindRNNBackwardDataAlgorithmEx, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Cfloat, Cint, Ptr{Cint}, Ptr{cudnnAlgorithmPerformance_t}, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t), handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
    end

@checked function cudnnGetRNNBackwardWeightsAlgorithmMaxCount(handle, rnnDesc, count)
        initialize_context()
        ccall((:cudnnGetRNNBackwardWeightsAlgorithmMaxCount, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{Cint}), handle, rnnDesc, count)
    end

@checked function cudnnFindRNNBackwardWeightsAlgorithmEx(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnFindRNNBackwardWeightsAlgorithmEx, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, Cfloat, Cint, Ptr{Cint}, Ptr{cudnnAlgorithmPerformance_t}, CuPtr{Cvoid}, Csize_t, cudnnFilterDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t), handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes)
    end

@checked function cudnnMultiHeadAttnBackwardData(handle, attnDesc, loWinIdx, hiWinIdx, devSeqLengthsDQDO, devSeqLengthsDKDV, doDesc, dout, dqDesc, dqueries, queries, dkDesc, dkeys, keys, dvDesc, dvalues, values, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace)
        initialize_context()
        ccall((:cudnnMultiHeadAttnBackwardData, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnAttnDescriptor_t, Ptr{Cint}, Ptr{Cint}, CuPtr{Cint}, CuPtr{Cint}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}), handle, attnDesc, loWinIdx, hiWinIdx, devSeqLengthsDQDO, devSeqLengthsDKDV, doDesc, dout, dqDesc, dqueries, queries, dkDesc, dkeys, keys, dvDesc, dvalues, values, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace)
    end

@checked function cudnnMultiHeadAttnBackwardWeights(handle, attnDesc, addGrad, qDesc, queries, kDesc, keys, vDesc, values, doDesc, dout, weightSizeInBytes, weights, dweights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace)
        initialize_context()
        ccall((:cudnnMultiHeadAttnBackwardWeights, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnAttnDescriptor_t, cudnnWgradMode_t, cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}), handle, attnDesc, addGrad, qDesc, queries, kDesc, keys, vDesc, values, doDesc, dout, weightSizeInBytes, weights, dweights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace)
    end

@cenum cudnnLossNormalizationMode_t::UInt32 begin
    CUDNN_LOSS_NORMALIZATION_NONE = 0
    CUDNN_LOSS_NORMALIZATION_SOFTMAX = 1
end

@checked function cudnnCreateCTCLossDescriptor(ctcLossDesc)
        initialize_context()
        ccall((:cudnnCreateCTCLossDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnCTCLossDescriptor_t},), ctcLossDesc)
    end

@checked function cudnnSetCTCLossDescriptor(ctcLossDesc, compType)
        initialize_context()
        ccall((:cudnnSetCTCLossDescriptor, libcudnn), cudnnStatus_t, (cudnnCTCLossDescriptor_t, cudnnDataType_t), ctcLossDesc, compType)
    end

@checked function cudnnSetCTCLossDescriptorEx(ctcLossDesc, compType, normMode, gradMode)
        initialize_context()
        ccall((:cudnnSetCTCLossDescriptorEx, libcudnn), cudnnStatus_t, (cudnnCTCLossDescriptor_t, cudnnDataType_t, cudnnLossNormalizationMode_t, cudnnNanPropagation_t), ctcLossDesc, compType, normMode, gradMode)
    end

@checked function cudnnSetCTCLossDescriptor_v8(ctcLossDesc, compType, normMode, gradMode, maxLabelLength)
        initialize_context()
        ccall((:cudnnSetCTCLossDescriptor_v8, libcudnn), cudnnStatus_t, (cudnnCTCLossDescriptor_t, cudnnDataType_t, cudnnLossNormalizationMode_t, cudnnNanPropagation_t, Cint), ctcLossDesc, compType, normMode, gradMode, maxLabelLength)
    end

@checked function cudnnGetCTCLossDescriptor(ctcLossDesc, compType)
        initialize_context()
        ccall((:cudnnGetCTCLossDescriptor, libcudnn), cudnnStatus_t, (cudnnCTCLossDescriptor_t, Ptr{cudnnDataType_t}), ctcLossDesc, compType)
    end

@checked function cudnnGetCTCLossDescriptorEx(ctcLossDesc, compType, normMode, gradMode)
        initialize_context()
        ccall((:cudnnGetCTCLossDescriptorEx, libcudnn), cudnnStatus_t, (cudnnCTCLossDescriptor_t, Ptr{cudnnDataType_t}, Ptr{cudnnLossNormalizationMode_t}, Ptr{cudnnNanPropagation_t}), ctcLossDesc, compType, normMode, gradMode)
    end

@checked function cudnnGetCTCLossDescriptor_v8(ctcLossDesc, compType, normMode, gradMode, maxLabelLength)
        initialize_context()
        ccall((:cudnnGetCTCLossDescriptor_v8, libcudnn), cudnnStatus_t, (cudnnCTCLossDescriptor_t, Ref{cudnnDataType_t}, Ref{cudnnLossNormalizationMode_t}, Ref{cudnnNanPropagation_t}, Ref{Cint}), ctcLossDesc, compType, normMode, gradMode, maxLabelLength)
    end

@checked function cudnnDestroyCTCLossDescriptor(ctcLossDesc)
        initialize_context()
        ccall((:cudnnDestroyCTCLossDescriptor, libcudnn), cudnnStatus_t, (cudnnCTCLossDescriptor_t,), ctcLossDesc)
    end

@checked function cudnnCTCLoss(handle, probsDesc, probs, hostLabels, hostLabelLengths, hostInputLengths, costs, gradientsDesc, gradients, algo, ctcLossDesc, workspace, workSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnCTCLoss, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnCTCLossAlgo_t, cudnnCTCLossDescriptor_t, CuPtr{Cvoid}, Csize_t), handle, probsDesc, probs, hostLabels, hostLabelLengths, hostInputLengths, costs, gradientsDesc, gradients, algo, ctcLossDesc, workspace, workSpaceSizeInBytes)
    end

@checked function cudnnCTCLoss_v8(handle, algo, ctcLossDesc, probsDesc, probs, labels, labelLengths, inputLengths, costs, gradientsDesc, gradients, workSpaceSizeInBytes, workspace)
        initialize_context()
        ccall((:cudnnCTCLoss_v8, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnCTCLossAlgo_t, cudnnCTCLossDescriptor_t, cudnnTensorDescriptor_t, Ptr{Cvoid}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cint}, Ptr{Cvoid}, cudnnTensorDescriptor_t, Ptr{Cvoid}, Csize_t, CuPtr{Cvoid}), handle, algo, ctcLossDesc, probsDesc, probs, labels, labelLengths, inputLengths, costs, gradientsDesc, gradients, workSpaceSizeInBytes, workspace)
    end

@checked function cudnnGetCTCLossWorkspaceSize(handle, probsDesc, gradientsDesc, labels, labelLengths, inputLengths, algo, ctcLossDesc, sizeInBytes)
        initialize_context()
        ccall((:cudnnGetCTCLossWorkspaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, cudnnCTCLossAlgo_t, cudnnCTCLossDescriptor_t, Ref{Csize_t}), handle, probsDesc, gradientsDesc, labels, labelLengths, inputLengths, algo, ctcLossDesc, sizeInBytes)
    end

@checked function cudnnGetCTCLossWorkspaceSize_v8(handle, algo, ctcLossDesc, probsDesc, gradientsDesc, sizeInBytes)
        initialize_context()
        ccall((:cudnnGetCTCLossWorkspaceSize_v8, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnCTCLossAlgo_t, cudnnCTCLossDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, Ptr{Csize_t}), handle, algo, ctcLossDesc, probsDesc, gradientsDesc, sizeInBytes)
    end

@checked function cudnnAdvTrainVersionCheck()
        initialize_context()
        ccall((:cudnnAdvTrainVersionCheck, libcudnn), cudnnStatus_t, ())
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
    reserved::NTuple{3, Cint}
end

const cudnnConvolutionFwdAlgoPerf_t = cudnnConvolutionFwdAlgoPerfStruct

@checked function cudnnCreateConvolutionDescriptor(convDesc)
        initialize_context()
        ccall((:cudnnCreateConvolutionDescriptor, libcudnn), cudnnStatus_t, (Ptr{cudnnConvolutionDescriptor_t},), convDesc)
    end

@checked function cudnnDestroyConvolutionDescriptor(convDesc)
        initialize_context()
        ccall((:cudnnDestroyConvolutionDescriptor, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t,), convDesc)
    end

@checked function cudnnSetConvolutionMathType(convDesc, mathType)
        initialize_context()
        ccall((:cudnnSetConvolutionMathType, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, cudnnMathType_t), convDesc, mathType)
    end

@checked function cudnnGetConvolutionMathType(convDesc, mathType)
        initialize_context()
        ccall((:cudnnGetConvolutionMathType, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Ptr{cudnnMathType_t}), convDesc, mathType)
    end

@checked function cudnnSetConvolutionGroupCount(convDesc, groupCount)
        initialize_context()
        ccall((:cudnnSetConvolutionGroupCount, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Cint), convDesc, groupCount)
    end

@checked function cudnnGetConvolutionGroupCount(convDesc, groupCount)
        initialize_context()
        ccall((:cudnnGetConvolutionGroupCount, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Ptr{Cint}), convDesc, groupCount)
    end

@checked function cudnnSetConvolutionReorderType(convDesc, reorderType)
        initialize_context()
        ccall((:cudnnSetConvolutionReorderType, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, cudnnReorderType_t), convDesc, reorderType)
    end

@checked function cudnnGetConvolutionReorderType(convDesc, reorderType)
        initialize_context()
        ccall((:cudnnGetConvolutionReorderType, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Ptr{cudnnReorderType_t}), convDesc, reorderType)
    end

@checked function cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType)
        initialize_context()
        ccall((:cudnnSetConvolution2dDescriptor, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Cint, Cint, Cint, Cint, Cint, Cint, cudnnConvolutionMode_t, cudnnDataType_t), convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType)
    end

@checked function cudnnGetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType)
        initialize_context()
        ccall((:cudnnGetConvolution2dDescriptor, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{cudnnConvolutionMode_t}, Ptr{cudnnDataType_t}), convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType)
    end

@checked function cudnnSetConvolutionNdDescriptor(convDesc, arrayLength, padA, filterStrideA, dilationA, mode, computeType)
        initialize_context()
        ccall((:cudnnSetConvolutionNdDescriptor, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, cudnnConvolutionMode_t, cudnnDataType_t), convDesc, arrayLength, padA, filterStrideA, dilationA, mode, computeType)
    end

@checked function cudnnGetConvolutionNdDescriptor(convDesc, arrayLengthRequested, arrayLength, padA, strideA, dilationA, mode, computeType)
        initialize_context()
        ccall((:cudnnGetConvolutionNdDescriptor, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{cudnnConvolutionMode_t}, Ptr{cudnnDataType_t}), convDesc, arrayLengthRequested, arrayLength, padA, strideA, dilationA, mode, computeType)
    end

@checked function cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensorDesc, filterDesc, n, c, h, w)
        initialize_context()
        ccall((:cudnnGetConvolution2dForwardOutputDim, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), convDesc, inputTensorDesc, filterDesc, n, c, h, w)
    end

@checked function cudnnGetConvolutionNdForwardOutputDim(convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA)
        initialize_context()
        ccall((:cudnnGetConvolutionNdForwardOutputDim, libcudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, Cint, Ptr{Cint}), convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA)
    end

@checked function cudnnGetConvolutionForwardAlgorithmMaxCount(handle, count)
        initialize_context()
        ccall((:cudnnGetConvolutionForwardAlgorithmMaxCount, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Cint}), handle, count)
    end

@checked function cudnnGetConvolutionForwardAlgorithm_v7(handle, srcDesc, filterDesc, convDesc, destDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
        initialize_context()
        ccall((:cudnnGetConvolutionForwardAlgorithm_v7, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint}, Ptr{cudnnConvolutionFwdAlgoPerf_t}), handle, srcDesc, filterDesc, convDesc, destDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
    end

@checked function cudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
        initialize_context()
        ccall((:cudnnFindConvolutionForwardAlgorithm, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint}, Ptr{cudnnConvolutionFwdAlgoPerf_t}), handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
    end

@checked function cudnnFindConvolutionForwardAlgorithmEx(handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnFindConvolutionForwardAlgorithmEx, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Cint, Ptr{Cint}, Ptr{cudnnConvolutionFwdAlgoPerf_t}, CuPtr{Cvoid}, Csize_t), handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
    end

@checked function cudnnIm2Col(handle, xDesc, x, wDesc, convDesc, colBuffer)
        initialize_context()
        ccall((:cudnnIm2Col, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnFilterDescriptor_t, cudnnConvolutionDescriptor_t, CuPtr{Cvoid}), handle, xDesc, x, wDesc, convDesc, colBuffer)
    end

@checked function cudnnReorderFilterAndBias(handle, filterDesc, reorderType, filterData, reorderedFilterData, reorderBias, biasData, reorderedBiasData)
        initialize_context()
        ccall((:cudnnReorderFilterAndBias, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnReorderType_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Cint, CuPtr{Cvoid}, CuPtr{Cvoid}), handle, filterDesc, reorderType, filterData, reorderedFilterData, reorderBias, biasData, reorderedBiasData)
    end

@checked function cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes)
        initialize_context()
        ccall((:cudnnGetConvolutionForwardWorkspaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionFwdAlgo_t, Ref{Csize_t}), handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes)
    end

@checked function cudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y)
        initialize_context()
        ccall((:cudnnConvolutionForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}), handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y)
    end

@checked function cudnnConvolutionBiasActivationForward(handle, alpha1, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, alpha2, zDesc, z, biasDesc, bias, activationDesc, yDesc, y)
        initialize_context()
        ccall((:cudnnConvolutionBiasActivationForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnActivationDescriptor_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}), handle, alpha1, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, alpha2, zDesc, z, biasDesc, bias, activationDesc, yDesc, y)
    end

struct cudnnConvolutionBwdDataAlgoPerfStruct
    algo::cudnnConvolutionBwdDataAlgo_t
    status::cudnnStatus_t
    time::Cfloat
    memory::Csize_t
    determinism::cudnnDeterminism_t
    mathType::cudnnMathType_t
    reserved::NTuple{3, Cint}
end

const cudnnConvolutionBwdDataAlgoPerf_t = cudnnConvolutionBwdDataAlgoPerfStruct

@checked function cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, count)
        initialize_context()
        ccall((:cudnnGetConvolutionBackwardDataAlgorithmMaxCount, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Cint}), handle, count)
    end

@checked function cudnnFindConvolutionBackwardDataAlgorithm(handle, wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
        initialize_context()
        ccall((:cudnnFindConvolutionBackwardDataAlgorithm, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint}, Ptr{cudnnConvolutionBwdDataAlgoPerf_t}), handle, wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
    end

@checked function cudnnFindConvolutionBackwardDataAlgorithmEx(handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnFindConvolutionBackwardDataAlgorithmEx, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Cint, Ptr{Cint}, Ptr{cudnnConvolutionBwdDataAlgoPerf_t}, CuPtr{Cvoid}, Csize_t), handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
    end

@checked function cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, filterDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
        initialize_context()
        ccall((:cudnnGetConvolutionBackwardDataAlgorithm_v7, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint}, Ptr{cudnnConvolutionBwdDataAlgoPerf_t}), handle, filterDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
    end

@checked function cudnnGetConvolutionBackwardDataWorkspaceSize(handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes)
        initialize_context()
        ccall((:cudnnGetConvolutionBackwardDataWorkspaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionBwdDataAlgo_t, Ref{Csize_t}), handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes)
    end

@checked function cudnnConvolutionBackwardData(handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx)
        initialize_context()
        ccall((:cudnnConvolutionBackwardData, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnConvolutionDescriptor_t, cudnnConvolutionBwdDataAlgo_t, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}), handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx)
    end

@checked function cudnnGetFoldedConvBackwardDataDescriptors(handle, filterDesc, diffDesc, convDesc, gradDesc, transformFormat, foldedFilterDesc, paddedDiffDesc, foldedConvDesc, foldedGradDesc, filterFoldTransDesc, diffPadTransDesc, gradFoldTransDesc, gradUnfoldTransDesc)
        initialize_context()
        ccall((:cudnnGetFoldedConvBackwardDataDescriptors, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorTransformDescriptor_t, cudnnTensorTransformDescriptor_t, cudnnTensorTransformDescriptor_t, cudnnTensorTransformDescriptor_t), handle, filterDesc, diffDesc, convDesc, gradDesc, transformFormat, foldedFilterDesc, paddedDiffDesc, foldedConvDesc, foldedGradDesc, filterFoldTransDesc, diffPadTransDesc, gradFoldTransDesc, gradUnfoldTransDesc)
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
        ccall((:cudnnCnnInferVersionCheck, libcudnn), cudnnStatus_t, ())
    end

struct cudnnConvolutionBwdFilterAlgoPerfStruct
    algo::cudnnConvolutionBwdFilterAlgo_t
    status::cudnnStatus_t
    time::Cfloat
    memory::Csize_t
    determinism::cudnnDeterminism_t
    mathType::cudnnMathType_t
    reserved::NTuple{3, Cint}
end

const cudnnConvolutionBwdFilterAlgoPerf_t = cudnnConvolutionBwdFilterAlgoPerfStruct

@checked function cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, count)
        initialize_context()
        ccall((:cudnnGetConvolutionBackwardFilterAlgorithmMaxCount, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Cint}), handle, count)
    end

@checked function cudnnFindConvolutionBackwardFilterAlgorithm(handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
        initialize_context()
        ccall((:cudnnFindConvolutionBackwardFilterAlgorithm, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, Cint, Ptr{Cint}, Ptr{cudnnConvolutionBwdFilterAlgoPerf_t}), handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
    end

@checked function cudnnFindConvolutionBackwardFilterAlgorithmEx(handle, xDesc, x, dyDesc, y, convDesc, dwDesc, dw, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
        initialize_context()
        ccall((:cudnnFindConvolutionBackwardFilterAlgorithmEx, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, CuPtr{Cvoid}, Cint, Ptr{Cint}, Ptr{cudnnConvolutionBwdFilterAlgoPerf_t}, CuPtr{Cvoid}, Csize_t), handle, xDesc, x, dyDesc, y, convDesc, dwDesc, dw, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
    end

@checked function cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, srcDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
        initialize_context()
        ccall((:cudnnGetConvolutionBackwardFilterAlgorithm_v7, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, Cint, Ptr{Cint}, Ptr{cudnnConvolutionBwdFilterAlgoPerf_t}), handle, srcDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
    end

@checked function cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes)
        initialize_context()
        ccall((:cudnnGetConvolutionBackwardFilterWorkspaceSize, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, cudnnConvolutionBwdFilterAlgo_t, Ref{Csize_t}), handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes)
    end

@checked function cudnnConvolutionBackwardFilter(handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw)
        initialize_context()
        ccall((:cudnnConvolutionBackwardFilter, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnConvolutionDescriptor_t, cudnnConvolutionBwdFilterAlgo_t, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}), handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw)
    end

@checked function cudnnConvolutionBackwardBias(handle, alpha, dyDesc, dy, beta, dbDesc, db)
        initialize_context()
        ccall((:cudnnConvolutionBackwardBias, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}), handle, alpha, dyDesc, dy, beta, dbDesc, db)
    end

@checked function cudnnCreateFusedOpsConstParamPack(constPack, ops)
        initialize_context()
        ccall((:cudnnCreateFusedOpsConstParamPack, libcudnn), cudnnStatus_t, (Ptr{cudnnFusedOpsConstParamPack_t}, cudnnFusedOps_t), constPack, ops)
    end

@checked function cudnnDestroyFusedOpsConstParamPack(constPack)
        initialize_context()
        ccall((:cudnnDestroyFusedOpsConstParamPack, libcudnn), cudnnStatus_t, (cudnnFusedOpsConstParamPack_t,), constPack)
    end

@checked function cudnnSetFusedOpsConstParamPackAttribute(constPack, paramLabel, param)
        initialize_context()
        ccall((:cudnnSetFusedOpsConstParamPackAttribute, libcudnn), cudnnStatus_t, (cudnnFusedOpsConstParamPack_t, cudnnFusedOpsConstParamLabel_t, Ptr{Cvoid}), constPack, paramLabel, param)
    end

@checked function cudnnGetFusedOpsConstParamPackAttribute(constPack, paramLabel, param, isNULL)
        initialize_context()
        ccall((:cudnnGetFusedOpsConstParamPackAttribute, libcudnn), cudnnStatus_t, (cudnnFusedOpsConstParamPack_t, cudnnFusedOpsConstParamLabel_t, Ptr{Cvoid}, Ptr{Cint}), constPack, paramLabel, param, isNULL)
    end

@checked function cudnnCreateFusedOpsVariantParamPack(varPack, ops)
        initialize_context()
        ccall((:cudnnCreateFusedOpsVariantParamPack, libcudnn), cudnnStatus_t, (Ptr{cudnnFusedOpsVariantParamPack_t}, cudnnFusedOps_t), varPack, ops)
    end

@checked function cudnnDestroyFusedOpsVariantParamPack(varPack)
        initialize_context()
        ccall((:cudnnDestroyFusedOpsVariantParamPack, libcudnn), cudnnStatus_t, (cudnnFusedOpsVariantParamPack_t,), varPack)
    end

@checked function cudnnSetFusedOpsVariantParamPackAttribute(varPack, paramLabel, ptr)
        initialize_context()
        ccall((:cudnnSetFusedOpsVariantParamPackAttribute, libcudnn), cudnnStatus_t, (cudnnFusedOpsVariantParamPack_t, cudnnFusedOpsVariantParamLabel_t, PtrOrCuPtr{Cvoid}), varPack, paramLabel, ptr)
    end

@checked function cudnnGetFusedOpsVariantParamPackAttribute(varPack, paramLabel, ptr)
        initialize_context()
        ccall((:cudnnGetFusedOpsVariantParamPackAttribute, libcudnn), cudnnStatus_t, (cudnnFusedOpsVariantParamPack_t, cudnnFusedOpsVariantParamLabel_t, PtrOrCuPtr{Cvoid}), varPack, paramLabel, ptr)
    end

@checked function cudnnCreateFusedOpsPlan(plan, ops)
        initialize_context()
        ccall((:cudnnCreateFusedOpsPlan, libcudnn), cudnnStatus_t, (Ptr{cudnnFusedOpsPlan_t}, cudnnFusedOps_t), plan, ops)
    end

@checked function cudnnDestroyFusedOpsPlan(plan)
        initialize_context()
        ccall((:cudnnDestroyFusedOpsPlan, libcudnn), cudnnStatus_t, (cudnnFusedOpsPlan_t,), plan)
    end

@checked function cudnnMakeFusedOpsPlan(handle, plan, constPack, workspaceSizeInBytes)
        initialize_context()
        ccall((:cudnnMakeFusedOpsPlan, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnFusedOpsPlan_t, cudnnFusedOpsConstParamPack_t, Ptr{Csize_t}), handle, plan, constPack, workspaceSizeInBytes)
    end

@checked function cudnnFusedOpsExecute(handle, plan, varPack)
        initialize_context()
        ccall((:cudnnFusedOpsExecute, libcudnn), cudnnStatus_t, (cudnnHandle_t, cudnnFusedOpsPlan_t, cudnnFusedOpsVariantParamPack_t), handle, plan, varPack)
    end

@checked function cudnnCnnTrainVersionCheck()
        initialize_context()
        ccall((:cudnnCnnTrainVersionCheck, libcudnn), cudnnStatus_t, ())
    end

const CUDNN_MAJOR = 8

const CUDNN_MINOR = 6

const CUDNN_PATCHLEVEL = 0

const CUDNN_VERSION = CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL

const CUDNN_MAX_SM_MAJOR_NUMBER = 9

const CUDNN_MAX_SM_MINOR_NUMBER = 0

const CUDNN_MAX_DEVICE_VERSION = CUDNN_MAX_SM_MAJOR_NUMBER * 100 + CUDNN_MAX_SM_MINOR_NUMBER * 10

const CUDNN_OPS_INFER_MAJOR = 8

const CUDNN_OPS_INFER_MINOR = 6

const CUDNN_OPS_INFER_PATCH = 0

const CUDNN_DIM_MAX = 8

const CUDNN_LRN_MIN_N = 1

const CUDNN_LRN_MAX_N = 16

const CUDNN_LRN_MIN_K = 1.0e-5

const CUDNN_LRN_MIN_BETA = 0.01

const CUDNN_BN_MIN_EPSILON = 0.0

const CUDNN_OPS_TRAIN_MAJOR = 8

const CUDNN_OPS_TRAIN_MINOR = 6

const CUDNN_OPS_TRAIN_PATCH = 0

const CUDNN_ADV_INFER_MAJOR = 8

const CUDNN_ADV_INFER_MINOR = 6

const CUDNN_ADV_INFER_PATCH = 0

const CUDNN_RNN_PADDED_IO_DISABLED = 0

const CUDNN_RNN_PADDED_IO_ENABLED = Cuint(1) << 0

const CUDNN_SEQDATA_DIM_COUNT = 4

const CUDNN_ATTN_QUERYMAP_ALL_TO_ONE = 0

const CUDNN_ATTN_QUERYMAP_ONE_TO_ONE = Cuint(1) << 0

const CUDNN_ATTN_DISABLE_PROJ_BIASES = 0

const CUDNN_ATTN_ENABLE_PROJ_BIASES = Cuint(1) << 1

const CUDNN_ATTN_WKIND_COUNT = 8

const CUDNN_ADV_TRAIN_MAJOR = 8

const CUDNN_ADV_TRAIN_MINOR = 6

const CUDNN_ADV_TRAIN_PATCH = 0

const CUDNN_CNN_INFER_MAJOR = 8

const CUDNN_CNN_INFER_MINOR = 6

const CUDNN_CNN_INFER_PATCH = 0

const CUDNN_CNN_TRAIN_MAJOR = 8

const CUDNN_CNN_TRAIN_MINOR = 6

const CUDNN_CNN_TRAIN_PATCH = 0

