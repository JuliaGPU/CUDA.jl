# Julia wrapper for header: cudnn.h
# Automatically generated using Clang.jl


function cudnnGetVersion()
    ccall((:cudnnGetVersion, @libcudnn), Csize_t, ())
end

function cudnnGetCudartVersion()
    ccall((:cudnnGetCudartVersion, @libcudnn), Csize_t, ())
end

function cudnnGetErrorString(status)
    ccall((:cudnnGetErrorString, @libcudnn), Cstring,
        (cudnnStatus_t,),
        status)
end

function cudnnQueryRuntimeError(handle, rstatus, mode, tag)
    @check ccall((:cudnnQueryRuntimeError, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, Ptr{cudnnStatus_t}, cudnnErrQueryMode_t, Ptr{cudnnRuntimeTag_t}),
        handle, rstatus, mode, tag)
end

function cudnnGetProperty(type, value)
    @check ccall((:cudnnGetProperty, @libcudnn), cudnnStatus_t,
        (libraryPropertyType, Ptr{Cint}),
        type, value)
end

function cudnnCreate(handle)
    @check ccall((:cudnnCreate, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnHandle_t},),
        handle)
end

function cudnnDestroy(handle)
    @check ccall((:cudnnDestroy, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t,),
        handle)
end

function cudnnSetStream(handle, streamId)
    @check ccall((:cudnnSetStream, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, CuStream_t),
        handle, streamId)
end

function cudnnGetStream(handle, streamId)
    @check ccall((:cudnnGetStream, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, Ptr{CuStream_t}),
        handle, streamId)
end

function cudnnCreateTensorDescriptor(tensorDesc)
    @check ccall((:cudnnCreateTensorDescriptor, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnTensorDescriptor_t},),
        tensorDesc)
end

function cudnnSetTensor4dDescriptor(tensorDesc, format, dataType, n, c, h, w)
    @check ccall((:cudnnSetTensor4dDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, Cint, Cint, Cint,
         Cint),
        tensorDesc, format, dataType, n, c, h, w)
end

function cudnnSetTensor4dDescriptorEx(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride)
    @check ccall((:cudnnSetTensor4dDescriptorEx, @libcudnn), cudnnStatus_t,
        (cudnnTensorDescriptor_t, cudnnDataType_t, Cint, Cint, Cint, Cint, Cint, Cint,
         Cint, Cint),
        tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride)
end

function cudnnGetTensor4dDescriptor(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride)
    @check ccall((:cudnnGetTensor4dDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnTensorDescriptor_t, Ptr{cudnnDataType_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
         Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
        tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride)
end

function cudnnSetTensorNdDescriptor(tensorDesc, dataType, nbDims, dimA, strideA)
    @check ccall((:cudnnSetTensorNdDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnTensorDescriptor_t, cudnnDataType_t, Cint, Ptr{Cint}, Ptr{Cint}),
        tensorDesc, dataType, nbDims, dimA, strideA)
end

function cudnnSetTensorNdDescriptorEx(tensorDesc, format, dataType, nbDims, dimA)
    @check ccall((:cudnnSetTensorNdDescriptorEx, @libcudnn), cudnnStatus_t,
        (cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, Cint, Ptr{Cint}),
        tensorDesc, format, dataType, nbDims, dimA)
end

function cudnnGetTensorNdDescriptor(tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA)
    @check ccall((:cudnnGetTensorNdDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnTensorDescriptor_t, Cint, Ptr{cudnnDataType_t}, Ptr{Cint}, Ptr{Cint},
         Ptr{Cint}),
        tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA)
end

function cudnnGetTensorSizeInBytes(tensorDesc, size)
    @check ccall((:cudnnGetTensorSizeInBytes, @libcudnn), cudnnStatus_t,
        (cudnnTensorDescriptor_t, Ptr{Csize_t}),
        tensorDesc, size)
end

function cudnnDestroyTensorDescriptor(tensorDesc)
    @check ccall((:cudnnDestroyTensorDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnTensorDescriptor_t,),
        tensorDesc)
end

function cudnnInitTransformDest(transformDesc, srcDesc, destDesc, destSizeInBytes)
    @check ccall((:cudnnInitTransformDest, @libcudnn), cudnnStatus_t,
        (cudnnTensorTransformDescriptor_t, cudnnTensorDescriptor_t,
         cudnnTensorDescriptor_t, Ptr{Csize_t}),
        transformDesc, srcDesc, destDesc, destSizeInBytes)
end

function cudnnCreateTensorTransformDescriptor(transformDesc)
    @check ccall((:cudnnCreateTensorTransformDescriptor, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnTensorTransformDescriptor_t},),
        transformDesc)
end

function cudnnSetTensorTransformDescriptor(transformDesc, nbDims, destFormat, padBeforeA, padAfterA, foldA, direction)
    @check ccall((:cudnnSetTensorTransformDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnTensorTransformDescriptor_t, UInt32, cudnnTensorFormat_t, Ptr{Int32},
         Ptr{Int32}, Ptr{UInt32}, cudnnFoldingDirection_t),
        transformDesc, nbDims, destFormat, padBeforeA, padAfterA, foldA, direction)
end

function cudnnGetTensorTransformDescriptor(transformDesc, nbDimsRequested, destFormat, padBeforeA, padAfterA, foldA, direction)
    @check ccall((:cudnnGetTensorTransformDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnTensorTransformDescriptor_t, UInt32, Ptr{cudnnTensorFormat_t}, Ptr{Int32},
         Ptr{Int32}, Ptr{UInt32}, Ptr{cudnnFoldingDirection_t}),
        transformDesc, nbDimsRequested, destFormat, padBeforeA, padAfterA, foldA,
        direction)
end

function cudnnDestroyTensorTransformDescriptor(transformDesc)
    @check ccall((:cudnnDestroyTensorTransformDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnTensorTransformDescriptor_t,),
        transformDesc)
end

function cudnnTransformTensor(handle, alpha, xDesc, x, beta, yDesc, y)
    @check ccall((:cudnnTransformTensor, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, Ptr{Cvoid}, Ptr{Cvoid},
         cudnnTensorDescriptor_t, Ptr{Cvoid}),
        handle, alpha, xDesc, x, beta, yDesc, y)
end

function cudnnTransformTensorEx(handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData)
    @check ccall((:cudnnTransformTensorEx, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnTensorTransformDescriptor_t, Ptr{Cvoid},
         cudnnTensorDescriptor_t, Ptr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t,
         Ptr{Cvoid}),
        handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData)
end

function cudnnGetFoldedConvBackwardDataDescriptors(handle, filterDesc, diffDesc, convDesc, gradDesc, transformFormat, foldedFilterDesc, paddedDiffDesc, foldedConvDesc, foldedGradDesc, filterFoldTransDesc, diffPadTransDesc, gradFoldTransDesc, gradUnfoldTransDesc)
    @check ccall((:cudnnGetFoldedConvBackwardDataDescriptors, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t,
         cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorFormat_t,
         cudnnFilterDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t,
         cudnnTensorDescriptor_t, cudnnTensorTransformDescriptor_t,
         cudnnTensorTransformDescriptor_t, cudnnTensorTransformDescriptor_t,
         cudnnTensorTransformDescriptor_t),
        handle, filterDesc, diffDesc, convDesc, gradDesc, transformFormat,
        foldedFilterDesc, paddedDiffDesc, foldedConvDesc, foldedGradDesc,
        filterFoldTransDesc, diffPadTransDesc, gradFoldTransDesc, gradUnfoldTransDesc)
end

function cudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C)
    @check ccall((:cudnnAddTensor, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}),
        handle, alpha, aDesc, A, beta, cDesc, C)
end

function cudnnCreateOpTensorDescriptor(opTensorDesc)
    @check ccall((:cudnnCreateOpTensorDescriptor, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnOpTensorDescriptor_t},),
        opTensorDesc)
end

function cudnnSetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)
    @check ccall((:cudnnSetOpTensorDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnOpTensorDescriptor_t, cudnnOpTensorOp_t, cudnnDataType_t,
         cudnnNanPropagation_t),
        opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)
end

function cudnnGetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)
    @check ccall((:cudnnGetOpTensorDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnOpTensorDescriptor_t, Ptr{cudnnOpTensorOp_t}, Ptr{cudnnDataType_t},
         Ptr{cudnnNanPropagation_t}),
        opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)
end

function cudnnDestroyOpTensorDescriptor(opTensorDesc)
    @check ccall((:cudnnDestroyOpTensorDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnOpTensorDescriptor_t,),
        opTensorDesc)
end

function cudnnOpTensor(handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C)
    @check ccall((:cudnnOpTensor, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnOpTensorDescriptor_t, Ptr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}),
        handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C)
end

function cudnnCreateReduceTensorDescriptor(reduceTensorDesc)
    @check ccall((:cudnnCreateReduceTensorDescriptor, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnReduceTensorDescriptor_t},),
        reduceTensorDesc)
end

function cudnnSetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType)
    @check ccall((:cudnnSetReduceTensorDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnReduceTensorDescriptor_t, cudnnReduceTensorOp_t, cudnnDataType_t,
         cudnnNanPropagation_t, cudnnReduceTensorIndices_t, cudnnIndicesType_t),
        reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt,
        reduceTensorIndices, reduceTensorIndicesType)
end

function cudnnGetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType)
    @check ccall((:cudnnGetReduceTensorDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnReduceTensorDescriptor_t, Ptr{cudnnReduceTensorOp_t}, Ptr{cudnnDataType_t},
         Ptr{cudnnNanPropagation_t}, Ptr{cudnnReduceTensorIndices_t},
         Ptr{cudnnIndicesType_t}),
        reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt,
        reduceTensorIndices, reduceTensorIndicesType)
end

function cudnnDestroyReduceTensorDescriptor(reduceTensorDesc)
    @check ccall((:cudnnDestroyReduceTensorDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnReduceTensorDescriptor_t,),
        reduceTensorDesc)
end

function cudnnGetReductionIndicesSize(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes)
    @check ccall((:cudnnGetReductionIndicesSize, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnReduceTensorDescriptor_t, cudnnTensorDescriptor_t,
         cudnnTensorDescriptor_t, Ptr{Csize_t}),
        handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes)
end

function cudnnGetReductionWorkspaceSize(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes)
    @check ccall((:cudnnGetReductionWorkspaceSize, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnReduceTensorDescriptor_t, cudnnTensorDescriptor_t,
         cudnnTensorDescriptor_t, Ptr{Csize_t}),
        handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes)
end

function cudnnReduceTensor(handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace, workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C)
    @check ccall((:cudnnReduceTensor, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnReduceTensorDescriptor_t, Ptr{Cvoid}, Csize_t, CuPtr{Cvoid},
         Csize_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}),
        handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace,
        workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C)
end

function cudnnSetTensor(handle, yDesc, y, valuePtr)
    @check ccall((:cudnnSetTensor, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}),
        handle, yDesc, y, valuePtr)
end

function cudnnScaleTensor(handle, yDesc, y, alpha)
    @check ccall((:cudnnScaleTensor, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}),
        handle, yDesc, y, alpha)
end

function cudnnCreateFilterDescriptor(filterDesc)
    @check ccall((:cudnnCreateFilterDescriptor, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnFilterDescriptor_t},),
        filterDesc)
end

function cudnnSetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w)
    @check ccall((:cudnnSetFilter4dDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnFilterDescriptor_t, cudnnDataType_t, cudnnTensorFormat_t, Cint, Cint, Cint,
         Cint),
        filterDesc, dataType, format, k, c, h, w)
end

function cudnnGetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w)
    @check ccall((:cudnnGetFilter4dDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnFilterDescriptor_t, Ptr{cudnnDataType_t}, Ptr{cudnnTensorFormat_t},
         Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
        filterDesc, dataType, format, k, c, h, w)
end

function cudnnSetFilterNdDescriptor(filterDesc, dataType, format, nbDims, filterDimA)
    @check ccall((:cudnnSetFilterNdDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnFilterDescriptor_t, cudnnDataType_t, cudnnTensorFormat_t, Cint, Ptr{Cint}),
        filterDesc, dataType, format, nbDims, filterDimA)
end

function cudnnGetFilterNdDescriptor(filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA)
    @check ccall((:cudnnGetFilterNdDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnFilterDescriptor_t, Cint, Ptr{cudnnDataType_t}, Ptr{cudnnTensorFormat_t},
         Ptr{Cint}, Ptr{Cint}),
        filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA)
end

function cudnnGetFilterSizeInBytes(filterDesc, size)
    @check ccall((:cudnnGetFilterSizeInBytes, @libcudnn), cudnnStatus_t,
        (cudnnFilterDescriptor_t, Ptr{Csize_t}),
        filterDesc, size)
end

function cudnnTransformFilter(handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData)
    @check ccall((:cudnnTransformFilter, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnTensorTransformDescriptor_t, Ptr{Cvoid},
         cudnnFilterDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnFilterDescriptor_t,
         CuPtr{Cvoid}),
        handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData)
end

function cudnnDestroyFilterDescriptor(filterDesc)
    @check ccall((:cudnnDestroyFilterDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnFilterDescriptor_t,),
        filterDesc)
end

function cudnnReorderFilterAndBias(handle, filterDesc, reorderType, filterData, reorderedFilterData, reorderBias, biasData, reorderedBiasData)
    @check ccall((:cudnnReorderFilterAndBias, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnReorderType_t, CuPtr{Cvoid},
         CuPtr{Cvoid}, Cint, CuPtr{Cvoid}, CuPtr{Cvoid}),
        handle, filterDesc, reorderType, filterData, reorderedFilterData, reorderBias,
        biasData, reorderedBiasData)
end

function cudnnCreateConvolutionDescriptor(convDesc)
    @check ccall((:cudnnCreateConvolutionDescriptor, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnConvolutionDescriptor_t},),
        convDesc)
end

function cudnnSetConvolutionMathType(convDesc, mathType)
    @check ccall((:cudnnSetConvolutionMathType, @libcudnn), cudnnStatus_t,
        (cudnnConvolutionDescriptor_t, cudnnMathType_t),
        convDesc, mathType)
end

function cudnnGetConvolutionMathType(convDesc, mathType)
    @check ccall((:cudnnGetConvolutionMathType, @libcudnn), cudnnStatus_t,
        (cudnnConvolutionDescriptor_t, Ptr{cudnnMathType_t}),
        convDesc, mathType)
end

function cudnnSetConvolutionGroupCount(convDesc, groupCount)
    @check ccall((:cudnnSetConvolutionGroupCount, @libcudnn), cudnnStatus_t,
        (cudnnConvolutionDescriptor_t, Cint),
        convDesc, groupCount)
end

function cudnnGetConvolutionGroupCount(convDesc, groupCount)
    @check ccall((:cudnnGetConvolutionGroupCount, @libcudnn), cudnnStatus_t,
        (cudnnConvolutionDescriptor_t, Ptr{Cint}),
        convDesc, groupCount)
end

function cudnnSetConvolutionReorderType(convDesc, reorderType)
    @check ccall((:cudnnSetConvolutionReorderType, @libcudnn), cudnnStatus_t,
        (cudnnConvolutionDescriptor_t, cudnnReorderType_t),
        convDesc, reorderType)
end

function cudnnGetConvolutionReorderType(convDesc, reorderType)
    @check ccall((:cudnnGetConvolutionReorderType, @libcudnn), cudnnStatus_t,
        (cudnnConvolutionDescriptor_t, Ptr{cudnnReorderType_t}),
        convDesc, reorderType)
end

function cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType)
    @check ccall((:cudnnSetConvolution2dDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnConvolutionDescriptor_t, Cint, Cint, Cint, Cint, Cint, Cint,
         cudnnConvolutionMode_t, cudnnDataType_t),
        convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType)
end

function cudnnGetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType)
    @check ccall((:cudnnGetConvolution2dDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnConvolutionDescriptor_t, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
         Ptr{Cint}, Ptr{Cint}, Ptr{cudnnConvolutionMode_t}, Ptr{cudnnDataType_t}),
        convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType)
end

function cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensorDesc, filterDesc, n, c, h, w)
    @check ccall((:cudnnGetConvolution2dForwardOutputDim, @libcudnn), cudnnStatus_t,
        (cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t,
         Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
        convDesc, inputTensorDesc, filterDesc, n, c, h, w)
end

function cudnnSetConvolutionNdDescriptor(convDesc, arrayLength, padA, filterStrideA, dilationA, mode, computeType)
    @check ccall((:cudnnSetConvolutionNdDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnConvolutionDescriptor_t, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
         cudnnConvolutionMode_t, cudnnDataType_t),
        convDesc, arrayLength, padA, filterStrideA, dilationA, mode, computeType)
end

function cudnnGetConvolutionNdDescriptor(convDesc, arrayLengthRequested, arrayLength, padA, strideA, dilationA, mode, computeType)
    @check ccall((:cudnnGetConvolutionNdDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnConvolutionDescriptor_t, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
         Ptr{cudnnConvolutionMode_t}, Ptr{cudnnDataType_t}),
        convDesc, arrayLengthRequested, arrayLength, padA, strideA, dilationA, mode,
        computeType)
end

function cudnnGetConvolutionNdForwardOutputDim(convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA)
    @check ccall((:cudnnGetConvolutionNdForwardOutputDim, @libcudnn), cudnnStatus_t,
        (cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t,
         Cint, Ptr{Cint}),
        convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA)
end

function cudnnDestroyConvolutionDescriptor(convDesc)
    @check ccall((:cudnnDestroyConvolutionDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnConvolutionDescriptor_t,),
        convDesc)
end

function cudnnGetConvolutionForwardAlgorithmMaxCount(handle, count)
    @check ccall((:cudnnGetConvolutionForwardAlgorithmMaxCount, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, Ptr{Cint}),
        handle, count)
end

function cudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
    @check ccall((:cudnnFindConvolutionForwardAlgorithm, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t,
         cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint},
         Ptr{cudnnConvolutionFwdAlgoPerf_t}),
        handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, returnedAlgoCount,
        perfResults)
end

function cudnnFindConvolutionForwardAlgorithmEx(handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
    @check ccall((:cudnnFindConvolutionForwardAlgorithmEx, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnFilterDescriptor_t,
         CuPtr{Cvoid}, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         Cint, Ptr{Cint}, Ptr{cudnnConvolutionFwdAlgoPerf_t}, CuPtr{Cvoid}, Csize_t),
        handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestedAlgoCount,
        returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
end

function cudnnGetConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes, algo)
    @check ccall((:cudnnGetConvolutionForwardAlgorithm, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t,
         cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t,
         cudnnConvolutionFwdPreference_t, Csize_t, Ptr{cudnnConvolutionFwdAlgo_t}),
        handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes, algo)
end

function cudnnGetConvolutionForwardAlgorithm_v7(handle, srcDesc, filterDesc, convDesc, destDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
    @check ccall((:cudnnGetConvolutionForwardAlgorithm_v7, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t,
         cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint},
         Ptr{cudnnConvolutionFwdAlgoPerf_t}),
        handle, srcDesc, filterDesc, convDesc, destDesc, requestedAlgoCount,
        returnedAlgoCount, perfResults)
end

function cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes)
    @check ccall((:cudnnGetConvolutionForwardWorkspaceSize, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t,
         cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionFwdAlgo_t,
         Ptr{Csize_t}),
        handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes)
end

function cudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y)
    @check ccall((:cudnnConvolutionForward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnConvolutionDescriptor_t,
         cudnnConvolutionFwdAlgo_t, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}),
        handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes,
        beta, yDesc, y)
end

function cudnnConvolutionBiasActivationForward(handle, alpha1, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, alpha2, zDesc, z, biasDesc, bias, activationDesc, yDesc, y)
    @check ccall((:cudnnConvolutionBiasActivationForward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnConvolutionDescriptor_t,
         cudnnConvolutionFwdAlgo_t, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnActivationDescriptor_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}),
        handle, alpha1, xDesc, x, wDesc, w, convDesc, algo, workSpace,
        workSpaceSizeInBytes, alpha2, zDesc, z, biasDesc, bias, activationDesc, yDesc, y)
end

function cudnnConvolutionBackwardBias(handle, alpha, dyDesc, dy, beta, dbDesc, db)
    @check ccall((:cudnnConvolutionBackwardBias, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}),
        handle, alpha, dyDesc, dy, beta, dbDesc, db)
end

function cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, count)
    @check ccall((:cudnnGetConvolutionBackwardFilterAlgorithmMaxCount, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, Ptr{Cint}),
        handle, count)
end

function cudnnFindConvolutionBackwardFilterAlgorithm(handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
    @check ccall((:cudnnFindConvolutionBackwardFilterAlgorithm, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
         cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, Cint, Ptr{Cint},
         Ptr{cudnnConvolutionBwdFilterAlgoPerf_t}),
        handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount, returnedAlgoCount,
        perfResults)
end

function cudnnFindConvolutionBackwardFilterAlgorithmEx(handle, xDesc, x, dyDesc, y, convDesc, dwDesc, dw, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
    @check ccall((:cudnnFindConvolutionBackwardFilterAlgorithmEx, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, CuPtr{Cvoid},
         Cint, Ptr{Cint}, Ptr{cudnnConvolutionBwdFilterAlgoPerf_t}, CuPtr{Cvoid}, Csize_t),
        handle, xDesc, x, dyDesc, y, convDesc, dwDesc, dw, requestedAlgoCount,
        returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
end

function cudnnGetConvolutionBackwardFilterAlgorithm(handle, xDesc, dyDesc, convDesc, dwDesc, preference, memoryLimitInBytes, algo)
    @check ccall((:cudnnGetConvolutionBackwardFilterAlgorithm, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
         cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t,
         cudnnConvolutionBwdFilterPreference_t, Csize_t,
         Ptr{cudnnConvolutionBwdFilterAlgo_t}),
        handle, xDesc, dyDesc, convDesc, dwDesc, preference, memoryLimitInBytes, algo)
end

function cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, srcDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
    @check ccall((:cudnnGetConvolutionBackwardFilterAlgorithm_v7, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
         cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, Cint, Ptr{Cint},
         Ptr{cudnnConvolutionBwdFilterAlgoPerf_t}),
        handle, srcDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount,
        returnedAlgoCount, perfResults)
end

function cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes)
    @check ccall((:cudnnGetConvolutionBackwardFilterWorkspaceSize, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
         cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t,
         cudnnConvolutionBwdFilterAlgo_t, Ptr{Csize_t}),
        handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes)
end

function cudnnConvolutionBackwardFilter(handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw)
    @check ccall((:cudnnConvolutionBackwardFilter, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnConvolutionDescriptor_t,
         cudnnConvolutionBwdFilterAlgo_t, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid},
         cudnnFilterDescriptor_t, CuPtr{Cvoid}),
        handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace,
        workSpaceSizeInBytes, beta, dwDesc, dw)
end

function cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, count)
    @check ccall((:cudnnGetConvolutionBackwardDataAlgorithmMaxCount, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, Ptr{Cint}),
        handle, count)
end

function cudnnFindConvolutionBackwardDataAlgorithm(handle, wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
    @check ccall((:cudnnFindConvolutionBackwardDataAlgorithm, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t,
         cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint},
         Ptr{cudnnConvolutionBwdDataAlgoPerf_t}),
        handle, wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount, returnedAlgoCount,
        perfResults)
end

function cudnnFindConvolutionBackwardDataAlgorithmEx(handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
    @check ccall((:cudnnFindConvolutionBackwardDataAlgorithmEx, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         Cint, Ptr{Cint}, Ptr{cudnnConvolutionBwdDataAlgoPerf_t}, CuPtr{Cvoid}, Csize_t),
        handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx, requestedAlgoCount,
        returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
end

function cudnnGetConvolutionBackwardDataAlgorithm(handle, wDesc, dyDesc, convDesc, dxDesc, preference, memoryLimitInBytes, algo)
    @check ccall((:cudnnGetConvolutionBackwardDataAlgorithm, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t,
         cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t,
         cudnnConvolutionBwdDataPreference_t, Csize_t, Ptr{cudnnConvolutionBwdDataAlgo_t}),
        handle, wDesc, dyDesc, convDesc, dxDesc, preference, memoryLimitInBytes, algo)
end

function cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, filterDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
    @check ccall((:cudnnGetConvolutionBackwardDataAlgorithm_v7, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t,
         cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint},
         Ptr{cudnnConvolutionBwdDataAlgoPerf_t}),
        handle, filterDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount,
        returnedAlgoCount, perfResults)
end

function cudnnGetConvolutionBackwardDataWorkspaceSize(handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes)
    @check ccall((:cudnnGetConvolutionBackwardDataWorkspaceSize, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t,
         cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t,
         cudnnConvolutionBwdDataAlgo_t, Ptr{Csize_t}),
        handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes)
end

function cudnnConvolutionBackwardData(handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx)
    @check ccall((:cudnnConvolutionBackwardData, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, Ptr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnConvolutionDescriptor_t,
         cudnnConvolutionBwdDataAlgo_t, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}),
        handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace,
        workSpaceSizeInBytes, beta, dxDesc, dx)
end

function cudnnIm2Col(handle, xDesc, x, wDesc, convDesc, colBuffer)
    @check ccall((:cudnnIm2Col, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnFilterDescriptor_t,
         cudnnConvolutionDescriptor_t, CuPtr{Cvoid}),
        handle, xDesc, x, wDesc, convDesc, colBuffer)
end

function cudnnSoftmaxForward(handle, algo, mode, alpha, xDesc, x, beta, yDesc, y)
    @check ccall((:cudnnSoftmaxForward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, Ptr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}),
        handle, algo, mode, alpha, xDesc, x, beta, yDesc, y)
end

function cudnnSoftmaxBackward(handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx)
    @check ccall((:cudnnSoftmaxBackward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, Ptr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}),
        handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx)
end

function cudnnCreatePoolingDescriptor(poolingDesc)
    @check ccall((:cudnnCreatePoolingDescriptor, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnPoolingDescriptor_t},),
        poolingDesc)
end

function cudnnSetPooling2dDescriptor(poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)
    @check ccall((:cudnnSetPooling2dDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnNanPropagation_t, Cint, Cint,
         Cint, Cint, Cint, Cint),
        poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding,
        horizontalPadding, verticalStride, horizontalStride)
end

function cudnnGetPooling2dDescriptor(poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)
    @check ccall((:cudnnGetPooling2dDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnPoolingDescriptor_t, Ptr{cudnnPoolingMode_t}, Ptr{cudnnNanPropagation_t},
         Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
        poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding,
        horizontalPadding, verticalStride, horizontalStride)
end

function cudnnSetPoolingNdDescriptor(poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA)
    @check ccall((:cudnnSetPoolingNdDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnNanPropagation_t, Cint,
         Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
        poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA)
end

function cudnnGetPoolingNdDescriptor(poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA)
    @check ccall((:cudnnGetPoolingNdDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnPoolingDescriptor_t, Cint, Ptr{cudnnPoolingMode_t},
         Ptr{cudnnNanPropagation_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
        poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA,
        strideA)
end

function cudnnGetPoolingNdForwardOutputDim(poolingDesc, inputTensorDesc, nbDims, outputTensorDimA)
    @check ccall((:cudnnGetPoolingNdForwardOutputDim, @libcudnn), cudnnStatus_t,
        (cudnnPoolingDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint}),
        poolingDesc, inputTensorDesc, nbDims, outputTensorDimA)
end

function cudnnGetPooling2dForwardOutputDim(poolingDesc, inputTensorDesc, n, c, h, w)
    @check ccall((:cudnnGetPooling2dForwardOutputDim, @libcudnn), cudnnStatus_t,
        (cudnnPoolingDescriptor_t, cudnnTensorDescriptor_t, Ptr{Cint}, Ptr{Cint},
         Ptr{Cint}, Ptr{Cint}),
        poolingDesc, inputTensorDesc, n, c, h, w)
end

function cudnnDestroyPoolingDescriptor(poolingDesc)
    @check ccall((:cudnnDestroyPoolingDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnPoolingDescriptor_t,),
        poolingDesc)
end

function cudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y)
    @check ccall((:cudnnPoolingForward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnPoolingDescriptor_t, Ptr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}),
        handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y)
end

function cudnnPoolingBackward(handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
    @check ccall((:cudnnPoolingBackward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnPoolingDescriptor_t, Ptr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}),
        handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
end

function cudnnCreateActivationDescriptor(activationDesc)
    @check ccall((:cudnnCreateActivationDescriptor, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnActivationDescriptor_t},),
        activationDesc)
end

function cudnnSetActivationDescriptor(activationDesc, mode, reluNanOpt, coef)
    @check ccall((:cudnnSetActivationDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnActivationDescriptor_t, cudnnActivationMode_t, cudnnNanPropagation_t, Cdouble),
        activationDesc, mode, reluNanOpt, coef)
end

function cudnnGetActivationDescriptor(activationDesc, mode, reluNanOpt, coef)
    @check ccall((:cudnnGetActivationDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnActivationDescriptor_t, Ptr{cudnnActivationMode_t},
         Ptr{cudnnNanPropagation_t}, Ptr{Cdouble}),
        activationDesc, mode, reluNanOpt, coef)
end

function cudnnDestroyActivationDescriptor(activationDesc)
    @check ccall((:cudnnDestroyActivationDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnActivationDescriptor_t,),
        activationDesc)
end

function cudnnActivationForward(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y)
    @check ccall((:cudnnActivationForward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnActivationDescriptor_t, Ptr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}),
        handle, activationDesc, alpha, xDesc, x, beta, yDesc, y)
end

function cudnnActivationBackward(handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
    @check ccall((:cudnnActivationBackward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnActivationDescriptor_t, Ptr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}),
        handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
end

function cudnnCreateLRNDescriptor(normDesc)
    @check ccall((:cudnnCreateLRNDescriptor, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnLRNDescriptor_t},),
        normDesc)
end

function cudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
    @check ccall((:cudnnSetLRNDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnLRNDescriptor_t, UInt32, Cdouble, Cdouble, Cdouble),
        normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
end

function cudnnGetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
    @check ccall((:cudnnGetLRNDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnLRNDescriptor_t, Ptr{UInt32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
        normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
end

function cudnnDestroyLRNDescriptor(lrnDesc)
    @check ccall((:cudnnDestroyLRNDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnLRNDescriptor_t,),
        lrnDesc)
end

function cudnnLRNCrossChannelForward(handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y)
    @check ccall((:cudnnLRNCrossChannelForward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnLRNMode_t, Ptr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}),
        handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y)
end

function cudnnLRNCrossChannelBackward(handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
    @check ccall((:cudnnLRNCrossChannelBackward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnLRNMode_t, Ptr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}),
        handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
end

function cudnnDivisiveNormalizationForward(handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y)
    @check ccall((:cudnnDivisiveNormalizationForward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnDivNormMode_t, Ptr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid},
         Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}),
        handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y)
end

function cudnnDivisiveNormalizationBackward(handle, normDesc, mode, alpha, xDesc, x, means, dy, temp, temp2, beta, dXdMeansDesc, dx, dMeans)
    @check ccall((:cudnnDivisiveNormalizationBackward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnDivNormMode_t, Ptr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid},
         CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}),
        handle, normDesc, mode, alpha, xDesc, x, means, dy, temp, temp2, beta,
        dXdMeansDesc, dx, dMeans)
end

function cudnnDeriveBNTensorDescriptor(derivedBnDesc, xDesc, mode)
    @check ccall((:cudnnDeriveBNTensorDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnBatchNormMode_t),
        derivedBnDesc, xDesc, mode)
end

function cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle, mode, bnOps, xDesc, zDesc, yDesc, bnScaleBiasMeanVarDesc, activationDesc, sizeInBytes)
    @check ccall((:cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, cudnnTensorDescriptor_t,
         cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
         cudnnActivationDescriptor_t, Ptr{Csize_t}),
        handle, mode, bnOps, xDesc, zDesc, yDesc, bnScaleBiasMeanVarDesc, activationDesc,
        sizeInBytes)
end

function cudnnGetBatchNormalizationBackwardExWorkspaceSize(handle, mode, bnOps, xDesc, yDesc, dyDesc, dzDesc, dxDesc, dBnScaleBiasDesc, activationDesc, sizeInBytes)
    @check ccall((:cudnnGetBatchNormalizationBackwardExWorkspaceSize, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, cudnnTensorDescriptor_t,
         cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
         cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnActivationDescriptor_t,
         Ptr{Csize_t}),
        handle, mode, bnOps, xDesc, yDesc, dyDesc, dzDesc, dxDesc, dBnScaleBiasDesc,
        activationDesc, sizeInBytes)
end

function cudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle, mode, bnOps, activationDesc, xDesc, sizeInBytes)
    @check ccall((:cudnnGetBatchNormalizationTrainingExReserveSpaceSize, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
         cudnnActivationDescriptor_t, cudnnTensorDescriptor_t, Ptr{Csize_t}),
        handle, mode, bnOps, activationDesc, xDesc, sizeInBytes)
end

function cudnnBatchNormalizationForwardTraining(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance)
    @check ccall((:cudnnBatchNormalizationForwardTraining, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnBatchNormMode_t, Ptr{Cvoid}, Ptr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Cdouble, CuPtr{Cvoid},
         CuPtr{Cvoid}, Cdouble, CuPtr{Cvoid}, CuPtr{Cvoid}),
        handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale,
        bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance,
        epsilon, resultSaveMean, resultSaveInvVariance)
end

function cudnnBatchNormalizationForwardTrainingEx(handle, mode, bnOps, alpha, beta, xDesc, xData, zDesc, zData, yDesc, yData, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance, activationDesc, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
    @check ccall((:cudnnBatchNormalizationForwardTrainingEx, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, Ptr{Cvoid}, Ptr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         CuPtr{Cvoid}, Cdouble, CuPtr{Cvoid}, CuPtr{Cvoid}, Cdouble, CuPtr{Cvoid},
         CuPtr{Cvoid}, cudnnActivationDescriptor_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid},
         Csize_t),
        handle, mode, bnOps, alpha, beta, xDesc, xData, zDesc, zData, yDesc, yData,
        bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor,
        resultRunningMean, resultRunningVariance, epsilon, resultSaveMean,
        resultSaveInvVariance, activationDesc, workspace, workSpaceSizeInBytes,
        reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnBatchNormalizationForwardInference(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon)
    @check ccall((:cudnnBatchNormalizationForwardInference, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnBatchNormMode_t, Ptr{Cvoid}, Ptr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
         Cdouble),
        handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale,
        bnBias, estimatedMean, estimatedVariance, epsilon)
end

function cudnnBatchNormalizationBackward(handle, mode, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, x, dyDesc, dy, dxDesc, dx, dBnScaleBiasDesc, bnScale, dBnScaleResult, dBnBiasResult, epsilon, savedMean, savedInvVariance)
    @check ccall((:cudnnBatchNormalizationBackward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnBatchNormMode_t, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
         Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, Cdouble, CuPtr{Cvoid}, CuPtr{Cvoid}),
        handle, mode, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, x,
        dyDesc, dy, dxDesc, dx, dBnScaleBiasDesc, bnScale, dBnScaleResult, dBnBiasResult,
        epsilon, savedMean, savedInvVariance)
end

function cudnnBatchNormalizationBackwardEx(handle, mode, bnOps, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData, dxDesc, dxData, dBnScaleBiasDesc, bnScaleData, bnBiasData, dBnScaleData, dBnBiasData, epsilon, savedMean, savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
    @check ccall((:cudnnBatchNormalizationBackwardEx, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, Ptr{Cvoid}, Ptr{Cvoid},
         Ptr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid},
         Cdouble, CuPtr{Cvoid}, CuPtr{Cvoid}, cudnnActivationDescriptor_t, CuPtr{Cvoid},
         Csize_t, CuPtr{Cvoid}, Csize_t),
        handle, mode, bnOps, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff,
        xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData, dxDesc, dxData,
        dBnScaleBiasDesc, bnScaleData, bnBiasData, dBnScaleData, dBnBiasData, epsilon,
        savedMean, savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes,
        reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnCreateSpatialTransformerDescriptor(stDesc)
    @check ccall((:cudnnCreateSpatialTransformerDescriptor, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnSpatialTransformerDescriptor_t},),
        stDesc)
end

function cudnnSetSpatialTransformerNdDescriptor(stDesc, samplerType, dataType, nbDims, dimA)
    @check ccall((:cudnnSetSpatialTransformerNdDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnSpatialTransformerDescriptor_t, cudnnSamplerType_t, cudnnDataType_t, Cint,
         Ptr{Cint}),
        stDesc, samplerType, dataType, nbDims, dimA)
end

function cudnnDestroySpatialTransformerDescriptor(stDesc)
    @check ccall((:cudnnDestroySpatialTransformerDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnSpatialTransformerDescriptor_t,),
        stDesc)
end

function cudnnSpatialTfGridGeneratorForward(handle, stDesc, theta, grid)
    @check ccall((:cudnnSpatialTfGridGeneratorForward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}),
        handle, stDesc, theta, grid)
end

function cudnnSpatialTfGridGeneratorBackward(handle, stDesc, dgrid, dtheta)
    @check ccall((:cudnnSpatialTfGridGeneratorBackward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}),
        handle, stDesc, dgrid, dtheta)
end

function cudnnSpatialTfSamplerForward(handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y)
    @check ccall((:cudnnSpatialTfSamplerForward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, Ptr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Ptr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}),
        handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y)
end

function cudnnSpatialTfSamplerBackward(handle, stDesc, alpha, xDesc, x, beta, dxDesc, dx, alphaDgrid, dyDesc, dy, grid, betaDgrid, dgrid)
    @check ccall((:cudnnSpatialTfSamplerBackward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, Ptr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid},
         Ptr{Cvoid}, CuPtr{Cvoid}),
        handle, stDesc, alpha, xDesc, x, beta, dxDesc, dx, alphaDgrid, dyDesc, dy, grid,
        betaDgrid, dgrid)
end

function cudnnCreateDropoutDescriptor(dropoutDesc)
    @check ccall((:cudnnCreateDropoutDescriptor, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnDropoutDescriptor_t},),
        dropoutDesc)
end

function cudnnDestroyDropoutDescriptor(dropoutDesc)
    @check ccall((:cudnnDestroyDropoutDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnDropoutDescriptor_t,),
        dropoutDesc)
end

function cudnnDropoutGetStatesSize(handle, sizeInBytes)
    @check ccall((:cudnnDropoutGetStatesSize, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, Ptr{Csize_t}),
        handle, sizeInBytes)
end

function cudnnDropoutGetReserveSpaceSize(xdesc, sizeInBytes)
    @check ccall((:cudnnDropoutGetReserveSpaceSize, @libcudnn), cudnnStatus_t,
        (cudnnTensorDescriptor_t, Ptr{Csize_t}),
        xdesc, sizeInBytes)
end

function cudnnSetDropoutDescriptor(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed)
    @check ccall((:cudnnSetDropoutDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnDropoutDescriptor_t, cudnnHandle_t, Cfloat, CuPtr{Cvoid}, Csize_t, Culonglong),
        dropoutDesc, handle, dropout, states, stateSizeInBytes, seed)
end

function cudnnRestoreDropoutDescriptor(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed)
    @check ccall((:cudnnRestoreDropoutDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnDropoutDescriptor_t, cudnnHandle_t, Cfloat, CuPtr{Cvoid}, Csize_t, Culonglong),
        dropoutDesc, handle, dropout, states, stateSizeInBytes, seed)
end

function cudnnGetDropoutDescriptor(dropoutDesc, handle, dropout, states, seed)
    @check ccall((:cudnnGetDropoutDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnDropoutDescriptor_t, cudnnHandle_t, CuPtr{Cfloat}, Ptr{Ptr{Cvoid}},
         Ptr{Culonglong}),
        dropoutDesc, handle, dropout, states, seed)
end

function cudnnDropoutForward(handle, dropoutDesc, xdesc, x, ydesc, y, reserveSpace, reserveSpaceSizeInBytes)
    @check ccall((:cudnnDropoutForward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnDropoutDescriptor_t, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t),
        handle, dropoutDesc, xdesc, x, ydesc, y, reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnDropoutBackward(handle, dropoutDesc, dydesc, dy, dxdesc, dx, reserveSpace, reserveSpaceSizeInBytes)
    @check ccall((:cudnnDropoutBackward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnDropoutDescriptor_t, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t),
        handle, dropoutDesc, dydesc, dy, dxdesc, dx, reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnCreateRNNDescriptor(rnnDesc)
    @check ccall((:cudnnCreateRNNDescriptor, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnRNNDescriptor_t},),
        rnnDesc)
end

function cudnnDestroyRNNDescriptor(rnnDesc)
    @check ccall((:cudnnDestroyRNNDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnRNNDescriptor_t,),
        rnnDesc)
end

function cudnnSetRNNDescriptor(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, algo, mathPrec)
    @check ccall((:cudnnSetRNNDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Cint, cudnnDropoutDescriptor_t,
         cudnnRNNInputMode_t, cudnnDirectionMode_t, cudnnRNNMode_t, cudnnRNNAlgo_t,
         cudnnDataType_t),
        handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode,
        algo, mathPrec)
end

function cudnnGetRNNDescriptor(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, algo, mathPrec)
    @check ccall((:cudnnGetRNNDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{Cint}, Ptr{Cint},
         Ptr{cudnnDropoutDescriptor_t}, Ptr{cudnnRNNInputMode_t},
         Ptr{cudnnDirectionMode_t}, Ptr{cudnnRNNMode_t}, Ptr{cudnnRNNAlgo_t},
         Ptr{cudnnDataType_t}),
        handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode,
        algo, mathPrec)
end

function cudnnSetRNNMatrixMathType(rnnDesc, mType)
    @check ccall((:cudnnSetRNNMatrixMathType, @libcudnn), cudnnStatus_t,
        (cudnnRNNDescriptor_t, cudnnMathType_t),
        rnnDesc, mType)
end

function cudnnGetRNNMatrixMathType(rnnDesc, mType)
    @check ccall((:cudnnGetRNNMatrixMathType, @libcudnn), cudnnStatus_t,
        (cudnnRNNDescriptor_t, Ptr{cudnnMathType_t}),
        rnnDesc, mType)
end

function cudnnSetRNNBiasMode(rnnDesc, biasMode)
    @check ccall((:cudnnSetRNNBiasMode, @libcudnn), cudnnStatus_t,
        (cudnnRNNDescriptor_t, cudnnRNNBiasMode_t),
        rnnDesc, biasMode)
end

function cudnnGetRNNBiasMode(rnnDesc, biasMode)
    @check ccall((:cudnnGetRNNBiasMode, @libcudnn), cudnnStatus_t,
        (cudnnRNNDescriptor_t, Ptr{cudnnRNNBiasMode_t}),
        rnnDesc, biasMode)
end

function cudnnRNNSetClip(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip)
    @check ccall((:cudnnRNNSetClip, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnRNNClipMode_t, cudnnNanPropagation_t,
         Cdouble, Cdouble),
        handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip)
end

function cudnnRNNGetClip(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip)
    @check ccall((:cudnnRNNGetClip, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{cudnnRNNClipMode_t},
         Ptr{cudnnNanPropagation_t}, Ptr{Cdouble}, Ptr{Cdouble}),
        handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip)
end

function cudnnSetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize)
    @check ccall((:cudnnSetRNNProjectionLayers, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Cint),
        handle, rnnDesc, recProjSize, outProjSize)
end

function cudnnGetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize)
    @check ccall((:cudnnGetRNNProjectionLayers, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{Cint}, Ptr{Cint}),
        handle, rnnDesc, recProjSize, outProjSize)
end

function cudnnCreatePersistentRNNPlan(rnnDesc, minibatch, dataType, plan)
    @check ccall((:cudnnCreatePersistentRNNPlan, @libcudnn), cudnnStatus_t,
        (cudnnRNNDescriptor_t, Cint, cudnnDataType_t, Ptr{cudnnPersistentRNNPlan_t}),
        rnnDesc, minibatch, dataType, plan)
end

function cudnnDestroyPersistentRNNPlan(plan)
    @check ccall((:cudnnDestroyPersistentRNNPlan, @libcudnn), cudnnStatus_t,
        (cudnnPersistentRNNPlan_t,),
        plan)
end

function cudnnSetPersistentRNNPlan(rnnDesc, plan)
    @check ccall((:cudnnSetPersistentRNNPlan, @libcudnn), cudnnStatus_t,
        (cudnnRNNDescriptor_t, cudnnPersistentRNNPlan_t),
        rnnDesc, plan)
end

function cudnnGetRNNWorkspaceSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes)
    @check ccall((:cudnnGetRNNWorkspaceSize, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t},
         Ptr{Csize_t}),
        handle, rnnDesc, seqLength, xDesc, sizeInBytes)
end

function cudnnGetRNNTrainingReserveSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes)
    @check ccall((:cudnnGetRNNTrainingReserveSize, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t},
         Ptr{Csize_t}),
        handle, rnnDesc, seqLength, xDesc, sizeInBytes)
end

function cudnnGetRNNParamsSize(handle, rnnDesc, xDesc, sizeInBytes, dataType)
    @check ccall((:cudnnGetRNNParamsSize, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnTensorDescriptor_t, Ptr{Csize_t},
         cudnnDataType_t),
        handle, rnnDesc, xDesc, sizeInBytes, dataType)
end

function cudnnGetRNNLinLayerMatrixParams(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerMatDesc, linLayerMat)
    @check ccall((:cudnnGetRNNLinLayerMatrixParams, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, cudnnTensorDescriptor_t,
         cudnnFilterDescriptor_t, CuPtr{Cvoid}, Cint, cudnnFilterDescriptor_t,
         Ptr{Ptr{Cvoid}}),
        handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerMatDesc,
        linLayerMat)
end

function cudnnGetRNNLinLayerBiasParams(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerBiasDesc, linLayerBias)
    @check ccall((:cudnnGetRNNLinLayerBiasParams, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, cudnnTensorDescriptor_t,
         cudnnFilterDescriptor_t, CuPtr{Cvoid}, Cint, cudnnFilterDescriptor_t,
         Ptr{Ptr{Cvoid}}),
        handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerBiasDesc,
        linLayerBias)
end

function cudnnRNNForwardInference(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes)
    @check ccall((:cudnnRNNForwardInference, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t},
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t},
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t),
        handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y,
        hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes)
end

function cudnnRNNForwardTraining(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
    @check ccall((:cudnnRNNForwardTraining, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t},
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t},
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t),
        handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y,
        hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes, reserveSpace,
        reserveSpaceSizeInBytes)
end

function cudnnRNNBackwardData(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
    @check ccall((:cudnnRNNBackwardData, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t},
         CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnFilterDescriptor_t,
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t,
         CuPtr{Cvoid}, Csize_t),
        handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy,
        wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx,
        workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnRNNBackwardWeights(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, workspace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes)
    @check ccall((:cudnnRNNBackwardWeights, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t},
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t},
         CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, cudnnFilterDescriptor_t, CuPtr{Cvoid},
         CuPtr{Cvoid}, Csize_t),
        handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, workspace,
        workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnSetRNNPaddingMode(rnnDesc, paddingMode)
    @check ccall((:cudnnSetRNNPaddingMode, @libcudnn), cudnnStatus_t,
        (cudnnRNNDescriptor_t, cudnnRNNPaddingMode_t),
        rnnDesc, paddingMode)
end

function cudnnGetRNNPaddingMode(rnnDesc, paddingMode)
    @check ccall((:cudnnGetRNNPaddingMode, @libcudnn), cudnnStatus_t,
        (cudnnRNNDescriptor_t, Ptr{cudnnRNNPaddingMode_t}),
        rnnDesc, paddingMode)
end

function cudnnCreateRNNDataDescriptor(rnnDataDesc)
    @check ccall((:cudnnCreateRNNDataDescriptor, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnRNNDataDescriptor_t},),
        rnnDataDesc)
end

function cudnnDestroyRNNDataDescriptor(rnnDataDesc)
    @check ccall((:cudnnDestroyRNNDataDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnRNNDataDescriptor_t,),
        rnnDataDesc)
end

function cudnnSetRNNDataDescriptor(rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, seqLengthArray, paddingFill)
    @check ccall((:cudnnSetRNNDataDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnRNNDataDescriptor_t, cudnnDataType_t, cudnnRNNDataLayout_t, Cint, Cint, Cint,
         Ptr{Cint}, Ptr{Cvoid}),
        rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, seqLengthArray,
        paddingFill)
end

function cudnnGetRNNDataDescriptor(rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, arrayLengthRequested, seqLengthArray, paddingFill)
    @check ccall((:cudnnGetRNNDataDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnRNNDataDescriptor_t, Ptr{cudnnDataType_t}, Ptr{cudnnRNNDataLayout_t},
         Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cint}, Ptr{Cvoid}),
        rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize,
        arrayLengthRequested, seqLengthArray, paddingFill)
end

function cudnnRNNForwardTrainingEx(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
    @check ccall((:cudnnRNNForwardTrainingEx, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnRNNDataDescriptor_t, CuPtr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid},
         cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid},
         CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t),
        handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy,
        cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace,
        workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnRNNForwardInferenceEx(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace, workSpaceSizeInBytes)
    @check ccall((:cudnnRNNForwardInferenceEx, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnRNNDataDescriptor_t, CuPtr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnRNNDataDescriptor_t, Ptr{Cvoid}, cudnnRNNDataDescriptor_t, Ptr{Cvoid},
         cudnnRNNDataDescriptor_t, Ptr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid},
         CuPtr{Cvoid}, Csize_t),
        handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy,
        cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace,
        workSpaceSizeInBytes)
end

function cudnnRNNBackwardDataEx(handle, rnnDesc, yDesc, y, dyDesc, dy, dcDesc, dcAttn, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, dkDesc, dkeys, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
    @check ccall((:cudnnRNNBackwardDataEx, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnRNNDataDescriptor_t, CuPtr{Cvoid},
         cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid},
         cudnnTensorDescriptor_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnRNNDataDescriptor_t, Ptr{Cvoid}, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t),
        handle, rnnDesc, yDesc, y, dyDesc, dy, dcDesc, dcAttn, dhyDesc, dhy, dcyDesc, dcy,
        wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, dkDesc,
        dkeys, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnRNNBackwardWeightsEx(handle, rnnDesc, xDesc, x, hxDesc, hx, yDesc, y, workSpace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes)
    @check ccall((:cudnnRNNBackwardWeightsEx, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnRNNDataDescriptor_t, CuPtr{Cvoid},
         cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid},
         CuPtr{Cvoid}, Csize_t, cudnnFilterDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid},
         Csize_t),
        handle, rnnDesc, xDesc, x, hxDesc, hx, yDesc, y, workSpace, workSpaceSizeInBytes,
        dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnSetRNNAlgorithmDescriptor(handle, rnnDesc, algoDesc)
    @check ccall((:cudnnSetRNNAlgorithmDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnAlgorithmDescriptor_t),
        handle, rnnDesc, algoDesc)
end

function cudnnGetRNNForwardInferenceAlgorithmMaxCount(handle, rnnDesc, count)
    @check ccall((:cudnnGetRNNForwardInferenceAlgorithmMaxCount, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{Cint}),
        handle, rnnDesc, count)
end

function cudnnFindRNNForwardInferenceAlgorithmEx(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes)
    @check ccall((:cudnnFindRNNForwardInferenceAlgorithmEx, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t},
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t},
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, Cfloat, Cint, Ptr{Cint}, Ptr{cudnnAlgorithmPerformance_t},
         CuPtr{Cvoid}, Csize_t),
        handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y,
        hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount, returnedAlgoCount,
        perfResults, workspace, workSpaceSizeInBytes)
end

function cudnnGetRNNForwardTrainingAlgorithmMaxCount(handle, rnnDesc, count)
    @check ccall((:cudnnGetRNNForwardTrainingAlgorithmMaxCount, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{Cint}),
        handle, rnnDesc, count)
end

function cudnnFindRNNForwardTrainingAlgorithmEx(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
    @check ccall((:cudnnFindRNNForwardTrainingAlgorithmEx, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t},
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t},
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, Cfloat, Cint, Ptr{Cint}, Ptr{cudnnAlgorithmPerformance_t},
         CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t),
        handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y,
        hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount, returnedAlgoCount,
        perfResults, workspace, workSpaceSizeInBytes, reserveSpace,
        reserveSpaceSizeInBytes)
end

function cudnnGetRNNBackwardDataAlgorithmMaxCount(handle, rnnDesc, count)
    @check ccall((:cudnnGetRNNBackwardDataAlgorithmMaxCount, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{Cint}),
        handle, rnnDesc, count)
end

function cudnnFindRNNBackwardDataAlgorithmEx(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
    @check ccall((:cudnnFindRNNBackwardDataAlgorithmEx, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t},
         CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnFilterDescriptor_t,
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Cfloat, Cint, Ptr{Cint},
         Ptr{cudnnAlgorithmPerformance_t}, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t),
        handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy,
        wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx,
        findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace,
        workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnGetRNNBackwardWeightsAlgorithmMaxCount(handle, rnnDesc, count)
    @check ccall((:cudnnGetRNNBackwardWeightsAlgorithmMaxCount, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{Cint}),
        handle, rnnDesc, count)
end

function cudnnFindRNNBackwardWeightsAlgorithmEx(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes)
    @check ccall((:cudnnFindRNNBackwardWeightsAlgorithmEx, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t},
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t},
         CuPtr{Cvoid}, Cfloat, Cint, Ptr{Cint}, Ptr{cudnnAlgorithmPerformance_t},
         CuPtr{Cvoid}, Csize_t, cudnnFilterDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid},
         Csize_t),
        handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, findIntensity,
        requestedAlgoCount, returnedAlgoCount, perfResults, workspace,
        workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnCreateSeqDataDescriptor(seqDataDesc)
    @check ccall((:cudnnCreateSeqDataDescriptor, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnSeqDataDescriptor_t},),
        seqDataDesc)
end

function cudnnDestroySeqDataDescriptor(seqDataDesc)
    @check ccall((:cudnnDestroySeqDataDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnSeqDataDescriptor_t,),
        seqDataDesc)
end

function cudnnSetSeqDataDescriptor(seqDataDesc, dataType, nbDims, dimA, axes, seqLengthArraySize, seqLengthArray, paddingFill)
    @check ccall((:cudnnSetSeqDataDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnSeqDataDescriptor_t, cudnnDataType_t, Cint, Ptr{Cint},
         Ptr{cudnnSeqDataAxis_t}, Csize_t, Ptr{Cint}, Ptr{Cvoid}),
        seqDataDesc, dataType, nbDims, dimA, axes, seqLengthArraySize, seqLengthArray,
        paddingFill)
end

function cudnnGetSeqDataDescriptor(seqDataDesc, dataType, nbDims, nbDimsRequested, dimA, axes, seqLengthArraySize, seqLengthSizeRequested, seqLengthArray, paddingFill)
    @check ccall((:cudnnGetSeqDataDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnSeqDataDescriptor_t, Ptr{cudnnDataType_t}, Ptr{Cint}, Cint, Ptr{Cint},
         Ptr{cudnnSeqDataAxis_t}, Ptr{Csize_t}, Csize_t, Ptr{Cint}, Ptr{Cvoid}),
        seqDataDesc, dataType, nbDims, nbDimsRequested, dimA, axes, seqLengthArraySize,
        seqLengthSizeRequested, seqLengthArray, paddingFill)
end

function cudnnCreateAttnDescriptor(attnDesc)
    @check ccall((:cudnnCreateAttnDescriptor, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnAttnDescriptor_t},),
        attnDesc)
end

function cudnnDestroyAttnDescriptor(attnDesc)
    @check ccall((:cudnnDestroyAttnDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnAttnDescriptor_t,),
        attnDesc)
end

function cudnnSetAttnDescriptor(attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize)
    @check ccall((:cudnnSetAttnDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnAttnDescriptor_t, UInt32, Cint, Cdouble, cudnnDataType_t, cudnnDataType_t,
         cudnnMathType_t, cudnnDropoutDescriptor_t, cudnnDropoutDescriptor_t, Cint, Cint,
         Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint),
        attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType,
        attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize,
        vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize)
end

function cudnnGetAttnDescriptor(attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize)
    @check ccall((:cudnnGetAttnDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnAttnDescriptor_t, Ptr{UInt32}, Ptr{Cint}, Ptr{Cdouble}, Ptr{cudnnDataType_t},
         Ptr{cudnnDataType_t}, Ptr{cudnnMathType_t}, Ptr{cudnnDropoutDescriptor_t},
         Ptr{cudnnDropoutDescriptor_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
         Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
        attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType,
        attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize,
        vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize)
end

function cudnnGetMultiHeadAttnBuffers(handle, attnDesc, weightSizeInBytes, workSpaceSizeInBytes, reserveSpaceSizeInBytes)
    @check ccall((:cudnnGetMultiHeadAttnBuffers, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnAttnDescriptor_t, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}),
        handle, attnDesc, weightSizeInBytes, workSpaceSizeInBytes, reserveSpaceSizeInBytes)
end

function cudnnGetMultiHeadAttnWeights(handle, attnDesc, wKind, weightSizeInBytes, weights, wDesc, wAddr)
    @check ccall((:cudnnGetMultiHeadAttnWeights, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnAttnDescriptor_t, cudnnMultiHeadAttnWeightKind_t, Csize_t,
         CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Ptr{Cvoid}}),
        handle, attnDesc, wKind, weightSizeInBytes, weights, wDesc, wAddr)
end

function cudnnMultiHeadAttnForward(handle, attnDesc, currIdx, loWinIdx, hiWinIdx, seqLengthArrayQRO, seqLengthArrayKV, qDesc, queries, residuals, kDesc, keys, vDesc, values, oDesc, out, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace)
    @check ccall((:cudnnMultiHeadAttnForward, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnAttnDescriptor_t, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
         Ptr{Cint}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid},
         cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid},
         cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t,
         CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}),
        handle, attnDesc, currIdx, loWinIdx, hiWinIdx, seqLengthArrayQRO, seqLengthArrayKV,
        qDesc, queries, residuals, kDesc, keys, vDesc, values, oDesc, out,
        weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace,
        reserveSpaceSizeInBytes, reserveSpace)
end

function cudnnMultiHeadAttnBackwardData(handle, attnDesc, loWinIdx, hiWinIdx, seqLengthArrayDQDO, seqLengthArrayDKDV, doDesc, dout, dqDesc, dqueries, queries, dkDesc, dkeys, keys, dvDesc, dvalues, values, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace)
    @check ccall((:cudnnMultiHeadAttnBackwardData, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnAttnDescriptor_t, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
         cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid},
         CuPtr{Cvoid}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid},
         cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid},
         Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}),
        handle, attnDesc, loWinIdx, hiWinIdx, seqLengthArrayDQDO, seqLengthArrayDKDV,
        doDesc, dout, dqDesc, dqueries, queries, dkDesc, dkeys, keys, dvDesc, dvalues,
        values, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace,
        reserveSpaceSizeInBytes, reserveSpace)
end

function cudnnMultiHeadAttnBackwardWeights(handle, attnDesc, addGrad, qDesc, queries, kDesc, keys, vDesc, values, doDesc, dout, weightSizeInBytes, weights, dweights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace)
    @check ccall((:cudnnMultiHeadAttnBackwardWeights, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnAttnDescriptor_t, cudnnWgradMode_t, cudnnSeqDataDescriptor_t,
         CuPtr{Cvoid}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, cudnnSeqDataDescriptor_t,
         CuPtr{Cvoid}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid},
         CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}),
        handle, attnDesc, addGrad, qDesc, queries, kDesc, keys, vDesc, values, doDesc,
        dout, weightSizeInBytes, weights, dweights, workSpaceSizeInBytes, workSpace,
        reserveSpaceSizeInBytes, reserveSpace)
end

function cudnnCreateCTCLossDescriptor(ctcLossDesc)
    @check ccall((:cudnnCreateCTCLossDescriptor, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnCTCLossDescriptor_t},),
        ctcLossDesc)
end

function cudnnSetCTCLossDescriptor(ctcLossDesc, compType)
    @check ccall((:cudnnSetCTCLossDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnCTCLossDescriptor_t, cudnnDataType_t),
        ctcLossDesc, compType)
end

function cudnnSetCTCLossDescriptorEx(ctcLossDesc, compType, normMode, gradMode)
    @check ccall((:cudnnSetCTCLossDescriptorEx, @libcudnn), cudnnStatus_t,
        (cudnnCTCLossDescriptor_t, cudnnDataType_t, cudnnLossNormalizationMode_t,
         cudnnNanPropagation_t),
        ctcLossDesc, compType, normMode, gradMode)
end

function cudnnGetCTCLossDescriptor(ctcLossDesc, compType)
    @check ccall((:cudnnGetCTCLossDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnCTCLossDescriptor_t, Ptr{cudnnDataType_t}),
        ctcLossDesc, compType)
end

function cudnnGetCTCLossDescriptorEx(ctcLossDesc, compType, normMode, gradMode)
    @check ccall((:cudnnGetCTCLossDescriptorEx, @libcudnn), cudnnStatus_t,
        (cudnnCTCLossDescriptor_t, Ptr{cudnnDataType_t}, Ptr{cudnnLossNormalizationMode_t},
         Ptr{cudnnNanPropagation_t}),
        ctcLossDesc, compType, normMode, gradMode)
end

function cudnnDestroyCTCLossDescriptor(ctcLossDesc)
    @check ccall((:cudnnDestroyCTCLossDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnCTCLossDescriptor_t,),
        ctcLossDesc)
end

function cudnnCTCLoss(handle, probsDesc, probs, labels, labelLengths, inputLengths, costs, gradientsDesc, gradients, algo, ctcLossDesc, workspace, workSpaceSizeInBytes)
    @check ccall((:cudnnCTCLoss, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cint}, Ptr{Cint},
         Ptr{Cint}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
         cudnnCTCLossAlgo_t, cudnnCTCLossDescriptor_t, CuPtr{Cvoid}, Csize_t),
        handle, probsDesc, probs, labels, labelLengths, inputLengths, costs, gradientsDesc,
        gradients, algo, ctcLossDesc, workspace, workSpaceSizeInBytes)
end

function cudnnGetCTCLossWorkspaceSize(handle, probsDesc, gradientsDesc, labels, labelLengths, inputLengths, algo, ctcLossDesc, sizeInBytes)
    @check ccall((:cudnnGetCTCLossWorkspaceSize, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, Ptr{Cint},
         Ptr{Cint}, Ptr{Cint}, cudnnCTCLossAlgo_t, cudnnCTCLossDescriptor_t, Ptr{Csize_t}),
        handle, probsDesc, gradientsDesc, labels, labelLengths, inputLengths, algo,
        ctcLossDesc, sizeInBytes)
end

function cudnnCreateAlgorithmDescriptor(algoDesc)
    @check ccall((:cudnnCreateAlgorithmDescriptor, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnAlgorithmDescriptor_t},),
        algoDesc)
end

function cudnnSetAlgorithmDescriptor(algoDesc, algorithm)
    @check ccall((:cudnnSetAlgorithmDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnAlgorithmDescriptor_t, cudnnAlgorithm_t),
        algoDesc, algorithm)
end

function cudnnGetAlgorithmDescriptor(algoDesc, algorithm)
    @check ccall((:cudnnGetAlgorithmDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnAlgorithmDescriptor_t, Ptr{cudnnAlgorithm_t}),
        algoDesc, algorithm)
end

function cudnnCopyAlgorithmDescriptor(src, dest)
    @check ccall((:cudnnCopyAlgorithmDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnAlgorithmDescriptor_t, cudnnAlgorithmDescriptor_t),
        src, dest)
end

function cudnnDestroyAlgorithmDescriptor(algoDesc)
    @check ccall((:cudnnDestroyAlgorithmDescriptor, @libcudnn), cudnnStatus_t,
        (cudnnAlgorithmDescriptor_t,),
        algoDesc)
end

function cudnnCreateAlgorithmPerformance(algoPerf, numberToCreate)
    @check ccall((:cudnnCreateAlgorithmPerformance, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnAlgorithmPerformance_t}, Cint),
        algoPerf, numberToCreate)
end

function cudnnSetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory)
    @check ccall((:cudnnSetAlgorithmPerformance, @libcudnn), cudnnStatus_t,
        (cudnnAlgorithmPerformance_t, cudnnAlgorithmDescriptor_t, cudnnStatus_t, Cfloat,
         Csize_t),
        algoPerf, algoDesc, status, time, memory)
end

function cudnnGetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory)
    @check ccall((:cudnnGetAlgorithmPerformance, @libcudnn), cudnnStatus_t,
        (cudnnAlgorithmPerformance_t, Ptr{cudnnAlgorithmDescriptor_t}, Ptr{cudnnStatus_t},
         Ptr{Cfloat}, Ptr{Csize_t}),
        algoPerf, algoDesc, status, time, memory)
end

function cudnnDestroyAlgorithmPerformance(algoPerf, numberToDestroy)
    @check ccall((:cudnnDestroyAlgorithmPerformance, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnAlgorithmPerformance_t}, Cint),
        algoPerf, numberToDestroy)
end

function cudnnGetAlgorithmSpaceSize(handle, algoDesc, algoSpaceSizeInBytes)
    @check ccall((:cudnnGetAlgorithmSpaceSize, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnAlgorithmDescriptor_t, Ptr{Csize_t}),
        handle, algoDesc, algoSpaceSizeInBytes)
end

function cudnnSaveAlgorithm(handle, algoDesc, algoSpace, algoSpaceSizeInBytes)
    @check ccall((:cudnnSaveAlgorithm, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnAlgorithmDescriptor_t, Ptr{Cvoid}, Csize_t),
        handle, algoDesc, algoSpace, algoSpaceSizeInBytes)
end

function cudnnRestoreAlgorithm(handle, algoSpace, algoSpaceSizeInBytes, algoDesc)
    @check ccall((:cudnnRestoreAlgorithm, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, Ptr{Cvoid}, Csize_t, cudnnAlgorithmDescriptor_t),
        handle, algoSpace, algoSpaceSizeInBytes, algoDesc)
end

function cudnnSetCallback(mask, udata, fptr)
    @check ccall((:cudnnSetCallback, @libcudnn), cudnnStatus_t,
        (UInt32, Ptr{Cvoid}, cudnnCallback_t),
        mask, udata, fptr)
end

function cudnnGetCallback(mask, udata, fptr)
    @check ccall((:cudnnGetCallback, @libcudnn), cudnnStatus_t,
        (Ptr{UInt32}, Ptr{Ptr{Cvoid}}, Ptr{cudnnCallback_t}),
        mask, udata, fptr)
end

function cudnnCreateFusedOpsConstParamPack(constPack, ops)
    @check ccall((:cudnnCreateFusedOpsConstParamPack, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnFusedOpsConstParamPack_t}, cudnnFusedOps_t),
        constPack, ops)
end

function cudnnDestroyFusedOpsConstParamPack(constPack)
    @check ccall((:cudnnDestroyFusedOpsConstParamPack, @libcudnn), cudnnStatus_t,
        (cudnnFusedOpsConstParamPack_t,),
        constPack)
end

function cudnnSetFusedOpsConstParamPackAttribute(constPack, paramLabel, param)
    @check ccall((:cudnnSetFusedOpsConstParamPackAttribute, @libcudnn), cudnnStatus_t,
        (cudnnFusedOpsConstParamPack_t, cudnnFusedOpsConstParamLabel_t, Ptr{Cvoid}),
        constPack, paramLabel, param)
end

function cudnnGetFusedOpsConstParamPackAttribute(constPack, paramLabel, param, isNULL)
    @check ccall((:cudnnGetFusedOpsConstParamPackAttribute, @libcudnn), cudnnStatus_t,
        (cudnnFusedOpsConstParamPack_t, cudnnFusedOpsConstParamLabel_t, Ptr{Cvoid},
         Ptr{Cint}),
        constPack, paramLabel, param, isNULL)
end

function cudnnCreateFusedOpsVariantParamPack(varPack, ops)
    @check ccall((:cudnnCreateFusedOpsVariantParamPack, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnFusedOpsVariantParamPack_t}, cudnnFusedOps_t),
        varPack, ops)
end

function cudnnDestroyFusedOpsVariantParamPack(varPack)
    @check ccall((:cudnnDestroyFusedOpsVariantParamPack, @libcudnn), cudnnStatus_t,
        (cudnnFusedOpsVariantParamPack_t,),
        varPack)
end

function cudnnSetFusedOpsVariantParamPackAttribute(varPack, paramLabel, ptr)
    @check ccall((:cudnnSetFusedOpsVariantParamPackAttribute, @libcudnn), cudnnStatus_t,
        (cudnnFusedOpsVariantParamPack_t, cudnnFusedOpsVariantParamLabel_t,
         PtrOrCuPtr{Cvoid}),
        varPack, paramLabel, ptr)
end

function cudnnGetFusedOpsVariantParamPackAttribute(varPack, paramLabel, ptr)
    @check ccall((:cudnnGetFusedOpsVariantParamPackAttribute, @libcudnn), cudnnStatus_t,
        (cudnnFusedOpsVariantParamPack_t, cudnnFusedOpsVariantParamLabel_t,
         PtrOrCuPtr{Cvoid}),
        varPack, paramLabel, ptr)
end

function cudnnCreateFusedOpsPlan(plan, ops)
    @check ccall((:cudnnCreateFusedOpsPlan, @libcudnn), cudnnStatus_t,
        (Ptr{cudnnFusedOpsPlan_t}, cudnnFusedOps_t),
        plan, ops)
end

function cudnnDestroyFusedOpsPlan(plan)
    @check ccall((:cudnnDestroyFusedOpsPlan, @libcudnn), cudnnStatus_t,
        (cudnnFusedOpsPlan_t,),
        plan)
end

function cudnnMakeFusedOpsPlan(handle, plan, constPack, workspaceSizeInBytes)
    @check ccall((:cudnnMakeFusedOpsPlan, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnFusedOpsPlan_t, cudnnFusedOpsConstParamPack_t, Ptr{Csize_t}),
        handle, plan, constPack, workspaceSizeInBytes)
end

function cudnnFusedOpsExecute(handle, plan, varPack)
    @check ccall((:cudnnFusedOpsExecute, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnFusedOpsPlan_t, cudnnFusedOpsVariantParamPack_t),
        handle, plan, varPack)
end

function cudnnSetRNNDescriptor_v6(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, algo, mathPrec)
    @check ccall((:cudnnSetRNNDescriptor_v6, @libcudnn), cudnnStatus_t,
        (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Cint, cudnnDropoutDescriptor_t,
         cudnnRNNInputMode_t, cudnnDirectionMode_t, cudnnRNNMode_t, cudnnRNNAlgo_t,
         cudnnDataType_t),
        handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode,
        algo, mathPrec)
end

function cudnnSetRNNDescriptor_v5(rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, mathPrec)
    @check ccall((:cudnnSetRNNDescriptor_v5, @libcudnn), cudnnStatus_t,
        (cudnnRNNDescriptor_t, Cint, Cint, cudnnDropoutDescriptor_t, cudnnRNNInputMode_t,
         cudnnDirectionMode_t, cudnnRNNMode_t, cudnnDataType_t),
        rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, mathPrec)
end
