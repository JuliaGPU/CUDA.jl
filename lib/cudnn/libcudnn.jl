# Julia wrapper for header: cudnn.h
# Automatically generated using Clang.jl

function cudnnGetVersion()
    ccall((:cudnnGetVersion, libcudnn()), Csize_t, ())
end

function cudnnGetCudartVersion()
    ccall((:cudnnGetCudartVersion, libcudnn()), Csize_t, ())
end

function cudnnGetErrorString(status)
    ccall((:cudnnGetErrorString, libcudnn()), Cstring,
                   (cudnnStatus_t,),
                   status)
end

@checked function cudnnQueryRuntimeError(handle, rstatus, mode, tag)
    initialize_api()
    ccall((:cudnnQueryRuntimeError, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, Ptr{cudnnStatus_t}, cudnnErrQueryMode_t,
                    Ptr{cudnnRuntimeTag_t}),
                   handle, rstatus, mode, tag)
end

@checked function cudnnGetProperty(type, value)
    ccall((:cudnnGetProperty, libcudnn()), cudnnStatus_t,
                   (libraryPropertyType, Ptr{Cint}),
                   type, value)
end

@checked function cudnnCreate(handle)
    initialize_api()
    ccall((:cudnnCreate, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnHandle_t},),
                   handle)
end

@checked function cudnnDestroy(handle)
    initialize_api()
    ccall((:cudnnDestroy, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t,),
                   handle)
end

@checked function cudnnSetStream(handle, streamId)
    initialize_api()
    ccall((:cudnnSetStream, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, CUstream),
                   handle, streamId)
end

@checked function cudnnGetStream(handle, streamId)
    initialize_api()
    ccall((:cudnnGetStream, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, Ptr{CUstream}),
                   handle, streamId)
end

@checked function cudnnCreateTensorDescriptor(tensorDesc)
    initialize_api()
    ccall((:cudnnCreateTensorDescriptor, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnTensorDescriptor_t},),
                   tensorDesc)
end

@checked function cudnnSetTensor4dDescriptor(tensorDesc, format, dataType, n, c, h, w)
    initialize_api()
    ccall((:cudnnSetTensor4dDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, Cint,
                    Cint, Cint, Cint),
                   tensorDesc, format, dataType, n, c, h, w)
end

@checked function cudnnSetTensor4dDescriptorEx(tensorDesc, dataType, n, c, h, w, nStride,
                                               cStride, hStride, wStride)
    initialize_api()
    ccall((:cudnnSetTensor4dDescriptorEx, libcudnn()), cudnnStatus_t,
                   (cudnnTensorDescriptor_t, cudnnDataType_t, Cint, Cint, Cint, Cint,
                    Cint, Cint, Cint, Cint),
                   tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride)
end

@checked function cudnnGetTensor4dDescriptor(tensorDesc, dataType, n, c, h, w, nStride,
                                             cStride, hStride, wStride)
    initialize_api()
    ccall((:cudnnGetTensor4dDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnTensorDescriptor_t, Ptr{cudnnDataType_t}, Ptr{Cint}, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                   tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride)
end

@checked function cudnnSetTensorNdDescriptor(tensorDesc, dataType, nbDims, dimA, strideA)
    initialize_api()
    ccall((:cudnnSetTensorNdDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnTensorDescriptor_t, cudnnDataType_t, Cint, Ptr{Cint}, Ptr{Cint}),
                   tensorDesc, dataType, nbDims, dimA, strideA)
end

@checked function cudnnSetTensorNdDescriptorEx(tensorDesc, format, dataType, nbDims, dimA)
    initialize_api()
    ccall((:cudnnSetTensorNdDescriptorEx, libcudnn()), cudnnStatus_t,
                   (cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, Cint,
                    Ptr{Cint}),
                   tensorDesc, format, dataType, nbDims, dimA)
end

@checked function cudnnGetTensorNdDescriptor(tensorDesc, nbDimsRequested, dataType, nbDims,
                                             dimA, strideA)
    initialize_api()
    ccall((:cudnnGetTensorNdDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnTensorDescriptor_t, Cint, Ptr{cudnnDataType_t}, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}),
                   tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA)
end

@checked function cudnnGetTensorSizeInBytes(tensorDesc, size)
    initialize_api()
    ccall((:cudnnGetTensorSizeInBytes, libcudnn()), cudnnStatus_t,
                   (cudnnTensorDescriptor_t, Ptr{Csize_t}),
                   tensorDesc, size)
end

@checked function cudnnDestroyTensorDescriptor(tensorDesc)
    initialize_api()
    ccall((:cudnnDestroyTensorDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnTensorDescriptor_t,),
                   tensorDesc)
end

@checked function cudnnInitTransformDest(transformDesc, srcDesc, destDesc, destSizeInBytes)
    initialize_api()
    ccall((:cudnnInitTransformDest, libcudnn()), cudnnStatus_t,
                   (cudnnTensorTransformDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnTensorDescriptor_t, Ptr{Csize_t}),
                   transformDesc, srcDesc, destDesc, destSizeInBytes)
end

@checked function cudnnCreateTensorTransformDescriptor(transformDesc)
    initialize_api()
    ccall((:cudnnCreateTensorTransformDescriptor, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnTensorTransformDescriptor_t},),
                   transformDesc)
end

@checked function cudnnSetTensorTransformDescriptor(transformDesc, nbDims, destFormat,
                                                    padBeforeA, padAfterA, foldA, direction)
    initialize_api()
    ccall((:cudnnSetTensorTransformDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnTensorTransformDescriptor_t, UInt32, cudnnTensorFormat_t,
                    Ptr{Int32}, Ptr{Int32}, Ptr{UInt32}, cudnnFoldingDirection_t),
                   transformDesc, nbDims, destFormat, padBeforeA, padAfterA, foldA,
                   direction)
end

@checked function cudnnGetTensorTransformDescriptor(transformDesc, nbDimsRequested,
                                                    destFormat, padBeforeA, padAfterA,
                                                    foldA, direction)
    initialize_api()
    ccall((:cudnnGetTensorTransformDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnTensorTransformDescriptor_t, UInt32, Ptr{cudnnTensorFormat_t},
                    Ptr{Int32}, Ptr{Int32}, Ptr{UInt32}, Ptr{cudnnFoldingDirection_t}),
                   transformDesc, nbDimsRequested, destFormat, padBeforeA, padAfterA,
                   foldA, direction)
end

@checked function cudnnDestroyTensorTransformDescriptor(transformDesc)
    initialize_api()
    ccall((:cudnnDestroyTensorTransformDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnTensorTransformDescriptor_t,),
                   transformDesc)
end

@checked function cudnnTransformTensor(handle, alpha, xDesc, x, beta, yDesc, y)
    initialize_api()
    ccall((:cudnnTransformTensor, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, Ptr{Cvoid},
                    Ptr{Cvoid}, cudnnTensorDescriptor_t, Ptr{Cvoid}),
                   handle, alpha, xDesc, x, beta, yDesc, y)
end

@checked function cudnnTransformTensorEx(handle, transDesc, alpha, srcDesc, srcData, beta,
                                         destDesc, destData)
    initialize_api()
    ccall((:cudnnTransformTensorEx, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnTensorTransformDescriptor_t, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, Ptr{Cvoid}, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, Ptr{Cvoid}),
                   handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData)
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
    initialize_api()
    ccall((:cudnnGetFoldedConvBackwardDataDescriptors, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnTensorFormat_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnTensorTransformDescriptor_t, cudnnTensorTransformDescriptor_t,
                    cudnnTensorTransformDescriptor_t, cudnnTensorTransformDescriptor_t),
                   handle, filterDesc, diffDesc, convDesc, gradDesc, transformFormat,
                   foldedFilterDesc, paddedDiffDesc, foldedConvDesc, foldedGradDesc,
                   filterFoldTransDesc, diffPadTransDesc, gradFoldTransDesc,
                   gradUnfoldTransDesc)
end

@checked function cudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C)
    initialize_api()
    ccall((:cudnnAddTensor, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}),
                   handle, alpha, aDesc, A, beta, cDesc, C)
end

@checked function cudnnCreateOpTensorDescriptor(opTensorDesc)
    initialize_api()
    ccall((:cudnnCreateOpTensorDescriptor, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnOpTensorDescriptor_t},),
                   opTensorDesc)
end

@checked function cudnnSetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType,
                                             opTensorNanOpt)
    initialize_api()
    ccall((:cudnnSetOpTensorDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnOpTensorDescriptor_t, cudnnOpTensorOp_t, cudnnDataType_t,
                    cudnnNanPropagation_t),
                   opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)
end

@checked function cudnnGetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType,
                                             opTensorNanOpt)
    initialize_api()
    ccall((:cudnnGetOpTensorDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnOpTensorDescriptor_t, Ptr{cudnnOpTensorOp_t},
                    Ptr{cudnnDataType_t}, Ptr{cudnnNanPropagation_t}),
                   opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)
end

@checked function cudnnDestroyOpTensorDescriptor(opTensorDesc)
    initialize_api()
    ccall((:cudnnDestroyOpTensorDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnOpTensorDescriptor_t,),
                   opTensorDesc)
end

@checked function cudnnOpTensor(handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B,
                                beta, cDesc, C)
    initialize_api()
    ccall((:cudnnOpTensor, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnOpTensorDescriptor_t, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}),
                   handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc,
                   C)
end

@checked function cudnnCreateReduceTensorDescriptor(reduceTensorDesc)
    initialize_api()
    ccall((:cudnnCreateReduceTensorDescriptor, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnReduceTensorDescriptor_t},),
                   reduceTensorDesc)
end

@checked function cudnnSetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp,
                                                 reduceTensorCompType, reduceTensorNanOpt,
                                                 reduceTensorIndices,
                                                 reduceTensorIndicesType)
    initialize_api()
    ccall((:cudnnSetReduceTensorDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnReduceTensorDescriptor_t, cudnnReduceTensorOp_t, cudnnDataType_t,
                    cudnnNanPropagation_t, cudnnReduceTensorIndices_t, cudnnIndicesType_t),
                   reduceTensorDesc, reduceTensorOp, reduceTensorCompType,
                   reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType)
end

@checked function cudnnGetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp,
                                                 reduceTensorCompType, reduceTensorNanOpt,
                                                 reduceTensorIndices,
                                                 reduceTensorIndicesType)
    initialize_api()
    ccall((:cudnnGetReduceTensorDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnReduceTensorDescriptor_t, Ptr{cudnnReduceTensorOp_t},
                    Ptr{cudnnDataType_t}, Ptr{cudnnNanPropagation_t},
                    Ptr{cudnnReduceTensorIndices_t}, Ptr{cudnnIndicesType_t}),
                   reduceTensorDesc, reduceTensorOp, reduceTensorCompType,
                   reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType)
end

@checked function cudnnDestroyReduceTensorDescriptor(reduceTensorDesc)
    initialize_api()
    ccall((:cudnnDestroyReduceTensorDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnReduceTensorDescriptor_t,),
                   reduceTensorDesc)
end

@checked function cudnnGetReductionIndicesSize(handle, reduceTensorDesc, aDesc, cDesc,
                                               sizeInBytes)
    initialize_api()
    ccall((:cudnnGetReductionIndicesSize, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnReduceTensorDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnTensorDescriptor_t, Ptr{Csize_t}),
                   handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes)
end

@checked function cudnnGetReductionWorkspaceSize(handle, reduceTensorDesc, aDesc, cDesc,
                                                 sizeInBytes)
    initialize_api()
    ccall((:cudnnGetReductionWorkspaceSize, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnReduceTensorDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnTensorDescriptor_t, Ref{Csize_t}),
                   handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes)
end

@checked function cudnnReduceTensor(handle, reduceTensorDesc, indices, indicesSizeInBytes,
                                    workspace, workspaceSizeInBytes, alpha, aDesc, A, beta,
                                    cDesc, C)
    initialize_api()
    ccall((:cudnnReduceTensor, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnReduceTensorDescriptor_t, Ptr{Cvoid}, Csize_t,
                    CuPtr{Cvoid}, Csize_t, Ptr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}),
                   handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace,
                   workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C)
end

@checked function cudnnSetTensor(handle, yDesc, y, valuePtr)
    initialize_api()
    ccall((:cudnnSetTensor, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}),
                   handle, yDesc, y, valuePtr)
end

@checked function cudnnScaleTensor(handle, yDesc, y, alpha)
    initialize_api()
    ccall((:cudnnScaleTensor, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid}),
                   handle, yDesc, y, alpha)
end

@checked function cudnnCreateFilterDescriptor(filterDesc)
    initialize_api()
    ccall((:cudnnCreateFilterDescriptor, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnFilterDescriptor_t},),
                   filterDesc)
end

@checked function cudnnSetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w)
    initialize_api()
    ccall((:cudnnSetFilter4dDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnFilterDescriptor_t, cudnnDataType_t, cudnnTensorFormat_t, Cint,
                    Cint, Cint, Cint),
                   filterDesc, dataType, format, k, c, h, w)
end

@checked function cudnnGetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w)
    initialize_api()
    ccall((:cudnnGetFilter4dDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnFilterDescriptor_t, Ptr{cudnnDataType_t},
                    Ptr{cudnnTensorFormat_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                   filterDesc, dataType, format, k, c, h, w)
end

@checked function cudnnSetFilterNdDescriptor(filterDesc, dataType, format, nbDims,
                                             filterDimA)
    initialize_api()
    ccall((:cudnnSetFilterNdDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnFilterDescriptor_t, cudnnDataType_t, cudnnTensorFormat_t, Cint,
                    Ptr{Cint}),
                   filterDesc, dataType, format, nbDims, filterDimA)
end

@checked function cudnnGetFilterNdDescriptor(filterDesc, nbDimsRequested, dataType, format,
                                             nbDims, filterDimA)
    initialize_api()
    ccall((:cudnnGetFilterNdDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnFilterDescriptor_t, Cint, Ptr{cudnnDataType_t},
                    Ptr{cudnnTensorFormat_t}, Ptr{Cint}, Ptr{Cint}),
                   filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA)
end

@checked function cudnnGetFilterSizeInBytes(filterDesc, size)
    initialize_api()
    ccall((:cudnnGetFilterSizeInBytes, libcudnn()), cudnnStatus_t,
                   (cudnnFilterDescriptor_t, Ptr{Csize_t}),
                   filterDesc, size)
end

@checked function cudnnTransformFilter(handle, transDesc, alpha, srcDesc, srcData, beta,
                                       destDesc, destData)
    initialize_api()
    ccall((:cudnnTransformFilter, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnTensorTransformDescriptor_t, Ptr{Cvoid},
                    cudnnFilterDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid},
                    cudnnFilterDescriptor_t, CuPtr{Cvoid}),
                   handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData)
end

@checked function cudnnDestroyFilterDescriptor(filterDesc)
    initialize_api()
    ccall((:cudnnDestroyFilterDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnFilterDescriptor_t,),
                   filterDesc)
end

@checked function cudnnReorderFilterAndBias(handle, filterDesc, reorderType, filterData,
                                            reorderedFilterData, reorderBias, biasData,
                                            reorderedBiasData)
    initialize_api()
    ccall((:cudnnReorderFilterAndBias, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnReorderType_t,
                    CuPtr{Cvoid}, CuPtr{Cvoid}, Cint, CuPtr{Cvoid}, CuPtr{Cvoid}),
                   handle, filterDesc, reorderType, filterData, reorderedFilterData,
                   reorderBias, biasData, reorderedBiasData)
end

@checked function cudnnCreateConvolutionDescriptor(convDesc)
    initialize_api()
    ccall((:cudnnCreateConvolutionDescriptor, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnConvolutionDescriptor_t},),
                   convDesc)
end

@checked function cudnnSetConvolutionMathType(convDesc, mathType)
    initialize_api()
    ccall((:cudnnSetConvolutionMathType, libcudnn()), cudnnStatus_t,
                   (cudnnConvolutionDescriptor_t, cudnnMathType_t),
                   convDesc, mathType)
end

@checked function cudnnGetConvolutionMathType(convDesc, mathType)
    initialize_api()
    ccall((:cudnnGetConvolutionMathType, libcudnn()), cudnnStatus_t,
                   (cudnnConvolutionDescriptor_t, Ptr{cudnnMathType_t}),
                   convDesc, mathType)
end

@checked function cudnnSetConvolutionGroupCount(convDesc, groupCount)
    initialize_api()
    ccall((:cudnnSetConvolutionGroupCount, libcudnn()), cudnnStatus_t,
                   (cudnnConvolutionDescriptor_t, Cint),
                   convDesc, groupCount)
end

@checked function cudnnGetConvolutionGroupCount(convDesc, groupCount)
    initialize_api()
    ccall((:cudnnGetConvolutionGroupCount, libcudnn()), cudnnStatus_t,
                   (cudnnConvolutionDescriptor_t, Ptr{Cint}),
                   convDesc, groupCount)
end

@checked function cudnnSetConvolutionReorderType(convDesc, reorderType)
    initialize_api()
    ccall((:cudnnSetConvolutionReorderType, libcudnn()), cudnnStatus_t,
                   (cudnnConvolutionDescriptor_t, cudnnReorderType_t),
                   convDesc, reorderType)
end

@checked function cudnnGetConvolutionReorderType(convDesc, reorderType)
    initialize_api()
    ccall((:cudnnGetConvolutionReorderType, libcudnn()), cudnnStatus_t,
                   (cudnnConvolutionDescriptor_t, Ptr{cudnnReorderType_t}),
                   convDesc, reorderType)
end

@checked function cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, dilation_h,
                                                  dilation_w, mode, computeType)
    initialize_api()
    ccall((:cudnnSetConvolution2dDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnConvolutionDescriptor_t, Cint, Cint, Cint, Cint, Cint, Cint,
                    cudnnConvolutionMode_t, cudnnDataType_t),
                   convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType)
end

@checked function cudnnGetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, dilation_h,
                                                  dilation_w, mode, computeType)
    initialize_api()
    ccall((:cudnnGetConvolution2dDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnConvolutionDescriptor_t, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{cudnnConvolutionMode_t},
                    Ptr{cudnnDataType_t}),
                   convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType)
end

@checked function cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensorDesc,
                                                        filterDesc, n, c, h, w)
    initialize_api()
    ccall((:cudnnGetConvolution2dForwardOutputDim, libcudnn()), cudnnStatus_t,
                   (cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnFilterDescriptor_t, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                   convDesc, inputTensorDesc, filterDesc, n, c, h, w)
end

@checked function cudnnSetConvolutionNdDescriptor(convDesc, arrayLength, padA,
                                                  filterStrideA, dilationA, mode,
                                                  computeType)
    initialize_api()
    ccall((:cudnnSetConvolutionNdDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnConvolutionDescriptor_t, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                    cudnnConvolutionMode_t, cudnnDataType_t),
                   convDesc, arrayLength, padA, filterStrideA, dilationA, mode,
                   computeType)
end

@checked function cudnnGetConvolutionNdDescriptor(convDesc, arrayLengthRequested,
                                                  arrayLength, padA, strideA, dilationA,
                                                  mode, computeType)
    initialize_api()
    ccall((:cudnnGetConvolutionNdDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnConvolutionDescriptor_t, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                    Ptr{Cint}, Ptr{cudnnConvolutionMode_t}, Ptr{cudnnDataType_t}),
                   convDesc, arrayLengthRequested, arrayLength, padA, strideA, dilationA,
                   mode, computeType)
end

@checked function cudnnGetConvolutionNdForwardOutputDim(convDesc, inputTensorDesc,
                                                        filterDesc, nbDims, tensorOuputDimA)
    initialize_api()
    ccall((:cudnnGetConvolutionNdForwardOutputDim, libcudnn()), cudnnStatus_t,
                   (cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnFilterDescriptor_t, Cint, Ptr{Cint}),
                   convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA)
end

@checked function cudnnDestroyConvolutionDescriptor(convDesc)
    initialize_api()
    ccall((:cudnnDestroyConvolutionDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnConvolutionDescriptor_t,),
                   convDesc)
end

@checked function cudnnGetConvolutionForwardAlgorithmMaxCount(handle, count)
    initialize_api()
    ccall((:cudnnGetConvolutionForwardAlgorithmMaxCount, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, Ptr{Cint}),
                   handle, count)
end

@checked function cudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc,
                                                       yDesc, requestedAlgoCount,
                                                       returnedAlgoCount, perfResults)
    initialize_api()
    ccall((:cudnnFindConvolutionForwardAlgorithm, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t,
                    cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint},
                    Ptr{cudnnConvolutionFwdAlgoPerf_t}),
                   handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount,
                   returnedAlgoCount, perfResults)
end

@checked function cudnnFindConvolutionForwardAlgorithmEx(handle, xDesc, x, wDesc, w,
                                                         convDesc, yDesc, y,
                                                         requestedAlgoCount,
                                                         returnedAlgoCount, perfResults,
                                                         workSpace, workSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnFindConvolutionForwardAlgorithmEx, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnConvolutionDescriptor_t,
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, Cint, Ptr{Cint},
                    Ptr{cudnnConvolutionFwdAlgoPerf_t}, CuPtr{Cvoid}, Csize_t),
                   handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestedAlgoCount,
                   returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
end

@checked function cudnnGetConvolutionForwardAlgorithm_v7(handle, srcDesc, filterDesc,
                                                         convDesc, destDesc,
                                                         requestedAlgoCount,
                                                         returnedAlgoCount, perfResults)
    initialize_api()
    ccall((:cudnnGetConvolutionForwardAlgorithm_v7, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t,
                    cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint},
                    Ptr{cudnnConvolutionFwdAlgoPerf_t}),
                   handle, srcDesc, filterDesc, convDesc, destDesc, requestedAlgoCount,
                   returnedAlgoCount, perfResults)
end

@checked function cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc,
                                                          yDesc, algo, sizeInBytes)
    initialize_api()
    ccall((:cudnnGetConvolutionForwardWorkspaceSize, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t,
                    cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnConvolutionFwdAlgo_t, Ref{Csize_t}),
                   handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes)
end

@checked function cudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc,
                                          algo, workSpace, workSpaceSizeInBytes, beta,
                                          yDesc, y)
    initialize_api()
    ccall((:cudnnConvolutionForward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnConvolutionDescriptor_t,
                    cudnnConvolutionFwdAlgo_t, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}),
                   handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace,
                   workSpaceSizeInBytes, beta, yDesc, y)
end

@checked function cudnnConvolutionBiasActivationForward(handle, alpha1, xDesc, x, wDesc, w,
                                                        convDesc, algo, workSpace,
                                                        workSpaceSizeInBytes, alpha2,
                                                        zDesc, z, biasDesc, bias,
                                                        activationDesc, yDesc, y)
    initialize_api()
    ccall((:cudnnConvolutionBiasActivationForward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnConvolutionDescriptor_t,
                    cudnnConvolutionFwdAlgo_t, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnActivationDescriptor_t, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}),
                   handle, alpha1, xDesc, x, wDesc, w, convDesc, algo, workSpace,
                   workSpaceSizeInBytes, alpha2, zDesc, z, biasDesc, bias, activationDesc,
                   yDesc, y)
end

@checked function cudnnConvolutionBackwardBias(handle, alpha, dyDesc, dy, beta, dbDesc, db)
    initialize_api()
    ccall((:cudnnConvolutionBackwardBias, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}),
                   handle, alpha, dyDesc, dy, beta, dbDesc, db)
end

@checked function cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, count)
    initialize_api()
    ccall((:cudnnGetConvolutionBackwardFilterAlgorithmMaxCount, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, Ptr{Cint}),
                   handle, count)
end

@checked function cudnnFindConvolutionBackwardFilterAlgorithm(handle, xDesc, dyDesc,
                                                              convDesc, dwDesc,
                                                              requestedAlgoCount,
                                                              returnedAlgoCount, perfResults)
    initialize_api()
    ccall((:cudnnFindConvolutionBackwardFilterAlgorithm, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, Cint, Ptr{Cint},
                    Ptr{cudnnConvolutionBwdFilterAlgoPerf_t}),
                   handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount,
                   returnedAlgoCount, perfResults)
end

@checked function cudnnFindConvolutionBackwardFilterAlgorithmEx(handle, xDesc, x, dyDesc,
                                                                y, convDesc, dwDesc, dw,
                                                                requestedAlgoCount,
                                                                returnedAlgoCount,
                                                                perfResults, workSpace,
                                                                workSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnFindConvolutionBackwardFilterAlgorithmEx, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnConvolutionDescriptor_t,
                    cudnnFilterDescriptor_t, CuPtr{Cvoid}, Cint, Ptr{Cint},
                    Ptr{cudnnConvolutionBwdFilterAlgoPerf_t}, CuPtr{Cvoid}, Csize_t),
                   handle, xDesc, x, dyDesc, y, convDesc, dwDesc, dw, requestedAlgoCount,
                   returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
end

@checked function cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, srcDesc, diffDesc,
                                                                convDesc, gradDesc,
                                                                requestedAlgoCount,
                                                                returnedAlgoCount,
                                                                perfResults)
    initialize_api()
    ccall((:cudnnGetConvolutionBackwardFilterAlgorithm_v7, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, Cint, Ptr{Cint},
                    Ptr{cudnnConvolutionBwdFilterAlgoPerf_t}),
                   handle, srcDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount,
                   returnedAlgoCount, perfResults)
end

@checked function cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, xDesc, dyDesc,
                                                                 convDesc, gradDesc, algo,
                                                                 sizeInBytes)
    initialize_api()
    ccall((:cudnnGetConvolutionBackwardFilterWorkspaceSize, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t,
                    cudnnConvolutionBwdFilterAlgo_t, Ref{Csize_t}),
                   handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes)
end

@checked function cudnnConvolutionBackwardFilter(handle, alpha, xDesc, x, dyDesc, dy,
                                                 convDesc, algo, workSpace,
                                                 workSpaceSizeInBytes, beta, dwDesc, dw)
    initialize_api()
    ccall((:cudnnConvolutionBackwardFilter, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnConvolutionDescriptor_t,
                    cudnnConvolutionBwdFilterAlgo_t, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid},
                    cudnnFilterDescriptor_t, CuPtr{Cvoid}),
                   handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace,
                   workSpaceSizeInBytes, beta, dwDesc, dw)
end

@checked function cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, count)
    initialize_api()
    ccall((:cudnnGetConvolutionBackwardDataAlgorithmMaxCount, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, Ptr{Cint}),
                   handle, count)
end

@checked function cudnnFindConvolutionBackwardDataAlgorithm(handle, wDesc, dyDesc,
                                                            convDesc, dxDesc,
                                                            requestedAlgoCount,
                                                            returnedAlgoCount, perfResults)
    initialize_api()
    ccall((:cudnnFindConvolutionBackwardDataAlgorithm, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint},
                    Ptr{cudnnConvolutionBwdDataAlgoPerf_t}),
                   handle, wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount,
                   returnedAlgoCount, perfResults)
end

@checked function cudnnFindConvolutionBackwardDataAlgorithmEx(handle, wDesc, w, dyDesc, dy,
                                                              convDesc, dxDesc, dx,
                                                              requestedAlgoCount,
                                                              returnedAlgoCount,
                                                              perfResults, workSpace,
                                                              workSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnFindConvolutionBackwardDataAlgorithmEx, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnFilterDescriptor_t, CuPtr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnConvolutionDescriptor_t,
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, Cint, Ptr{Cint},
                    Ptr{cudnnConvolutionBwdDataAlgoPerf_t}, CuPtr{Cvoid}, Csize_t),
                   handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx, requestedAlgoCount,
                   returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
end

@checked function cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, filterDesc, diffDesc,
                                                              convDesc, gradDesc,
                                                              requestedAlgoCount,
                                                              returnedAlgoCount, perfResults)
    initialize_api()
    ccall((:cudnnGetConvolutionBackwardDataAlgorithm_v7, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint},
                    Ptr{cudnnConvolutionBwdDataAlgoPerf_t}),
                   handle, filterDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount,
                   returnedAlgoCount, perfResults)
end

@checked function cudnnGetConvolutionBackwardDataWorkspaceSize(handle, wDesc, dyDesc,
                                                               convDesc, dxDesc, algo,
                                                               sizeInBytes)
    initialize_api()
    ccall((:cudnnGetConvolutionBackwardDataWorkspaceSize, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnConvolutionBwdDataAlgo_t, Ref{Csize_t}),
                   handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes)
end

@checked function cudnnConvolutionBackwardData(handle, alpha, wDesc, w, dyDesc, dy,
                                               convDesc, algo, workSpace,
                                               workSpaceSizeInBytes, beta, dxDesc, dx)
    initialize_api()
    ccall((:cudnnConvolutionBackwardData, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, Ptr{Cvoid}, cudnnFilterDescriptor_t, CuPtr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnConvolutionDescriptor_t,
                    cudnnConvolutionBwdDataAlgo_t, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}),
                   handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace,
                   workSpaceSizeInBytes, beta, dxDesc, dx)
end

@checked function cudnnIm2Col(handle, xDesc, x, wDesc, convDesc, colBuffer)
    initialize_api()
    ccall((:cudnnIm2Col, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnFilterDescriptor_t, cudnnConvolutionDescriptor_t, CuPtr{Cvoid}),
                   handle, xDesc, x, wDesc, convDesc, colBuffer)
end

@checked function cudnnSoftmaxForward(handle, algo, mode, alpha, xDesc, x, beta, yDesc, y)
    initialize_api()
    ccall((:cudnnSoftmaxForward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t,
                    Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}),
                   handle, algo, mode, alpha, xDesc, x, beta, yDesc, y)
end

@checked function cudnnSoftmaxBackward(handle, algo, mode, alpha, yDesc, y, dyDesc, dy,
                                       beta, dxDesc, dx)
    initialize_api()
    ccall((:cudnnSoftmaxBackward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t,
                    Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}),
                   handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx)
end

@checked function cudnnCreatePoolingDescriptor(poolingDesc)
    initialize_api()
    ccall((:cudnnCreatePoolingDescriptor, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnPoolingDescriptor_t},),
                   poolingDesc)
end

@checked function cudnnSetPooling2dDescriptor(poolingDesc, mode, maxpoolingNanOpt,
                                              windowHeight, windowWidth, verticalPadding,
                                              horizontalPadding, verticalStride,
                                              horizontalStride)
    initialize_api()
    ccall((:cudnnSetPooling2dDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnNanPropagation_t,
                    Cint, Cint, Cint, Cint, Cint, Cint),
                   poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth,
                   verticalPadding, horizontalPadding, verticalStride, horizontalStride)
end

@checked function cudnnGetPooling2dDescriptor(poolingDesc, mode, maxpoolingNanOpt,
                                              windowHeight, windowWidth, verticalPadding,
                                              horizontalPadding, verticalStride,
                                              horizontalStride)
    initialize_api()
    ccall((:cudnnGetPooling2dDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnPoolingDescriptor_t, Ptr{cudnnPoolingMode_t},
                    Ptr{cudnnNanPropagation_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}),
                   poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth,
                   verticalPadding, horizontalPadding, verticalStride, horizontalStride)
end

@checked function cudnnSetPoolingNdDescriptor(poolingDesc, mode, maxpoolingNanOpt, nbDims,
                                              windowDimA, paddingA, strideA)
    initialize_api()
    ccall((:cudnnSetPoolingNdDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnNanPropagation_t,
                    Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                   poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA,
                   strideA)
end

@checked function cudnnGetPoolingNdDescriptor(poolingDesc, nbDimsRequested, mode,
                                              maxpoolingNanOpt, nbDims, windowDimA,
                                              paddingA, strideA)
    initialize_api()
    ccall((:cudnnGetPoolingNdDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnPoolingDescriptor_t, Cint, Ptr{cudnnPoolingMode_t},
                    Ptr{cudnnNanPropagation_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                   poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt, nbDims,
                   windowDimA, paddingA, strideA)
end

@checked function cudnnGetPoolingNdForwardOutputDim(poolingDesc, inputTensorDesc, nbDims,
                                                    outputTensorDimA)
    initialize_api()
    ccall((:cudnnGetPoolingNdForwardOutputDim, libcudnn()), cudnnStatus_t,
                   (cudnnPoolingDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint}),
                   poolingDesc, inputTensorDesc, nbDims, outputTensorDimA)
end

@checked function cudnnGetPooling2dForwardOutputDim(poolingDesc, inputTensorDesc, n, c, h, w)
    initialize_api()
    ccall((:cudnnGetPooling2dForwardOutputDim, libcudnn()), cudnnStatus_t,
                   (cudnnPoolingDescriptor_t, cudnnTensorDescriptor_t, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                   poolingDesc, inputTensorDesc, n, c, h, w)
end

@checked function cudnnDestroyPoolingDescriptor(poolingDesc)
    initialize_api()
    ccall((:cudnnDestroyPoolingDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnPoolingDescriptor_t,),
                   poolingDesc)
end

@checked function cudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y)
    initialize_api()
    ccall((:cudnnPoolingForward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnPoolingDescriptor_t, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}),
                   handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y)
end

@checked function cudnnPoolingBackward(handle, poolingDesc, alpha, yDesc, y, dyDesc, dy,
                                       xDesc, x, beta, dxDesc, dx)
    initialize_api()
    ccall((:cudnnPoolingBackward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnPoolingDescriptor_t, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}),
                   handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta,
                   dxDesc, dx)
end

@checked function cudnnCreateActivationDescriptor(activationDesc)
    initialize_api()
    ccall((:cudnnCreateActivationDescriptor, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnActivationDescriptor_t},),
                   activationDesc)
end

@checked function cudnnSetActivationDescriptor(activationDesc, mode, reluNanOpt, coef)
    initialize_api()
    ccall((:cudnnSetActivationDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnActivationDescriptor_t, cudnnActivationMode_t,
                    cudnnNanPropagation_t, Cdouble),
                   activationDesc, mode, reluNanOpt, coef)
end

@checked function cudnnGetActivationDescriptor(activationDesc, mode, reluNanOpt, coef)
    initialize_api()
    ccall((:cudnnGetActivationDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnActivationDescriptor_t, Ptr{cudnnActivationMode_t},
                    Ptr{cudnnNanPropagation_t}, Ptr{Cdouble}),
                   activationDesc, mode, reluNanOpt, coef)
end

@checked function cudnnDestroyActivationDescriptor(activationDesc)
    initialize_api()
    ccall((:cudnnDestroyActivationDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnActivationDescriptor_t,),
                   activationDesc)
end

@checked function cudnnActivationForward(handle, activationDesc, alpha, xDesc, x, beta,
                                         yDesc, y)
    initialize_api()
    ccall((:cudnnActivationForward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnActivationDescriptor_t, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}),
                   handle, activationDesc, alpha, xDesc, x, beta, yDesc, y)
end

@checked function cudnnActivationBackward(handle, activationDesc, alpha, yDesc, y, dyDesc,
                                          dy, xDesc, x, beta, dxDesc, dx)
    initialize_api()
    ccall((:cudnnActivationBackward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnActivationDescriptor_t, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}),
                   handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta,
                   dxDesc, dx)
end

@checked function cudnnCreateLRNDescriptor(normDesc)
    initialize_api()
    ccall((:cudnnCreateLRNDescriptor, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnLRNDescriptor_t},),
                   normDesc)
end

@checked function cudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
    initialize_api()
    ccall((:cudnnSetLRNDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnLRNDescriptor_t, UInt32, Cdouble, Cdouble, Cdouble),
                   normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
end

@checked function cudnnGetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
    initialize_api()
    ccall((:cudnnGetLRNDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnLRNDescriptor_t, Ptr{UInt32}, Ptr{Cdouble}, Ptr{Cdouble},
                    Ptr{Cdouble}),
                   normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
end

@checked function cudnnDestroyLRNDescriptor(lrnDesc)
    initialize_api()
    ccall((:cudnnDestroyLRNDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnLRNDescriptor_t,),
                   lrnDesc)
end

@checked function cudnnLRNCrossChannelForward(handle, normDesc, lrnMode, alpha, xDesc, x,
                                              beta, yDesc, y)
    initialize_api()
    ccall((:cudnnLRNCrossChannelForward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnLRNMode_t, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}),
                   handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y)
end

@checked function cudnnLRNCrossChannelBackward(handle, normDesc, lrnMode, alpha, yDesc, y,
                                               dyDesc, dy, xDesc, x, beta, dxDesc, dx)
    initialize_api()
    ccall((:cudnnLRNCrossChannelBackward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnLRNMode_t, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}),
                   handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta,
                   dxDesc, dx)
end

@checked function cudnnDivisiveNormalizationForward(handle, normDesc, mode, alpha, xDesc,
                                                    x, means, temp, temp2, beta, yDesc, y)
    initialize_api()
    ccall((:cudnnDivisiveNormalizationForward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnDivNormMode_t, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid},
                    CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}),
                   handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta,
                   yDesc, y)
end

@checked function cudnnDivisiveNormalizationBackward(handle, normDesc, mode, alpha, xDesc,
                                                     x, means, dy, temp, temp2, beta,
                                                     dXdMeansDesc, dx, dMeans)
    initialize_api()
    ccall((:cudnnDivisiveNormalizationBackward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnDivNormMode_t, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid},
                    CuPtr{Cvoid}, CuPtr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, CuPtr{Cvoid}),
                   handle, normDesc, mode, alpha, xDesc, x, means, dy, temp, temp2, beta,
                   dXdMeansDesc, dx, dMeans)
end

@checked function cudnnDeriveBNTensorDescriptor(derivedBnDesc, xDesc, mode)
    initialize_api()
    ccall((:cudnnDeriveBNTensorDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnBatchNormMode_t),
                   derivedBnDesc, xDesc, mode)
end

@checked function cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle, mode,
                                                                           bnOps, xDesc,
                                                                           zDesc, yDesc,
                                                                           bnScaleBiasMeanVarDesc,
                                                                           activationDesc,
                                                                           sizeInBytes)
    initialize_api()
    ccall((:cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
                    cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnActivationDescriptor_t, Ref{Csize_t}),
                   handle, mode, bnOps, xDesc, zDesc, yDesc, bnScaleBiasMeanVarDesc,
                   activationDesc, sizeInBytes)
end

@checked function cudnnGetBatchNormalizationBackwardExWorkspaceSize(handle, mode, bnOps,
                                                                    xDesc, yDesc, dyDesc,
                                                                    dzDesc, dxDesc,
                                                                    dBnScaleBiasDesc,
                                                                    activationDesc,
                                                                    sizeInBytes)
    initialize_api()
    ccall((:cudnnGetBatchNormalizationBackwardExWorkspaceSize, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
                    cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnActivationDescriptor_t, Ref{Csize_t}),
                   handle, mode, bnOps, xDesc, yDesc, dyDesc, dzDesc, dxDesc,
                   dBnScaleBiasDesc, activationDesc, sizeInBytes)
end

@checked function cudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle, mode, bnOps,
                                                                       activationDesc,
                                                                       xDesc, sizeInBytes)
    initialize_api()
    ccall((:cudnnGetBatchNormalizationTrainingExReserveSpaceSize, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
                    cudnnActivationDescriptor_t, cudnnTensorDescriptor_t, Ref{Csize_t}),
                   handle, mode, bnOps, activationDesc, xDesc, sizeInBytes)
end

@checked function cudnnBatchNormalizationForwardTraining(handle, mode, alpha, beta, xDesc,
                                                         x, yDesc, y,
                                                         bnScaleBiasMeanVarDesc, bnScale,
                                                         bnBias, exponentialAverageFactor,
                                                         resultRunningMean,
                                                         resultRunningVariance, epsilon,
                                                         resultSaveMean,
                                                         resultSaveInvVariance)
    initialize_api()
    ccall((:cudnnBatchNormalizationForwardTraining, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnBatchNormMode_t, Ptr{Cvoid}, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid},
                    Cdouble, CuPtr{Cvoid}, CuPtr{Cvoid}, Cdouble, CuPtr{Cvoid},
                    CuPtr{Cvoid}),
                   handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc,
                   bnScale, bnBias, exponentialAverageFactor, resultRunningMean,
                   resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance)
end

@checked function cudnnBatchNormalizationForwardTrainingEx(handle, mode, bnOps, alpha,
                                                           beta, xDesc, xData, zDesc,
                                                           zData, yDesc, yData,
                                                           bnScaleBiasMeanVarDesc, bnScale,
                                                           bnBias,
                                                           exponentialAverageFactor, resultRunningMean,
                                                           resultRunningVariance, epsilon,
                                                           resultSaveMean,
                                                           resultSaveInvVariance,
                                                           activationDesc, workspace,
                                                           workSpaceSizeInBytes,
                                                           reserveSpace,
                                                           reserveSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnBatchNormalizationForwardTrainingEx, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, Ptr{Cvoid},
                    Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid},
                    Cdouble, CuPtr{Cvoid}, CuPtr{Cvoid}, Cdouble, CuPtr{Cvoid},
                    CuPtr{Cvoid}, cudnnActivationDescriptor_t, CuPtr{Cvoid}, Csize_t,
                    CuPtr{Cvoid}, Csize_t),
                   handle, mode, bnOps, alpha, beta, xDesc, xData, zDesc, zData, yDesc,
                   yData, bnScaleBiasMeanVarDesc, bnScale, bnBias,
                   exponentialAverageFactor, resultRunningMean, resultRunningVariance,
                   epsilon, resultSaveMean, resultSaveInvVariance, activationDesc,
                   workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
end

@checked function cudnnBatchNormalizationForwardInference(handle, mode, alpha, beta, xDesc,
                                                          x, yDesc, y,
                                                          bnScaleBiasMeanVarDesc, bnScale,
                                                          bnBias, estimatedMean,
                                                          estimatedVariance, epsilon)
    initialize_api()
    ccall((:cudnnBatchNormalizationForwardInference, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnBatchNormMode_t, Ptr{Cvoid}, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid},
                    CuPtr{Cvoid}, CuPtr{Cvoid}, Cdouble),
                   handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc,
                   bnScale, bnBias, estimatedMean, estimatedVariance, epsilon)
end

@checked function cudnnBatchNormalizationBackward(handle, mode, alphaDataDiff,
                                                  betaDataDiff, alphaParamDiff,
                                                  betaParamDiff, xDesc, x, dyDesc, dy,
                                                  dxDesc, dx, dBnScaleBiasDesc, bnScale,
                                                  dBnScaleResult, dBnBiasResult, epsilon,
                                                  savedMean, savedInvVariance)
    initialize_api()
    ccall((:cudnnBatchNormalizationBackward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnBatchNormMode_t, Ptr{Cvoid}, Ptr{Cvoid},
                    Ptr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid},
                    CuPtr{Cvoid}, Cdouble, CuPtr{Cvoid}, CuPtr{Cvoid}),
                   handle, mode, alphaDataDiff, betaDataDiff, alphaParamDiff,
                   betaParamDiff, xDesc, x, dyDesc, dy, dxDesc, dx, dBnScaleBiasDesc,
                   bnScale, dBnScaleResult, dBnBiasResult, epsilon, savedMean,
                   savedInvVariance)
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
    initialize_api()
    ccall((:cudnnBatchNormalizationBackwardEx, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, Ptr{Cvoid},
                    Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid},
                    CuPtr{Cvoid}, Cdouble, CuPtr{Cvoid}, CuPtr{Cvoid},
                    cudnnActivationDescriptor_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid},
                    Csize_t),
                   handle, mode, bnOps, alphaDataDiff, betaDataDiff, alphaParamDiff,
                   betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc,
                   dzData, dxDesc, dxData, dBnScaleBiasDesc, bnScaleData, bnBiasData,
                   dBnScaleData, dBnBiasData, epsilon, savedMean, savedInvVariance,
                   activationDesc, workSpace, workSpaceSizeInBytes, reserveSpace,
                   reserveSpaceSizeInBytes)
end

@checked function cudnnCreateSpatialTransformerDescriptor(stDesc)
    initialize_api()
    ccall((:cudnnCreateSpatialTransformerDescriptor, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnSpatialTransformerDescriptor_t},),
                   stDesc)
end

@checked function cudnnSetSpatialTransformerNdDescriptor(stDesc, samplerType, dataType,
                                                         nbDims, dimA)
    initialize_api()
    ccall((:cudnnSetSpatialTransformerNdDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnSpatialTransformerDescriptor_t, cudnnSamplerType_t,
                    cudnnDataType_t, Cint, Ptr{Cint}),
                   stDesc, samplerType, dataType, nbDims, dimA)
end

@checked function cudnnDestroySpatialTransformerDescriptor(stDesc)
    initialize_api()
    ccall((:cudnnDestroySpatialTransformerDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnSpatialTransformerDescriptor_t,),
                   stDesc)
end

@checked function cudnnSpatialTfGridGeneratorForward(handle, stDesc, theta, grid)
    initialize_api()
    ccall((:cudnnSpatialTfGridGeneratorForward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, CuPtr{Cvoid},
                    CuPtr{Cvoid}),
                   handle, stDesc, theta, grid)
end

@checked function cudnnSpatialTfGridGeneratorBackward(handle, stDesc, dgrid, dtheta)
    initialize_api()
    ccall((:cudnnSpatialTfGridGeneratorBackward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, CuPtr{Cvoid},
                    CuPtr{Cvoid}),
                   handle, stDesc, dgrid, dtheta)
end

@checked function cudnnSpatialTfSamplerForward(handle, stDesc, alpha, xDesc, x, grid, beta,
                                               yDesc, y)
    initialize_api()
    ccall((:cudnnSpatialTfSamplerForward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}),
                   handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y)
end

@checked function cudnnSpatialTfSamplerBackward(handle, stDesc, alpha, xDesc, x, beta,
                                                dxDesc, dx, alphaDgrid, dyDesc, dy, grid,
                                                betaDgrid, dgrid)
    initialize_api()
    ccall((:cudnnSpatialTfSamplerBackward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Ptr{Cvoid},
                    CuPtr{Cvoid}),
                   handle, stDesc, alpha, xDesc, x, beta, dxDesc, dx, alphaDgrid, dyDesc,
                   dy, grid, betaDgrid, dgrid)
end

@checked function cudnnCreateDropoutDescriptor(dropoutDesc)
    initialize_api()
    ccall((:cudnnCreateDropoutDescriptor, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnDropoutDescriptor_t},),
                   dropoutDesc)
end

@checked function cudnnDestroyDropoutDescriptor(dropoutDesc)
    initialize_api()
    ccall((:cudnnDestroyDropoutDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnDropoutDescriptor_t,),
                   dropoutDesc)
end

@checked function cudnnDropoutGetStatesSize(handle, sizeInBytes)
    initialize_api()
    ccall((:cudnnDropoutGetStatesSize, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, Ptr{Csize_t}),
                   handle, sizeInBytes)
end

@checked function cudnnDropoutGetReserveSpaceSize(xdesc, sizeInBytes)
    initialize_api()
    ccall((:cudnnDropoutGetReserveSpaceSize, libcudnn()), cudnnStatus_t,
                   (cudnnTensorDescriptor_t, Ref{Csize_t}),
                   xdesc, sizeInBytes)
end

@checked function cudnnSetDropoutDescriptor(dropoutDesc, handle, dropout, states,
                                            stateSizeInBytes, seed)
    initialize_api()
    ccall((:cudnnSetDropoutDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnDropoutDescriptor_t, cudnnHandle_t, Cfloat, CuPtr{Cvoid},
                    Csize_t, Culonglong),
                   dropoutDesc, handle, dropout, states, stateSizeInBytes, seed)
end

@checked function cudnnRestoreDropoutDescriptor(dropoutDesc, handle, dropout, states,
                                                stateSizeInBytes, seed)
    initialize_api()
    ccall((:cudnnRestoreDropoutDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnDropoutDescriptor_t, cudnnHandle_t, Cfloat, CuPtr{Cvoid},
                    Csize_t, Culonglong),
                   dropoutDesc, handle, dropout, states, stateSizeInBytes, seed)
end

@checked function cudnnGetDropoutDescriptor(dropoutDesc, handle, dropout, states, seed)
    initialize_api()
    ccall((:cudnnGetDropoutDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnDropoutDescriptor_t, cudnnHandle_t, Ptr{Cfloat},
                    Ptr{CuPtr{Cvoid}}, Ptr{Culonglong}),
                   dropoutDesc, handle, dropout, states, seed)
end

@checked function cudnnDropoutForward(handle, dropoutDesc, xdesc, x, ydesc, y,
                                      reserveSpace, reserveSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnDropoutForward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnDropoutDescriptor_t, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid},
                    Csize_t),
                   handle, dropoutDesc, xdesc, x, ydesc, y, reserveSpace,
                   reserveSpaceSizeInBytes)
end

@checked function cudnnDropoutBackward(handle, dropoutDesc, dydesc, dy, dxdesc, dx,
                                       reserveSpace, reserveSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnDropoutBackward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnDropoutDescriptor_t, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid},
                    Csize_t),
                   handle, dropoutDesc, dydesc, dy, dxdesc, dx, reserveSpace,
                   reserveSpaceSizeInBytes)
end

@checked function cudnnCreateRNNDescriptor(rnnDesc)
    initialize_api()
    ccall((:cudnnCreateRNNDescriptor, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnRNNDescriptor_t},),
                   rnnDesc)
end

@checked function cudnnDestroyRNNDescriptor(rnnDesc)
    initialize_api()
    ccall((:cudnnDestroyRNNDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnRNNDescriptor_t,),
                   rnnDesc)
end

@checked function cudnnSetRNNMatrixMathType(rnnDesc, mType)
    initialize_api()
    ccall((:cudnnSetRNNMatrixMathType, libcudnn()), cudnnStatus_t,
                   (cudnnRNNDescriptor_t, cudnnMathType_t),
                   rnnDesc, mType)
end

@checked function cudnnGetRNNMatrixMathType(rnnDesc, mType)
    initialize_api()
    ccall((:cudnnGetRNNMatrixMathType, libcudnn()), cudnnStatus_t,
                   (cudnnRNNDescriptor_t, Ptr{cudnnMathType_t}),
                   rnnDesc, mType)
end

@checked function cudnnSetRNNBiasMode(rnnDesc, biasMode)
    initialize_api()
    ccall((:cudnnSetRNNBiasMode, libcudnn()), cudnnStatus_t,
                   (cudnnRNNDescriptor_t, cudnnRNNBiasMode_t),
                   rnnDesc, biasMode)
end

@checked function cudnnGetRNNBiasMode(rnnDesc, biasMode)
    initialize_api()
    ccall((:cudnnGetRNNBiasMode, libcudnn()), cudnnStatus_t,
                   (cudnnRNNDescriptor_t, Ptr{cudnnRNNBiasMode_t}),
                   rnnDesc, biasMode)
end

@checked function cudnnRNNSetClip(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip)
    initialize_api()
    ccall((:cudnnRNNSetClip, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnRNNClipMode_t,
                    cudnnNanPropagation_t, Cdouble, Cdouble),
                   handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip)
end

@checked function cudnnRNNGetClip(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip)
    initialize_api()
    ccall((:cudnnRNNGetClip, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{cudnnRNNClipMode_t},
                    Ptr{cudnnNanPropagation_t}, Ptr{Cdouble}, Ptr{Cdouble}),
                   handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip)
end

@checked function cudnnSetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize)
    initialize_api()
    ccall((:cudnnSetRNNProjectionLayers, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Cint),
                   handle, rnnDesc, recProjSize, outProjSize)
end

@checked function cudnnGetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize)
    initialize_api()
    ccall((:cudnnGetRNNProjectionLayers, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{Cint}, Ptr{Cint}),
                   handle, rnnDesc, recProjSize, outProjSize)
end

@checked function cudnnCreatePersistentRNNPlan(rnnDesc, minibatch, dataType, plan)
    initialize_api()
    ccall((:cudnnCreatePersistentRNNPlan, libcudnn()), cudnnStatus_t,
                   (cudnnRNNDescriptor_t, Cint, cudnnDataType_t,
                    Ptr{cudnnPersistentRNNPlan_t}),
                   rnnDesc, minibatch, dataType, plan)
end

@checked function cudnnDestroyPersistentRNNPlan(plan)
    initialize_api()
    ccall((:cudnnDestroyPersistentRNNPlan, libcudnn()), cudnnStatus_t,
                   (cudnnPersistentRNNPlan_t,),
                   plan)
end

@checked function cudnnSetPersistentRNNPlan(rnnDesc, plan)
    initialize_api()
    ccall((:cudnnSetPersistentRNNPlan, libcudnn()), cudnnStatus_t,
                   (cudnnRNNDescriptor_t, cudnnPersistentRNNPlan_t),
                   rnnDesc, plan)
end

@checked function cudnnGetRNNWorkspaceSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes)
    initialize_api()
    ccall((:cudnnGetRNNWorkspaceSize, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Cint,
                    Ptr{cudnnTensorDescriptor_t}, Ref{Csize_t}),
                   handle, rnnDesc, seqLength, xDesc, sizeInBytes)
end

@checked function cudnnGetRNNTrainingReserveSize(handle, rnnDesc, seqLength, xDesc,
                                                 sizeInBytes)
    initialize_api()
    ccall((:cudnnGetRNNTrainingReserveSize, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Cint,
                    Ptr{cudnnTensorDescriptor_t}, Ref{Csize_t}),
                   handle, rnnDesc, seqLength, xDesc, sizeInBytes)
end

@checked function cudnnGetRNNParamsSize(handle, rnnDesc, xDesc, sizeInBytes, dataType)
    initialize_api()
    ccall((:cudnnGetRNNParamsSize, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnTensorDescriptor_t,
                    Ref{Csize_t}, cudnnDataType_t),
                   handle, rnnDesc, xDesc, sizeInBytes, dataType)
end

@checked function cudnnGetRNNLinLayerMatrixParams(handle, rnnDesc, pseudoLayer, xDesc,
                                                  wDesc, w, linLayerID, linLayerMatDesc,
                                                  linLayerMat)
    initialize_api()
    ccall((:cudnnGetRNNLinLayerMatrixParams, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, cudnnTensorDescriptor_t,
                    cudnnFilterDescriptor_t, CuPtr{Cvoid}, Cint, cudnnFilterDescriptor_t,
                    Ptr{Ptr{Cvoid}}),
                   handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID,
                   linLayerMatDesc, linLayerMat)
end

@checked function cudnnGetRNNLinLayerBiasParams(handle, rnnDesc, pseudoLayer, xDesc, wDesc,
                                                w, linLayerID, linLayerBiasDesc,
                                                linLayerBias)
    initialize_api()
    ccall((:cudnnGetRNNLinLayerBiasParams, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, cudnnTensorDescriptor_t,
                    cudnnFilterDescriptor_t, CuPtr{Cvoid}, Cint, cudnnFilterDescriptor_t,
                    Ptr{Ptr{Cvoid}}),
                   handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID,
                   linLayerBiasDesc, linLayerBias)
end

@checked function cudnnRNNForwardInference(handle, rnnDesc, seqLength, xDesc, x, hxDesc,
                                           hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy,
                                           cyDesc, cy, workspace, workSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnRNNForwardInference, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Cint,
                    Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnFilterDescriptor_t, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t},
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t),
                   handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w,
                   yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes)
end

@checked function cudnnRNNForwardTraining(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx,
                                          cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy,
                                          cyDesc, cy, workspace, workSpaceSizeInBytes,
                                          reserveSpace, reserveSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnRNNForwardTraining, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Cint,
                    Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnFilterDescriptor_t, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t},
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t,
                    CuPtr{Cvoid}, Csize_t),
                   handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w,
                   yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes,
                   reserveSpace, reserveSpaceSizeInBytes)
end

@checked function cudnnRNNBackwardData(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy,
                                       dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx,
                                       cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx,
                                       workspace, workSpaceSizeInBytes, reserveSpace,
                                       reserveSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnRNNBackwardData, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Cint,
                    Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid},
                    Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid},
                    Csize_t, CuPtr{Cvoid}, Csize_t),
                   handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc,
                   dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx,
                   dcxDesc, dcx, workspace, workSpaceSizeInBytes, reserveSpace,
                   reserveSpaceSizeInBytes)
end

@checked function cudnnRNNBackwardWeights(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx,
                                          yDesc, y, workspace, workSpaceSizeInBytes,
                                          dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnRNNBackwardWeights, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Cint,
                    Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, CuPtr{Cvoid},
                    Csize_t, cudnnFilterDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t),
                   handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, workspace,
                   workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes)
end

@checked function cudnnSetRNNPaddingMode(rnnDesc, paddingMode)
    initialize_api()
    ccall((:cudnnSetRNNPaddingMode, libcudnn()), cudnnStatus_t,
                   (cudnnRNNDescriptor_t, cudnnRNNPaddingMode_t),
                   rnnDesc, paddingMode)
end

@checked function cudnnGetRNNPaddingMode(rnnDesc, paddingMode)
    initialize_api()
    ccall((:cudnnGetRNNPaddingMode, libcudnn()), cudnnStatus_t,
                   (cudnnRNNDescriptor_t, Ptr{cudnnRNNPaddingMode_t}),
                   rnnDesc, paddingMode)
end

@checked function cudnnCreateRNNDataDescriptor(rnnDataDesc)
    initialize_api()
    ccall((:cudnnCreateRNNDataDescriptor, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnRNNDataDescriptor_t},),
                   rnnDataDesc)
end

@checked function cudnnDestroyRNNDataDescriptor(rnnDataDesc)
    initialize_api()
    ccall((:cudnnDestroyRNNDataDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnRNNDataDescriptor_t,),
                   rnnDataDesc)
end

@checked function cudnnSetRNNDataDescriptor(rnnDataDesc, dataType, layout, maxSeqLength,
                                            batchSize, vectorSize, seqLengthArray,
                                            paddingFill)
    initialize_api()
    ccall((:cudnnSetRNNDataDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnRNNDataDescriptor_t, cudnnDataType_t, cudnnRNNDataLayout_t, Cint,
                    Cint, Cint, Ptr{Cint}, Ptr{Cvoid}),
                   rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize,
                   seqLengthArray, paddingFill)
end

@checked function cudnnGetRNNDataDescriptor(rnnDataDesc, dataType, layout, maxSeqLength,
                                            batchSize, vectorSize, arrayLengthRequested,
                                            seqLengthArray, paddingFill)
    initialize_api()
    ccall((:cudnnGetRNNDataDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnRNNDataDescriptor_t, Ptr{cudnnDataType_t},
                    Ptr{cudnnRNNDataLayout_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint,
                    Ptr{Cint}, Ptr{Cvoid}),
                   rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize,
                   arrayLengthRequested, seqLengthArray, paddingFill)
end

@checked function cudnnRNNForwardTrainingEx(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc,
                                            cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy,
                                            kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc,
                                            queries, workSpace, workSpaceSizeInBytes,
                                            reserveSpace, reserveSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnRNNForwardTrainingEx, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnRNNDataDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnFilterDescriptor_t,
                    CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid},
                    cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t,
                    CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid},
                    Csize_t, CuPtr{Cvoid}, Csize_t),
                   handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y,
                   hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc,
                   queries, workSpace, workSpaceSizeInBytes, reserveSpace,
                   reserveSpaceSizeInBytes)
end

@checked function cudnnRNNForwardInferenceEx(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc,
                                             cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc,
                                             cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn,
                                             qDesc, queries, workSpace, workSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnRNNForwardInferenceEx, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnRNNDataDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnFilterDescriptor_t,
                    CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, Ptr{Cvoid},
                    cudnnRNNDataDescriptor_t, Ptr{Cvoid}, cudnnRNNDataDescriptor_t,
                    Ptr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid},
                    Csize_t),
                   handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y,
                   hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc,
                   queries, workSpace, workSpaceSizeInBytes)
end

@checked function cudnnRNNBackwardDataEx(handle, rnnDesc, yDesc, y, dyDesc, dy, dcDesc,
                                         dcAttn, dhyDesc, dhy, dcyDesc, dcy, wDesc, w,
                                         hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx,
                                         dcxDesc, dcx, dkDesc, dkeys, workSpace,
                                         workSpaceSizeInBytes, reserveSpace,
                                         reserveSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnRNNBackwardDataEx, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnRNNDataDescriptor_t,
                    CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid},
                    cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnRNNDataDescriptor_t, Ptr{Cvoid}, CuPtr{Cvoid}, Csize_t,
                    CuPtr{Cvoid}, Csize_t),
                   handle, rnnDesc, yDesc, y, dyDesc, dy, dcDesc, dcAttn, dhyDesc, dhy,
                   dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc,
                   dhx, dcxDesc, dcx, dkDesc, dkeys, workSpace, workSpaceSizeInBytes,
                   reserveSpace, reserveSpaceSizeInBytes)
end

@checked function cudnnRNNBackwardWeightsEx(handle, rnnDesc, xDesc, x, hxDesc, hx, yDesc,
                                            y, workSpace, workSpaceSizeInBytes, dwDesc, dw,
                                            reserveSpace, reserveSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnRNNBackwardWeightsEx, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnRNNDataDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t,
                    cudnnFilterDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t),
                   handle, rnnDesc, xDesc, x, hxDesc, hx, yDesc, y, workSpace,
                   workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes)
end

@checked function cudnnSetRNNAlgorithmDescriptor(handle, rnnDesc, algoDesc)
    initialize_api()
    ccall((:cudnnSetRNNAlgorithmDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnAlgorithmDescriptor_t),
                   handle, rnnDesc, algoDesc)
end

@checked function cudnnGetRNNForwardInferenceAlgorithmMaxCount(handle, rnnDesc, count)
    initialize_api()
    ccall((:cudnnGetRNNForwardInferenceAlgorithmMaxCount, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{Cint}),
                   handle, rnnDesc, count)
end

@checked function cudnnFindRNNForwardInferenceAlgorithmEx(handle, rnnDesc, seqLength,
                                                          xDesc, x, hxDesc, hx, cxDesc, cx,
                                                          wDesc, w, yDesc, y, hyDesc, hy,
                                                          cyDesc, cy, findIntensity,
                                                          requestedAlgoCount,
                                                          returnedAlgoCount, perfResults,
                                                          workspace, workSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnFindRNNForwardInferenceAlgorithmEx, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Cint,
                    Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnFilterDescriptor_t, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t},
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, Cfloat, Cint, Ptr{Cint},
                    Ptr{cudnnAlgorithmPerformance_t}, CuPtr{Cvoid}, Csize_t),
                   handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w,
                   yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount,
                   returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes)
end

@checked function cudnnGetRNNForwardTrainingAlgorithmMaxCount(handle, rnnDesc, count)
    initialize_api()
    ccall((:cudnnGetRNNForwardTrainingAlgorithmMaxCount, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{Cint}),
                   handle, rnnDesc, count)
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
    initialize_api()
    ccall((:cudnnFindRNNForwardTrainingAlgorithmEx, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Cint,
                    Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnFilterDescriptor_t, CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t},
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnTensorDescriptor_t, CuPtr{Cvoid}, Cfloat, Cint, Ptr{Cint},
                    Ptr{cudnnAlgorithmPerformance_t}, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid},
                    Csize_t),
                   handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w,
                   yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount,
                   returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes,
                   reserveSpace, reserveSpaceSizeInBytes)
end

@checked function cudnnGetRNNBackwardDataAlgorithmMaxCount(handle, rnnDesc, count)
    initialize_api()
    ccall((:cudnnGetRNNBackwardDataAlgorithmMaxCount, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{Cint}),
                   handle, rnnDesc, count)
end

@checked function cudnnFindRNNBackwardDataAlgorithmEx(handle, rnnDesc, seqLength, yDesc, y,
                                                      dyDesc, dy, dhyDesc, dhy, dcyDesc,
                                                      dcy, wDesc, w, hxDesc, hx, cxDesc,
                                                      cx, dxDesc, dx, dhxDesc, dhx,
                                                      dcxDesc, dcx, findIntensity,
                                                      requestedAlgoCount,
                                                      returnedAlgoCount, perfResults,
                                                      workspace, workSpaceSizeInBytes,
                                                      reserveSpace, reserveSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnFindRNNBackwardDataAlgorithmEx, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Cint,
                    Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid},
                    Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    cudnnFilterDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid},
                    Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Cfloat, Cint,
                    Ptr{Cint}, Ptr{cudnnAlgorithmPerformance_t}, CuPtr{Cvoid}, Csize_t,
                    CuPtr{Cvoid}, Csize_t),
                   handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc,
                   dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx,
                   dcxDesc, dcx, findIntensity, requestedAlgoCount, returnedAlgoCount,
                   perfResults, workspace, workSpaceSizeInBytes, reserveSpace,
                   reserveSpaceSizeInBytes)
end

@checked function cudnnGetRNNBackwardWeightsAlgorithmMaxCount(handle, rnnDesc, count)
    initialize_api()
    ccall((:cudnnGetRNNBackwardWeightsAlgorithmMaxCount, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{Cint}),
                   handle, rnnDesc, count)
end

@checked function cudnnFindRNNBackwardWeightsAlgorithmEx(handle, rnnDesc, seqLength, xDesc,
                                                         x, hxDesc, hx, yDesc, y,
                                                         findIntensity, requestedAlgoCount,
                                                         returnedAlgoCount, perfResults,
                                                         workspace, workSpaceSizeInBytes,
                                                         dwDesc, dw, reserveSpace,
                                                         reserveSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnFindRNNBackwardWeightsAlgorithmEx, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Cint,
                    Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, Ptr{cudnnTensorDescriptor_t}, CuPtr{Cvoid}, Cfloat, Cint,
                    Ptr{Cint}, Ptr{cudnnAlgorithmPerformance_t}, CuPtr{Cvoid}, Csize_t,
                    cudnnFilterDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t),
                   handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y,
                   findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults,
                   workspace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace,
                   reserveSpaceSizeInBytes)
end

@checked function cudnnCreateSeqDataDescriptor(seqDataDesc)
    initialize_api()
    ccall((:cudnnCreateSeqDataDescriptor, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnSeqDataDescriptor_t},),
                   seqDataDesc)
end

@checked function cudnnDestroySeqDataDescriptor(seqDataDesc)
    initialize_api()
    ccall((:cudnnDestroySeqDataDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnSeqDataDescriptor_t,),
                   seqDataDesc)
end

@checked function cudnnSetSeqDataDescriptor(seqDataDesc, dataType, nbDims, dimA, axes,
                                            seqLengthArraySize, seqLengthArray, paddingFill)
    initialize_api()
    ccall((:cudnnSetSeqDataDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnSeqDataDescriptor_t, cudnnDataType_t, Cint, Ptr{Cint},
                    Ptr{cudnnSeqDataAxis_t}, Csize_t, Ptr{Cint}, Ptr{Cvoid}),
                   seqDataDesc, dataType, nbDims, dimA, axes, seqLengthArraySize,
                   seqLengthArray, paddingFill)
end

@checked function cudnnGetSeqDataDescriptor(seqDataDesc, dataType, nbDims, nbDimsRequested,
                                            dimA, axes, seqLengthArraySize,
                                            seqLengthSizeRequested, seqLengthArray,
                                            paddingFill)
    initialize_api()
    ccall((:cudnnGetSeqDataDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnSeqDataDescriptor_t, Ptr{cudnnDataType_t}, Ptr{Cint}, Cint,
                    Ptr{Cint}, Ptr{cudnnSeqDataAxis_t}, Ptr{Csize_t}, Csize_t, Ptr{Cint},
                    Ptr{Cvoid}),
                   seqDataDesc, dataType, nbDims, nbDimsRequested, dimA, axes,
                   seqLengthArraySize, seqLengthSizeRequested, seqLengthArray, paddingFill)
end

@checked function cudnnCreateAttnDescriptor(attnDesc)
    initialize_api()
    ccall((:cudnnCreateAttnDescriptor, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnAttnDescriptor_t},),
                   attnDesc)
end

@checked function cudnnDestroyAttnDescriptor(attnDesc)
    initialize_api()
    ccall((:cudnnDestroyAttnDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnAttnDescriptor_t,),
                   attnDesc)
end

@checked function cudnnSetAttnDescriptor(attnDesc, attnMode, nHeads, smScaler, dataType,
                                         computePrec, mathType, attnDropoutDesc,
                                         postDropoutDesc, qSize, kSize, vSize, qProjSize,
                                         kProjSize, vProjSize, oProjSize, qoMaxSeqLength,
                                         kvMaxSeqLength, maxBatchSize, maxBeamSize)
    initialize_api()
    ccall((:cudnnSetAttnDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnAttnDescriptor_t, UInt32, Cint, Cdouble, cudnnDataType_t,
                    cudnnDataType_t, cudnnMathType_t, cudnnDropoutDescriptor_t,
                    cudnnDropoutDescriptor_t, Cint, Cint, Cint, Cint, Cint, Cint, Cint,
                    Cint, Cint, Cint, Cint),
                   attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType,
                   attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize,
                   kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength,
                   maxBatchSize, maxBeamSize)
end

@checked function cudnnGetAttnDescriptor(attnDesc, attnMode, nHeads, smScaler, dataType,
                                         computePrec, mathType, attnDropoutDesc,
                                         postDropoutDesc, qSize, kSize, vSize, qProjSize,
                                         kProjSize, vProjSize, oProjSize, qoMaxSeqLength,
                                         kvMaxSeqLength, maxBatchSize, maxBeamSize)
    initialize_api()
    ccall((:cudnnGetAttnDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnAttnDescriptor_t, Ptr{UInt32}, Ptr{Cint}, Ptr{Cdouble},
                    Ptr{cudnnDataType_t}, Ptr{cudnnDataType_t}, Ptr{cudnnMathType_t},
                    Ptr{cudnnDropoutDescriptor_t}, Ptr{cudnnDropoutDescriptor_t},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                   attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType,
                   attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize,
                   kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength,
                   maxBatchSize, maxBeamSize)
end

@checked function cudnnGetMultiHeadAttnBuffers(handle, attnDesc, weightSizeInBytes,
                                               workSpaceSizeInBytes, reserveSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnGetMultiHeadAttnBuffers, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnAttnDescriptor_t, Ptr{Csize_t}, Ptr{Csize_t},
                    Ptr{Csize_t}),
                   handle, attnDesc, weightSizeInBytes, workSpaceSizeInBytes,
                   reserveSpaceSizeInBytes)
end

@checked function cudnnGetMultiHeadAttnWeights(handle, attnDesc, wKind, weightSizeInBytes,
                                               weights, wDesc, wAddr)
    initialize_api()
    ccall((:cudnnGetMultiHeadAttnWeights, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnAttnDescriptor_t, cudnnMultiHeadAttnWeightKind_t,
                    Csize_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Ptr{Cvoid}}),
                   handle, attnDesc, wKind, weightSizeInBytes, weights, wDesc, wAddr)
end

@checked function cudnnMultiHeadAttnForward(handle, attnDesc, currIdx, loWinIdx, hiWinIdx,
                                            devSeqLengthsQO, devSeqLengthsKV, qDesc,
                                            queries, residuals, kDesc, keys, vDesc, values,
                                            oDesc, out, weightSizeInBytes, weights,
                                            workSpaceSizeInBytes, workSpace,
                                            reserveSpaceSizeInBytes, reserveSpace)
    initialize_api()
    ccall((:cudnnMultiHeadAttnForward, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnAttnDescriptor_t, Cint, Ptr{Cint}, Ptr{Cint},
                    CuPtr{Cint}, CuPtr{Cint}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid},
                    CuPtr{Cvoid}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid},
                    cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, cudnnSeqDataDescriptor_t,
                    CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t,
                    CuPtr{Cvoid}),
                   handle, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO,
                   devSeqLengthsKV, qDesc, queries, residuals, kDesc, keys, vDesc, values,
                   oDesc, out, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace,
                   reserveSpaceSizeInBytes, reserveSpace)
end

@checked function cudnnMultiHeadAttnBackwardData(handle, attnDesc, loWinIdx, hiWinIdx,
                                                 devSeqLengthsDQDO, devSeqLengthsDKDV,
                                                 doDesc, dout, dqDesc, dqueries, queries,
                                                 dkDesc, dkeys, keys, dvDesc, dvalues,
                                                 values, weightSizeInBytes, weights,
                                                 workSpaceSizeInBytes, workSpace,
                                                 reserveSpaceSizeInBytes, reserveSpace)
    initialize_api()
    ccall((:cudnnMultiHeadAttnBackwardData, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnAttnDescriptor_t, Ptr{Cint}, Ptr{Cint},
                    CuPtr{Cint}, CuPtr{Cint}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid},
                    cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid},
                    cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid},
                    cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t,
                    CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}),
                   handle, attnDesc, loWinIdx, hiWinIdx, devSeqLengthsDQDO,
                   devSeqLengthsDKDV, doDesc, dout, dqDesc, dqueries, queries, dkDesc,
                   dkeys, keys, dvDesc, dvalues, values, weightSizeInBytes, weights,
                   workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace)
end

@checked function cudnnMultiHeadAttnBackwardWeights(handle, attnDesc, addGrad, qDesc,
                                                    queries, kDesc, keys, vDesc, values,
                                                    doDesc, dout, weightSizeInBytes,
                                                    weights, dweights,
                                                    workSpaceSizeInBytes, workSpace,
                                                    reserveSpaceSizeInBytes, reserveSpace)
    initialize_api()
    ccall((:cudnnMultiHeadAttnBackwardWeights, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnAttnDescriptor_t, cudnnWgradMode_t,
                    cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, cudnnSeqDataDescriptor_t,
                    CuPtr{Cvoid}, cudnnSeqDataDescriptor_t, CuPtr{Cvoid},
                    cudnnSeqDataDescriptor_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid},
                    CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}),
                   handle, attnDesc, addGrad, qDesc, queries, kDesc, keys, vDesc, values,
                   doDesc, dout, weightSizeInBytes, weights, dweights,
                   workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace)
end

@checked function cudnnCreateCTCLossDescriptor(ctcLossDesc)
    initialize_api()
    ccall((:cudnnCreateCTCLossDescriptor, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnCTCLossDescriptor_t},),
                   ctcLossDesc)
end

@checked function cudnnSetCTCLossDescriptor(ctcLossDesc, compType)
    initialize_api()
    ccall((:cudnnSetCTCLossDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnCTCLossDescriptor_t, cudnnDataType_t),
                   ctcLossDesc, compType)
end

@checked function cudnnSetCTCLossDescriptorEx(ctcLossDesc, compType, normMode, gradMode)
    initialize_api()
    ccall((:cudnnSetCTCLossDescriptorEx, libcudnn()), cudnnStatus_t,
                   (cudnnCTCLossDescriptor_t, cudnnDataType_t,
                    cudnnLossNormalizationMode_t, cudnnNanPropagation_t),
                   ctcLossDesc, compType, normMode, gradMode)
end

@checked function cudnnGetCTCLossDescriptor(ctcLossDesc, compType)
    initialize_api()
    ccall((:cudnnGetCTCLossDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnCTCLossDescriptor_t, Ptr{cudnnDataType_t}),
                   ctcLossDesc, compType)
end

@checked function cudnnGetCTCLossDescriptorEx(ctcLossDesc, compType, normMode, gradMode)
    initialize_api()
    ccall((:cudnnGetCTCLossDescriptorEx, libcudnn()), cudnnStatus_t,
                   (cudnnCTCLossDescriptor_t, Ptr{cudnnDataType_t},
                    Ptr{cudnnLossNormalizationMode_t}, Ptr{cudnnNanPropagation_t}),
                   ctcLossDesc, compType, normMode, gradMode)
end

@checked function cudnnDestroyCTCLossDescriptor(ctcLossDesc)
    initialize_api()
    ccall((:cudnnDestroyCTCLossDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnCTCLossDescriptor_t,),
                   ctcLossDesc)
end

@checked function cudnnCTCLoss(handle, probsDesc, probs, labels, labelLengths,
                               inputLengths, costs, gradientsDesc, gradients, algo,
                               ctcLossDesc, workspace, workSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnCTCLoss, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}, CuPtr{Cvoid}, cudnnTensorDescriptor_t,
                    CuPtr{Cvoid}, cudnnCTCLossAlgo_t, cudnnCTCLossDescriptor_t,
                    CuPtr{Cvoid}, Csize_t),
                   handle, probsDesc, probs, labels, labelLengths, inputLengths, costs,
                   gradientsDesc, gradients, algo, ctcLossDesc, workspace,
                   workSpaceSizeInBytes)
end

@checked function cudnnGetCTCLossWorkspaceSize(handle, probsDesc, gradientsDesc, labels,
                                               labelLengths, inputLengths, algo,
                                               ctcLossDesc, sizeInBytes)
    initialize_api()
    ccall((:cudnnGetCTCLossWorkspaceSize, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, cudnnCTCLossAlgo_t,
                    cudnnCTCLossDescriptor_t, Ref{Csize_t}),
                   handle, probsDesc, gradientsDesc, labels, labelLengths, inputLengths,
                   algo, ctcLossDesc, sizeInBytes)
end

@checked function cudnnCreateAlgorithmDescriptor(algoDesc)
    initialize_api()
    ccall((:cudnnCreateAlgorithmDescriptor, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnAlgorithmDescriptor_t},),
                   algoDesc)
end

@checked function cudnnSetAlgorithmDescriptor(algoDesc, algorithm)
    initialize_api()
    ccall((:cudnnSetAlgorithmDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnAlgorithmDescriptor_t, cudnnAlgorithm_t),
                   algoDesc, algorithm)
end

@checked function cudnnGetAlgorithmDescriptor(algoDesc, algorithm)
    initialize_api()
    ccall((:cudnnGetAlgorithmDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnAlgorithmDescriptor_t, Ptr{cudnnAlgorithm_t}),
                   algoDesc, algorithm)
end

@checked function cudnnCopyAlgorithmDescriptor(src, dest)
    initialize_api()
    ccall((:cudnnCopyAlgorithmDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnAlgorithmDescriptor_t, cudnnAlgorithmDescriptor_t),
                   src, dest)
end

@checked function cudnnDestroyAlgorithmDescriptor(algoDesc)
    initialize_api()
    ccall((:cudnnDestroyAlgorithmDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnAlgorithmDescriptor_t,),
                   algoDesc)
end

@checked function cudnnCreateAlgorithmPerformance(algoPerf, numberToCreate)
    initialize_api()
    ccall((:cudnnCreateAlgorithmPerformance, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnAlgorithmPerformance_t}, Cint),
                   algoPerf, numberToCreate)
end

@checked function cudnnSetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory)
    initialize_api()
    ccall((:cudnnSetAlgorithmPerformance, libcudnn()), cudnnStatus_t,
                   (cudnnAlgorithmPerformance_t, cudnnAlgorithmDescriptor_t,
                    cudnnStatus_t, Cfloat, Csize_t),
                   algoPerf, algoDesc, status, time, memory)
end

@checked function cudnnGetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory)
    initialize_api()
    ccall((:cudnnGetAlgorithmPerformance, libcudnn()), cudnnStatus_t,
                   (cudnnAlgorithmPerformance_t, Ptr{cudnnAlgorithmDescriptor_t},
                    Ptr{cudnnStatus_t}, Ptr{Cfloat}, Ptr{Csize_t}),
                   algoPerf, algoDesc, status, time, memory)
end

@checked function cudnnDestroyAlgorithmPerformance(algoPerf, numberToDestroy)
    initialize_api()
    ccall((:cudnnDestroyAlgorithmPerformance, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnAlgorithmPerformance_t}, Cint),
                   algoPerf, numberToDestroy)
end

@checked function cudnnGetAlgorithmSpaceSize(handle, algoDesc, algoSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnGetAlgorithmSpaceSize, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnAlgorithmDescriptor_t, Ptr{Csize_t}),
                   handle, algoDesc, algoSpaceSizeInBytes)
end

@checked function cudnnSaveAlgorithm(handle, algoDesc, algoSpace, algoSpaceSizeInBytes)
    initialize_api()
    ccall((:cudnnSaveAlgorithm, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnAlgorithmDescriptor_t, Ptr{Cvoid}, Csize_t),
                   handle, algoDesc, algoSpace, algoSpaceSizeInBytes)
end

@checked function cudnnRestoreAlgorithm(handle, algoSpace, algoSpaceSizeInBytes, algoDesc)
    initialize_api()
    ccall((:cudnnRestoreAlgorithm, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, Ptr{Cvoid}, Csize_t, cudnnAlgorithmDescriptor_t),
                   handle, algoSpace, algoSpaceSizeInBytes, algoDesc)
end

@checked function cudnnSetCallback(mask, udata, fptr)
    initialize_api()
    ccall((:cudnnSetCallback, libcudnn()), cudnnStatus_t,
                   (UInt32, Ptr{Cvoid}, cudnnCallback_t),
                   mask, udata, fptr)
end

@checked function cudnnGetCallback(mask, udata, fptr)
    initialize_api()
    ccall((:cudnnGetCallback, libcudnn()), cudnnStatus_t,
                   (Ptr{UInt32}, Ptr{Ptr{Cvoid}}, Ptr{cudnnCallback_t}),
                   mask, udata, fptr)
end

@checked function cudnnCreateFusedOpsConstParamPack(constPack, ops)
    initialize_api()
    ccall((:cudnnCreateFusedOpsConstParamPack, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnFusedOpsConstParamPack_t}, cudnnFusedOps_t),
                   constPack, ops)
end

@checked function cudnnDestroyFusedOpsConstParamPack(constPack)
    initialize_api()
    ccall((:cudnnDestroyFusedOpsConstParamPack, libcudnn()), cudnnStatus_t,
                   (cudnnFusedOpsConstParamPack_t,),
                   constPack)
end

@checked function cudnnSetFusedOpsConstParamPackAttribute(constPack, paramLabel, param)
    initialize_api()
    ccall((:cudnnSetFusedOpsConstParamPackAttribute, libcudnn()), cudnnStatus_t,
                   (cudnnFusedOpsConstParamPack_t, cudnnFusedOpsConstParamLabel_t,
                    Ptr{Cvoid}),
                   constPack, paramLabel, param)
end

@checked function cudnnGetFusedOpsConstParamPackAttribute(constPack, paramLabel, param,
                                                          isNULL)
    initialize_api()
    ccall((:cudnnGetFusedOpsConstParamPackAttribute, libcudnn()), cudnnStatus_t,
                   (cudnnFusedOpsConstParamPack_t, cudnnFusedOpsConstParamLabel_t,
                    Ptr{Cvoid}, Ptr{Cint}),
                   constPack, paramLabel, param, isNULL)
end

@checked function cudnnCreateFusedOpsVariantParamPack(varPack, ops)
    initialize_api()
    ccall((:cudnnCreateFusedOpsVariantParamPack, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnFusedOpsVariantParamPack_t}, cudnnFusedOps_t),
                   varPack, ops)
end

@checked function cudnnDestroyFusedOpsVariantParamPack(varPack)
    initialize_api()
    ccall((:cudnnDestroyFusedOpsVariantParamPack, libcudnn()), cudnnStatus_t,
                   (cudnnFusedOpsVariantParamPack_t,),
                   varPack)
end

@checked function cudnnSetFusedOpsVariantParamPackAttribute(varPack, paramLabel, ptr)
    initialize_api()
    ccall((:cudnnSetFusedOpsVariantParamPackAttribute, libcudnn()), cudnnStatus_t,
                   (cudnnFusedOpsVariantParamPack_t, cudnnFusedOpsVariantParamLabel_t,
                    PtrOrCuPtr{Cvoid}),
                   varPack, paramLabel, ptr)
end

@checked function cudnnGetFusedOpsVariantParamPackAttribute(varPack, paramLabel, ptr)
    initialize_api()
    ccall((:cudnnGetFusedOpsVariantParamPackAttribute, libcudnn()), cudnnStatus_t,
                   (cudnnFusedOpsVariantParamPack_t, cudnnFusedOpsVariantParamLabel_t,
                    PtrOrCuPtr{Cvoid}),
                   varPack, paramLabel, ptr)
end

@checked function cudnnCreateFusedOpsPlan(plan, ops)
    initialize_api()
    ccall((:cudnnCreateFusedOpsPlan, libcudnn()), cudnnStatus_t,
                   (Ptr{cudnnFusedOpsPlan_t}, cudnnFusedOps_t),
                   plan, ops)
end

@checked function cudnnDestroyFusedOpsPlan(plan)
    initialize_api()
    ccall((:cudnnDestroyFusedOpsPlan, libcudnn()), cudnnStatus_t,
                   (cudnnFusedOpsPlan_t,),
                   plan)
end

@checked function cudnnMakeFusedOpsPlan(handle, plan, constPack, workspaceSizeInBytes)
    initialize_api()
    ccall((:cudnnMakeFusedOpsPlan, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnFusedOpsPlan_t, cudnnFusedOpsConstParamPack_t,
                    Ptr{Csize_t}),
                   handle, plan, constPack, workspaceSizeInBytes)
end

@checked function cudnnFusedOpsExecute(handle, plan, varPack)
    initialize_api()
    ccall((:cudnnFusedOpsExecute, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnFusedOpsPlan_t, cudnnFusedOpsVariantParamPack_t),
                   handle, plan, varPack)
end

@checked function cudnnSetRNNDescriptor_v6(handle, rnnDesc, hiddenSize, numLayers,
                                           dropoutDesc, inputMode, direction, mode, algo,
                                           mathPrec)
    initialize_api()
    ccall((:cudnnSetRNNDescriptor_v6, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Cint,
                    cudnnDropoutDescriptor_t, cudnnRNNInputMode_t, cudnnDirectionMode_t,
                    cudnnRNNMode_t, cudnnRNNAlgo_t, cudnnDataType_t),
                   handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode,
                   direction, mode, algo, mathPrec)
end


## added in CUDNN 8.0

@checked function cudnnDeriveNormTensorDescriptor(derivedNormScaleBiasDesc, derivedNormMeanVarDesc, xDesc, mode, groupCnt)
    initialize_api()
    ccall((:cudnnDeriveNormTensorDescriptor, libcudnn()), cudnnStatus_t, (cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnNormMode_t, Cint), derivedNormScaleBiasDesc, derivedNormMeanVarDesc, xDesc, mode, groupCnt)
end

@checked function cudnnAdvTrainVersionCheck()
    initialize_api()
    ccall((:cudnnAdvTrainVersionCheck, libcudnn()), cudnnStatus_t, ())
end

@checked function cudnnOpsTrainVersionCheck()
    initialize_api()
    ccall((:cudnnOpsTrainVersionCheck, libcudnn()), cudnnStatus_t, ())
end

@checked function cudnnGetRNNWeightSpaceSize(handle, rnnDesc, weightSpaceSize)
    initialize_api()
    ccall((:cudnnGetRNNWeightSpaceSize, libcudnn()), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Ref{Csize_t}), handle, rnnDesc, weightSpaceSize)
end

@checked function cudnnGetRNNDescriptor_v6(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, cellMode, algo, mathPrec)
    initialize_api()
    ccall((:cudnnGetRNNDescriptor_v6, libcudnn()), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Ref{Cint}, Ref{Cint}, Ref{cudnnDropoutDescriptor_t}, Ref{cudnnRNNInputMode_t}, Ref{cudnnDirectionMode_t}, Ref{cudnnRNNMode_t}, Ref{cudnnRNNAlgo_t}, Ref{cudnnDataType_t}), handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, cellMode, algo, mathPrec)
end

@checked function cudnnGetCTCLossDescriptor_v8(ctcLossDesc, compType, normMode, gradMode, maxLabelLength)
    initialize_api()
    ccall((:cudnnGetCTCLossDescriptor_v8, libcudnn()), cudnnStatus_t, (cudnnCTCLossDescriptor_t, Ref{cudnnDataType_t}, Ref{cudnnLossNormalizationMode_t}, Ref{cudnnNanPropagation_t}, Ref{Cint}), ctcLossDesc, compType, normMode, gradMode, maxLabelLength)
end

@checked function cudnnSetRNNDescriptor_v8(rnnDesc, algo, cellMode, biasMode, dirMode, inputMode, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropoutDesc, auxFlags)
    initialize_api()
    ccall((:cudnnSetRNNDescriptor_v8, libcudnn()), cudnnStatus_t, (cudnnRNNDescriptor_t, cudnnRNNAlgo_t, cudnnRNNMode_t, cudnnRNNBiasMode_t, cudnnDirectionMode_t, cudnnRNNInputMode_t, cudnnDataType_t, cudnnDataType_t, cudnnMathType_t, Int32, Int32, Int32, Int32, cudnnDropoutDescriptor_t, UInt32), rnnDesc, algo, cellMode, biasMode, dirMode, inputMode, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropoutDesc, auxFlags)
end

@checked function cudnnGetRNNTempSpaceSizes(handle, rnnDesc, fMode, xDesc, workSpaceSize, reserveSpaceSize)
    initialize_api()
    ccall((:cudnnGetRNNTempSpaceSizes, libcudnn()), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnForwardMode_t, cudnnRNNDataDescriptor_t, Ref{Csize_t}, Ref{Csize_t}), handle, rnnDesc, fMode, xDesc, workSpaceSize, reserveSpaceSize)
end

@checked function cudnnCTCLoss_v8(handle, algo, ctcLossDesc, probsDesc, probs, labels, labelLengths, inputLengths, costs, gradientsDesc, gradients, workSpaceSizeInBytes, workspace)
    initialize_api()
    ccall((:cudnnCTCLoss_v8, libcudnn()), cudnnStatus_t, (cudnnHandle_t, cudnnCTCLossAlgo_t, cudnnCTCLossDescriptor_t, cudnnTensorDescriptor_t, Ptr{Cvoid}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cint}, Ptr{Cvoid}, cudnnTensorDescriptor_t, Ptr{Cvoid}, Csize_t, CuPtr{Cvoid}), handle, algo, ctcLossDesc, probsDesc, probs, labels, labelLengths, inputLengths, costs, gradientsDesc, gradients, workSpaceSizeInBytes, workspace)
end

@checked function cudnnAdvInferVersionCheck()
    initialize_api()
    ccall((:cudnnAdvInferVersionCheck, libcudnn()), cudnnStatus_t, ())
end

@checked function cudnnSetCTCLossDescriptor_v8(ctcLossDesc, compType, normMode, gradMode, maxLabelLength)
    initialize_api()
    ccall((:cudnnSetCTCLossDescriptor_v8, libcudnn()), cudnnStatus_t, (cudnnCTCLossDescriptor_t, cudnnDataType_t, cudnnLossNormalizationMode_t, cudnnNanPropagation_t, Cint), ctcLossDesc, compType, normMode, gradMode, maxLabelLength)
end

@checked function cudnnGetNormalizationTrainingReserveSpaceSize(handle, mode, normOps, algo, activationDesc, xDesc, sizeInBytes, groupCnt)
    initialize_api()
    ccall((:cudnnGetNormalizationTrainingReserveSpaceSize, libcudnn()), cudnnStatus_t, (cudnnHandle_t, cudnnNormMode_t, cudnnNormOps_t, cudnnNormAlgo_t, cudnnActivationDescriptor_t, cudnnTensorDescriptor_t, Ref{Csize_t}, Cint), handle, mode, normOps, algo, activationDesc, xDesc, sizeInBytes, groupCnt)
end

@checked function cudnnGetRNNWeightParams(handle, rnnDesc, pseudoLayer, weightSpaceSize, weightSpace, linLayerID, mDesc, mAddr, bDesc, bAddr)
    initialize_api()
    ccall((:cudnnGetRNNWeightParams, libcudnn()), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Int32, Csize_t, Ptr{Cvoid}, Int32, cudnnTensorDescriptor_t, Ptr{Ptr{Cvoid}}, cudnnTensorDescriptor_t, Ptr{Ptr{Cvoid}}), handle, rnnDesc, pseudoLayer, weightSpaceSize, weightSpace, linLayerID, mDesc, mAddr, bDesc, bAddr)
    # not sure about memory residency here, isn't clearly documented
end

@checked function cudnnGetNormalizationForwardTrainingWorkspaceSize(handle, mode, normOps, algo, xDesc, zDesc, yDesc, normScaleBiasDesc, activationDesc, normMeanVarDesc, sizeInBytes, groupCnt)
    initialize_api()
    ccall((:cudnnGetNormalizationForwardTrainingWorkspaceSize, libcudnn()), cudnnStatus_t, (cudnnHandle_t, cudnnNormMode_t, cudnnNormOps_t, cudnnNormAlgo_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnActivationDescriptor_t, cudnnTensorDescriptor_t, Ref{Csize_t}, Cint), handle, mode, normOps, algo, xDesc, zDesc, yDesc, normScaleBiasDesc, activationDesc, normMeanVarDesc, sizeInBytes, groupCnt)
end

@checked function cudnnRNNBackwardWeights_v8(handle, rnnDesc, addGrad, devSeqLengths, xDesc, x, hDesc, hx, yDesc, y, weightSpaceSize, dweightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace)
    initialize_api()
    ccall((:cudnnRNNBackwardWeights_v8, libcudnn()), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnWgradMode_t, CuPtr{Int32}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}), handle, rnnDesc, addGrad, devSeqLengths, xDesc, x, hDesc, hx, yDesc, y, weightSpaceSize, dweightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace)
end

@checked function cudnnCnnInferVersionCheck()
    initialize_api()
    ccall((:cudnnCnnInferVersionCheck, libcudnn()), cudnnStatus_t, ())
end

@checked function cudnnCnnTrainVersionCheck()
    initialize_api()
    ccall((:cudnnCnnTrainVersionCheck, libcudnn()), cudnnStatus_t, ())
end

@checked function cudnnRNNGetClip_v8(rnnDesc, clipMode, clipNanOpt, lclip, rclip)
    initialize_api()
    ccall((:cudnnRNNGetClip_v8, libcudnn()), cudnnStatus_t, (cudnnRNNDescriptor_t, Ref{cudnnRNNClipMode_t}, Ref{cudnnNanPropagation_t}, Ref{Cdouble}, Ref{Cdouble}), rnnDesc, clipMode, clipNanOpt, lclip, rclip)
end

@checked function cudnnGetRNNDescriptor_v8(rnnDesc, algo, cellMode, biasMode, dirMode, inputMode, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropoutDesc, auxFlags)
    initialize_api()
    ccall((:cudnnGetRNNDescriptor_v8, libcudnn()), cudnnStatus_t, (cudnnRNNDescriptor_t, Ref{cudnnRNNAlgo_t}, Ref{cudnnRNNMode_t}, Ref{cudnnRNNBiasMode_t}, Ref{cudnnDirectionMode_t}, Ref{cudnnRNNInputMode_t}, Ref{cudnnDataType_t}, Ref{cudnnDataType_t}, Ref{cudnnMathType_t}, Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{cudnnDropoutDescriptor_t}, Ref{UInt32}), rnnDesc, algo, cellMode, biasMode, dirMode, inputMode, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropoutDesc, auxFlags)
end

@checked function cudnnGetNormalizationBackwardWorkspaceSize(handle, mode, normOps, algo, xDesc, yDesc, dyDesc, dzDesc, dxDesc, dNormScaleBiasDesc, activationDesc, normMeanVarDesc, sizeInBytes, groupCnt)
    initialize_api()
    ccall((:cudnnGetNormalizationBackwardWorkspaceSize, libcudnn()), cudnnStatus_t, (cudnnHandle_t, cudnnNormMode_t, cudnnNormOps_t, cudnnNormAlgo_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnActivationDescriptor_t, cudnnTensorDescriptor_t, Ref{Csize_t}, Cint), handle, mode, normOps, algo, xDesc, yDesc, dyDesc, dzDesc, dxDesc, dNormScaleBiasDesc, activationDesc, normMeanVarDesc, sizeInBytes, groupCnt)
end

@checked function cudnnNormalizationForwardInference(handle, mode, normOps, algo, alpha, beta, xDesc, x, normScaleBiasDesc, normScale, normBias, normMeanVarDesc, estimatedMean, estimatedVariance, zDesc, z, activationDesc, yDesc, y, epsilon, groupCnt)
    initialize_api()
    ccall((:cudnnNormalizationForwardInference, libcudnn()), cudnnStatus_t, (cudnnHandle_t, cudnnNormMode_t, cudnnNormOps_t, cudnnNormAlgo_t, Ptr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnActivationDescriptor_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, Cdouble, Cint), handle, mode, normOps, algo, alpha, beta, xDesc, x, normScaleBiasDesc, normScale, normBias, normMeanVarDesc, estimatedMean, estimatedVariance, zDesc, z, activationDesc, yDesc, y, epsilon, groupCnt)
end

@checked function cudnnOpsInferVersionCheck()
    initialize_api()
    ccall((:cudnnOpsInferVersionCheck, libcudnn()), cudnnStatus_t, ())
end

@checked function cudnnNormalizationForwardTraining(handle, mode, normOps, algo, alpha, beta, xDesc, xData, normScaleBiasDesc, normScale, normBias, exponentialAverageFactor, normMeanVarDesc, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance, activationDesc, zDesc, zData, yDesc, yData, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes, groupCnt)
    initialize_api()
    ccall((:cudnnNormalizationForwardTraining, libcudnn()), cudnnStatus_t, (cudnnHandle_t, cudnnNormMode_t, cudnnNormOps_t, cudnnNormAlgo_t, Ptr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Cdouble, cudnnTensorDescriptor_t, Ptr{Cvoid}, Ptr{Cvoid}, Cdouble, Ptr{Cvoid}, Ptr{Cvoid}, cudnnActivationDescriptor_t, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, Cint), handle, mode, normOps, algo, alpha, beta, xDesc, xData, normScaleBiasDesc, normScale, normBias, exponentialAverageFactor, normMeanVarDesc, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance, activationDesc, zDesc, zData, yDesc, yData, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes, groupCnt)
    # not sure about residency of resultSaveMean and resultSaveInvVariance: host or device?
end

@checked function cudnnGetCTCLossWorkspaceSize_v8(handle, algo, ctcLossDesc, probsDesc, gradientsDesc, sizeInBytes)
    initialize_api()
    ccall((:cudnnGetCTCLossWorkspaceSize_v8, libcudnn()), cudnnStatus_t, (cudnnHandle_t, cudnnCTCLossAlgo_t, cudnnCTCLossDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, Ptr{Csize_t}), handle, algo, ctcLossDesc, probsDesc, gradientsDesc, sizeInBytes)
end

@checked function cudnnRNNSetClip_v8(rnnDesc, clipMode, clipNanOpt, lclip, rclip)
    initialize_api()
    ccall((:cudnnRNNSetClip_v8, libcudnn()), cudnnStatus_t, (cudnnRNNDescriptor_t, cudnnRNNClipMode_t, cudnnNanPropagation_t, Cdouble, Cdouble), rnnDesc, clipMode, clipNanOpt, lclip, rclip)
end

@checked function cudnnRNNForward(handle, rnnDesc, fwdMode, devSeqLengths, xDesc, x, yDesc, y, hDesc, hx, hy, cDesc, cx, cy, weightSpaceSize, weightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace)
    initialize_api()
    ccall((:cudnnRNNForward, libcudnn()), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnForwardMode_t, CuPtr{Int32}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}), handle, rnnDesc, fwdMode, devSeqLengths, xDesc, x, yDesc, y, hDesc, hx, hy, cDesc, cx, cy, weightSpaceSize, weightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace)
end

@checked function cudnnBuildRNNDynamic(handle, rnnDesc, miniBatch)
    initialize_api()
    ccall((:cudnnBuildRNNDynamic, libcudnn()), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint), handle, rnnDesc, miniBatch)
end

@checked function cudnnRNNBackwardData_v8(handle, rnnDesc, devSeqLengths, yDesc, y, dy, xDesc, dx, hDesc, hx, dhy, dhx, cDesc, cx, dcy, dcx, weightSpaceSize, weightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace)
    initialize_api()
    ccall((:cudnnRNNBackwardData_v8, libcudnn()), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, CuPtr{Int32}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, cudnnRNNDataDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}), handle, rnnDesc, devSeqLengths, yDesc, y, dy, xDesc, dx, hDesc, hx, dhy, dhx, cDesc, cx, dcy, dcx, weightSpaceSize, weightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace)
end

@checked function cudnnNormalizationBackward(handle, mode, normOps, algo, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData, dxDesc, dxData, dNormScaleBiasDesc, normScaleData, normBiasData, dNormScaleData, dNormBiasData, epsilon, normMeanVarDesc, savedMean, savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes, groupCnt)
    initialize_api()
    ccall((:cudnnNormalizationBackward, libcudnn()), cudnnStatus_t, (cudnnHandle_t, cudnnNormMode_t, cudnnNormOps_t, cudnnNormAlgo_t, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, cudnnTensorDescriptor_t, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid}, Cdouble, cudnnTensorDescriptor_t, Ptr{Cvoid}, Ptr{Cvoid}, cudnnActivationDescriptor_t, CuPtr{Cvoid}, Csize_t, CuPtr{Cvoid}, Csize_t, Cint), handle, mode, normOps, algo, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData, dxDesc, dxData, dNormScaleBiasDesc, normScaleData, normBiasData, dNormScaleData, dNormBiasData, epsilon, normMeanVarDesc, savedMean, savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes, groupCnt)
    # savedMean and savedInvVariance in host or device memory?
end
