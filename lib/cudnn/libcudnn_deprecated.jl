# old and deprecated wrappers

## deprecated in CUDNN 8.0

@cenum cudnnConvolutionFwdPreference_t::UInt32 begin
    CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = 0
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2
end

@cenum cudnnConvolutionBwdFilterPreference_t::UInt32 begin
    CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = 1
    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2
end

@cenum cudnnConvolutionBwdDataPreference_t::UInt32 begin
    CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = 0
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = 1
    CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2
end

@checked function cudnnGetConvolutionBackwardDataAlgorithm(handle, wDesc, dyDesc, convDesc,
                                                           dxDesc, preference,
                                                           memoryLimitInBytes, algo)
    initialize_api()
    ccall((:cudnnGetConvolutionBackwardDataAlgorithm, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnConvolutionBwdDataPreference_t, Csize_t,
                    Ptr{cudnnConvolutionBwdDataAlgo_t}),
                   handle, wDesc, dyDesc, convDesc, dxDesc, preference, memoryLimitInBytes,
                   algo)
end

@checked function cudnnSetRNNDescriptor(handle, rnnDesc, hiddenSize, numLayers,
                                        dropoutDesc, inputMode, direction, mode, algo,
                                        mathPrec)
    initialize_api()
    ccall((:cudnnSetRNNDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Cint,
                    cudnnDropoutDescriptor_t, cudnnRNNInputMode_t, cudnnDirectionMode_t,
                    cudnnRNNMode_t, cudnnRNNAlgo_t, cudnnDataType_t),
                   handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode,
                   direction, mode, algo, mathPrec)
end

@checked function cudnnGetRNNDescriptor(handle, rnnDesc, hiddenSize, numLayers,
                                        dropoutDesc, inputMode, direction, mode, algo,
                                        mathPrec)
    initialize_api()
    ccall((:cudnnGetRNNDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnRNNDescriptor_t, Ptr{Cint}, Ptr{Cint},
                    Ptr{cudnnDropoutDescriptor_t}, Ptr{cudnnRNNInputMode_t},
                    Ptr{cudnnDirectionMode_t}, Ptr{cudnnRNNMode_t}, Ptr{cudnnRNNAlgo_t},
                    Ptr{cudnnDataType_t}),
                   handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode,
                   direction, mode, algo, mathPrec)
end

@checked function cudnnSetRNNDescriptor_v5(rnnDesc, hiddenSize, numLayers, dropoutDesc,
                                           inputMode, direction, mode, mathPrec)
    initialize_api()
    ccall((:cudnnSetRNNDescriptor_v5, libcudnn()), cudnnStatus_t,
                   (cudnnRNNDescriptor_t, Cint, Cint, cudnnDropoutDescriptor_t,
                    cudnnRNNInputMode_t, cudnnDirectionMode_t, cudnnRNNMode_t,
                    cudnnDataType_t),
                   rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode,
                   mathPrec)
end

@checked function cudnnGetConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc,
                                                      yDesc, preference,
                                                      memoryLimitInBytes, algo)
    initialize_api()
    ccall((:cudnnGetConvolutionForwardAlgorithm, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t,
                    cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnConvolutionFwdPreference_t, Csize_t,
                    Ptr{cudnnConvolutionFwdAlgo_t}),
                   handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes,
                   algo)
end

@checked function cudnnGetConvolutionBackwardFilterAlgorithm(handle, xDesc, dyDesc,
                                                             convDesc, dwDesc, preference,
                                                             memoryLimitInBytes, algo)
    initialize_api()
    ccall((:cudnnGetConvolutionBackwardFilterAlgorithm, libcudnn()), cudnnStatus_t,
                   (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
                    cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t,
                    cudnnConvolutionBwdFilterPreference_t, Csize_t,
                    Ptr{cudnnConvolutionBwdFilterAlgo_t}),
                   handle, xDesc, dyDesc, convDesc, dwDesc, preference, memoryLimitInBytes,
                   algo)
end
