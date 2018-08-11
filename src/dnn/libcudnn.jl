function cudnnGetVersion()
    ccall((:cudnnGetVersion,libcudnn),Cint,())
end

function cudnnGetErrorString(status)
    ccall((:cudnnGetErrorString,libcudnn),Ptr{UInt8},(cudnnStatus_t,),status)
end

function cudnnCreate(handle)
    @check ccall((:cudnnCreate,libcudnn),cudnnStatus_t,(Ptr{cudnnHandle_t},),handle)
end

function cudnnDestroy(handle)
    @check ccall((:cudnnDestroy,libcudnn),cudnnStatus_t,(cudnnHandle_t,),handle)
end

function cudnnCreateTensorDescriptor(tensorDesc)
    @check ccall((:cudnnCreateTensorDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnTensorDescriptor_t},),tensorDesc)
end

function cudnnSetTensorNdDescriptor(tensorDesc,dataType,nbDims,dimA,strideA)
    @check ccall((:cudnnSetTensorNdDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,cudnnDataType_t,Cint,Ptr{Cint},Ptr{Cint}),tensorDesc,dataType,nbDims,dimA,strideA)
end

function cudnnGetTensorNdDescriptor(tensorDesc,nbDimsRequested,dataType,nbDims,dimA,strideA)
    @check ccall((:cudnnGetTensorNdDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,Cint,Ptr{cudnnDataType_t},Ptr{Cint},Ptr{Cint},Ptr{Cint}),tensorDesc,nbDimsRequested,dataType,nbDims,dimA,strideA)
end

function cudnnDestroyTensorDescriptor(tensorDesc)
    @check ccall((:cudnnDestroyTensorDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,),tensorDesc)
end

function cudnnCreateFilterDescriptor(filterDesc)
    @check ccall((:cudnnCreateFilterDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnFilterDescriptor_t},),filterDesc)
end

function cudnnSetFilterNdDescriptor(filterDesc,dataType,nbDims,filterDimA)
    @check ccall((:cudnnSetFilterNdDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,cudnnDataType_t,Cint,Ptr{Cint}),filterDesc,dataType,nbDims,filterDimA)
end

function cudnnSetFilterNdDescriptor(filterDesc,dataType,format,nbDims,filterDimA)
    @check ccall((:cudnnSetFilterNdDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,cudnnDataType_t,cudnnTensorFormat_t,Cint,Ptr{Cint}),filterDesc,dataType,format,nbDims,filterDimA)
end

function cudnnSetFilterNdDescriptor_v4(filterDesc,dataType,format,nbDims,filterDimA)
    @check ccall((:cudnnSetFilterNdDescriptor_v4,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,cudnnDataType_t,cudnnTensorFormat_t,Cint,Ptr{Cint}),filterDesc,dataType,format,nbDims,filterDimA)
end

function cudnnGetFilterNdDescriptor(filterDesc,nbDimsRequested,dataType,nbDims,filterDimA)
    @check ccall((:cudnnGetFilterNdDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,Cint,Ptr{cudnnDataType_t},Ptr{Cint},Ptr{Cint}),filterDesc,nbDimsRequested,dataType,nbDims,filterDimA)
end

function cudnnGetFilterNdDescriptor_v4(filterDesc,nbDimsRequested,dataType,format,nbDims,filterDimA)
    @check ccall((:cudnnGetFilterNdDescriptor_v4,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,Cint,Ptr{cudnnDataType_t},Ptr{cudnnTensorFormat_t},Ptr{Cint},Ptr{Cint}),filterDesc,nbDimsRequested,dataType,format,nbDims,filterDimA)
end

function cudnnDestroyFilterDescriptor(filterDesc)
    @check ccall((:cudnnDestroyFilterDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,),filterDesc)
end

function cudnnCreateConvolutionDescriptor(convDesc)
    @check ccall((:cudnnCreateConvolutionDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnConvolutionDescriptor_t},),convDesc)
end

function cudnnSetConvolutionNdDescriptor(convDesc,arrayLength,padA,filterStrideA,upscaleA,mode,dataType)
    @check ccall((:cudnnSetConvolutionNdDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},cudnnConvolutionMode_t,cudnnDataType_t),convDesc,arrayLength,padA,filterStrideA,upscaleA,mode,dataType)
end

function cudnnGetConvolutionNdDescriptor(convDesc,arrayLengthRequested,arrayLength,padA,strideA,upscaleA,mode,dataType)
    @check ccall((:cudnnGetConvolutionNdDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{cudnnConvolutionMode_t},Ptr{cudnnDataType_t}),convDesc,arrayLengthRequested,arrayLength,padA,strideA,upscaleA,mode,dataType)
end

function cudnnDestroyConvolutionDescriptor(convDesc)
    @check ccall((:cudnnDestroyConvolutionDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,),convDesc)
end

function cudnnCreatePoolingDescriptor(poolingDesc)
    ccall((:cudnnCreatePoolingDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnPoolingDescriptor_t},),poolingDesc)
end

function cudnnSetPoolingNdDescriptor(poolingDesc,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA)
    ccall((:cudnnSetPoolingNdDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnPoolingMode_t,cudnnNanPropagation_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA)
end

function cudnnGetPoolingNdDescriptor(poolingDesc,nbDimsRequested,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA)
    ccall((:cudnnGetPoolingNdDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,Cint,Ptr{cudnnPoolingMode_t},Ptr{cudnnNanPropagation_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,nbDimsRequested,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA)
end

function cudnnDestroyPoolingDescriptor(poolingDesc)
    ccall((:cudnnDestroyPoolingDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,),poolingDesc)
end

function cudnnSoftmaxForward(handle,algo,mode,alpha,xDesc,x,beta,yDesc,y)
    @check ccall((:cudnnSoftmaxForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnSoftmaxAlgorithm_t,cudnnSoftmaxMode_t,Ptr{Nothing},cudnnTensorDescriptor_t,Ptr{Nothing},Ptr{Nothing},cudnnTensorDescriptor_t,Ptr{Nothing}),handle,algo,mode,alpha,xDesc,x,beta,yDesc,y)
end

function cudnnSoftmaxForward(src::CuArray{T,4}, dest::CuArray{T,4}=src;
                             handle=libcudnn_handle[],
                             algorithm=CUDNN_SOFTMAX_ACCURATE, # or CUDNN_SOFTMAX_FAST
                             mode=CUDNN_SOFTMAX_MODE_INSTANCE, # or CUDNN_SOFTMAX_MODE_CHANNEL
                             alpha=1.0, beta=0.0) where T
    cudnnSoftmaxForward(handle, algorithm, mode,
                        cptr(alpha, src), TensorDesc(src), src,
                        cptr(beta, dest), TensorDesc(dest), dest)
    return dest
end

function cudnnSoftmaxBackward(handle,algo,mode,alpha,yDesc,y,dyDesc,dy,beta,dxDesc,dx)
    @check ccall((:cudnnSoftmaxBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnSoftmaxAlgorithm_t,cudnnSoftmaxMode_t,Ptr{Nothing},cudnnTensorDescriptor_t,Ptr{Nothing},cudnnTensorDescriptor_t,Ptr{Nothing},Ptr{Nothing},cudnnTensorDescriptor_t,Ptr{Nothing}),handle,algo,mode,alpha,yDesc,y,dyDesc,dy,beta,dxDesc,dx)
end

function cudnnSoftmaxBackward(src::CuArray{T,4}, srcDiff::CuArray{T,4}, destDiff::CuArray=srcDiff;
                              handle=libcudnn_handle[],
                              algorithm=CUDNN_SOFTMAX_ACCURATE, # or CUDNN_SOFTMAX_FAST
                              mode=CUDNN_SOFTMAX_MODE_INSTANCE, # or CUDNN_SOFTMAX_MODE_CHANNEL
                              alpha=1.0, beta=0.0) where T
    cudnnSoftmaxBackward(handle, algorithm, mode,
                         cptr(alpha, src), TensorDesc(src), src,
                         TensorDesc(srcDiff), srcDiff,
                         cptr(beta, destDiff), TensorDesc(destDiff), destDiff)
    return destDiff
end

function cudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y)
    @check ccall((:cudnnConvolutionForward, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Nothing}, cudnnTensorDescriptor_t, Ptr{Nothing}, cudnnFilterDescriptor_t, Ptr{Nothing}, cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, Ptr{Nothing}, Cint, Ptr{Nothing}, cudnnTensorDescriptor_t, Ptr{Nothing}), handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y)
end

function cudnnConvolutionForward(y::CuArray{T,N}, x::CuArray{T,N}, w::CuArray{T,N};
                                 handle=libcudnn_handle[], algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0,
                                 alpha=1, beta=0, padding=0, stride=1, upscale=1, mode=0) where {T,N}
    cd = ConvDesc(T, N-2, padding, stride, upscale, mode)
    cudnnConvolutionForward(
      handle,Ref(T(alpha)),TensorDesc(x),x,FilterDesc(w),w,cd,algo,workSpace,
      workSpaceSizeInBytes,Ref(T(beta)),TensorDesc(y),y)
    return y
end

function cudnnConvolutionBackwardData(handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx)
    @check ccall((:cudnnConvolutionBackwardData, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Nothing}, cudnnFilterDescriptor_t, Ptr{Nothing}, cudnnTensorDescriptor_t, Ptr{Nothing}, cudnnConvolutionDescriptor_t, cudnnConvolutionBwdDataAlgo_t, Ptr{Nothing}, Cint, Ptr{Nothing}, cudnnTensorDescriptor_t, Ptr{Nothing}), handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx)
end

function cudnnConvolutionBackwardData(dx::CuArray{T,N}, x::CuArray{T,N}, w::CuArray{T,N}, dy::CuArray{T,N};
                                      handle=libcudnn_handle[], algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0,
                                      alpha=1, beta=0, padding=0, stride=1, upscale=1, mode=0) where {T,N}
    cd = ConvDesc(T, N-2, padding, stride, upscale, mode)
    cudnnConvolutionBackwardData(
        handle,Ref(T(alpha)),FilterDesc(w),w,TensorDesc(dy),dy,cd,algo,workSpace,
        workSpaceSizeInBytes,Ref(T(beta)),TensorDesc(dx),dx)
    return dx
end

function cudnnConvolutionBackwardFilter(handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw)
    @check ccall((:cudnnConvolutionBackwardFilter, libcudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Nothing}, cudnnTensorDescriptor_t, Ptr{Nothing}, cudnnTensorDescriptor_t, Ptr{Nothing}, cudnnConvolutionDescriptor_t, cudnnConvolutionBwdFilterAlgo_t, Ptr{Nothing}, Cint, Ptr{Nothing}, cudnnFilterDescriptor_t, Ptr{Nothing}), handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw)
end

function cudnnConvolutionBackwardFilter(dw::CuArray{T,N}, x::CuArray{T,N}, w::CuArray{T,N}, dy::CuArray{T,N};
                                        handle=libcudnn_handle[], algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0,
                                        alpha=1, beta=0, padding=0, stride=1, upscale=1, mode=0) where {T,N}
    cd = ConvDesc(T, N-2, padding, stride, upscale, mode)
    cudnnConvolutionBackwardFilter(
        handle,Ref(T(alpha)),TensorDesc(x),x,TensorDesc(dy),dy,cd,algo,workSpace,
        workSpaceSizeInBytes,Ref(T(beta)),FilterDesc(dw),dw)
    return dw
end

function cudnnPoolingForward(handle,poolingDesc,alpha,xDesc,x,beta,yDesc,y)
    ccall((:cudnnPoolingForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnPoolingDescriptor_t,Ptr{Nothing},cudnnTensorDescriptor_t,Ptr{Nothing},Ptr{Nothing},cudnnTensorDescriptor_t,Ptr{Nothing}),handle,poolingDesc,alpha,xDesc,x,beta,yDesc,y)
end

function cudnnPoolingBackward(handle,poolingDesc,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx)
    ccall((:cudnnPoolingBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnPoolingDescriptor_t,Ptr{Nothing},cudnnTensorDescriptor_t,Ptr{Nothing},cudnnTensorDescriptor_t,Ptr{Nothing},cudnnTensorDescriptor_t,Ptr{Nothing},Ptr{Nothing},cudnnTensorDescriptor_t,Ptr{Nothing}),handle,poolingDesc,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx)
end

function cudnnPoolingForward(y::CuArray{T,N}, x::CuArray{T,N}; handle=libcudnn_handle[], alpha=1,
                                window=2, padding=0, stride=window, mode=0) where {T,N}
    beta = 0
    pd = PoolDesc(N-2, window, padding, stride, mode)
    cudnnPoolingForward(handle, pd, Ref(T(alpha)), TensorDesc(x), x, Ref(T(beta)), TensorDesc(y), y)
    return y
end

function cudnnPoolingBackward(dx::CuArray{T,N}, dy::CuArray{T,N}, x::CuArray{T,N}, y::CuArray{T,N};
                                 handle=libcudnn_handle[], alpha=1,
                                 window=2, padding=0, stride=window, mode=0) where {T,N}
    if alpha!=1 && mode==0; error("Gradient of pool(alpha!=1,mode=0) broken in CUDNN"); end
    beta = 0
    pd = PoolDesc(N-2, window, padding, stride, mode)
    cudnnPoolingBackward(handle, pd, Ref(T(alpha)), TensorDesc(y), y,
                         TensorDesc(dy), dy, TensorDesc(x), x, Ref(T(beta)), TensorDesc(dx), dx)
    return dx
end
