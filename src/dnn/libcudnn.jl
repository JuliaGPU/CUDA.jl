function cudnnGetVersion()
    ccall((:cudnnGetVersion,libcudnn),Cint,())
end

function cudnnGetErrorString(status)
    ccall((:cudnnGetErrorString,libcudnn),Ptr{UInt8},(cudnnStatus_t,),status)
end

function cudnnCreate(handle)
    ccall((:cudnnCreate,libcudnn),cudnnStatus_t,(Ptr{cudnnHandle_t},),handle)
end

function cudnnDestroy(handle)
    ccall((:cudnnDestroy,libcudnn),cudnnStatus_t,(cudnnHandle_t,),handle)
end

function cudnnCreateTensorDescriptor(tensorDesc)
    ccall((:cudnnCreateTensorDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnTensorDescriptor_t},),tensorDesc)
end

function cudnnSetTensorNdDescriptor(tensorDesc,dataType,nbDims,dimA,strideA)
    ccall((:cudnnSetTensorNdDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,cudnnDataType_t,Cint,Ptr{Cint},Ptr{Cint}),tensorDesc,dataType,nbDims,dimA,strideA)
end

function cudnnGetTensorNdDescriptor(tensorDesc,nbDimsRequested,dataType,nbDims,dimA,strideA)
    ccall((:cudnnGetTensorNdDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,Cint,Ptr{cudnnDataType_t},Ptr{Cint},Ptr{Cint},Ptr{Cint}),tensorDesc,nbDimsRequested,dataType,nbDims,dimA,strideA)
end

function cudnnDestroyTensorDescriptor(tensorDesc)
    ccall((:cudnnDestroyTensorDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,),tensorDesc)
end

function cudnnSoftmaxForward(handle,algo,mode,alpha,xDesc,x,beta,yDesc,y)
    ccall((:cudnnSoftmaxForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnSoftmaxAlgorithm_t,cudnnSoftmaxMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,algo,mode,alpha,xDesc,x,beta,yDesc,y)
end

function cudnnSoftmaxForward(src::CuArray, dest::CuArray=src;
                             handle=cudnnHandle,
                             algorithm=CUDNN_SOFTMAX_ACCURATE, # or CUDNN_SOFTMAX_FAST
                             mode=CUDNN_SOFTMAX_MODE_INSTANCE, # or CUDNN_SOFTMAX_MODE_CHANNEL
                             alpha=1.0, beta=0.0)
    cudnnSoftmaxForward(handle, algorithm, mode,
                        cptr(alpha, src), TD(src,4), src,
                        cptr(beta, dest), TD(dest,4), dest)
    return dest
end

function cudnnSoftmaxBackward(handle,algo,mode,alpha,yDesc,y,dyDesc,dy,beta,dxDesc,dx)
    ccall((:cudnnSoftmaxBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnSoftmaxAlgorithm_t,cudnnSoftmaxMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,algo,mode,alpha,yDesc,y,dyDesc,dy,beta,dxDesc,dx)
end

function cudnnSoftmaxBackward(src::CuArray, srcDiff::CuArray, destDiff::CuArray=srcDiff;
                              handle=cudnnHandle,
                              algorithm=CUDNN_SOFTMAX_ACCURATE, # or CUDNN_SOFTMAX_FAST
                              mode=CUDNN_SOFTMAX_MODE_INSTANCE, # or CUDNN_SOFTMAX_MODE_CHANNEL
                              alpha=1.0, beta=0.0)
    cudnnSoftmaxBackward(handle, algorithm, mode,
                         cptr(alpha, src), TD(src,4), src,
                         TD(srcDiff,4), srcDiff,
                         cptr(beta, destDiff), TD(destDiff,4), destDiff)
    return destDiff
end
