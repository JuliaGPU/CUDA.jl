# wrappers

function cudnnSoftmaxForward(x::DenseCuArray{T,4}, y::DenseCuArray{T,4}=x;
                             algo=CUDNN_SOFTMAX_FAST, # or CUDNN_SOFTMAX_ACCURATE
                             mode=CUDNN_SOFTMAX_MODE_INSTANCE, # or CUDNN_SOFTMAX_MODE_CHANNEL
                             alpha=1.0, beta=0.0) where T
    cudnnSoftmaxForward(handle(), algo, mode,
                        scalingParameter(T, alpha), TensorDesc(x), x,
                        scalingParameter(T, beta ), TensorDesc(y), y)
    return y
end

function cudnnSoftmaxBackward(y::DenseCuArray{T,4}, dy::DenseCuArray{T,4}, dx::DenseCuArray{T,4}=dy;
                              algo=CUDNN_SOFTMAX_FAST, # or CUDNN_SOFTMAX_ACCURATE
                              mode=CUDNN_SOFTMAX_MODE_INSTANCE, # or CUDNN_SOFTMAX_MODE_CHANNEL
                              alpha=1.0, beta=0.0) where T
    cudnnSoftmaxBackward(handle(), algo, mode,
                         scalingParameter(T, alpha), TensorDesc(y), y,
                         TensorDesc(dy), dy,
                         scalingParameter(T, beta ), TensorDesc(dx), dx)
    return dx
end
