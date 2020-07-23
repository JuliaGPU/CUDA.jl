# wrappers

function cudnnSoftmaxForward(x::CuArray{T,4}, y::CuArray{T,4}=x;
                             algo=CUDNN_SOFTMAX_FAST, # or CUDNN_SOFTMAX_ACCURATE
                             mode=CUDNN_SOFTMAX_MODE_INSTANCE, # or CUDNN_SOFTMAX_MODE_CHANNEL
                             alpha=1.0, beta=0.0) where T
    cudnnSoftmaxForward(handle(), algo, mode,
                        Ref(T(alpha)), TensorDesc(x), x,
                        Ref(T(beta )), TensorDesc(y), y)
    return y
end

function cudnnSoftmaxBackward(y::CuArray{T,4}, dy::CuArray{T,4}, dx::CuArray{T,4}=dy;
                              algo=CUDNN_SOFTMAX_FAST, # or CUDNN_SOFTMAX_ACCURATE
                              mode=CUDNN_SOFTMAX_MODE_INSTANCE, # or CUDNN_SOFTMAX_MODE_CHANNEL
                              alpha=1.0, beta=0.0) where T
    cudnnSoftmaxBackward(handle(), algo, mode,
                         Ref(T(alpha)), TensorDesc(y), y,
                         TensorDesc(dy), dy,
                         Ref(T(beta )), TensorDesc(dx), dx)
    return dx
end
