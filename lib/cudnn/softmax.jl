"""
    cudnnSoftmaxForward(x; algo, mode, alpha)
    cudnnSoftmaxForward!(y, x; algo, mode, alpha, beta)

Return the softmax or logsoftmax of the input `x` depending on the `algo` keyword argument.
The `y` argument holds the result and it should be similar to `x` if specified. Keyword
arguments:

* `algo = (CUDA.math_mode()===CUDA.FAST_MATH ? CUDNN_SOFTMAX_FAST : CUDNN_SOFTMAX_ACCURATE)`: Options are `CUDNN_SOFTMAX_ACCURATE` which subtracts max from every point to avoid overflow, `CUDNN_SOFTMAX_FAST` which doesn't and `CUDNN_SOFTMAX_LOG` which returns logsoftmax.
* `mode = CUDNN_SOFTMAX_MODE_INSTANCE`: Compute softmax per image (N) across the dimensions C,H,W. `CUDNN_SOFTMAX_MODE_CHANNEL` computes softmax per spatial location (H,W) per image (N) across the dimension C. 
* `alpha=1, beta=0` can be used for scaling, i.e. `y .= alpha*op(x1) .+ beta*y`
"""


# Public methods
cudnnSoftmaxForward(x; o...) = cudnnSoftmaxForwardWithDefaults(x; o...)
cudnnSoftmaxForward!(y, x; o...) = cudnnSoftmaxForwardWithDefaults(x; y, o...)


# Private method
function cudnnSoftmaxForwardWithDefaults(
    x;
    y = similar(x),
    algo::cudnnSoftmaxAlgorithm_t = (CUDA.math_mode()===CUDA.FAST_MATH ? CUDNN_SOFTMAX_FAST : CUDNN_SOFTMAX_ACCURATE),
    mode::cudnnSoftmaxMode_t = CUDNN_SOFTMAX_MODE_INSTANCE,
    alpha::Real = 1,
    beta::Real = 0,
    format::cudnnTensorFormat_t = CUDNN_TENSOR_NCHW,
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x; format),
    yDesc::cudnnTensorDescriptor = xDesc,
)
    @assert size(y) == size(x)
    T = eltype(x)
    alpha, beta = scalingParameter(T,alpha), scalingParameter(T,beta)
    cudnnSoftmaxForwardAD(x; algo, mode, alpha, xDesc, beta, yDesc, y)
end


# AD method
function cudnnSoftmaxForwardAD(x; algo, mode, alpha, xDesc, beta, yDesc, y)
    cudnnSoftmaxForward(handle(), algo, mode, alpha, xDesc, x, beta, yDesc, y)
    return y
end


# Deprecated methods
function cudnnSoftmaxForward(x::DenseCuArray{T,4}, y::DenseCuArray{T,4}; o...) where T
    @warn "`cudnnSoftmaxForward(x,y)` is deprecated, please use one of the methods in `@doc cudnnSoftmaxForward`." maxlog=1
    cudnnSoftmaxForward!(y, x; o...)
end

