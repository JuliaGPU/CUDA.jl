"""
    cudnnActivationForward(x; mode, nanOpt, coef, alpha)
    cudnnActivationForward(x, d::cudnnActivationDescriptor; alpha)
    cudnnActivationForward!(y, x; mode, nanOpt, coef, alpha, beta)
    cudnnActivationForward!(y, x, d::cudnnActivationDescriptor; alpha, beta)

Return the result of the specified elementwise activation operation applied to `x`.
Optionally `y` holds the result and `d` specifies the operation. `y` should be similar to
`x` if specified. Keyword arguments `alpha=1, beta=0` can be used for scaling, i.e. `y .=
alpha*op.(x1) .+ beta*y`.  The following keyword arguments specify the operation if `d` is
not given:

* `mode = CUDNN_ACTIVATION_RELU`: Options are SIGMOID, RELU, TANH, CLIPPED_RELU, ELU, IDENTITY
* `nanOpt = CUDNN_NOT_PROPAGATE_NAN`: NAN propagation policy, the other option is `CUDNN_PROPAGATE_NAN`
* `coef=1`: When the activation mode is set to CUDNN_ACTIVATION_CLIPPED_RELU, this input specifies the clipping threshold; and when the activation mode is set to CUDNN_ACTIVATION_ELU, this input specifies the α parameter.
"""
cudnnActivationForward, cudnnActivationForward!


# Public methods
cudnnActivationForward(x; o...)     = cudnnActivationForwardWithDefaults(x; o...)
cudnnActivationForward!(y, x; o...) = cudnnActivationForwardWithDefaults(x; y, o...)
cudnnActivationForward(x, d::cudnnActivationDescriptor; o...)     = cudnnActivationForwardWithDefaults(x; activationDesc=d, o...)
cudnnActivationForward!(y, x, d::cudnnActivationDescriptor; o...) = cudnnActivationForwardWithDefaults(x; y, activationDesc=d, o...)


# Private method
function cudnnActivationForwardWithDefaults(
    x;
    y = similar(x),
    mode::cudnnActivationMode_t = CUDNN_ACTIVATION_RELU,
    nanOpt::cudnnNanPropagation_t = CUDNN_NOT_PROPAGATE_NAN,
    coef::Real=1,
    activationDesc::cudnnActivationDescriptor = cudnnActivationDescriptor(mode, nanOpt, Cdouble(coef)),
    alpha::Real=1,
    beta::Real=0,
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x),
    yDesc::cudnnTensorDescriptor = xDesc,
)
    T = eltype(x)
    alpha, beta = scalingParameter(T,alpha), scalingParameter(T,beta)
    cudnnActivationForwardAD(x; activationDesc, alpha, xDesc, beta, yDesc, y)
end


# AD method:
function cudnnActivationForwardAD(x; activationDesc, alpha, xDesc, beta, yDesc, y)
    cudnnActivationForward(handle(), activationDesc, alpha, xDesc, x, beta, yDesc, y)
    return y
end


# Deprecated:
function cudnnActivationForward(x::DenseCuArray{T,N}, y::DenseCuArray{T,N}; o...) where {T,N}
    @warn "`cudnnActivationForward(x,y)` is deprecated, please use one of the methods in `@doc cudnnActivationForward`." maxlog=1
    cudnnActivationForward!(y, x; o...)
end
