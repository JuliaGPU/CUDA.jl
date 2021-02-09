"""
    cudnnSetTensor!(x, s)

Set all elements of tensor `x` to scalar `s` and return `x`.
"""
function cudnnSetTensor!(
    x, s::Real;
    format::cudnnTensorFormat_t = CUDNN_TENSOR_NCHW,
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x; format)
)
    cudnnSetTensor(handle(), xDesc, x, Ref(eltype(x)(s)))
    return x
end


"""
    cudnnScaleTensor(x, s)
    cudnnScaleTensor!(y, x, s)

Scale all elements of tensor `x` with scale `s` and return the result. `cudnnScaleTensor`
allocates a new array for the answer, `cudnnScaleTensor!` overwrites `y`.
"""
cudnnScaleTensor, cudnnScaleTensor!

function cudnnScaleTensor!(
    y, x, s::Real;
    format::cudnnTensorFormat_t = CUDNN_TENSOR_NCHW,
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x; format)
)
    y === x || copyto!(y, x)
    cudnnScaleTensor(handle(), xDesc, y, Ref(eltype(y)(s)))
    return y
end

cudnnScaleTensor(x, s::Real; o...) = cudnnScaleTensor!(similar(x), x, s; o...)


# cudnnAddTensor does not support all broadcasting dimensions, use cudnnOpTensor instead.
# Compared to libknet8 x .+ b it is ~2x slower for (1,1,100,100), ~30% faster for (14,14,256,32)
# CUDA.jl x .+ b is 2x slower than both

"""
    cudnnAddTensor(x, b; alpha)
    cudnnAddTensor!(y, x, b; alpha, beta)

Broadcast-add tensor `b` to tensor `x`. `alpha=1, beta=1` are used for scaling, i.e. `y .=
alpha * b .+ beta * x`.  `cudnnAddTensor` allocates a new array for the answer,
`cudnnAddTensor!` overwrites `y`. Does not support all valid broadcasting dimensions.  For
more flexible broadcast operations see `cudnnOpTensor`.
"""
cudnnAddTensor, cudnnAddTensor!

function cudnnAddTensor!(
    y, x, b;
    alpha::Real=1,
    beta::Real=1,
    format::cudnnTensorFormat_t = CUDNN_TENSOR_NCHW,
    bDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(b; format),
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x; format),
)
    T = eltype(x)
    alpha, beta = scalingParameter(T, alpha), scalingParameter(T, beta)
    y === x || copyto!(y, x)
    cudnnAddTensor(handle(), alpha, bDesc, b, beta, xDesc, y)
    return y
end

cudnnAddTensor(x, b; o...) = cudnnAddTensor!(similar(x), x, b; o...)
