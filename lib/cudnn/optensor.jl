# Compared to cudnnAddTensor!(copy(a),b), cudnnOpTensor is ~50% faster on
# (14,14,256,32)+(1,1,256,1), ~50% slower on (1,1,100,100)+(1,1,100,1) Unlike cudnnAddTensor
# it supports all broadcasting shapes up to ndims=5 as described in the documentation.

"""
    cudnnOpTensor(x1, x2; op, compType, nanOpt, alpha1, alpha2)
    cudnnOpTensor(x1, x2, d::cudnnOpTensorDescriptor; alpha1, alpha2)
    cudnnOpTensor!(y, x1, x2; op, compType, nanOpt, alpha1, alpha2, beta)
    cudnnOpTensor!(y, x1, x2, d::cudnnOpTensorDescriptor; alpha1, alpha2, beta)

Return the result of the specified broadcasting operation applied to `x1` and `x2`.
Optionally `y` holds the result and `d` specifies the operation. Each dimension of the input
tensor `x1` must match the corresponding dimension of the destination tensor `y`, and each
dimension of the input tensor `x2` must match the corresponding dimension of the destination
tensor `y` or must be equal to 1. Keyword arguments:

* `alpha1=1, alpha2=1, beta=0` are used for scaling, i.e. `y .= beta*y .+ op.(alpha1*x1, alpha2*x2)`

Keyword arguments used when `cudnnOpTensorDescriptor` is not specified:

* `op = CUDNN_OP_TENSOR_ADD`, ADD can be replaced with MUL, MIN, MAX, SQRT, NOT; SQRT and NOT performed only on x1; NOT computes 1-x1
* `compType = (eltype(x1) <: Float64 ? Float64 : Float32)`: Computation datatype (see cudnn docs for available options)
* `nanOpt = CUDNN_NOT_PROPAGATE_NAN`: NAN propagation policy. The other option is `CUDNN_PROPAGATE_NAN`.
"""
cudnnOpTensor, cudnnOpTensor!


# Public methods:
cudnnOpTensor(x1,x2; o...)    = cudnnOpTensorWithDefaults(x1,x2; o...)
cudnnOpTensor!(y,x1,x2; o...) = cudnnOpTensorWithDefaults(x1,x2; y, o...)
cudnnOpTensor(x1,x2,d::cudnnOpTensorDescriptor; o...)    = cudnnOpTensorWithDefaults(x1,x2; opTensorDesc=d, o...)
cudnnOpTensor!(y,x1,x2,d::cudnnOpTensorDescriptor; o...) = cudnnOpTensorWithDefaults(x1,x2; y, opTensorDesc=d, o...)


# Private method:
function cudnnOpTensorWithDefaults(
    x1, x2;
    y = similar(x1),
    op::cudnnOpTensorOp_t = CUDNN_OP_TENSOR_ADD,
    compType::DataType = (eltype(x1) <: Float64 ? Float64 : Float32),
    nanOpt::cudnnNanPropagation_t = CUDNN_NOT_PROPAGATE_NAN,
    opTensorDesc::cudnnOpTensorDescriptor = cudnnOpTensorDescriptor(op, cudnnDataType(compType), nanOpt),
    alpha1::Real = 1,
    alpha2::Real = 1,
    beta::Real = 0,
    x1Desc::cudnnTensorDescriptor = cudnnTensorDescriptor(x1),
    x2Desc::cudnnTensorDescriptor = cudnnTensorDescriptor(x2),
    yDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(y)
)
    @assert ndims(x1) <= 5
    @assert size(y) == size(x1)
    @assert all(size(x2,i) == size(x1,i) || size(x2,i) == 1 for i in 1:ndims(x2))
    T = eltype(x1)
    alpha1, alpha2, beta = scalingParameter(T,alpha1), scalingParameter(T,alpha2), scalingParameter(T,beta)
    cudnnOpTensorAD(x1, x2; opTensorDesc, alpha1, x1Desc, alpha2, x2Desc, beta, yDesc, y)
end


# AD method: This method aids gradient definition, please do not remove!
function cudnnOpTensorAD(x1, x2; opTensorDesc, alpha1, x1Desc, alpha2, x2Desc, beta, yDesc, y)
    cudnnOpTensor(handle(), opTensorDesc, alpha1, x1Desc, x1, alpha2, x2Desc, x2, beta, yDesc, y)
    return y
end


# Deprecated:
function cudnnOpTensor(op::cudnnOpTensorOp_t, 
                       A::DenseCuArray{T,N}, B::DenseCuArray{T,N}, C::DenseCuArray{T,N};
                       alpha1=true, alpha2=true, beta=false) where {T,N}
    @warn "cudnnOpTensor(op,A,B,C) is deprecated, please use one of the methods in `@doc cudnnOpTensor`." maxlog=1
    cudnnOpTensorWithDefaults(A, B; y=C, op, alpha1, alpha2, beta)
end
