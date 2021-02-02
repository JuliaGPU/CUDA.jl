# This is unfortunately 10x slower than libknet8, 2x slower than CUDA.jl

"""
    cudnnReduceTensor(x; dims, op, compType, nanOpt, indices, alpha)
    cudnnReduceTensor(x, d::cudnnReduceTensorDescriptor; dims, indices, alpha)
    cudnnReduceTensor!(y, x; op, compType, nanOpt, indices, alpha, beta)
    cudnnReduceTensor!(y, x, d::cudnnReduceTensorDescriptor; indices, alpha, beta)

Return the result of the specified reduction operation applied to `x`.  Optionally `y` holds
the result and `d` specifies the operation.  Each dimension of the output tensor `y` must
match the corresponding dimension of the input tensor `x` or must be equal to 1. The
dimensions equal to 1 indicate the dimensions of `x` to be reduced.  Keyword arguments:

* `dims = ntuple(i->1,ndims(x))`: specifies the shape of the output when `y` is not given
* `indices = nothing`: previously allocated space for writing indices which can be generated for min and max ops only, can be a `CuArray` of `UInt8`, `UInt16`, `UInt32` or `UInt64`
* `alpha=1, beta=0` are used for scaling, i.e. `y .= alpha*op.(x1) .+ beta*y`

Keyword arguments that can be used when `reduceTensorDesc` is not specified:
* `op = CUDNN_REDUCE_TENSOR_ADD`: Reduction operation, ADD can be replaced with MUL, MIN, MAX, AMAX, AVG, NORM1, NORM2, MUL_NO_ZEROS
* `compType = (eltype(x) <: Float64 ? Float64 : Float32)`: Computation datatype
* `nanOpt = CUDNN_NOT_PROPAGATE_NAN`: NAN propagation policy, the other option is `CUDNN_PROPAGATE_NAN`
"""
cudnnReduceTensor, cudnnReduceTensor!


# Public methods
cudnnReduceTensor(x; o...)     = cudnnReduceTensorWithDefaults(x; o...)
cudnnReduceTensor!(y, x; o...) = cudnnReduceTensorWithDefaults(x; y, o...)
cudnnReduceTensor(x, d::cudnnReduceTensorDescriptor; o...)     = cudnnReduceTensorWithDefaults(x; reduceTensorDesc=d, o...)
cudnnReduceTensor!(y, x, d::cudnnReduceTensorDescriptor; o...) = cudnnReduceTensorWithDefaults(x; y, reduceTensorDesc=d, o...)


# Private method
function cudnnReduceTensorWithDefaults(
    x;
    op::cudnnReduceTensorOp_t = CUDNN_REDUCE_TENSOR_ADD,
    compType::DataType = (eltype(x) <: Float64 ? Float64 : Float32),
    nanOpt::cudnnNanPropagation_t = CUDNN_NOT_PROPAGATE_NAN,
    indices::Union{Vector{<:Unsigned},Nothing} = nothing,
    reduceTensorDesc::cudnnReduceTensorDescriptor = cudnnReduceTensorDescriptor(op, cudnnDataType(compType), nanOpt, cudnnReduceTensorIndices(op, indices), cudnnIndicesType(indices)),
    dims::Dims = ntuple(i->1,ndims(x)),
    y = similar(x, dims),
    alpha::Real = 1,
    beta::Real = 0,
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x),
    yDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(y),
)
    T = eltype(x)
    alpha, beta = scalingParameter(T,alpha), scalingParameter(T,beta)
    cudnnReduceTensorAD(x; reduceTensorDesc, alpha, xDesc, beta, yDesc, y, indices)
end

function cudnnReduceTensorIndices(op, indices)
    if indices !== nothing && op in (CUDNN_REDUCE_TENSOR_MIN, CUDNN_REDUCE_TENSOR_MAX)
        CUDNN_REDUCE_TENSOR_FLATTENED_INDICES
    else
        CUDNN_REDUCE_TENSOR_NO_INDICES
    end
end

cudnnIndicesType(::Nothing)=CUDNN_32BIT_INDICES
cudnnIndicesType(::Vector{UInt8})=CUDNN_8BIT_INDICES
cudnnIndicesType(::Vector{UInt16})=CUDNN_16BIT_INDICES
cudnnIndicesType(::Vector{UInt32})=CUDNN_32BIT_INDICES
cudnnIndicesType(::Vector{UInt64})=CUDNN_64BIT_INDICES
cudnnIndicesType(x)=error("Bad type $x for cudnnIndices, use Vector{UInt8, 16, 32 or 64}.")


# AD method
function cudnnReduceTensorAD(x; reduceTensorDesc, alpha, xDesc, beta, yDesc, y, indices)
    @workspace size=@argout(
        cudnnGetReductionWorkspaceSize(handle(), reduceTensorDesc, xDesc, yDesc, out(Ref{Csize_t}()))
    )[] workspace->begin
        cudnnReduceTensor(handle(), reduceTensorDesc, something(indices, C_NULL), sizeof(indices), workspace, sizeof(workspace), alpha, xDesc, x, beta, yDesc, y)
    end
    return y
end


# Deprecated
function cudnnReduceTensor(op::cudnnReduceTensorOp_t,
                           A::DenseCuArray{T,N}, C::DenseCuArray{T,N};
                           alpha=true, beta=false) where {T,N}
    @warn "cudnnReduceTensor(op,A,C) is deprecated, please use one of the methods in `@doc cudnnReduceTensor`." maxlog=1
    cudnnReduceTensor(A; y=C, op, alpha, beta)
end
