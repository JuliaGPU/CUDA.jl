# TensorDesc

# descriptor

mutable struct TensorDesc
    ptr::cudnnTensorDescriptor_t
end

unsafe_free!(td::TensorDesc) = cudnnDestroyTensorDescriptor(td.ptr)

Base.unsafe_convert(::Type{cudnnTensorDescriptor_t}, td::TensorDesc) = td.ptr

function TensorDesc(T::Type, size::NTuple{N,Integer}, strides::NTuple{N,Integer} = tuple_strides(size)) where N
    sz = Cint.(size) |> reverse |> collect
    st = Cint.(strides) |> reverse |> collect
    td = Ref{cudnnTensorDescriptor_t}()
    cudnnCreateTensorDescriptor(td)
    cudnnSetTensorNdDescriptor(td[], cudnnDataType(T), length(sz), sz, st)
    this = TensorDesc(td[])
    finalizer(unsafe_free!, this)
    return this
end

TensorDesc(a::DenseCuArray) = TensorDesc(eltype(a), size(a), strides(a))

# wrappers

function cudnnAddTensor(C::DenseCuArray{T,N}, A::DenseCuArray{T,N};
                        alpha=1, beta=1) where {T,N}
    cudnnAddTensor(handle(),
                   scalingParameter(T, alpha), TensorDesc(A), A,
                   scalingParameter(T, beta ), TensorDesc(C), C)
    return C
end


# OpTensorDesc

# descriptor

mutable struct OpTensorDesc
    ptr::cudnnOpTensorDescriptor_t
end

unsafe_free!(otd::OpTensorDesc) = cudnnDestroyOpTensorDescriptor(otd.ptr)

Base.unsafe_convert(::Type{cudnnOpTensorDescriptor_t}, otd::OpTensorDesc) = otd.ptr

function OpTensorDesc(op::cudnnOpTensorOp_t, T::Type;
                      opTensorNanOpt=CUDNN_NOT_PROPAGATE_NAN)
    otd = Ref{cudnnOpTensorDescriptor_t}()
    cudnnCreateOpTensorDescriptor(otd)
    cudnnSetOpTensorDescriptor(otd[], op, cudnnDataType(T), opTensorNanOpt)
    this = OpTensorDesc(otd[])
    finalizer(unsafe_free!, this)
    return this
end

OpTensorDesc(op::cudnnOpTensorOp_t, a::DenseCuArray) = OpTensorDesc(op, eltype(a))

# wrappers

function cudnnOpTensor(op::cudnnOpTensorOp_t,
                       A::DenseCuArray{T,N}, B::DenseCuArray{T,N}, C::DenseCuArray{T,N};
                       alpha1=1, alpha2=1, beta=0) where {T,N}
    cudnnOpTensor(handle(), OpTensorDesc(op, T),
                  scalingParameter(T, alpha1), TensorDesc(A), A,
                  scalingParameter(T, alpha2), TensorDesc(B), B,
                  scalingParameter(T, beta  ), TensorDesc(C), C)
    return C
end


# ReduceTensorDesc

# descriptor

mutable struct ReduceTensorDesc
    ptr::cudnnReduceTensorDescriptor_t
end

unsafe_free!(rtd::ReduceTensorDesc) = cudnnDestroyReduceTensorDescriptor(rtd.ptr)

Base.unsafe_convert(::Type{cudnnReduceTensorDescriptor_t}, rtd::ReduceTensorDesc) = rtd.ptr

function ReduceTensorDesc(op::cudnnReduceTensorOp_t, T::Type;
                          reduceTensorNanOpt=CUDNN_NOT_PROPAGATE_NAN,
                          reduceTensorIndices=CUDNN_REDUCE_TENSOR_NO_INDICES,
                          reduceTensorIndicesType=CUDNN_32BIT_INDICES)
    rtd = Ref{cudnnReduceTensorDescriptor_t}()
    cudnnCreateReduceTensorDescriptor(rtd)
    cudnnSetReduceTensorDescriptor(rtd[], op, cudnnDataType(T), reduceTensorNanOpt,
                                   reduceTensorIndices, reduceTensorIndicesType)
    this = ReduceTensorDesc(rtd[])
    finalizer(unsafe_free!, this)
    return this
end

ReduceTensorDesc(op::cudnnReduceTensorOp_t, a::DenseCuArray) = ReduceTensorDesc(op, eltype(a))

# wrappers

function cudnnGetReductionIndicesSize(op::cudnnReduceTensorOp_t,
                                      A::DenseCuArray{T,N}, C::DenseCuArray{T,N}) where {T,N}
    size=@argout(
        cudnnGetReductionIndicesSize(
            handle(), ReduceTensorDesc(op, A),
            TensorDesc(A), TensorDesc(C),
            out(Ref{Csize_t}()))
        )[]
    return size
end

function cudnnReduceTensor(op::cudnnReduceTensorOp_t,
                           A::DenseCuArray{T,N}, C::DenseCuArray{T,N};
                           alpha=1, beta=0) where {T,N}
    # indices = Array{UInt64, 1}(undef, N)
    indicesSizeInBytes = cudnnGetReductionIndicesSize(op, A, C)
    @workspace size=@argout(
        cudnnGetReductionWorkspaceSize(
            handle(), ReduceTensorDesc(op, A),
            TensorDesc(A), TensorDesc(C),
            out(Ref{Csize_t}()))
        )[] workspace->begin
            cudnnReduceTensor(handle(), ReduceTensorDesc(op, A),
                              C_NULL, indicesSizeInBytes,
                              workspace, sizeof(workspace),
                              scalingParameter(T, alpha), TensorDesc(A), A,
                              scalingParameter(T, beta ), TensorDesc(C), C)
        end
    return C
end
