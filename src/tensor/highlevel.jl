using LinearAlgebra
import LinearAlgebra: axpy!, axpby!, mul!

mutable struct CuTensor{T, N}
    data::CuArray{T, N}
    inds::Vector{Cwchar_t}
    function CuTensor{T, N}(data::CuArray{T, N}, inds::Vector{Cwchar_t}) where {T<:Number, N}
        new(data, inds)
    end
    function CuTensor{T, N}(data::CuArray{N, T}, inds::Vector{<:AbstractChar}) where {T<:Number, N}
        new(data, Cwchar_t.(inds))
    end
end
CuTensor(data::CuArray{T, N}, inds::Vector{<:AbstractChar}) where {T<:Number, N} =
    CuTensor{T, N}(data, convert(Vector{Cwchar_t}, inds))
CuTensor(data::CuArray{T, N}, inds::Vector{Cwchar_t}) where {T<:Number, N} =
    CuTensor{T, N}(data, inds)

Base.size(T::CuTensor) = size(T.data)
Base.size(T::CuTensor, i) = size(T.data, i)
Base.length(T::CuTensor) = length(T.data)
Base.ndims(T::CuTensor) = length(T.inds)
Base.strides(T::CuTensor) = strides(T.data)
Base.eltype(T::CuTensor) = eltype(T.data)
Base.similar(T::CuTensor{Tv, N}) where {Tv, N} = CuTensor{Tv, N}(similar(T.data), copy(T.inds))
Base.collect(T::CuTensor) = (collect(T.data), T.inds)

function Base.:(+)(A::CuTensor, B::CuTensor)
    α = convert(eltype(A), 1.0)
    γ = convert(eltype(B), 1.0)
    C = similar(B)
    elementwiseBinary!(α, A, CUTENSOR_OP_IDENTITY, γ, B, CUTENSOR_OP_IDENTITY, C, CUTENSOR_OP_ADD)
end

function Base.:(-)(A::CuTensor, B::CuTensor)
    α = convert(eltype(A), 1.0)
    γ = convert(eltype(B), -1.0)
    C = similar(B)
    elementwiseBinary!(α, A, CUTENSOR_OP_IDENTITY, γ, B, CUTENSOR_OP_IDENTITY, C, CUTENSOR_OP_ADD)
end

axpy!(a, X::CuTensor, Y::CuTensor) = elementwiseBinary!(a, X, CUTENSOR_OP_IDENTITY, one(eltype(Y)), Y, CUTENSOR_OP_IDENTITY, similar(Y), CUTENSOR_OP_ADD)
axpby!(a, X::CuTensor, b, Y::CuTensor) = elementwiseBinary!(a, X, CUTENSOR_OP_IDENTITY, b, Y, CUTENSOR_OP_IDENTITY, similar(Y), CUTENSOR_OP_ADD)

mul!(C::CuTensor, A::CuTensor, B::CuTensor) = contraction!(one(eltype(C)), A, CUTENSOR_OP_IDENTITY, B, CUTENSOR_OP_IDENTITY, zero(eltype(C)), C, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_IDENTITY)

function Base.:(*)(A::CuTensor, B::CuTensor)
    tC = promote_type(eltype(A), eltype(B))
    A_uniqs = [(idx, i) for (idx, i) in enumerate(A.inds) if !(i in B.inds)]
    B_uniqs = [(idx, i) for (idx, i) in enumerate(B.inds) if !(i in A.inds)]
    A_sizes = map(x->size(A,x[1]), A_uniqs)
    B_sizes = map(x->size(B,x[1]), B_uniqs)
    A_inds = map(x->Cwchar_t(x[2]), A_uniqs)
    B_inds = map(x->Cwchar_t(x[2]), B_uniqs)
    C = CuTensor(CuArrays.zeros(tC, Dims(vcat(A_sizes, B_sizes))), vcat(A_inds, B_inds))
    return mul!(C, A, B)
end
