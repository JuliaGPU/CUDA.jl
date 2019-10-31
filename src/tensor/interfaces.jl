# interfacing with other packages

## Base

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


## LinearAlgebra

using LinearAlgebra

LinearAlgebra.axpy!(a, X::CuTensor, Y::CuTensor) = elementwiseBinary!(a, X, CUTENSOR_OP_IDENTITY, one(eltype(Y)), Y, CUTENSOR_OP_IDENTITY, similar(Y), CUTENSOR_OP_ADD)
LinearAlgebra.axpby!(a, X::CuTensor, b, Y::CuTensor) = elementwiseBinary!(a, X, CUTENSOR_OP_IDENTITY, b, Y, CUTENSOR_OP_IDENTITY, similar(Y), CUTENSOR_OP_ADD)

LinearAlgebra.mul!(C::CuTensor, A::CuTensor, B::CuTensor) = contraction!(one(eltype(C)), A, CUTENSOR_OP_IDENTITY, B, CUTENSOR_OP_IDENTITY, zero(eltype(C)), C, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_IDENTITY)
