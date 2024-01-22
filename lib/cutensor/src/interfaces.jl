# interfacing with other packages

## Base

function Base.:(+)(A::CuTensor, B::CuTensor)
    α = convert(eltype(A), 1.0)
    γ = convert(eltype(B), 1.0)
    C = similar(B)
    elementwise_binary_execute!(α, A.data, A.inds, CUTENSOR_OP_IDENTITY,
                                γ, B.data, B.inds, CUTENSOR_OP_IDENTITY,
                                C.data, C.inds, CUTENSOR_OP_ADD)
    C
end

function Base.:(-)(A::CuTensor, B::CuTensor)
    α = convert(eltype(A), 1.0)
    γ = convert(eltype(B), -1.0)
    C = similar(B)
    elementwise_binary_execute!(α, A.data, A.inds, CUTENSOR_OP_IDENTITY,
                                γ, B.data, B.inds, CUTENSOR_OP_IDENTITY,
                                C.data, C.inds, CUTENSOR_OP_ADD)
    C
end

function Base.:(*)(A::CuTensor, B::CuTensor)
    tC = promote_type(eltype(A), eltype(B))
    A_uniqs = [(idx, i) for (idx, i) in enumerate(A.inds) if !(i in B.inds)]
    B_uniqs = [(idx, i) for (idx, i) in enumerate(B.inds) if !(i in A.inds)]
    A_sizes = map(x->size(A,x[1]), A_uniqs)
    B_sizes = map(x->size(B,x[1]), B_uniqs)
    A_inds = map(x->x[2], A_uniqs)
    B_inds = map(x->x[2], B_uniqs)
    C = CuTensor(CUDA.zeros(tC, Dims(vcat(A_sizes, B_sizes))), vcat(A_inds, B_inds))
    return mul!(C, A, B)
end


## LinearAlgebra

using LinearAlgebra

function LinearAlgebra.axpy!(a, X::CuTensor, Y::CuTensor)
    elementwise_binary_execute!(a, X.data, X.inds, CUTENSOR_OP_IDENTITY,
                                one(eltype(Y)), Y.data, Y.inds, CUTENSOR_OP_IDENTITY,
                                Y.data, Y.inds, CUTENSOR_OP_ADD)
    return Y
end

function LinearAlgebra.axpby!(a, X::CuTensor, b, Y::CuTensor)
    elementwise_binary_execute!(a, X.data, X.inds, CUTENSOR_OP_IDENTITY,
                                b, Y.data, Y.inds, CUTENSOR_OP_IDENTITY,
                                Y.data, Y.inds, CUTENSOR_OP_ADD)
    return Y
end

function LinearAlgebra.mul!(C::CuTensor, A::CuTensor, B::CuTensor, α::Number, β::Number)
   contract!(α, A.data, A.inds, CUTENSOR_OP_IDENTITY,
             B.data, B.inds, CUTENSOR_OP_IDENTITY, β,
             C.data, C.inds, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_IDENTITY;
             jit=CUTENSOR_JIT_MODE_DEFAULT)
   return C
end
