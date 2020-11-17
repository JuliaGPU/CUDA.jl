using NNlib

# Activation functions
@cufunc softplus(x::Real) = ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))

@cufunc logσ(x::Real) = -softplus(-x)

@cufunc function gelu(x::Real)
    p = oftype(x / 1, π)
    λ = oftype(x / 1, √(2 / p))
    α = oftype(x / 1, 0.044715)
    h = oftype(x / 1, 0.5)
    h * x * (one(x) + tanh(λ * (x + α * x^3)))
end

@cufunc lisht(x::Real) = x * tanh(x)

@cufunc logcosh(x::Real) = x + softplus(-2x) - log(oftype(x, 2))

@cufunc mish(x::Real) = x * tanh(softplus(x))

@cufunc tanhshrink(x::Real) = x - tanh(x)


# Batched matrix multiplication
# 1st argument is produced by NNlib.storage_type(A)
NNlib._batched_gemm!(::Type{<:CuArray}, transA::Char, transB::Char, α::Number, A, B, β::Number, C) =
     CUBLAS.gemm_strided_batched!(transA, transB, α, A, B, β, C)

Base.unsafe_convert(::Type{CuPtr{T}}, A::NNlib.BatchedAdjOrTrans{T}) where {T} =
    Base.unsafe_convert(CuPtr{T}, parent(A))
