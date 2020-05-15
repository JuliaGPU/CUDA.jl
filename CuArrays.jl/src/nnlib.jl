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

const batched_gemm_args = [
    (:(CuArray{T, 3}), 'N'),
    (:(NNlib.BatchedTranspose{T, <:CuArray{T, 3}}), 'T'),
    (:(NNlib.BatchedAdjoint{T, <:CuArray{T, 3}}), 'C')
]

for (TA, transA) in batched_gemm_args, (TB, transB) in batched_gemm_args
    @eval function NNlib.batched_mul!(C::CuArray{T, 3}, A::$TA, B::$TB) where {T<:CUBLAS.CublasFloat}
        CuArrays.CUBLAS.gemm_strided_batched!($transA, $transB, one(T), NNlib._unbatch(A), NNlib._unbatch(B), zero(T), C)
        C
    end
end
