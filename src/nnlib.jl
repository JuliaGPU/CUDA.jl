using NNlib

# Activation functions

@cufunc σ(x) = ifelse(x < -80, zero(x), one(x) / (one(x) + exp(-x)))

@cufunc function logσ(x)
  max_v = max(zero(x), -x)
  z = exp(-max_v) + exp(-x-max_v)
  -(max_v + log(z))
end

@cufunc elu(x, α = one(x)) =
  ifelse(x ≥ 0, x/1, α * (exp(x) - one(x)))

@cufunc swish(x) = x * σ(x)

@cufunc function gelu(x)
  λ = oftype(x/1, √(2/π))
  α = oftype(x/1, 0.044715)
  h = oftype(x/1, 0.5)
  h * x * (one(x) + tanh(λ * (x + α * x^3)))
end

@cufunc function selu(x)
  λ = oftype(x/1, 1.0507009873554804934193349852946)
  α = oftype(x/1, 1.6732632423543772848170429916717)
  λ * ifelse(x > 0, x/1, α * (exp(x) - 1))
end

@cufunc softplus(x) = ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))

# Batched matrix multiplication

_BATCHED_GEMM_LIST = [
    (:(CuArray{T, 3}), 'N'),
    (:(NNlib.BatchedTranspose{T, <:CuArray{T, 3}}), 'T'),
    (:(NNlib.BatchedAdjoint{T, <:CuArray{T, 3}}), 'C')
]

for (TA, transA) in _BATCHED_GEMM_LIST, (TB, transB) in _BATCHED_GEMM_LIST
    @eval function NNlib.batched_mul!(C::CuArray{T, 3}, A::$TA, B::$TB) where {T<:NNlib._GemmFloat}
        CuArrays.CUBLAS.gemm_strided_batched!($transA, $transB, one(T), NNlib._unbatch(A), NNlib._unbatch(B), zero(T), C)
        C
    end
end
