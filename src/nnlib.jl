using NNlib
import NNlib: conv!, ∇conv_filter!, ∇conv_data!,
  maxpool!, meanpool!, ∇maxpool!, ∇meanpool!,
  softmax, softmax!, ∇softmax!, logsoftmax, logsoftmax!, ∇logsoftmax
using CUDAnative

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

@cufunc softplus(x) = log1p(exp(x))
