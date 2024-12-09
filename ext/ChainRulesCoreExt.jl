# compatibility with ChainRulesCore

module ChainRulesCoreExt

using CUDA: CuArray, CUDA

isdefined(Base, :get_extension) ? (import ChainRulesCore) : (import ..ChainRulesCore)

## support ChainRulesCore inplaceability

ChainRulesCore.is_inplaceable_destination(::CuArray) = true

# allow usage of rand with Zygote
ChainRulesCore.@non_differentiable CUDA.randn(::Any...)

end
