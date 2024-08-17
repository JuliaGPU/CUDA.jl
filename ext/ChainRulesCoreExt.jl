# compatibility with ChainRulesCore

module ChainRulesCoreExt

using CUDA: CuArray

isdefined(Base, :get_extension) ? (import ChainRulesCore) : (import ..ChainRulesCore)

## support ChainRulesCore inplaceability

ChainRulesCore.is_inplaceable_destination(::CuArray) = true

end
