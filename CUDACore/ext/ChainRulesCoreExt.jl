# compatibility with ChainRulesCore

module ChainRulesCoreExt

using CUDACore: CuArray

import ChainRulesCore

## support ChainRulesCore inplaceability

ChainRulesCore.is_inplaceable_destination(::CuArray) = true

end
