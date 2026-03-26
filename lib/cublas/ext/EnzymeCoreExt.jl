# Enzyme rules for cuBLAS
module EnzymeCoreExt

using cuBLAS
using EnzymeCore

function EnzymeCore.EnzymeRules.inactive(::typeof(cuBLAS.handle))
    return nothing
end
function EnzymeCore.EnzymeRules.inactive_noinl(::typeof(cuBLAS.version))
    return nothing
end

end
