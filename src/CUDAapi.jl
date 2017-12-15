__precompile__()

module CUDAapi

using Compat

include("util.jl")

include("compatibility.jl")
include("discovery.jl")

function __init__()
    __init_logging__()
end

end
