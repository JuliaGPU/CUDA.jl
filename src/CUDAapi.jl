__precompile__()

module CUDAapi

using Compat

include("logging.jl")

include("compatibility.jl")
include("discovery.jl")

function __init__()
    __init_logging__()
end

end
