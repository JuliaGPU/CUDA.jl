__precompile__()

module CUDAapi

using Compat
VERSION >= v"0.7.0-DEV.3382" && using Libdl

include("util.jl")
include("compatibility.jl")
include("discovery.jl")

end
