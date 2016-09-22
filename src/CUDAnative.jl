isdefined(Base, :__precompile__) && __precompile__()

module CUDAnative

using CUDAdrv

# non-exported utility functions
import CUDAdrv: debug, DEBUG, trace, TRACE


include("util.jl")
include("types.jl")

include("execution.jl")
include("intrinsics.jl")
include("array.jl")


function __init__()
    __init_util__()
end

end
