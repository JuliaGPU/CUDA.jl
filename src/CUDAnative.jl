isdefined(Base, :__precompile__) && __precompile__()

module CUDAnative

using CUDAdrv
using LLVM

# non-exported utility functions
import CUDAdrv: debug, DEBUG, trace, TRACE


include("util.jl")
include("types.jl")

include("jit.jl")
include("execution.jl")
include("intrinsics.jl")
include("array.jl")


function __init__()
    __init_util__()
    __init_jit__()
end

end
