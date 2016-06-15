isdefined(Base, :__precompile__) && __precompile__()

module CUDAnative

using CUDAdrv


include("util.jl")

include("compilation.jl")

include("execution.jl")
include("intrinsics.jl")
include("arrays.jl")


function __init__()
    __init_util__()
end

end
