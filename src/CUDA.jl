# NOTE: precompilation is supported, but slows down development
__precompile__(false)

module CUDA

using Logging


include("errors.jl")

include("util.jl")
include("base.jl")
include("types.jl")
include("devices.jl")
include("context.jl")
include("module.jl")
include("stream.jl")
include("execution.jl")
include("compilation.jl")

include("memory.jl")
include("arrays.jl")

include("native/execution.jl")
include("native/intrinsics.jl")
include("native/arrays.jl")

include("profile.jl")


function __init__()
    __init_base__()
    __init_util__()
end

end
