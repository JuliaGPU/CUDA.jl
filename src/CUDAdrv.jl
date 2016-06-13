# NOTE: precompilation is supported, but slows down development
__precompile__(false)

module CUDAdrv

using Compat
import Compat.String


include("util/logging.jl")

include("errors.jl")
include("funmap.jl")
include("base.jl")
include("types.jl")
include("devices.jl")
include("context.jl")
include("module.jl")
include("stream.jl")
include("execution.jl")
include("jit.jl")
include("events.jl")
include("profile.jl")

include("memory.jl")
include("arrays.jl")


function __init__()
    __init_logging__()
    __init_base__()
end


end
