isdefined(Base, :__precompile__) && __precompile__()

module CUDAdrv

using Compat
import Compat.String

# TODO: organize according to the CUDA Driver API reference

include("util/logging.jl")

include("errors.jl")
include("funmap.jl")
include("base.jl")
include("pointer.jl")
include("devices.jl")
include("context.jl")
include("module.jl")
include("memory.jl")
include("stream.jl")
include("execution.jl")
include("events.jl")
include("profile.jl")

include("array.jl")


function __init__()
    __init_logging__()
    @static if !DEBUG
        __init_base__()
    end
end


end
