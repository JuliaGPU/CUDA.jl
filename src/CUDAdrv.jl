__precompile__()

module CUDAdrv

using Compat
import Compat.String

include("util/logging.jl")

include("pointer.jl")

include("errors.jl")
include("base.jl")
include("version.jl")
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
    __init_library__()
end


end
