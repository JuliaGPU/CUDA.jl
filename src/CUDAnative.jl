# NOTE: precompilation is supported, but slows down development
__precompile__(false)

module CUDAnative


include("errors.jl")

include("util.jl")
include("funmap.jl")
include("base.jl")
include("types.jl")
include("devices.jl")
include("context.jl")
include("jit.jl")
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
    __init_util__()
    __init_base__()
end

end
