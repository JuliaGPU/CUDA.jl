__precompile__()

module CUDAdrv

using Compat
using Compat.String

using CUDAapi

const ext = joinpath(dirname(@__DIR__), "deps", "ext.jl")
const configured = if isfile(ext)
    include(ext)
    true
else
    # enable CUDAdrv.jl to be loaded when the build failed, simplifying downstream use.
    # remove this when we have proper support for conditional modules.
    const libcuda_version = v"5.5"
    const libcuda_vendor = "none"
    const libcuda_path = nothing
    false
end
const libcuda = libcuda_path

include("base.jl")

# CUDA Driver API wrappers
include("init.jl")
include("errors.jl")
include("version.jl")
include("devices.jl")
include("context.jl")
include(joinpath("context", "primary.jl"))
include("pointer.jl")   # not a wrapper, but used by them
include("module.jl")
include("memory.jl")
include("stream.jl")
include("events.jl")
include("execution.jl")
include("profile.jl")

include("array.jl")

include("deprecated.jl")

end
