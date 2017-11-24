__precompile__()

module CUDAdrv

using Compat
using Compat.String

using CUDAapi

const ext = joinpath(dirname(@__DIR__), "deps", "ext.jl")
isfile(ext) || error("CUDAdrv.jl has not been built, please run Pkg.build(\"CUDAdrv\").")
include(ext)
if !configured
    # default (non-functional) values for critical variables,
    # making it possible to _load_ the package at all times.
    const libcuda_version = v"5.5"
    const libcuda_vendor = "none"
    const libcuda_path = nothing
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
include("module.jl")
include("stream.jl")
include("memory.jl")
include("events.jl")
include("execution.jl")
include("profile.jl")

include("array.jl")

include("deprecated.jl")

end
