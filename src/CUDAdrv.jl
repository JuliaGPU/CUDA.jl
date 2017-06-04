__precompile__()

module CUDAdrv

using Compat
using Compat.String

const ext = joinpath(dirname(@__DIR__), "deps", "ext.jl")
if isfile(ext)
    include(ext)
elseif haskey(ENV, "ONLY_LOAD")
    # special mode where the package is loaded without requiring a successful build.
    # this is useful for loading in unsupported environments, eg. Travis + Documenter.jl
    warn("Only loading the package, without activating any functionality.")
    const libcuda_path = ""
    const libcuda_version = v"999"  # make sure all functions are available
else
    error("Unable to load dependency file $ext.\nPlease run Pkg.build(\"CUDAdrv\") and restart Julia.")
end
const libcuda = libcuda_path

include(joinpath("util", "logging.jl"))

include("types.jl")
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
