__precompile__()

module CUDAdrv

using Compat
using Compat.String

const ext = joinpath(dirname(@__DIR__), "deps", "ext.jl")
if isfile(ext)
    include(ext)
else
    error("Unable to load dependency file $ext.\nPlease run Pkg.build(\"CUDAdrv\") and restart Julia.")
end
const libcuda = libcuda_path

include(joinpath("util", "logging.jl"))

include("types.jl")
include("base.jl")

# CUDA API wrappers
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
include("execution.jl")
include("events.jl")
include("profile.jl")

include("array.jl")


end
