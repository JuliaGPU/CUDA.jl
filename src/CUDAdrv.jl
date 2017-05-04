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

include("errors.jl")
include("base.jl")

# CUDA API wrappers
include("devices.jl")
include("context.jl")
include("pointer.jl")   # not a wrapper, but depends on context.jl
include(joinpath("context", "primary.jl"))
include("module.jl")
include("memory.jl")
include("stream.jl")
include("execution.jl")
include("events.jl")
include("profile.jl")

include("array.jl")

function __init__()
    # check validity of CUDA library
    @debug("Checking validity of $(libcuda_path)")
    if version() != libcuda_version
        error("CUDA library version has changed. Please re-run Pkg.build(\"CUDAdrv\") and restart Julia.")
    end

    __init_logging__()
    if haskey(ENV, "_") && basename(ENV["_"]) == "rr"
        warn("Running under rr, which is incompatible with CUDA; disabling initialization.")
    else
        @apicall(:cuInit, (Cint,), 0)
    end
end

end
