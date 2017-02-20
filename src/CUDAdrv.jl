__precompile__()

module CUDAdrv

using Compat
import Compat.String

ext = joinpath(dirname(@__FILE__), "..", "deps", "ext.jl")
isfile(ext) || error("Unable to load $ext\n\nPlease re-run Pkg.build(\"CUDAdrv\"), and restart Julia.")
include(ext)
const libcuda = libcuda_path

include("util/logging.jl")

# CUDA API wrappers
include("errors.jl")
include("base.jl")
include("devices.jl")
include("context.jl")
include("pointer.jl")
include("module.jl")
include("memory.jl")
include("stream.jl")
include("execution.jl")
include("events.jl")
include("profile.jl")

include("gc.jl")
include("array.jl")


function __init__()
    # check validity of CUDA library
    @debug("Checking validity of $(libcuda_path)")
    if version() != libcuda_version
        error("CUDA library version has changed. Please re-run Pkg.build(\"CUDA\") and restart Julia.")
    end

    __init_logging__()
    @apicall(:cuInit, (Cint,), 0)
end


end
