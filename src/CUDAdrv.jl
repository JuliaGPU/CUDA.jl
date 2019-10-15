module CUDAdrv

using CUDAapi

using CEnum

using Printf
using Libdl


## discovery

let
    # NOTE: on macOS, the driver is part of the toolkit
    toolkit_dirs = find_toolkit()

    global const libcuda = find_cuda_library("cuda", toolkit_dirs)
    if libcuda == nothing
        error("Could not find CUDA driver library")
    end
    Base.include_dependency(libcuda)

    @debug "Found CUDA at $libcuda"


    # backwards-compatible flags

    global const configured = true
end


## source code includes

# essential functionality
include("pointer.jl")
const CUdeviceptr = CuPtr{Cvoid}

# low-level wrappers
include("libcuda_common.jl")
include("error.jl")
include("libcuda.jl")
include("libcuda_aliases.jl")

# high-level wrappers
include("version.jl")
global const libcuda_version = version()
include("devices.jl")
include("context.jl")
include(joinpath("context", "primary.jl"))
include("stream.jl")
include("memory.jl")
include("module.jl")
include("events.jl")
include("execution.jl")
include("profile.jl")
include("occupancy.jl")

include("deprecated.jl")


## initialization

function __init__()
    if !ispath(libcuda) || version() != libcuda_version
        cachefile = if VERSION >= v"1.3-"
            Base.compilecache_path(Base.PkgId(CUDAdrv))
        else
            abspath(DEPOT_PATH[1], Base.cache_file_entry(Base.PkgId(CUDAdrv)))
        end
        rm(cachefile)
        error("Your set-up changed, and CUDAdrv.jl needs to be reconfigured. Please load the package again.")
    end

    if haskey(ENV, "_") && basename(ENV["_"]) == "rr"
        @warn "Running under rr, which is incompatible with CUDA; disabling initialization."
    else
        cuInit(0)
    end
end

end
