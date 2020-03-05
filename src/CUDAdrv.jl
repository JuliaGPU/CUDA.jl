module CUDAdrv

using CUDAapi

using CEnum

using Printf


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

# initialization of CUDA is deferred to run-time (to avoid package import latency,
# and to improve use on systems without a GPU), at the point of the first ccall.

const __initialized__ = Ref{Union{Nothing,Bool}}(nothing)
const __libcuda = Sys.iswindows() ? :nvcuda : :libcuda

"""
    functional(show_reason=false)

Check if the package has been initialized successfully and is ready to use.

This call is intended for packages that support conditionally using an available GPU. If you
fail to check whether CUDA is functional, actual use of functionality might warn and error.
"""
function functional(show_reason::Bool=false)
    if __initialized__[] === nothing
        __runtime_init__(show_reason)
    end
    __initialized__[]
end

function __runtime_init__(show_reason::Bool)
    __initialized__[] = false

    if haskey(ENV, "_") && basename(ENV["_"]) == "rr"
        show_reason && @error("Running under rr, which is incompatible with CUDA")
        return
    end

    @debug "Initializing CUDA driver"
    ccall((:cuInit, __libcuda), CUresult, (UInt32,), 0)
    __initialized__[] = true

    if version() < v"9"
        @warn "CUDAdrv.jl only supports NVIDIA drivers for CUDA 9.0 or higher (yours is for CUDA $(version()))"
    end
end

function libcuda()
    @assert functional(true) "CUDAdrv.jl is not functional"
    __libcuda
end

end
