module CUDAdrv

using CEnum

using Printf


## source code includes

# essential functionality
include("pointer.jl")
const CUdeviceptr = CuPtr{Cvoid}

# low-level wrappers
const libcuda = Sys.iswindows() ? :nvcuda : :libcuda
include("libcuda_common.jl")
include("error.jl")
include("libcuda.jl")
include("libcuda_aliases.jl")

include("util.jl")

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

const __initialized__ = Ref(false)
functional() = __initialized__[]

function __init__()
    try
        # barrier to avoid compiling `ccall`s to unavailable libraries
        inferencebarrier(__hidden_init__)()
        __initialized__[] = true
    catch ex
        # don't actually fail to keep the package loadable
        @debug("CUDAdrv.jl failed to initialize; the package will not be functional.",
               exception=(ex, catch_backtrace()))
    end
end

if VERSION >= v"1.3.0-DEV.35"
    using Base: inferencebarrier
else
    inferencebarrier(@nospecialize(x)) = Ref{Any}(x)[]
end

function __hidden_init__()
    if haskey(ENV, "_") && basename(ENV["_"]) == "rr"
        error("Running under rr, which is incompatible with CUDA")
    end

    cuInit(0)

    if version() <= v"9"
        @warn "CUDAdrv.jl only supports NVIDIA drivers for CUDA 9.0 or higher (yours is for CUDA $(version()))"
    end
end

end
