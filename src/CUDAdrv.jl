module CUDAdrv

using CUDAapi

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
    silent = parse(Bool, get(ENV, "JULIA_CUDA_SILENT", "false"))
    verbose = parse(Bool, get(ENV, "JULIA_CUDA_VERBOSE", "false"))
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0

    try
        if haskey(ENV, "_") && basename(ENV["_"]) == "rr"
            error("Running under rr, which is incompatible with CUDA")
        end

        cuInit(0)

        if version() <= v"9"
            @warn "CUDAdrv.jl only supports NVIDIA drivers for CUDA 9.0 or higher (yours is for CUDA $(version()))"
        end

        __initialized__[] = true
    catch ex
        # don't actually fail to keep the package loadable
        if !silent && !precompiling
            if verbose
                @error "CUDAdrv.jl failed to initialize" exception=(ex, catch_backtrace())
            else
                @info "CUDAdrv.jl failed to initialize, GPU functionality unavailable (set JULIA_CUDA_SILENT or JULIA_CUDA_VERBOSE to silence or expand this message)"
            end
        end
    end
end

end
