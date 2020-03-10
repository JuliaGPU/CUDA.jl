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


## deferred initialization

# CUDA packages require complex initialization (discover CUDA, download artifacts, etc)
# that can't happen at module load time, so defer that to run time upon actual use.

const configured = Ref{Union{Nothing,Bool}}(nothing)

"""
    functional(show_reason=false)

Check if the package has been configured successfully and is ready to use.

This call is intended for packages that support conditionally using an available GPU. If you
fail to check whether CUDA is functional, actual use of functionality might warn and error.
"""
function functional(show_reason::Bool=false)
    if configured[] === nothing
        _functional(show_reason)
    end
    configured[]::Bool
end

const configure_lock = ReentrantLock()
@noinline function _functional(show_reason::Bool=false)
    lock(configure_lock) do
        if configured[] === nothing
            if __configure__(show_reason)
                configured[] = true
                try
                    __runtime_init__()
                catch
                    configured[] = false
                    rethrow()
                end
            else
                configured[] = false
            end
        end
    end
end

# macro to guard code that only can run after the package has successfully initialized
macro after_init(ex)
    quote
        @assert functional(true) "CUDAdrv.jl did not successfully initialize, and is not usable."
        $(esc(ex))
    end
end


## initialization

const __libcuda = Sys.iswindows() ? :nvcuda : :libcuda
libcuda() = @after_init(__libcuda)

function __configure__(show_reason::Bool)
    if haskey(ENV, "_") && basename(ENV["_"]) == "rr"
        show_reason && @error("Running under rr, which is incompatible with CUDA")
        return false
    end

    @debug "Initializing CUDA driver"
    ccall((:cuInit, __libcuda), CUresult, (UInt32,), 0)

    return true
end

function __runtime_init__()
    if version() < v"9"
        @warn "CUDAdrv.jl only supports NVIDIA drivers for CUDA 9.0 or higher (yours is for CUDA $(version()))"
    end
end

end
