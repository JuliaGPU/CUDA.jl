module CUDAdrv

using CEnum

using Printf
using Libdl


## source code includes

# essential functionality
include("pointer.jl")
const CUdeviceptr = CuPtr{Cvoid}

# low-level wrappers
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

function __init__()
    if ccall(:jl_generating_output, Cint, ()) == 0
        if haskey(ENV, "_") && basename(ENV["_"]) == "rr"
            @warn "Running under rr, which is incompatible with CUDA; disabling initialization."
        else
            cuInit(0)
        end
    end
end

end
