module NVML


using ..GPUToolbox

using ..CUDA

using CEnum: @cenum

import Libdl


export has_nvml

# Compile-time constant for `ccall` (Julia 1.13 needs this)
const libnvml::String = @static Sys.iswindows() ? "nvml" : "libnvidia-ml.so.1"

function __init__()
    if Sys.iswindows()
        # NVSMI dir isn't added to PATH by the installer; add it to Julia's DLL search path.
        nvsmi = joinpath(get(ENV, "ProgramFiles", raw"C:\Program Files"), "NVIDIA Corporation", "NVSMI")
        if isdir(nvsmi) && !(nvsmi in Libdl.DL_LOAD_PATH)
            pushfirst!(Libdl.DL_LOAD_PATH, nvsmi)
        end
    end
end

function has_nvml()
    @memoize begin
        if CUDA.is_tegra()
            # XXX: even though Orin supports NVML, we don't know how to
            #      look up the device (CUDA.jl#2580)
            return false
        end

        if Libdl.dlopen(libnvml; throw_error=false) === nothing
            return false
        end

        # JuliaGPU/CUDA.jl#860: initialization can fail on Windows
        try
            initialize_context()
        catch err
            @error "Cannot use NVML, as it failed to initialize" exception=(err, catch_backtrace())
            return false
        end

        return true
    end::Bool
end


# core library
include("libnvml.jl")

# wrappers
include("error.jl")
include("system.jl")
include("device.jl")

end
