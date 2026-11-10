"""
Julia wrappers for NVML, the NVIDIA Management Library, which queries the state of the GPUs
in the system (clocks, power, temperature, memory, processes) without creating a CUDA
context. See [`has_nvml`](@ref) and [`NVML.Device`](@ref).
"""
module NVML

using GPUToolbox

using CUDACore
import CUDACore: name, uuid

using CEnum: @cenum

import Libdl


export has_nvml

const libnvml::String = if Sys.iswindows()
    # the NVSMI dir isn't added to PATH by the installer
    nvsmi = joinpath(ENV["ProgramFiles"], "NVIDIA Corporation", "NVSMI")
    if isdir(nvsmi)
        joinpath(nvsmi, "nvml.dll")
    else
        # let's just hope for the best
        "nvml"
    end
else
    "libnvidia-ml.so.1"
end

"""
    has_nvml() -> Bool

Whether NVML, the NVIDIA Management Library that ships with the driver, can be used. It
can't when the library is missing or fails to initialize, and CUDA.jl doesn't use it on
Tegra devices, where it can't look up the NVML device of a `CuDevice`. Check this before calling any other function in this module.
"""
function has_nvml()
    @memoize begin
        if CUDACore.is_tegra()
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

include("precompile.jl")

end
