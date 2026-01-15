module NVML


using ..GPUToolbox

using ..CUDA

using CEnum: @cenum

import Libdl


export has_nvml

#TODO function libnvml()
#TODO     @memoize begin
#TODO         if Sys.iswindows()
#TODO             # the NVSMI dir isn't added to PATH by the installer
#TODO             nvsmi = joinpath(ENV["ProgramFiles"], "NVIDIA Corporation", "NVSMI")
#TODO             if isdir(nvsmi)
#TODO                 joinpath(nvsmi, "nvml.dll")
#TODO             else
#TODO                 # let's just hope for the best
#TODO                 "nvml"
#TODO             end
#TODO         else
#TODO             "libnvidia-ml.so.1"
#TODO         end
#TODO     end::String
#TODO end
const libnvml::String = "libnvidia-ml.so.1"

function has_nvml()
    @memoize begin
        if CUDA.is_tegra()
            # XXX: even though Orin supports NVML, we don't know how to
            #      look up the device (CUDA.jl#2580)
            return false
        end

        if Libdl.dlopen(libnvml(); throw_error=false) === nothing
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
