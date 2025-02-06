module NVML

using ..APIUtils

using ..CUDA

using CEnum: @cenum

import Libdl


export has_nvml

function libnvml()
    @memoize begin
        if Sys.iswindows()
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
    end::String
end

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
