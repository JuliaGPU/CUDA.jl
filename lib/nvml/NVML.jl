module NVML

using ..APIUtils

using ..CUDA

using CEnum

using Libdl

function libnvml()
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
end
has_nvml() = Libdl.dlopen(libnvml(); throw_error=false) !== nothing

# core library
include("libnvml_common.jl")
include("error.jl")
include("libnvml.jl")
include("libnvml_deprecated.jl")

# wrappers
include("system.jl")
include("device.jl")

end
