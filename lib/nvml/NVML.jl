module NVML

using ..APIUtils

using ..CUDA

using CEnum

using Memoize

using Libdl


@memoize function libnvml()
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

@memoize function has_nvml()
    if Libdl.dlopen(libnvml(); throw_error=false) === nothing
        return false
    end

    # JuliaGPU/CUDA.jl#860: initialization can fail on Windows
    try
        initialize_api()
    catch err
        @error "Cannot use NVML, as it failed to initialize" exception=(err, catch_backtrace())
        return false
    end

    return true
end


# core library
include("libnvml_common.jl")
include("error.jl")
include("libnvml.jl")
include("libnvml_deprecated.jl")

# wrappers
include("system.jl")
include("device.jl")

end
