module NVML

using ..APIUtils

using ..CUDA

using CEnum

using Libdl

libnvml() = Sys.iswindows() ? :nvml : Symbol("libnvidia-ml")
has_nvml() = Libdl.dlopen(libnvml(); throw_error=false) !== nothing

# core library
include("libnvml_common.jl")
include("error.jl")
include("libnvml.jl")

# wrappers
include("system.jl")
include("device.jl")

end
