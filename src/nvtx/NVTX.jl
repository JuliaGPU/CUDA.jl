module NVTX

using CUDAapi

using CUDAdrv: CUstream, CUdevice, CUcontext, CUevent

using CEnum

using MacroTools

using ..CUDAnative
using ..CUDAnative: libnvtx

# core library
include("libnvtx_common.jl")
include("libnvtx.jl")

include("highlevel.jl")

end
