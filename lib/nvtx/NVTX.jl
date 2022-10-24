module NVTX

using ..CUDA_Runtime

using ..CUDA
using ..CUDA: @checked
using ..CUDA: CUstream, CUdevice, CUcontext, CUevent

using CEnum: @cenum

using ExprTools: splitdef, combinedef


# core library
include("libnvtx.jl")

include("highlevel.jl")

end
