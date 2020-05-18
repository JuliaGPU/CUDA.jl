module NVTX

using ..CUDA
using ..CUDA: libnvtx, @checked, @runtime_ccall
using ..CUDA: CUstream, CUdevice, CUcontext, CUevent

using CEnum

using ExprTools

# core library
initialize_api() = return
include("libnvtx_common.jl")
include("libnvtx.jl")

include("highlevel.jl")

end
