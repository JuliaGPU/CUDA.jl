module CUPTI

using ..APIUtils

using ..CUDA
using ..CUDA: libcupti
using ..CUDA: CUuuid, CUcontext, CUstream, CUdevice, CUdevice_attribute,
              CUgraph, CUgraphNode, CUgraphNodeType, CUgraphExec, CUaccessPolicyWindow

using CEnum

# core library
include("libcupti_common.jl")
include("error.jl")
include("libcupti.jl")

include("wrappers.jl")

end
