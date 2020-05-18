module CUPTI

using ..APIUtils

using ..CUDA
using ..CUDA: libcupti
using ..CUDA: CUcontext, CUstream, CUdevice, CUdevice_attribute,
              CUgraph, CUgraphNode, CUgraphNodeType, CUgraphExec

# TODO: move to APIUtils
struct CUuuid
    bytes::NTuple{16,Int8}
end

using CEnum

# core library
include("libcupti_common.jl")
include("error.jl")
include("libcupti.jl")

end
