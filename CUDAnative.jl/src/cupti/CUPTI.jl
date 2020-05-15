module CUPTI

using CUDAapi

using CUDAdrv: CUcontext, CUstream, CUdevice, CUdevice_attribute,
               CUgraph, CUgraphNode, CUgraphNodeType, CUgraphExec

# TODO: move to CUDAdrv
struct CUuuid
    bytes::NTuple{16,Int8}
end

using ..CUDAnative
using ..CUDAnative: libcupti

using CEnum

# core library
include("libcupti_common.jl")
include("error.jl")
include("libcupti.jl")

end
