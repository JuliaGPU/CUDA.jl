module CUDALIBMG

using CUDAapi

using CUDAdrv
using CUDAdrv: CUstream

using CUDAnative

using ..CuArrays
using ..CuArrays: libcudalibmg, unsafe_free!, @retry_reclaim
using LinearAlgebra

using CEnum

const cudaDataType_t = cudaDataType

# core library
include("libcudalibmg_common.jl")
include("error.jl")
include("libcudalibmg.jl")

# low-level wrappers
include("wrappers.jl")

end
