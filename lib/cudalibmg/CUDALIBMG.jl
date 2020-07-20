module CUDALIBMG

using ..CUDA
using ..CUDA: libcudalibmg, unsafe_free!, @retry_reclaim, @runtime_ccall, @checked, cudaDataType
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
