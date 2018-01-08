module BLAS

import Base: one, zero
using CUDAdrv
using ..CuArrays: CuArray, CuVector, CuMatrix, CuVecOrMat, libcublas, configured

const BlasChar = Char

include("util.jl")
include("libcublas_types.jl")
include("error.jl")

# Typedef needed by libcublas
const cudaStream_t = Ptr{Void}

include("libcublas.jl")

# setup cublas handle
const cublashandle = cublasHandle_t[0]

function __init__()
    configured || return

    cublasCreate_v2(cublashandle)
    # destroy cublas handle at julia exit
    atexit(()->cublasDestroy_v2(cublashandle[1]))
end

include("wrap.jl")
include("highlevel.jl")

end
