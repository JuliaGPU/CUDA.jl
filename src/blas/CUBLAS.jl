module BLAS

import Base: one, zero
using CUDAdrv
using ..CuArrays: CuArray, CuVector, CuMatrix, CuVecOrMat, libcublas, configured

const BlasChar = Char

include("util.jl")
include("libcublas_types.jl")
include("error.jl")

# Typedef needed by libcublas
const cudaStream_t = Ptr{Nothing}

include("libcublas.jl")

const libcublas_handle = Ref{cublasHandle_t}()
function __init__()
    configured || return

    cublasCreate_v2(libcublas_handle)
    atexit(()->cublasDestroy_v2(libcublas_handle[]))
end

include("wrap.jl")
include("highlevel.jl")

end
