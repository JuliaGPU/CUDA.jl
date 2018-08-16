__precompile__()

module CuArrays

using CUDAdrv, CUDAnative
import CUDAnative: cudaconvert

using GPUArrays

export CuArray, CuVector, CuMatrix, CuVecOrMat, cu, cuzeros, cuones

import LinearAlgebra

const ext = joinpath(dirname(@__DIR__), "deps", "ext.jl")
isfile(ext) || error("CuArrays.jl has not been built, please run Pkg.build(\"CuArrays\").")
include(ext)
if !configured
    # default (non-functional) values for critical variables,
    # making it possible to _load_ the package at all times.
    const libcublas = nothing
    const libcufft = nothing
    const libcusolver = nothing
    const libcudnn = nothing
end

include("memory.jl")
include("array.jl")
include("utils.jl")
include("indexing.jl")
include("broadcast.jl")
include("mapreduce.jl")

include("blas/CUBLAS.jl")
include("solver/CUSOLVER.jl")
include("fft/CUFFT.jl")
include("rand/CURAND.jl")
include("gpuarray_interface.jl")

cudnn_available() = libcudnn ≠ nothing
if cudnn_available()
  include("dnn/CUDNN.jl")
end

function __init__()
    if !configured
        @warn("CuArrays.jl has not been successfully built, and will not work properly.")
        @warn("Please run Pkg.build(\"CuArrays\") and restart Julia.")
        return
    end

    __init_memory__()
end

end # module
