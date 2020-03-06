module CuArrays

using CUDAapi, CUDAdrv, CUDAnative

using GPUArrays

export CuArray, CuVector, CuMatrix, CuVecOrMat, cu
export CUBLAS, CUSPARSE, CUSOLVER, CUFFT, CURAND, CUDNN, CUTENSOR

import LinearAlgebra

using Adapt

using Libdl

using Requires


## source code includes

include("bindeps.jl")

# core array functionality
include("memory.jl")
include("array.jl")
include("gpuarrays.jl")
include("subarray.jl")
include("utils.jl")

# integrations and specialized functionality
include("indexing.jl")
include("broadcast.jl")
include("mapreduce.jl")
include("accumulate.jl")
include("linalg.jl")
include("nnlib.jl")

# vendor libraries
include("blas/CUBLAS.jl")
include("sparse/CUSPARSE.jl")
include("solver/CUSOLVER.jl")
include("fft/CUFFT.jl")
include("rand/CURAND.jl")
include("dnn/CUDNN.jl")
include("tensor/CUTENSOR.jl")

include("deprecated.jl")


## initialization

function __init__()
    # package integrations
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" include("forwarddiff.jl")

    __init_memory__()

    # NOTE: we only perform minimal initialization here that does not require CUDA or a GPU.
    #       most of the actual initialization is deferred to run time:
    #       see bindeps.jl for initialization of CUDA binary dependencies.
end

end # module
