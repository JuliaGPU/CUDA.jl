__precompile__()

module CuArrays

using CUDAdrv, CUDAnative
import CUDAnative: cudaconvert

export CuArray, CuVector, CuMatrix, CuVecOrMat, cu

include("memory.jl")
include("array.jl")
include("utils.jl")
include("indexing.jl")
include("broadcast.jl")
include("reduction.jl")

include("../deps/ext.jl")
include("blas/CUBLAS.jl")
include("solver/CUSOLVER.jl")
include("gpuarray_interface.jl")

cudnn_available() = libcudnn ≠ nothing
if cudnn_available()
  include("dnn/CUDNN.jl")
end

end # module
