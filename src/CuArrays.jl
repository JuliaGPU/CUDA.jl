__precompile__()

module CuArrays

using CUDAdrv, CUDAnative
import CUDAnative: cudaconvert

export CuArray, CuVector, CuMatrix, cu

include("memory.jl")
include("array.jl")
include("utils.jl")
include("indexing.jl")
include("broadcast.jl")
include("reduction.jl")

include("../deps/ext.jl")
include("blas/BLAS.jl")
if libcudnn ≠ nothing
  include("cudnn/CUDNN.jl")
end

end # module
