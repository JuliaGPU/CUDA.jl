module CUDA

using GPUCompiler

using GPUArrays

using LLVM
using LLVM.Interop

using Adapt

using Requires

using LinearAlgebra


## source code includes

include("pointer.jl")

# core library
include("../lib/utils/APIUtils.jl")
include("../lib/cudadrv/CUDAdrv.jl")

# essential stuff
include("initialization.jl")
include("state.jl")

# binary dependencies
include("../deps/discovery.jl")
include("../deps/compatibility.jl")
include("../deps/bindeps.jl")

# device functionality (needs to be loaded first, because of generated functions)
include("device/pointer.jl")
include("device/array.jl")
include("device/intrinsics.jl")
include("device/llvm.jl")
include("device/runtime.jl")
include("device/texture.jl")

# compiler libraries
include("../lib/cupti/CUPTI.jl")
include("../lib/nvtx/NVTX.jl")
export CUPTI, NVTX

# compiler implementation
include("compiler/gpucompiler.jl")
include("compiler/execution.jl")
include("compiler/exceptions.jl")
include("compiler/reflection.jl")

# array abstraction
include("pool.jl")
include("array.jl")
include("gpuarrays.jl")
include("subarray.jl")
include("utilities.jl")
include("texture.jl")

# array libraries
include("../lib/complex.jl")
include("../lib/library_types.jl")
include("../lib/cublas/CUBLAS.jl")
include("../lib/cudalibmg/CUDALIBMG.jl")
include("../lib/blasmg/CUBLASMG.jl")
include("../lib/cusparse/CUSPARSE.jl")
include("../lib/cusolver/CUSOLVER.jl")
include("../lib/cufft/CUFFT.jl")
include("../lib/curand/CURAND.jl")
include("../lib/cudnn/CUDNN.jl")
include("../lib/cutensor/CUTENSOR.jl")
export CUBLAS, CUSPARSE, CUSOLVER, CUFFT, CURAND, CUDNN, CUTENSOR

# integrations and specialized functionality
include("indexing.jl")
include("broadcast.jl")
include("mapreduce.jl")
include("accumulate.jl")
include("linalg.jl")
include("nnlib.jl")
include("iterator.jl")
include("statistics.jl")

# other libraries
include("../lib/nvml/NVML.jl")
const has_nvml = NVML.has_nvml
export NVML, has_nvml

include("deprecated.jl")

end
