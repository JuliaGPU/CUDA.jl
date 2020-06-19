module CUDA

using GPUCompiler

using GPUArrays

using LLVM
using LLVM.Interop

using Adapt

using Requires

using LinearAlgebra


## source code includes

const root = dirname(@__DIR__)

include("$root/src/pointer.jl")

# core library
include("$root/lib/utils/APIUtils.jl")
include("$root/lib/cudadrv/CUDAdrv.jl")

# essential stuff
include("$root/src/initialization.jl")
include("$root/src/state.jl")

# binary dependencies
include("$root/deps/discovery.jl")
include("$root/deps/compatibility.jl")
include("$root/deps/bindeps.jl")

# device functionality (needs to be loaded first, because of generated functions)
include("$root/src/device/pointer.jl")
include("$root/src/device/array.jl")
include("$root/src/device/cuda.jl")
include("$root/src/device/llvm.jl")
include("$root/src/device/runtime.jl")

# compiler libraries
include("$root/lib/cupti/CUPTI.jl")
include("$root/lib/nvtx/NVTX.jl")
export CUPTI, NVTX

# compiler implementation
include("$root/src/compiler/gpucompiler.jl")
include("$root/src/compiler/execution.jl")
include("$root/src/compiler/exceptions.jl")
include("$root/src/compiler/reflection.jl")

# array abstraction
include("$root/src/memory.jl")
include("$root/src/array.jl")
include("$root/src/gpuarrays.jl")
include("$root/src/subarray.jl")
include("$root/src/utilities.jl")

# array libraries
include("$root/lib/complex.jl")
include("$root/lib/library_types.jl")
include("$root/lib/cublas/CUBLAS.jl")
include("$root/lib/cusparse/CUSPARSE.jl")
include("$root/lib/cusolver/CUSOLVER.jl")
include("$root/lib/cufft/CUFFT.jl")
include("$root/lib/curand/CURAND.jl")
include("$root/lib/cudnn/CUDNN.jl")
include("$root/lib/cutensor/CUTENSOR.jl")
export CUBLAS, CUSPARSE, CUSOLVER, CUFFT, CURAND, CUDNN, CUTENSOR

# integrations and specialized functionality
include("$root/src/indexing.jl")
include("$root/src/broadcast.jl")
include("$root/src/mapreduce.jl")
include("$root/src/accumulate.jl")
include("$root/src/linalg.jl")
include("$root/src/nnlib.jl")
include("$root/src/iterator.jl")
include("$root/src/statistics.jl")

include("$root/src/deprecated.jl")

end
