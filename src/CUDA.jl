module CUDA

using GPUCompiler

using GPUArrays

using LLVM
using LLVM.Interop
using Core: LLVMPtr

using Adapt: Adapt, adapt, WrappedArray

using Requires: @require

using LinearAlgebra

using BFloat16s: BFloat16

using ExprTools: splitdef, combinedef

# TODO: set lib versions in bindeps or so

# XXX: to be replaced by a JLL
include("../deps/Deps.jl")
using .Deps


## source code includes

include("pointer.jl")

# core library
include("../lib/utils/APIUtils.jl")
include("../lib/cudadrv/CUDAdrv.jl")

# essential stuff
include("initialization.jl")
include("debug.jl")

# device functionality (needs to be loaded first, because of generated functions)
include("device/utils.jl")
include("device/pointer.jl")
include("device/array.jl")
include("device/intrinsics.jl")
include("device/runtime.jl")
include("device/texture.jl")
include("device/random.jl")
include("device/quirks.jl")

# array essentials
include("pool.jl")
include("array.jl")

# compiler libraries
include("../lib/cupti/CUPTI.jl")
include("../lib/nvtx/NVTX.jl")
export CUPTI, NVTX

# compiler implementation
include("compiler/gpucompiler.jl")
include("compiler/execution.jl")
include("compiler/exceptions.jl")
include("compiler/reflection.jl")

# array implementation
include("gpuarrays.jl")
include("utilities.jl")
include("texture.jl")

# integrations and specialized functionality
include("indexing.jl")
include("broadcast.jl")
include("mapreduce.jl")
include("accumulate.jl")
include("reverse.jl")
include("iterator.jl")
include("sorting.jl")

# array libraries
include("../lib/complex.jl")
include("../lib/library_types.jl")
include("../lib/cublas/CUBLAS.jl")
include("../lib/cusparse/CUSPARSE.jl")
include("../lib/cusolver/CUSOLVER.jl")
include("../lib/cufft/CUFFT.jl")
include("../lib/curand/CURAND.jl")
include("../lib/cutensor/CUTENSOR.jl")

export CUBLAS, CUSPARSE, CUSOLVER, CUFFT, CURAND, CUTENSOR

# random depends on CURAND
include("random.jl")

# other libraries
include("../lib/nvml/NVML.jl")
const has_nvml = NVML.has_nvml
export NVML, has_nvml

include("deprecated.jl")
include("precompile.jl")

end
