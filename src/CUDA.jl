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

using LLVMLoopInfo

using CUDA_Driver_jll

import CUDA_Runtime_jll
const local_toolkit = CUDA_Runtime_jll.host_platform["cuda_local"] == "true"
const toolkit_version = if CUDA_Runtime_jll.host_platform["cuda"] == "none"
    nothing
else
    parse(VersionNumber, CUDA_Runtime_jll.host_platform["cuda"])
end
if local_toolkit
    using CUDA_Runtime_Discovery
    const CUDA_Runtime = CUDA_Runtime_Discovery
else
    using CUDA_Runtime_jll
    const CUDA_Runtime = CUDA_Runtime_jll
end

import Preferences

using Libdl

import NVTX

using Printf


## source code includes

include("pointer.jl")

# core library
include("../lib/utils/APIUtils.jl")
include("../lib/cudadrv/CUDAdrv.jl")

# essential stuff
include("initialization.jl")
include("compatibility.jl")
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
include("memory.jl")
include("array.jl")

# compiler libraries
include("../lib/cupti/CUPTI.jl")
export CUPTI

# compiler implementation
include("compiler/compilation.jl")
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
include("profile.jl")

# array libraries
include("../lib/complex.jl")
include("../lib/library_types.jl")
include("../lib/cublas/CUBLAS.jl")
include("../lib/cusparse/CUSPARSE.jl")
include("../lib/cusolver/CUSOLVER.jl")
include("../lib/cufft/CUFFT.jl")
include("../lib/curand/CURAND.jl")

export CUBLAS, CUSPARSE, CUSOLVER, CUFFT, CURAND
const has_cusolvermg = CUSOLVER.has_cusolvermg
export has_cusolvermg

# random depends on CURAND
include("random.jl")

# other libraries
include("../lib/nvml/NVML.jl")
const has_nvml = NVML.has_nvml
export NVML, has_nvml

# KernelAbstractions
include("CUDAKernels.jl")
import .CUDAKernels: CUDABackend
export CUDABackend

# StaticArrays is still a direct dependency, so directly include the extension
include("../ext/StaticArraysExt.jl")

include("precompile.jl")
include("deprecated.jl")

end
