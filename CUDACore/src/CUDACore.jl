module CUDACore

include("utils/public.jl")

using GPUCompiler
using GPUCompiler: InvalidIRError, KernelError
@public InvalidIRError, KernelError

using GPUArrays
using GPUArrays: allowscalar
@public allowscalar

using GPUToolbox
@public i32

using LLVM
using LLVM.Interop
using Core: LLVMPtr

import KernelAbstractions

using Adapt: Adapt, adapt, WrappedArray

using LinearAlgebra

using BFloat16s: BFloat16

using ExprTools: splitdef, combinedef

using LLVMLoopInfo

using CUDA_Driver_jll

using CUDA_Compiler_jll

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

# Julia has several notions of `sizeof`
# - Base.sizeof is the size of an object in memory
# - Base.aligned_sizeof is the size of an object in an array/inline alloced
# Both of them are equivalent for immutable objects, but differ for mutable singtons and Symbol
# We use `aligned_sizeof` since we care about the size of a type in an array
@generated function aligned_sizeof(::Type{T}) where T
    return :($(Base.aligned_sizeof(T)))
end

## source code includes

include("pointer.jl")

# core utilities
include("utils/call.jl")
include("utils/cache.jl")
include("utils/struct_size.jl")
include("../../lib/cudadrv/CUDAdrv.jl")

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
include("refpointer.jl")

# compiler libraries
include("../../lib/cupti/CUPTI.jl")
export CUPTI

# compiler implementation
include("compiler/compilation.jl")
include("compiler/execution.jl")
include("compiler/exceptions.jl")
include("compiler/reflection.jl")

# array implementation
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
include("random.jl")

# shared library types
include("complex.jl")
include("library_types.jl")

# other libraries
include("../../lib/nvml/NVML.jl")
const has_nvml = NVML.has_nvml
export NVML, has_nvml

# KernelAbstractions
include("CUDAKernels.jl")
import .CUDAKernels: CUDABackend
export CUDABackend

# StaticArrays is still a direct dependency, so directly include the extension
include("../ext/StaticArraysExt.jl")
# NOTE: StaticArrays is a direct dep, so extension is directly included

include("precompile.jl")

end
