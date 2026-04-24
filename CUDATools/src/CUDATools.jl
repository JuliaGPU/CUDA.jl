module CUDATools

using CUDACore

using Republic: @public

using GPUCompiler
using GPUCompiler: CompilerJob, methodinstance
using LLVM

using CUDA_Compiler_jll: CUDA_Compiler_jll

import Preferences
using Printf

using CUPTI
export CUPTI

using NVML
using NVML: has_nvml
export NVML, has_nvml

# Developer tools
include("reflection.jl")
include("profile.jl")
include("utilities.jl")

include("precompile.jl")

end
