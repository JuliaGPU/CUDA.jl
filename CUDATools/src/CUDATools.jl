module CUDATools

using CUDACore

# names marked `@public`, for the Julia 1.10 path of the CUDA.jl package
const PUBLIC_NAMES = Symbol[]

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
include("cubin.jl")
include("profile.jl")
include("utilities.jl")

include("precompile.jl")

end
