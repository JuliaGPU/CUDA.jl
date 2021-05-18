module CUDA

using GPUCompiler

using GPUArrays

using LLVM
using LLVM.Interop
using Core: LLVMPtr

using Adapt

using Requires

using LinearAlgebra

using BFloat16s

using Memoize

using ExprTools

# XXX: to be replaced by a JLL
include("../deps/Deps.jl")
using .Deps

# only use TimerOutputs on non latency-critical CI, in part because
# @timeit_debug isn't truely zero-cost (KristofferC/TimerOutputs.jl#120)
if getenv("CI", false) && !getenv("BENCHMARKS", false)
    using TimerOutputs
    const to = TimerOutput()

    macro timeit_ci(args...)
        TimerOutputs.timer_expr(CUDA, false, :($CUDA.to), args...)
    end
else
    macro timeit_ci(args...)
        esc(args[end])
    end
end


## source code includes

include("pointer.jl")

# core library
include("../lib/utils/APIUtils.jl")
include("../lib/cudadrv/CUDAdrv.jl")

# essential stuff
include("initialization.jl")
include("state.jl")
include("debug.jl")

# device functionality (needs to be loaded first, because of generated functions)
include("device/utils.jl")
include("device/pointer.jl")
include("device/array.jl")
include("device/intrinsics.jl")
include("device/llvm.jl")
include("device/runtime.jl")
include("device/texture.jl")
include("device/random.jl")

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

# array libraries
include("../lib/complex.jl")
include("../lib/library_types.jl")
include("../lib/cublas/CUBLAS.jl")
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
include("reverse.jl")
include("linalg.jl")
include("iterator.jl")
include("random.jl")
include("sorting.jl")

# other libraries
include("../lib/nvml/NVML.jl")
const has_nvml = NVML.has_nvml
export NVML, has_nvml

include("deprecated.jl")

end
