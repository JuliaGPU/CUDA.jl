module CUDAnative

using CUDAapi
using CUDAdrv

using LLVM
using LLVM.Interop

using Adapt
using TimerOutputs
using DataStructures


## source code includes

# needs to be loaded _before_ the compiler infrastructure, because of generated functions
include("device/tools.jl")
include("device/pointer.jl")
include("device/array.jl")
include("device/cuda.jl")
include("device/llvm.jl")
include("device/runtime.jl")

include("init.jl")
include("compatibility.jl")
include("bindeps.jl")

include("cupti/CUPTI.jl")
include("nvtx/NVTX.jl")

include("compiler.jl")
include("execution.jl")
include("exceptions.jl")
include("reflection.jl")

include("deprecated.jl")

export CUPTI, NVTX


## initialization

function __init__()
    # enable generation of FMA instructions to mimic behavior of nvcc
    LLVM.clopts("-nvptx-fma-level=1")

    TimerOutputs.reset_timer!(to)

    resize!(thread_contexts, Threads.nthreads())
    fill!(thread_contexts, nothing)

    CUDAdrv.initializer(maybe_initialize)

    # NOTE: we only perform minimal initialization here that does not require CUDA or a GPU.
    #       most of the actual initialization is deferred to run time:
    #       see bindeps.jl for initialization of CUDA binary dependencies,
    #       and init.jl for initialization of per-device/thread CUDA contexts.
end

end
