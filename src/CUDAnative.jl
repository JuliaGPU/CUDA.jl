module CUDAnative

using CUDAapi
using CUDAdrv

using LLVM
using LLVM.Interop

using Adapt
using TimerOutputs
using DataStructures


## source code includes

include("utils.jl")

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

const __initialized__ = Ref(false)
functional() = __initialized__[]

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0
    silent = parse(Bool, get(ENV, "JULIA_CUDA_SILENT", "false")) || precompiling
    verbose = parse(Bool, get(ENV, "JULIA_CUDA_VERBOSE", "false"))

    # if any dependent GPU package failed, expect it to have logged an error and bail out
    if !CUDAdrv.functional()
        verbose && @warn "CUDAnative.jl did not initialize because CUDAdrv.jl failed to"
        return
    end

    try
        __init_bindeps__(silent=silent, verbose=verbose)

        __init_compiler__()

        resize!(thread_contexts, Threads.nthreads())
        fill!(thread_contexts, nothing)
        CUDAdrv.initializer(maybe_initialize)

        __initialized__[] = true
    catch ex
        # don't actually fail to keep the package loadable
        if !silent
            if verbose
                @error "CUDAnative.jl failed to initialize" exception=(ex, catch_backtrace())
            else
                @info "CUDAnative.jl failed to initialize, GPU functionality unavailable (set JULIA_CUDA_SILENT or JULIA_CUDA_VERBOSE to silence or expand this message)"
            end
        end
    end
end

end
