module CUDAnative

using CUDAapi
using CUDAdrv

using LLVM
using LLVM.Interop

using GPUCompiler

using Adapt


## deferred initialization

# CUDA packages require complex initialization (discover CUDA, download artifacts, etc)
# that can't happen at module load time, so defer that to run time upon actual use.

const configured = Ref{Union{Nothing,Bool}}(nothing)

"""
    functional(show_reason=false)

Check if the package has been configured successfully and is ready to use.

This call is intended for packages that support conditionally using an available GPU. If you
fail to check whether CUDA is functional, actual use of functionality might warn and error.
"""
function functional(show_reason::Bool=false)
    if configured[] === nothing
        _functional(show_reason)
    end
    configured[]::Bool
end

const configure_lock = ReentrantLock()
@noinline function _functional(show_reason::Bool=false)
    lock(configure_lock) do
        if configured[] === nothing
            configured[] = false
            if __configure__(show_reason)
                configured[] = true
                try
                    __runtime_init__()
                catch
                    configured[] = false
                    rethrow()
                end
            end
        end
    end
end

# macro to guard code that only can run after the package has successfully initialized
macro after_init(ex)
    quote
        @assert functional(true) "CUDAnative.jl did not successfully initialize, and is not usable."
        $(esc(ex))
    end
end


## source code includes

include("init.jl")
include("compatibility.jl")
include("bindeps.jl")

include("cupti/CUPTI.jl")
include("nvtx/NVTX.jl")

# needs to be loaded _before_ the compiler infrastructure, because of generated functions
include("device/pointer.jl")
include("device/array.jl")
include("device/cuda.jl")
include("device/llvm.jl")
include("device/runtime.jl")

include("compiler.jl")
include("execution.jl")
include("exceptions.jl")
include("reflection.jl")
include("array.jl")

include("deprecated.jl")

export CUPTI, NVTX


## initialization

# device compatibility
const __target_support = Ref{Vector{VersionNumber}}()
const __ptx_support = Ref{Vector{VersionNumber}}()

target_support() = @after_init(__target_support[])
ptx_support() = @after_init(__ptx_support[])

function __init__()
    if Base.libllvm_version != LLVM.version()
        error("LLVM $(LLVM.version()) incompatible with Julia's LLVM $(Base.libllvm_version)")
    end

    # enable generation of FMA instructions to mimic behavior of nvcc
    LLVM.clopts("-nvptx-fma-level=1")

    resize!(thread_contexts, Threads.nthreads())
    fill!(thread_contexts, nothing)

    resize!(thread_tasks, Threads.nthreads())
    fill!(thread_tasks, nothing)

    CUDAdrv.initializer(prepare_cuda_call)
end

function __configure__(show_reason::Bool)
    # if any dependent GPU package failed, expect it to have logged an error and bail out
    if !CUDAdrv.functional(show_reason)
        show_reason && @warn "CUDAnative.jl did not initialize because CUDAdrv.jl failed to"
        return false
    end

    return __configure_dependencies__(show_reason)
end

function __runtime_init__()
    if release() < v"9"
        @warn "CUDAnative.jl only supports CUDA 9.0 or higher (your toolkit provides CUDA $(release()))"
    elseif release() > CUDAdrv.release()
        @warn """You are using CUDA toolkit $(release()) with a driver that only supports up to $(CUDAdrv.release()).
                 It is recommended to upgrade your driver, or switch to automatic installation of CUDA."""
    end

    # device compatibility

    llvm_support = llvm_compat()
    cuda_support = cuda_compat()

    __target_support[] = sort(collect(llvm_support.cap ∩ cuda_support.cap))
    isempty(__target_support[]) && error("Your toolchain does not support any device capability")

    __ptx_support[] = sort(collect(llvm_support.ptx ∩ cuda_support.ptx))
    isempty(__ptx_support[]) && error("Your toolchain does not support any PTX ISA")

    @debug("Toolchain with LLVM $(LLVM.version()), CUDA driver $(CUDAdrv.version()) and toolkit $(CUDAnative.version()) supports devices $(verlist(__target_support[])); PTX $(verlist(__ptx_support[]))")
end

end
