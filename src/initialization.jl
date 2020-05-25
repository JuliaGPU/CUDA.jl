# initialization

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
        @assert functional(true) "CUDA.jl did not successfully initialize, and is not usable."
        $(esc(ex))
    end
end


## deferred initialization API

const __libcuda = Sys.iswindows() ? :nvcuda : :libcuda
libcuda() = @after_init(__libcuda)

# load-time initialization: only perform mininal checks here
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

    initializer(prepare_cuda_call)

    __init_memory__()

    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" include("forwarddiff.jl")
end

# run-time configuration: try and initialize CUDA, but don't error
function __configure__(show_reason::Bool)
    if haskey(ENV, "_") && basename(ENV["_"]) == "rr"
        show_reason && @error("Running under rr, which is incompatible with CUDA")
        return false
    end

    try
        @debug "Initializing CUDA driver"
        res = @runtime_ccall((:cuInit, __libcuda), CUresult, (UInt32,), 0)
        if res == 0xffffffff
            error("Cannot use the CUDA stub libraries. You either don't have the NVIDIA driver installed, or it is not properly discoverable.")
        elseif res != SUCCESS
            throw_api_error(res)
        end
    catch ex
        show_reason && @error("Could not initialize CUDA", exception=(ex,catch_backtrace()))
        return false
    end

    return true
end

# run-time initialization: we have a driver, so try and discover CUDA
function __runtime_init__()
    if version() < v"9"
        @warn "CUDA.jl only supports NVIDIA drivers for CUDA 9.0 or higher (yours is for CUDA $(version()))"
    end

    __init_dependencies__() || error("Could not find a suitable CUDA installation")

    if toolkit_release() < v"9"
        @warn "CUDA.jl only supports CUDA 9.0 or higher (your toolkit provides CUDA $(toolkit_release()))"
    elseif toolkit_release() > release()
        @warn """You are using CUDA toolkit $(toolkit_release()) with a driver that only supports up to $(release()).
                 It is recommended to upgrade your driver, or switch to automatic installation of CUDA."""
    end

    __init_compatibility__()

    return
end


## convenience functions

# TODO: update docstrings

export has_cuda, has_cuda_gpu, usable_cuda_gpus

"""
    has_cuda()::Bool

Check whether the local system provides an installation of the CUDA driver and toolkit.
Use this function if your code loads packages that require CUDA.jl.

Note that CUDA-dependent packages might still fail to load if the installation is broken,
so it's recommended to guard against that and print a warning to inform the user:

```
using CUDA
if has_cuda()
    try
        using CuArrays
    catch ex
        @warn "CUDA is installed, but CuArrays.jl fails to load" exception=(ex,catch_backtrace())
    end
end
```
"""
has_cuda(show_reason::Bool=false) = functional(show_reason)

"""
    has_cuda_gpu()::Bool

Check whether the local system provides an installation of the CUDA driver and toolkit, and
if it contains a CUDA-capable GPU. See [`has_cuda`](@ref) for more details.

Note that this function initializes the CUDA API in order to check for the number of GPUs.
"""
has_cuda_gpu(show_reason::Bool=false) = has_cuda(show_reason) && length(devices()) > 0
