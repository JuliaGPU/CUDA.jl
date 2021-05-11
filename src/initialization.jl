# initialization

# CUDA packages require complex initialization (discover CUDA, download artifacts, etc)
# that can't happen at module load time, so defer that to run time upon actual use.

const configured = Threads.Atomic{Int}(-1)   # -1=unconfigured, -2=configuring,
                                             # 0=failed, 1=configured

"""
    functional(show_reason=false)

Check if the package has been configured successfully and is ready to use.

This call is intended for packages that support conditionally using an available GPU. If you
fail to check whether CUDA is functional, actual use of functionality might warn and error.
"""
function functional(show_reason::Bool=false)
    if configured[] < 0
        Bool(_functional(show_reason))
    else
        Bool(configured[])
    end
end

const configure_lock = ReentrantLock()
@noinline function _functional(show_reason::Bool=false)
    Base.@lock configure_lock begin
        if configured[] == -1
            configured[] = -2
            configured[] = try
                __runtime_init__()
                1
            catch ex
                show_reason && @error("Error during initialization of CUDA.jl", exception=(ex,catch_backtrace()))
                0
            end
        elseif configured[] == -2
            # claim that we're initialized _during_ initialization (this can only happen
            # on the same task that's currently initializing, due to the reentrant lock)
            1
        else
            # we got stuck on the lock while another thread was initializing
            configured[]
        end
    end
end

# macro to guard code that only can run after the package has successfully initialized
macro after_init(ex)
    quote
        if !functional(true)
            error("""CUDA.jl did not successfully initialize, and is not usable.
                     If you did not see any other error message, try again in a new session
                     with the JULIA_DEBUG environment variable set to 'CUDA'.""")
        end
        $(esc(ex))
    end
end


## deferred initialization API

const __libcuda = Sys.iswindows() ? "nvcuda" : ( Sys.islinux() ? "libcuda.so.1" : "libcuda" )
libcuda() = @after_init(__libcuda)

# load-time initialization: only perform mininal checks here
function __init__()
    if Base.libllvm_version != LLVM.version()
        error("LLVM $(LLVM.version()) incompatible with Julia's LLVM $(Base.libllvm_version)")
    end

    # enable generation of FMA instructions to mimic behavior of nvcc
    LLVM.clopts("-nvptx-fma-level=1")

    # ensure that operations executed by the REPL back-end finish before returning,
    # because displaying values happens on a different task (CUDA.jl#831)
    if isdefined(Base, :active_repl_backend)
        push!(Base.active_repl_backend.ast_transforms, ex->
            quote
                try
                    $(ex)
                finally
                    $configured[] == 1 && $synchronize()
                end
            end
        )
    end

    reset_timer!(to)
    atexit() do
        TimerOutputs.complement!(to)
        @debug """CUDA.jl timings:
                  $to"""
    end

    precompiling = ccall(:jl_generating_output, Cint, ()) != 0
    if !precompiling
        eval(overrides)
    end
end

function __runtime_init__()
    if haskey(ENV, "_") && basename(ENV["_"]) == "rr"
        error("Running under rr, which is incompatible with CUDA")
    end

    @debug "Initializing CUDA driver"
    res = ccall((:cuInit, __libcuda), CUresult, (UInt32,), 0)
    if res == 0xffffffff
        error("Cannot use the CUDA stub libraries. You either don't have the NVIDIA driver installed, or it is not properly discoverable.")
    elseif res != SUCCESS
        throw_api_error(res)
    end

    if version() < v"10.1"
        @warn "This version of CUDA.jl only supports NVIDIA drivers for CUDA 10.1 or higher (yours is for CUDA $(version()))"
    end

    __init_dependencies__() || error("Could not find a suitable CUDA installation")

    if toolkit_release() < v"10.1"
        @warn "This version of CUDA.jl only supports CUDA 10.1 or higher (your toolkit provides CUDA $(toolkit_release()))"
    elseif toolkit_release() > release()
        @warn """You are using CUDA toolkit $(toolkit_release()) with a driver that only supports up to $(release()).
                 It is recommended to upgrade your driver, or switch to automatic installation of CUDA."""
    end

    resize!(__device_contexts, ndevices())
    fill!(__device_contexts, nothing)

    __init_compatibility__()

    __init_pool__()

    CUBLAS.__runtime_init__()
    return
end


## convenience functions

# TODO: update docstrings

export has_cuda, has_cuda_gpu

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

# robustly get and parse an env var
function getenv(var, default::T) where T
    if haskey(ENV, var)
        result = tryparse(T, ENV[var])
        if result === nothing
            @warn "Could not parse $(var)=$(ENV[var]), using default value '$default'"
            default
        else
            result
        end
    else
        default
    end
end
