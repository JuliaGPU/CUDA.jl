# initialization

# XXX: we currently allow loading CUDA.jl even if the package is not functional, because
#      downstream packages can only unconditionally depend on CUDA.jl. that's why we have
#      the errors be non-fatal, sometimes even silencing them, and why we have the
#      `functional()` API that allows checking for successfull initialization.
# TODO: once we have conditional dependencies, remove this complexity and have __init__ fail

const _initialized = Ref{Bool}(false)
const _initialization_error = Ref{String}()

"""
    functional(show_reason=false)

Check if the package has been configured successfully and is ready to use.

This call is intended for packages that support conditionally using an available GPU. If you
fail to check whether CUDA is functional, actual use of functionality might warn and error.
"""
function functional(show_reason::Bool=false)
    _initialized[] && return true

    if show_reason && isassigned(_initialization_error)
        error(_initialization_error[])
    elseif show_reason
        error("unknown initialization error")
    end
    return false
end

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0
    precompiling && return

    # TODO: make errors here (and in submodules/subpackages like cuBLAS and cuDNN) fatal,
    #       and remove functional(), once people sufficiently use weak dependencies.

    # check that we have a driver
    global libcuda
    if CUDA_Driver_jll.is_available()
        if isdefined(CUDA_Driver_jll, :libcuda)
            libcuda = CUDA_Driver_jll.libcuda
        else
            _initialization_error[] = "CUDA driver not found"
            return
        end
    else
        # CUDA_Driver_jll only kicks in for supported platforms, so fall back to
        # a system search if the artifact isn't available (JLLWrappers.jl#50)
        library = if Sys.iswindows()
            Libdl.find_library("nvcuda")
        else
            Libdl.find_library(["libcuda.so.1", "libcuda.so"])
        end
        if library != ""
            libcuda = library
        else
            _initialization_error[] = "CUDA driver not found"
            return
        end
    end
    driver = driver_version()

    if driver < v"11"
        @error "This version of CUDA.jl only supports NVIDIA drivers for CUDA 11.x or higher (yours is for CUDA $driver)"
        _initialization_error[] = "CUDA driver too old"
        return
    end

    if driver < v"11.2"
        @warn """The NVIDIA driver on this system only supports up to CUDA $driver.
                 For performance reasons, it is recommended to upgrade to a driver that supports CUDA 11.2 or higher."""
    end

    # check that we have a runtime
    if !CUDA_Runtime.is_available()
        @error """CUDA.jl could not find an appropriate CUDA runtime to use.

                  This can have several reasons:
                  * you are using an unsupported platform: this version of CUDA.jl
                    only supports Linux (x86_64, aarch64, ppc64le) and Windows (x86_64),
                    while your platform was identified as $(Base.BinaryPlatforms.triplet(CUDA_Runtime_jll.host_platform));
                  * you precompiled CUDA.jl in an environment where the CUDA driver
                    was not available (i.e., a container, or an HPC login node).
                    in that case, you need to specify which CUDA version to use
                    by calling `CUDA.set_runtime_version!`;
                  * you requested use of a local CUDA toolkit, but not all
                    required components were discovered. try running with
                    JULIA_DEBUG=all in your environment for more details.

                  For more details, refer to the CUDA.jl documentation at
                  https://cuda.juliagpu.org/stable/installation/overview/"""
        _initialization_error[] = "CUDA runtime not found"
        return
    end
    runtime = try
        runtime_version()
    catch err
        if err isa CuError && err.code == ERROR_NO_DEVICE
            @error "No CUDA-capable device found"
            _initialization_error[] = "No CUDA-capable device found"
            return
        end
        rethrow()
    end

    if runtime < v"10.2"
        @error "This version of CUDA.jl only supports CUDA 11 or higher (your toolkit provides CUDA $runtime)"
        _initialization_error[] = "CUDA runtime too old"
        return
    end

    if runtime.major > driver.major
        @warn """You are using CUDA $runtime with a driver that only supports up to $(driver.major).x.
                 It is recommended to upgrade your driver, or switch to automatic installation of CUDA."""
    end

    # finally, initialize CUDA
    try
        cuInit(0)
    catch err
        @error "Failed to initialize CUDA" exception=(err,catch_backtrace())
        _initialization_error[] = "CUDA initialization failed"
        return
    end

    # register device overrides
    eval(Expr(:block, overrides...))
    empty!(overrides)
    @require SpecialFunctions="276daf66-3868-5448-9aa4-cd146d93841b" begin
        include("device/intrinsics/special_math.jl")
        eval(Expr(:block, overrides...))
        empty!(overrides)
    end

    # ensure that operations executed by the REPL back-end finish before returning,
    # because displaying values happens on a different task (CUDA.jl#831)
    if isdefined(Base, :active_repl_backend)
        push!(Base.active_repl_backend.ast_transforms, synchronize_cuda_tasks)
    end

    # enable generation of FMA instructions to mimic behavior of nvcc
    LLVM.clopts("-nvptx-fma-level=1")

    # warn about old, deprecated environment variables
    if haskey(ENV, "JULIA_CUDA_USE_BINARYBUILDER") && CUDA_Runtime == CUDA_Runtime_jll
        @error """JULIA_CUDA_USE_BINARYBUILDER is deprecated, and CUDA.jl always uses artifacts now.
                  To use a local installation, use overrides or preferences to customize the artifact.
                  Please check the CUDA.jl or Pkg.jl documentation for more details."""
        # we do not warn about this when we're already using the new preference,
        # because during the transition clusters will be deploying both mechanisms.
    end
    if haskey(ENV, "JULIA_CUDA_VERSION")
        @error """JULIA_CUDA_VERSION is deprecated. Call `CUDA.jl.set_runtime_version!` to use a different version instead."""
    end

    _initialized[] = true
end

function synchronize_cuda_tasks(ex)
    quote
        try
            $(ex)
        finally
            $task_local_state() !== nothing && $device_synchronize()
        end
    end
end


## convenience functions

# TODO: update docstrings

export has_cuda, has_cuda_gpu

"""
    has_cuda()::Bool

Check whether the local system provides an installation of the CUDA driver and runtime.
Use this function if your code loads packages that require CUDA.jl.
```
"""
has_cuda(show_reason::Bool=false) = functional(show_reason)

"""
    has_cuda_gpu()::Bool

Check whether the local system provides an installation of the CUDA driver and runtime, and
if it contains a CUDA-capable GPU. See [`has_cuda`](@ref) for more details.

Note that this function initializes the CUDA API in order to check for the number of GPUs.
"""
has_cuda_gpu(show_reason::Bool=false) = has_cuda(show_reason) && length(devices()) > 0
