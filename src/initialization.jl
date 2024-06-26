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

    # TODO: make errors here (and in submodules/subpackages like cuBLAS and cuDNN) fatal,
    #       and remove functional(), once people sufficiently use weak dependencies.

    # check that we have a driver
    global libcuda
    if CUDA_Driver_jll.is_available()
        if isnothing(CUDA_Driver_jll.libcuda)
            _initialization_error[] = "CUDA driver not found"
            return
        end
        libcuda = CUDA_Driver_jll.libcuda
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

    driver = try
        set_driver_version()
    catch err
        @debug "CUDA driver failed to report a version" exception=(err, catch_backtrace())
        _initialization_error[] = "CUDA driver not functional"
        return
    end

    if !(v"11" <= driver < v"13-")
        @error "This version of CUDA.jl only supports NVIDIA drivers for CUDA 11.x or 12.x (yours is for CUDA $driver)"
        _initialization_error[] = "CUDA driver unsupported"
        return
    end

    if driver < v"11.3"
        @warn """The NVIDIA driver on this system only supports up to CUDA $driver.
                 For performance reasons, it is recommended to upgrade to a driver that supports CUDA 11.3 or higher."""
    end

    # check that we have a runtime
    if !CUDA_Runtime.is_available()
        # try to find out why
        reason = if CUDA_Runtime != CUDA_Runtime_jll
            """You requested use of a local CUDA toolkit, but not all
               required components were discovered.

               Try running with `JULIA_DEBUG=CUDA_Runtime_Discovery` in
               your environment and re-loading CUDA.jl for more details."""
        elseif !Sys.iswindows() && !Sys.islinux() && !in(Sys.ARCH, [:x86_64, :aarch64])
            """You are using an unsupported platform: this version of CUDA.jl
               only supports Linux (x86_64, aarch64) and Windows (x86_64).

               Consider downgrading CUDA.jl (refer to the README for a list of
               supported platforms) or manually installing the CUDA toolkit and make
               CUDA.jl use it by calling `CUDA.set_runtime_version!(local_toolkit=true)`."""
        elseif CUDA_Runtime_jll.host_platform["cuda"] == "none"
            """CUDA.jl's JLLs were precompiled without an NVIDIA driver present.
               This can happen when installing CUDA.jl on an HPC log-in node,
               or in a container. In that case, you need to specify which CUDA
               version to use at run time by calling `CUDA.set_runtime_version!`
               or provisioning the preference it sets at compile time.

               If you are not running in a container or on an HPC log-in node,
               try re-compiling the CUDA runtime JLL and re-loading CUDA.jl:
                    pkg = Base.PkgId(Base.UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2"),
                                     "CUDA_Runtime_jll")
                    Base.compilecache(pkg)
                    # re-start Julia and re-load CUDA.jl"""
        else
            """Could not diagnose why the CUDA runtime is not available.

               If the issue persists, please file a support ticket with the following details:
               - host platform: $(Base.BinaryPlatforms.triplet(CUDA_Runtime_jll.host_platform))
               - libcuda: $libcuda (loaded through JLL: $(CUDA_Driver_jll.is_available()))
               - driver version: $driver
               """
        end
        @error """CUDA.jl could not find an appropriate CUDA runtime to use.

                  $reason

                  For more details, refer to the CUDA.jl documentation at
                  https://cuda.juliagpu.org/stable/installation/overview/"""
        _initialization_error[] = "CUDA runtime not found"
        return
    end
    runtime = try
        runtime_version()
    catch err
        if err isa CuError && err.code == ERROR_NO_DEVICE
            _initialization_error[] = "No CUDA-capable device found"
            return
        end
        rethrow()
    end

    # ensure the loaded runtime is supported
    if runtime < v"10.2"
        @error "This version of CUDA.jl only supports CUDA 11 or higher (your toolkit provides CUDA $runtime)"
    end
    if runtime.major > driver.major
        @warn """You are using CUDA $runtime with a driver that only supports up to $(driver.major).x.
                 It is recommended to upgrade your driver, or switch to automatic installation of CUDA."""
    end

    # ensure the loaded runtime matches what we precompiled for.
    if toolkit_version == nothing
        @error """CUDA.jl was precompiled without knowing the CUDA toolkit version. This is unsupported.
                  You should either precompile CUDA.jl in an environment where the CUDA toolkit is available,
                  or call `CUDA.set_runtime_version!` to specify which CUDA version to use."""
    elseif Base.thisminor(runtime) != Base.thisminor(toolkit_version)
        # this can only happen with a local toolkit, but let's always check to be sure
        if local_toolkit
            @error """You are using a local CUDA $(Base.thisminor(runtime)) toolkit, but CUDA.jl was precompiled for CUDA $(Base.thisminor(toolkit_version)). This is unsupported.
                      Call `CUDA.set_runtime_version!` to update the CUDA version to match your local installation."""
        else
            @error """You are using CUDA $(Base.thisminor(runtime)), but CUDA.jl was precompiled for CUDA $(Base.thisminor(toolkit_version)).
                      This is unexpected; please file an issue."""
        end
    end

    # if we're not running under an external profiler, let CUPTI handle NVTX events
    # XXX: JuliaGPU/NVTX.jl#37
    if !NVTX.isactive() && !Sys.iswindows()
        ENV["NVTX_INJECTION64_PATH"] = CUDA_Runtime.libcupti
        NVTX.activate()
    end

    # finally, initialize CUDA
    try
        cuInit(0)
    catch err
        _initialization_error[] = "CUDA initialization failed: " * sprint(showerror, err)
        return
    end

    # warn if we're not using an official build of Julia
    official_release = startswith(Base.TAGGED_RELEASE_BANNER, "Official")
    if !official_release
        @warn """You are using a non-official build of Julia. This may cause issues with CUDA.jl.
                 Please consider using an official build from https://julialang.org/downloads/."""
    end

    @static if !isdefined(Base, :get_extension)
        @require ChainRulesCore="d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4" begin
            include("../ext/ChainRulesCoreExt.jl")
        end
        @require SpecialFunctions="276daf66-3868-5448-9aa4-cd146d93841b" begin
            include("../ext/SpecialFunctionsExt.jl")
        end
        @require EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869" begin
            include("../ext/EnzymeCoreExt.jl")
        end
    end

    # enable generation of FMA instructions to mimic behavior of nvcc
    LLVM.clopts("-nvptx-fma-level=1")

    # warn about old, deprecated environment variables
    if haskey(ENV, "JULIA_CUDA_USE_BINARYBUILDER") && !local_toolkit
        @error """JULIA_CUDA_USE_BINARYBUILDER is deprecated. Call `CUDA.jl.set_runtime_version!` to use a local toolkit."""
        # we do not warn about this when we're already using the new preference,
        # because during the transition clusters will be deploying both mechanisms.
    end
    if haskey(ENV, "JULIA_CUDA_VERSION")
        @error """JULIA_CUDA_VERSION is deprecated. Call `CUDA.jl.set_runtime_version!` to use a different version."""
    end

    if !local_toolkit
        # scan for CUDA libraries that may have been loaded from system paths
        # note that this must cover more that the libraries provided by the
        # runtime JLL, in order to detect possible conditional dependencies.
        runtime_libraries = ["cudart",
                             "nvperf", "nvvm", "nvrtc", "nvJitLink",
                             "cublas", "cupti", "cusparse", "cufft", "curand", "cusolver"]
        for lib in Libdl.dllist()
            contains(lib, "artifacts") && continue
            if any(rtlib -> contains(lib, rtlib), runtime_libraries)
                @warn """CUDA runtime library `$(basename(lib))` was loaded from a system path, `$lib`.

                         This may cause errors. Ensure that you have not set the LD_LIBRARY_PATH
                         environment variable, or that it does not contain paths to CUDA libraries.

                         In any other case, please file an issue."""
            end
        end
    end

    _initialized[] = true
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
