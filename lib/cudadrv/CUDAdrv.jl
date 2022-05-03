using .APIUtils

using CEnum: @cenum

using Printf

import Libdl

using LazyArtifacts


# low-level wrappers
const CUdeviceptr = CuPtr{Cvoid}
const CUarray = CuArrayPtr{Cvoid}
const GLuint = Cuint    # FIXME: get these from somewhere
const GLenum = Cuint
include("libcuda_common.jl")
include("error.jl")
include("libcuda.jl")
include("libcuda_deprecated.jl")

@noinline function find_libcuda()
    # NOTE: we can't use function wrappers from CUDAdrv.jl here,
    #       as those depend on `libcuda()` which is not set yet.

    # find the local system library
    system_driver = if Sys.iswindows()
        Libdl.find_library("nvcuda")
    else
        Libdl.find_library(["libcuda.so.1", "libcuda.so"])
    end
    if system_driver == ""
        if Sys.iswindows()
            error("""Could not find the CUDA driver library. Please make sure you have installed the NVIDIA driver for your GPU.
                     If you're sure it's installed, look for `nvcuda.dll` in your system and make sure it's discoverable by the linker.
                     Typically, that involves adding an entry to PATH.""")
        else
            error("""Could not find the CUDA driver library. Please make sure you have installed the NVIDIA driver for your GPU.
                     If you're sure it's installed, look for `libcuda.so` in your system and make sure it's discoverable by the linker.
                     Typically, that involves an entry in '/etc/ld.so.conf', or setting LD_LIBRARY_PATH.""")
        end
    end

    # query the system driver version
    function get_version(driver)
        library_handle = Libdl.dlopen(driver)
        try
            function_handle = Libdl.dlsym(library_handle, "cuDriverGetVersion")
            version_ref = Ref{Cint}()
            @check ccall(function_handle, CUresult, (Ptr{Cint},), version_ref)
            major, ver = divrem(version_ref[], 1000)
            minor, patch = divrem(ver, 10)
            return VersionNumber(major, minor, patch)
        finally
            Libdl.dlclose(library_handle)
        end
    end

    # XXX: apparently cuDriverGetVersion can be used before cuInit,
    #      despite the docs stating "any function [...] will return
    #      CUDA_ERROR_NOT_INITIALIZED"; is this a recent change?
    _system_version[] = get_version(system_driver)

    # check if we managed to unload the system driver.
    # if we didn't, we can't consider a forward compatible library because that would
    # risk having multiple copies of libcuda.so loaded (also see NVIDIA bug #3418723)
    system_driver_loaded = Libdl.dlopen(system_driver, Libdl.RTLD_NOLOAD; throw_error=false) !== nothing
    if system_driver_loaded
        @debug "Could not unload the system CUDA library; this will prevent use of the forward-compatible package"
        return system_driver
    end

    # check if this process is hooked by CUDA's injection libraries, which prevents
    # unloading libcuda after dlopening. this is problematic, because we might want to
    # after loading a forwards-compatible libcuda and realizing we can't use it. without
    # being able to unload the library, we'd run into issues (see NVIDIA bug #3418723)
    hooked = haskey(ENV, "CUDA_INJECTION64_PATH")
    if hooked
        @debug "Running under CUDA injection tools; this will prevent use of the forward-compatible package"
        return system_driver
    end

    # if we're using an older driver; consider using forward compatibility
    function do_init(driver)
        library_handle = Libdl.dlopen(driver)
        try
            function_handle = Libdl.dlsym(library_handle, "cuInit")
            @check ccall(function_handle, CUresult, (UInt32,), 0)
        finally
            Libdl.dlclose(library_handle)
        end
        return
    end
    if getenv("JULIA_CUDA_USE_COMPAT", !hooked && !system_driver_loaded) && _system_version[] < v"11.6"
        artifact = try
            # work around artifact"" eagerness on unsupported platforms by passing a variable
            f = id -> @artifact_str(id)
            f("CUDA_compat")
        catch ex
            @debug "Could not download forward compatibility package" exception=(ex,catch_backtrace())
            nothing
        end

        if artifact !== nothing
            # TODO: do we need to dlopen the JIT compiler library for it to be discoverable?
            #       doesn't that clash with a system one if compat cuInit fails? or should
            #       we load it _after_ the compat driver initialization succeeds?
            #compat_compiler = joinpath(artifact, "lib", "libnvidia-ptxjitcompiler.so")
            #Libdl.dlopen(compat_compiler)

            compat_driver = joinpath(artifact, "lib", "libcuda.so")
            try
                # just calling cuDriverGetVersion doesn't trigger errors,
                # so actually perform driver intialization here already.
                # XXX: reuse __init_driver__, which detects e.g. running under rr
                do_init(compat_driver)
                return compat_driver
            catch ex
                @debug "Could not use forward compatibility package" exception=(ex,catch_backtrace())

                # see comment above about unloading the system driver
                compat_driver_loaded = Libdl.dlopen(compat_driver, Libdl.RTLD_NOLOAD; throw_error=false) !== nothing
                if compat_driver_loaded
                    error("""Could not unload the forward compatible CUDA driver library.

                                This is probably caused by running Julia under a tool that hooks CUDA API calls.
                                In that case, prevent CUDA.jl from loading multiple drivers by setting JULIA_CUDA_USE_COMPAT=false in your environment.""")
                end
            end
        end
    end

    return system_driver
end

const _libcuda = Ref{Union{String,Exception}}()
function libcuda()
    # libcuda discovery has not been triggered yet
    if !isassigned(_libcuda)
        try
            _libcuda[] = find_libcuda()
            __init_driver__()
        catch err
            _libcuda[] = err
            rethrow()
        end
    end

    # libcuda was not found, or initialization failed.
    # since we may have cached a value of libcuda (through ccalls from __init_driver__)
    # we cannot re-initialize in this session, so re-throw the original error again.
    if _libcuda[] isa Exception
        throw(_libcuda[])
    end

    _libcuda[]::String
end

# high-level wrappers
include("types.jl")
include("version.jl")
include("devices.jl")
include("context.jl")
include("stream.jl")
include("pool.jl")
include("memory.jl")
include("module.jl")
include("events.jl")
include("execution.jl")
include("profile.jl")
include("occupancy.jl")
include("graph.jl")

# global state (CUDA.jl's driver wrappers behave like CUDA's runtime library)
include("state.jl")
