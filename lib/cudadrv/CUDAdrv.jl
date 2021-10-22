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

const _libcuda = Ref{String}()
function libcuda()
    if !isassigned(_libcuda)
        # find the local system library
        system_driver = if Sys.iswindows()
            Libdl.find_library("nvcuda")
        else
            Libdl.find_library(["libcuda.so.1", "libcuda.so"])
        end
        if name == ""
            if Sys.iswindows()
                error("""Could not find the CUDA driver library. Please make sure you have installed the NVIDIA driver for your GPU.
                         If you're sure it's installed, look for `libcuda.so` in your system and make sure it's discoverable by the linker.
                         Typically, that involves an entry in '/etc/ld.so.conf', or setting LD_LIBRARY_PATH.""")
            else
                error("""Could not find the CUDA driver library. Please make sure you have installed the NVIDIA driver for your GPU.
                         If you're sure it's installed, look for `nvcuda.dll` in your system and make sure it's discoverable by the linker.
                         Typically, that involves adding an entry to PATH.""")
            end
        end

        # NOTE: we can't re-use the function wrappers from libcuda.so,
        #       as those use libcuda() whose value is cached by ccall.

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
        _system_version[] = get_version(system_driver)
        _libcuda[] = system_driver
        # XXX: apparently cuDriverGetVersion can be used before cuInit,
        #      despite the docs stating "any function [...] will return
        #      CUDA_ERROR_NOT_INITIALIZED"; is this a recent change?

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
        if _system_version[] < v"11.4"
            platform = Base.BinaryPlatforms.HostPlatform()
            platform.tags["cuda"] = "11.4"

            artifact = try
                @artifact_str("CUDA_compat", platform)
            catch ex
                @debug "Could not download forward compatibility package" exception=(ex,catch_backtrace())
                nothing
            end

            if artifact !== nothing
                # TODO: do we need to dlopen the JIT compiler library for it to
                #       be discoverable? doesn't that clash with a system one
                #       if compat cuInit fails? or should we load it _after_
                #       the compat driver initialization succeeds?
                #compat_compiler = joinpath(artifact, "lib", "libnvidia-ptxjitcompiler.so")
                #Libdl.dlopen(compat_compiler)

                compat_driver = joinpath(artifact, "lib", "libcuda.so")
                try
                    # just calling cuDriverGetVersion doesn't trigger errors,
                    # so perform driver intialization here
                    do_init(compat_driver)
                    _libcuda[] = compat_driver
                catch ex
                    @debug "Could not use forward compatibility package" exception=(ex,catch_backtrace())
                end
            end
        end

        cuInit(0)
    end

    # memoized because otherwise each ccall would perform discovery again
    _libcuda[]
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

# TODO: figure out if these wrappers may use the runtime-esque state (stream(), context()).
#       it's inconsitent now: @finalize_in_ctx doesn't, memory.jl does use stream(), etc.
