using .APIUtils

using CEnum

using Printf

using Libdl

const _libcuda = Ref{String}()
function libcuda()
    if !isassigned(_libcuda)
        if Sys.iswindows()
            _libcuda[] = "nvcuda"
        else
            _libcuda[] = Libdl.find_library(["libcuda.so.1", "libcuda.so"])
            if isempty(_libcuda[])
                error("""Could not find the CUDA driver library 'libcuda.so'. Please make sure you have installed the NVIDIA driver for your GPU.
                         If you're sure it's installed, look for `libcuda.so` in your system and make sure it's discoverable by the dynamic linker.
                         Typically, that involves an entry in '/etc/ld.so.conf', or setting LD_LIBRARY_PATH.""")
            end
        end
    end

    # memoized because otherwise each ccall would perform discovery again
    _libcuda[]
end


# low-level wrappers
const CUdeviceptr = CuPtr{Cvoid}
const CUarray = CuArrayPtr{Cvoid}
const GLuint = Cuint    # FIXME: get these from somewhere
const GLenum = Cuint
include("libcuda_common.jl")
include("error.jl")
include("libcuda.jl")
include("libcuda_deprecated.jl")

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
