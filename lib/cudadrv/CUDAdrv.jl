using .APIUtils

using CEnum

using Memoize

using Printf

const libcuda = Sys.iswindows() ? "nvcuda" : ( Sys.islinux() ? "libcuda.so.1" : "libcuda" )


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
