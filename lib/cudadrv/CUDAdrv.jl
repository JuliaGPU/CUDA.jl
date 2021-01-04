using .APIUtils

using CEnum

using Printf

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
include(joinpath("context", "primary.jl"))
include("stream.jl")
include("memory.jl")
include("module.jl")
include("events.jl")
include("execution.jl")
include("profile.jl")
include("occupancy.jl")
