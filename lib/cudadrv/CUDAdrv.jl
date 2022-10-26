using .APIUtils

using CEnum: @cenum

using Printf

using LazyArtifacts


# low-level wrappers
include("libcuda.jl")
include("libcuda_deprecated.jl")

# high-level wrappers
include("error.jl")
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
