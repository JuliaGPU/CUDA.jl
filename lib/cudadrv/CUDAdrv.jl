using .APIUtils

using CEnum: @cenum

using Printf

using LazyArtifacts

# Julia has several notions of `sizeof`
# - Base.sizeof is the size of an object in memory
# - Base.aligned_sizeof is the size of an object in an array/inline alloced
# Both of them are equivalent for immutable objects, but differ for mutable singtons and Symbol
# We use `aligned_sizeof` since we care about the size of a type in an array
import Base: aligned_sizeof

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
include("occupancy.jl")
include("graph.jl")

# global state (CUDA.jl's driver wrappers behave like CUDA's runtime library)
include("state.jl")

# support for concurrent programming
include("synchronization.jl")
