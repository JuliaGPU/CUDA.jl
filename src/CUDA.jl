module CUDA

using Logging
if haskey(ENV, "DEBUG")
    @Logging.configure(level=DEBUG)
else
    @Logging.configure(level=ERROR)
end

include("errors.jl")

include("util.jl")
include("base.jl")
include("types.jl")
include("devices.jl")
include("context.jl")
include("module.jl")
include("stream.jl")
include("execution.jl")
include("compilation.jl")

include("memory.jl")
include("arrays.jl")

include("native/execution.jl")
include("native/intrinsics.jl")
include("native/arrays.jl")

include("profile.jl")

function initialize()
	initialize_api()
end
initialize()

end
