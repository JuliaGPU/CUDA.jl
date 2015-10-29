module CUDA

using Logging
@Logging.configure(level=ERROR)

include("errors.jl")

include("base.jl")
include("types.jl")
include("devices.jl")
include("context.jl")
include("module.jl")
include("stream.jl")
include("execution.jl")

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
