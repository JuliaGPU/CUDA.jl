@testset "CUDA driver" begin

include("cuda/errors.jl")
include("cuda/version.jl")
include("cuda/devices.jl")
include("cuda/context.jl")
include("cuda/module.jl")
include("cuda/memory.jl")
include("cuda/stream.jl")
include("cuda/execution.jl")
include("cuda/events.jl")
include("cuda/profile.jl")
include("cuda/occupancy.jl")

end
