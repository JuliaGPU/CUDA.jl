include("setup.jl")
@test cuFFT.version() isa VersionNumber

include("complex.jl")
include("real.jl")
include("integer.jl")
include("issues.jl")
