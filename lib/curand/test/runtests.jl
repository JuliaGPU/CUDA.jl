include("setup.jl")
@test cuRAND.version() isa VersionNumber

include("high_level.jl")
