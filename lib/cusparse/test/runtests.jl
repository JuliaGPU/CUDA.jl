include("setup.jl")
@test cuSPARSE.version() isa VersionNumber

include("array.jl")
include("construction.jl")
include("conversion.jl")
include("preconditioners.jl")
include("solvers.jl")
include("operations.jl")
include("gtsv2.jl")
include("misc.jl")
include("kernelabstractions.jl")
