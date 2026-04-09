include("setup.jl")

@testset verbose=true "cuSOLVER" begin

include("base.jl")
include("dense.jl")
include("dense_generic.jl")
include("sparse.jl")
include("sparse_factorizations.jl")
include("multigpu.jl")

end
