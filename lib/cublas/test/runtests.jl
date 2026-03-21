include("setup.jl")
@test cuBLAS.has_cublas()

@testset verbose=true "cuBLAS" begin

include("level1.jl")
include("level2.jl")
include("level3.jl")
include("extensions.jl")
include("xt.jl")
include("linalg.jl")

end
