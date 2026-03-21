include("setup.jl")
@test cuTENSOR.has_cutensor()
@test cuTensorNet.has_cutensornet()

@testset verbose=true "cuTensorNet" begin
    include("helpers.jl")
    include("errors.jl")
    include("contractions.jl")
end
