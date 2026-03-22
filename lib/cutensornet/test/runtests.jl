include("setup.jl")
@test cuTENSOR.functional()
@test cuTensorNet.functional()

@testset verbose=true "cuTensorNet" begin
    include("helpers.jl")
    include("errors.jl")
    include("contractions.jl")
end
