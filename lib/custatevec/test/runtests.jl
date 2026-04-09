include("setup.jl")
@test cuStateVec.functional()

@testset verbose=true "cuStateVec" begin
    include("errors.jl")
    include("apply_matrix.jl")
    include("apply_pauli.jl")
    include("measure.jl")
    include("collapse.jl")
    include("utilities.jl")
end

include("multigpu.jl")
