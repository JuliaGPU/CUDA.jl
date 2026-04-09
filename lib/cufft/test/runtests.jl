include("setup.jl")
@test cuFFT.version() isa VersionNumber

@testset "ensure_raising" begin
    @test (1,2,3) == cuFFT.ensure_raising((1,2,3))
    @test (1,2,3) == cuFFT.ensure_raising((2,1,3))
    @test (1,2,3) == cuFFT.ensure_raising((1,3,2))
    @test (1,2,3) == cuFFT.ensure_raising((3,1,2))
    @test (1,2,3) == cuFFT.ensure_raising((2,3,1))
    @test (10,20,30) == cuFFT.ensure_raising((30,20,10))
end

include("complex.jl")
include("real.jl")
include("integer.jl")
include("issues.jl")
