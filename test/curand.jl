using CUDA.CURAND

using Random

@testset "CURAND" begin

@testcase "essentials" begin
    @test CURAND.version() isa VersionNumber
end

@testcase "seeding" begin
    rng = CURAND.default_rng()
    Random.seed!(rng)
    Random.seed!(rng, nothing)
    Random.seed!(rng, 1)
    Random.seed!(rng, 1, 0)
end

end
