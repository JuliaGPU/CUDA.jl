@testset "deprecated seed! with positional offset" begin
    rng = cuRAND.LibraryRNG()

    Random.seed!(rng, 1; offset=100)
    A = Array(rand(rng, Float32, 16))

    @test_deprecated Random.seed!(rng, 1, 100)
    B = Array(rand(rng, Float32, 16))
    @test A == B

    Random.seed!(rng, 1)
    C = Array(rand(rng, Float32, 16))
    @test A != C
end
