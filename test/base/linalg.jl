using LinearAlgebra

@testset "normalize!" begin
    x = rand(ComplexF32, 10)
    dx = CuVector{ComplexF32}(x)
    @test isreal(norm(dx, 2))
    @test norm(normalize!(dx)) ≈ 1
end

@testset "dot" begin
    @testset for T in [Int16, Int32, Int64,
                       Float16, Float32, Float64,
                       ComplexF16, ComplexF32, ComplexF64]
        @test testf(dot, rand(T, 256), rand(Bool, 256))
        @test testf(dot, rand(Bool, 256), rand(T, 256))
    end

    @test testf(dot, rand(Bool, 1024, 1024), rand(Float64, 1024, 1024))
end
