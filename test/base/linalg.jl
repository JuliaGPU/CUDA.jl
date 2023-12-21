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

@testset "kron" begin
    dim1A = 50
    dim2A = 80
    dim1B = 90
    dim2B = 40

    @testset for T in [Int16, Int32, Int64,
                       Float16, Float32, Float64,
                       ComplexF16, ComplexF32, ComplexF64]

        A = CUDA.rand(T, dim1A, dim2A)
        B = CUDA.rand(T, dim1B, dim2B);
        @test Array(kron(A, B)) ≈ kron(Array(A), Array(B))
        @test Array(kron(B, A)) ≈ kron(Array(B), Array(A))
    end
end