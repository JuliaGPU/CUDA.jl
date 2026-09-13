using LinearAlgebra
using StaticArrays: SVector

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

        @test testf(dot, rand(T, 256), rand(T, 256, 256), rand(Bool, 256))
        @test testf(dot, rand(Bool, 256), rand(T, 256, 256), rand(T, 256))
    end

    @test testf(dot, rand(Bool, 1024, 1024), rand(Float64, 1024, 1024))

    # https://discourse.julialang.org/t/result-of-inner-product-of-two-cuarray-with-views-is-incorrect/121539
    @test testf(dot, view(rand(Float32, 100, 100), 2:99, 2:99),
                     view(rand(Float32, 100, 100), 2:99, 2:99))

    # The fallback must preserve dot's result type and integer overflow behavior.
    @testset "deterministic fallback" begin
        old_mode = CUDACore.math_mode()
        CUDACore.math_mode!(CUDACore.PEDANTIC_MATH)
        try
            @testset for T in [Int16, Int32, Int64, Float32, Float64]
                @test testf(dot, rand(T, 256), rand(T, 256))
                @test testf(dot, rand(T, 256), rand(T, 256, 256), rand(T, 256))
            end
            # The scalar result type need not match the input element types.
            x = [SVector(1f0, 2f0), SVector(3f0, 4f0)]
            y = [SVector(5f0, 6f0), SVector(7f0, 8f0)]
            @test testf(dot, x, Float32[1 2; 3 4], y)
            for T in (Int16, Int32, Float32), (m, n) in ((0, 0), (0, 3), (3, 0))
                @test dot(CUDA.zeros(T, m), CUDA.zeros(T, m, n), CUDA.zeros(T, n)) === zero(T)
            end
        finally
            CUDACore.math_mode!(old_mode)
        end
    end
end

@testset "kron" begin
    dim1A = 50
    dim2A = 80
    dim1B = 90
    dim2B = 40

    @testset for T in [Int16, Int32, Int64,
                       Float16, Float32, Float64,
                       ComplexF16, ComplexF32, ComplexF64]

        A = CuArray(rand(T, dim1A, dim2A))
        B = CuArray(rand(T, dim1B, dim2B))
        @test Array(kron(A, B)) ≈ kron(Array(A), Array(B))
        @test Array(kron(B, A)) ≈ kron(Array(B), Array(A))
    end
end
