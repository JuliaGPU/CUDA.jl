using LinearAlgebra
using StaticArrays: SVector

@testset "normalize!" begin
    x = rand(ComplexF32, 10)
    dx = CuVector{ComplexF32}(x)
    @test isreal(norm(dx, 2))
    @test norm(normalize!(dx)) ≈ 1
end

@testset "dot" begin
    # one eltype per code path of the generic fallback: the atomic kernel
    # (Int16 needs sm_70+, otherwise it falls back) and the mapreduce path (complex)
    @testset for T in [Int16, Int64, Float16, ComplexF32]
        @test testf(dot, rand(T, 256), rand(Bool, 256))
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
            # cuBLAS-backed dot also has to honour the pedantic math mode
            @test testf(dot, rand(Float32, 256), rand(Float32, 256))
            @testset for T in [Int16]
                @test testf(dot, rand(T, 256), rand(T, 256))
                @test testf(dot, rand(T, 256), rand(T, 256, 256), rand(T, 256))
            end
            # The scalar result type need not match the input element types.
            x = [SVector(1f0, 2f0), SVector(3f0, 4f0)]
            y = [SVector(5f0, 6f0), SVector(7f0, 8f0)]
            @test testf(dot, x, Float32[1 2; 3 4], y)
            for T in (Int16,), (m, n) in ((0, 0), (0, 3), (3, 0))
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

    # one type with and one without cached (ldg) loads
    @testset for T in [Float32, ComplexF32]
        A = CuArray(rand(T, dim1A, dim2A))
        B = CuArray(rand(T, dim1B, dim2B))
        @test Array(kron(A, B)) ≈ kron(Array(A), Array(B))
        @test Array(kron(B, A)) ≈ kron(Array(B), Array(A))
    end
end

@testset "storage-level mul! falls back where cuBLAS can't go" begin
    A = rand(Float32, 8, 6); B = rand(Float32, 6, 5); x = rand(Float32, 6)
    dA, dB, dx = CuArray(A), CuArray(B), CuArray(x)

    # columns that aren't contiguous (row step 2)
    C = CuArray(zeros(Float32, 4, 5))
    mul!(C, 'N', 'N', view(dA, 1:2:8, :), dB, true, false)
    @test Array(C) ≈ A[1:2:8, :] * B
    y = CuArray(zeros(Float32, 4))
    mul!(y, 'N', view(dA, 1:2:8, :), dx, true, false)
    @test Array(y) ≈ A[1:2:8, :] * x

    # element types cuBLAS has no symm/hemm for
    S = rand(Float16, 4, 4); B16 = rand(Float16, 2, 3)
    C16 = CuArray(zeros(Float16, 2, 3))
    mul!(C16, 'S', 'N', view(CuArray(S), 1:2, 1:2), CuArray(B16), true, false)
    @test Array(C16) ≈ Symmetric(S[1:2, 1:2]) * B16
    S32 = rand(Float32, 4, 4); B64 = rand(Float64, 4, 3)
    C64 = CuArray(zeros(Float64, 4, 3))
    mul!(C64, 'S', 'N', CuArray(S32), CuArray(B64), true, false)
    @test Array(C64) ≈ Symmetric(S32) * B64
end
