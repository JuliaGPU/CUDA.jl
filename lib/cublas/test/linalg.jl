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

    # reversed views, which cuBLAS would get as a negative leading dimension or increment
    C = CuArray(zeros(Float32, 8, 5))
    mul!(C, 'N', 'N', view(dA, :, 6:-1:1), dB, true, false)
    @test Array(C) ≈ A[:, 6:-1:1] * B
    y = CuArray(zeros(Float32, 8))
    mul!(y, 'N', view(dA, :, 6:-1:1), dx, true, false)
    @test Array(y) ≈ A[:, 6:-1:1] * x
    mul!(y, 'N', dA, view(dx, 6:-1:1), true, false)
    @test Array(y) ≈ A * reverse(x)
    mul!(view(y, 8:-1:1), 'N', dA, dx, true, false)
    @test Array(y) ≈ reverse(A * x)
    S = rand(ComplexF32, 6, 6); dS = CuArray(S)
    for (tA, wrapper) in (('S', Symmetric), ('H', Hermitian))
        Cc = CuArray(zeros(ComplexF32, 6, 5))
        mul!(Cc, tA, 'N', view(dS, :, 6:-1:1), CuArray{ComplexF32}(dB), true, false)
        @test Array(Cc) ≈ wrapper(S[:, 6:-1:1]) * B
        yc = CuArray(zeros(ComplexF32, 6))
        mul!(yc, tA, dS, view(CuArray{ComplexF32}(dx), 6:-1:1), true, false)
        @test Array(yc) ≈ wrapper(S) * reverse(x)
    end

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

@testset "triangular products and solves with reversed views" begin
    S = rand(Float32, 4, 4) + 4I; B = rand(Float32, 4, 5); x = rand(Float32, 4)
    dS = CuArray(S)
    U, dU = UpperTriangular(S[:, 4:-1:1]), UpperTriangular(view(dS, :, 4:-1:1))

    # reversed vectors
    z = CuArray(x)
    lmul!(UpperTriangular(dS), view(z, 4:-1:1))
    @test Array(z) ≈ reverse(UpperTriangular(S) * reverse(x))
    z = CuArray(x)
    ldiv!(UpperTriangular(dS), view(z, 4:-1:1))
    @test Array(z) ≈ reverse(UpperTriangular(S) \ reverse(x))

    # reversed triangular matrix
    z = CuArray(x)
    @test Array(lmul!(dU, z)) ≈ U * x
    z = CuArray(x)
    @test Array(ldiv!(dU, z)) ≈ U \ x
    Z = CuArray(B)
    @test Array(lmul!(dU, Z)) ≈ U * B
    Z = CuArray(B)
    @test Array(ldiv!(dU, Z)) ≈ U \ B
    Z = CuArray(copy(B'))
    @test Array(rmul!(Z, dU)) ≈ B' * U
    Z = CuArray(copy(B'))
    @test Array(rdiv!(Z, dU)) ≈ B' / U

    # reversed right-hand side
    Z = CuArray(B)
    lmul!(UpperTriangular(dS), view(Z, :, 5:-1:1))
    @test Array(Z) ≈ (UpperTriangular(S) * B[:, 5:-1:1])[:, 5:-1:1]
    Z = CuArray(B)
    ldiv!(UpperTriangular(dS), view(Z, :, 5:-1:1))
    @test Array(Z) ≈ (UpperTriangular(S) \ B[:, 5:-1:1])[:, 5:-1:1]
    Z = CuArray(copy(B'))
    rmul!(view(Z, 5:-1:1, :), UpperTriangular(dS))
    @test Array(Z) ≈ (B'[5:-1:1, :] * UpperTriangular(S))[5:-1:1, :]
    Z = CuArray(copy(B'))
    rdiv!(view(Z, 5:-1:1, :), UpperTriangular(dS))
    @test Array(Z) ≈ (B'[5:-1:1, :] / UpperTriangular(S))[5:-1:1, :]
end
