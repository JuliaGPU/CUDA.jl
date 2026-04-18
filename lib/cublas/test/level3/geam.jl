using cuBLAS
using LinearAlgebra

@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    m = 20
    n = 35

    @testset "geam!" begin
        alpha = rand(elty)
        beta = rand(elty)
        A = rand(elty, m, n)
        B = rand(elty, m, n)
        C = zeros(elty, m, n)
        d_A = CuArray(A)
        d_B = CuArray(B)

        d_C = CuArray(C)
        cuBLAS.geam!('N', 'N', alpha, d_A, beta, d_B, d_C)
        @test alpha * A + beta * B ≈ Array(d_C)

        # test in place versions too
        d_C = CuArray(C)
        cuBLAS.geam!('N', 'N', alpha, d_C, beta, d_B, d_C)
        @test alpha * C + beta * B ≈ Array(d_C)

        d_C = CuArray(C)
        cuBLAS.geam!('N', 'N', alpha, d_A, beta, d_C, d_C)
        @test alpha * A + beta * C ≈ Array(d_C)

        # test setting C to zero
        cuBLAS.geam!('N', 'N', zero(elty), d_A, zero(elty), d_B, d_C)
        @test Array(d_C) ≈ zeros(elty, m, n)

        # bounds checking
        @test_throws DimensionMismatch cuBLAS.geam!('N', 'T', alpha, d_A, beta, d_B, d_C)
        @test_throws DimensionMismatch cuBLAS.geam!('T', 'T', alpha, d_A, beta, d_B, d_C)
        @test_throws DimensionMismatch cuBLAS.geam!('T', 'N', alpha, d_A, beta, d_B, d_C)
    end

    @testset "geam" begin
        alpha = rand(elty)
        beta = rand(elty)
        A = rand(elty, m, n)
        B = rand(elty, m, n)
        d_A = CuArray(A)
        d_B = CuArray(B)

        d_C = cuBLAS.geam('N', 'N', alpha, d_A, beta, d_B)
        @test alpha * A + beta * B ≈ Array(d_C)

        d_C = cuBLAS.geam('N', 'N', d_A, d_B)
        @test A + B ≈ Array(d_C)
    end

    @testset "CuMatrix -- A ± B" begin
        for opa in (identity, transpose, adjoint),
            opb in (identity, transpose, adjoint)

            p, q = 10, 20
            A = opa == identity ? rand(elty, p, q) : rand(elty, q, p)
            B = opb == identity ? rand(elty, p, q) : rand(elty, q, p)
            dA = CuMatrix{elty}(A)
            dB = CuMatrix{elty}(B)

            @test opa(A) + opb(B) ≈ collect(opa(dA) + opb(dB))
            @test opa(A) - opb(B) ≈ collect(opa(dA) - opb(dB))
        end
    end

    @testset "diagm" begin
        A = rand(elty, m)
        B = rand(elty, n)
        d_A = CuArray(A)
        d_B = CuArray(B)

        diagA = diagm(d_A)
        diagB = diagm(2 => d_B)
        @test A ≈ Array(diagA[diagind(diagA, 0)])
        @test B ≈ Array(diagB[diagind(diagB, 2)])

        diagA = diagm(m, m, 0 => d_A)
        @test A ≈ Array(diagA[diagind(diagA, 0)])

        diagA = diagm(m, m, d_A)
        @test A ≈ Array(diagA[diagind(diagA, 0)])
    end
end
