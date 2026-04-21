using cuBLAS
using LinearAlgebra

@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    m = 20
    n = 35
    k = 13

    @testset "syrkx!" begin
        alpha = rand(elty)
        beta = rand(elty)
        A = rand(elty, n, k)
        B = rand(elty, n, k)
        C = rand(elty, n, n)
        C += C'
        d_A = CuArray(A)
        d_B = CuArray(B)
        d_C = CuArray(C)
        # C = (alpha*A)*transpose(B) + beta*C
        d_C = cuBLAS.syrkx!('U', 'N', alpha, d_A, d_B, beta, d_C)
        final_C = (alpha * A) * transpose(B) + beta * C
        @test triu(final_C) ≈ triu(Array(d_C))

        @test_throws DimensionMismatch cuBLAS.syrkx!('U', 'N', alpha, d_A, d_B, beta, CuArray(rand(elty, m, n)))
        @test_throws DimensionMismatch cuBLAS.syrkx!('U', 'N', alpha, d_A, d_B, beta, CuArray(rand(elty, n + 1, n + 1)))
    end

    @testset "syrkx" begin
        A = rand(elty, n, k)
        B = rand(elty, n, k)
        d_A = CuArray(A)
        d_B = CuArray(B)
        d_C = cuBLAS.syrkx('U', 'N', d_A, d_B)
        @test triu(A * transpose(B)) ≈ triu(Array(d_C))
    end

    @testset "syrk" begin
        A = rand(elty, m, k)
        d_A = CuArray(A)
        d_C = cuBLAS.syrk('U', 'N', d_A)
        @test triu(A * transpose(A)) ≈ triu(Array(d_C))
    end

    @testset "syr2k!" begin
        alpha = rand(elty)
        beta = rand(elty)
        A = rand(elty, m, k)
        B = rand(elty, m, k)
        C = rand(elty, m, m)
        C = C + transpose(C)
        d_A = CuArray(A)
        d_B = CuArray(B)
        d_C = CuArray(C)

        C = alpha * (A * transpose(B) + B * transpose(A)) + beta * C
        cuBLAS.syr2k!('U', 'N', alpha, d_A, d_B, beta, d_C)
        @test triu(C) ≈ triu(Array(d_C))

        @test_throws DimensionMismatch cuBLAS.syr2k!('U', 'N', alpha, d_A, CuArray(rand(elty, m + 1, k + 1)), beta, d_C)
    end

    @testset "syr2k" begin
        alpha = rand(elty)
        A = rand(elty, m, k)
        B = rand(elty, m, k)
        d_A = CuArray(A)
        d_B = CuArray(B)

        C = alpha * (A * transpose(B) + B * transpose(A))
        d_C = cuBLAS.syr2k('U', 'N', alpha, d_A, d_B)
        @test triu(C) ≈ triu(Array(d_C))

        C = A * transpose(B) + B * transpose(A)
        d_C = cuBLAS.syr2k('U', 'N', d_A, d_B)
        @test triu(C) ≈ triu(Array(d_C))
    end

    if elty <: Complex
        @testset "herk!" begin
            alpha = rand(real(elty))
            beta = rand(real(elty))
            A = rand(elty, m, m)
            hA = A + A'
            d_A = CuArray(A)
            d_C = CuArray(hA)
            cuBLAS.herk!('U', 'N', alpha, d_A, beta, d_C)
            C = real(alpha) * (A * A') + real(beta) * hA
            @test triu(C) ≈ triu(Array(d_C))
        end

        @testset "herk" begin
            A = rand(elty, m, m)
            d_A = CuArray(A)
            d_C = cuBLAS.herk('U', 'N', d_A)
            @test triu(A * A') ≈ triu(Array(d_C))
        end

        @testset "her2k!" begin
            α = rand(elty)
            β = rand(real(elty))
            A = rand(elty, m, k)
            B = rand(elty, m, k)
            C = rand(elty, m, m)
            C = C + C'
            d_A = CuArray(A)
            d_B = CuArray(B)
            d_C = CuArray(C)

            C = α * (A * B') + conj(α) * (B * A') + β * C
            cuBLAS.her2k!('U', 'N', α, d_A, d_B, β, d_C)
            @test triu(C) ≈ triu(Array(d_C))

            @test_throws DimensionMismatch cuBLAS.her2k!('U', 'N', α, d_A, CuArray(rand(elty, m + 1, k + 1)), β, d_C)
            @test_throws DimensionMismatch cuBLAS.her2k!('U', 'N', α, d_A, CuArray(rand(elty, m, k + 1)), β, d_C)
        end

        @testset "her2k" begin
            α = rand(elty)
            A = rand(elty, m, k)
            B = rand(elty, m, k)
            d_A = CuArray(A)
            d_B = CuArray(B)

            C = α * A * B' + conj(α) * B * A'
            d_C = cuBLAS.her2k('U', 'N', α, d_A, d_B)
            @test triu(C) ≈ triu(Array(d_C))

            C = A * B' + B * A'
            d_C = cuBLAS.her2k('U', 'N', d_A, d_B)
            @test triu(C) ≈ triu(Array(d_C))
        end
    end
end
