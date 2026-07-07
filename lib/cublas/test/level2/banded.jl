using cuBLAS
using cuBLAS: band, bandex
using LinearAlgebra

@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    m = 20
    n = 35

    @testset "gbmv!" begin
        ku = 2
        kl = 3
        A = bandex(rand(elty, m, n), kl, ku)
        Ab = band(A, kl, ku)
        d_Ab = CuArray(Ab)
        alpha = rand(elty)
        beta = rand(elty)

        for trans in ('N', 'T', 'C')
            if trans == 'N'
                x = rand(elty, n); y = rand(elty, m)
            else
                x = rand(elty, m); y = rand(elty, n)
            end
            d_x = CuArray(x)
            d_y = CuArray(y)
            cuBLAS.gbmv!(trans, m, kl, ku, alpha, d_Ab, d_x, beta, d_y)
            BLAS.gbmv!(trans, m, kl, ku, alpha, Ab, x, beta, y)
            @test y ≈ Array(d_y)
        end

        # test alpha=1 version without y
        x = rand(elty, n)
        d_x = CuArray(x)
        d_y = cuBLAS.gbmv('N', m, kl, ku, d_Ab, d_x)
        y = BLAS.gbmv('N', m, kl, ku, Ab, x)
        @test y ≈ Array(d_y)
    end

    @testset "gbmv" begin
        ku = 2
        kl = 3
        A = bandex(rand(elty, m, n), kl, ku)
        Ab = band(A, kl, ku)
        d_Ab = CuArray(Ab)
        x = rand(elty, n)
        d_x = CuArray(x)
        alpha = rand(elty)
        # test y = alpha*A*x
        d_y = cuBLAS.gbmv('N', m, kl, ku, alpha, d_Ab, d_x)
        y = BLAS.gbmv('N', m, kl, ku, alpha, Ab, x)
        @test y ≈ Array(d_y)
    end

    # upper banded storage of a symmetric/hermitian matrix
    function sym_banded(elty, m, nbands)
        A = rand(elty, m, m)
        A = A + A'
        A = bandex(A, nbands, nbands)
        AB = band(A, 0, nbands)
        A, AB
    end

    # upper banded storage of a triangular matrix
    function tri_banded(elty, m, nbands)
        A = bandex(rand(elty, m, m), 0, nbands)
        AB = band(A, 0, nbands)
        A, AB
    end

    if elty <: Real
        @testset "sbmv!" begin
            nbands = 3
            @test m >= 1 + nbands
            A, AB = sym_banded(elty, m, nbands)
            x = rand(elty, m)
            y = rand(elty, m)
            d_AB = CuArray(AB)
            d_x = CuArray(x)
            d_y = CuArray(y)
            alpha = rand(elty)
            beta = rand(elty)
            cuBLAS.sbmv!('U', nbands, alpha, d_AB, d_x, beta, d_y)
            @test alpha * (A * x) + beta * y ≈ Array(d_y)
        end

        @testset "sbmv" begin
            nbands = 3
            @test m >= 1 + nbands
            A, AB = sym_banded(elty, m, nbands)
            x = rand(elty, m)
            d_AB = CuArray(AB)
            d_x = CuArray(x)
            d_y = cuBLAS.sbmv('U', nbands, d_AB, d_x)
            @test A * x ≈ Array(d_y)
        end
    else
        @testset "hbmv!" begin
            nbands = 3
            @test m >= 1 + nbands
            A, AB = sym_banded(elty, m, nbands)
            x = rand(elty, m)
            y = rand(elty, m)
            d_AB = CuArray(AB)
            d_x = CuArray(x)
            d_y = CuArray(y)
            alpha = rand(elty)
            beta = rand(elty)
            cuBLAS.hbmv!('U', nbands, alpha, d_AB, d_x, beta, d_y)
            @test alpha * (A * x) + beta * y ≈ Array(d_y)
        end

        @testset "hbmv" begin
            nbands = 3
            @test m >= 1 + nbands
            A, AB = sym_banded(elty, m, nbands)
            x = rand(elty, m)
            d_AB = CuArray(AB)
            d_x = CuArray(x)
            d_y = cuBLAS.hbmv('U', nbands, d_AB, d_x)
            @test A * x ≈ Array(d_y)
        end
    end

    @testset "tbmv!" begin
        nbands = 3
        @test m >= 1 + nbands
        A, AB = tri_banded(elty, m, nbands)
        d_AB = CuArray(AB)
        y = rand(elty, m)
        d_y = CuArray(y)
        cuBLAS.tbmv!('U', 'N', 'N', nbands, d_AB, d_y)
        @test A * y ≈ Array(d_y)
    end

    @testset "tbmv" begin
        nbands = 3
        @test m >= 1 + nbands
        A, AB = tri_banded(elty, m, nbands)
        d_AB = CuArray(AB)
        x = rand(elty, m)
        d_x = CuArray(x)
        d_y = cuBLAS.tbmv('U', 'N', 'N', nbands, d_AB, d_x)
        @test A * x ≈ Array(d_y)
    end

    @testset "tbsv!" begin
        nbands = 3
        @test m >= 1 + nbands
        A, AB = tri_banded(elty, m, nbands)
        d_AB = CuArray(AB)
        x = rand(elty, m)
        d_x = CuArray(x)
        d_y = copy(d_x)
        cuBLAS.tbsv!('U', 'N', 'N', nbands, d_AB, d_y)
        @test A \ x ≈ Array(d_y)
    end

    @testset "tbsv" begin
        nbands = 3
        @test m >= 1 + nbands
        A, AB = tri_banded(elty, m, nbands)
        d_AB = CuArray(AB)
        x = rand(elty, m)
        d_x = CuArray(x)
        d_y = cuBLAS.tbsv('U', 'N', 'N', nbands, d_AB, d_x)
        @test A \ x ≈ Array(d_y)
    end
end
