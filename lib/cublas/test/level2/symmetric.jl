using cuBLAS
using LinearAlgebra

function packed(A::AbstractMatrix{T}, uplo) where {T}
    n = size(A, 1)
    AP = Vector{T}(undef, (n * (n + 1)) >> 1)
    k = 1
    for j in 1:n
        for i in (uplo == :L ? (j:n) : (1:j))
            AP[k] = A[i, j]
            k += 1
        end
    end
    return AP
end

@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    m = 20

    if elty <: Real
        @testset "spmv!" begin
            alpha = rand(elty)
            beta = rand(elty)
            sA = rand(elty, m, m)
            sA = sA + transpose(sA)
            x = rand(elty, m)
            for uplo in ('U', 'L')
                sAP = packed(sA, uplo == 'U' ? :U : :L)
                dsAP = CuVector(sAP)
                dx = CuArray(x)
                y = rand(elty, m)
                dy = CuArray(y)
                BLAS.spmv!(uplo, alpha, sAP, x, beta, y)
                cuBLAS.spmv!(uplo, alpha, dsAP, dx, beta, dy)
                @test y ≈ Array(dy)
            end
        end

        @testset "spmv" begin
            sA = rand(elty, m, m)
            sA = sA + transpose(sA)
            x = rand(elty, m)
            dx = CuArray(x)
            for uplo in ('U', 'L')
                sAP = packed(sA, uplo == 'U' ? :U : :L)
                dsAP = CuVector(sAP)
                y = zeros(elty, m)
                BLAS.spmv!(uplo, one(elty), sAP, x, zero(elty), y)
                dy = cuBLAS.spmv(uplo, dsAP, dx)
                @test y ≈ Array(dy)
            end
        end

        @testset "spr!" begin
            alpha = rand(elty)
            sA = rand(elty, m, m)
            sA = sA + transpose(sA)
            x = rand(elty, m)
            dx = CuArray(x)
            for uplo in ('U', 'L')
                sAP = packed(sA, uplo == 'U' ? :U : :L)
                dsAP = CuVector(sAP)
                BLAS.spr!(uplo, alpha, x, sAP)
                cuBLAS.spr!(uplo, alpha, dx, dsAP)
                @test sAP ≈ Array(dsAP)
            end
        end
    end

    @testset "symv!" begin
        alpha = rand(elty)
        beta = rand(elty)
        sA = rand(elty, m, m)
        sA = sA + transpose(sA)
        dsA = CuArray(sA)
        x = rand(elty, m)
        dx = CuArray(x)
        y = rand(elty, m)
        dy = CuArray(y)
        BLAS.symv!('U', alpha, sA, x, beta, y)
        cuBLAS.symv!('U', alpha, dsA, dx, beta, dy)
        @test y ≈ Array(dy)
    end

    @testset "symv" begin
        sA = rand(elty, m, m)
        sA = sA + transpose(sA)
        dsA = CuArray(sA)
        x = rand(elty, m)
        dx = CuArray(x)
        y = BLAS.symv('U', sA, x)
        dy = cuBLAS.symv('U', dsA, dx)
        @test y ≈ Array(dy)
    end

    if elty <: Complex
        @testset "hemv!" begin
            alpha = rand(elty)
            beta = rand(elty)
            hA = rand(elty, m, m)
            hA = hA + hA'
            dhA = CuArray(hA)
            x = rand(elty, m)
            dx = CuArray(x)
            y = rand(elty, m)
            dy = CuArray(y)
            BLAS.hemv!('U', alpha, hA, x, beta, y)
            cuBLAS.hemv!('U', alpha, dhA, dx, beta, dy)
            @test y ≈ Array(dy)
        end

        @testset "hemv" begin
            hA = rand(elty, m, m)
            hA = hA + hA'
            dhA = CuArray(hA)
            x = rand(elty, m)
            dx = CuArray(x)
            y = BLAS.hemv('U', hA, x)
            dy = cuBLAS.hemv('U', dhA, dx)
            @test y ≈ Array(dy)
        end
    end
end
