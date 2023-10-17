using CUDA.CUSOLVER
using LinearAlgebra

m = 15
n = 10
p = 5

@testset "elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    @testset "sytrs!" begin
        for uplo in ('L', 'U')
            A = rand(elty,n,n)
            B = rand(elty,n,p)
            A = A + A'
            d_A = CuMatrix(A)
            d_B = CuMatrix(B)
            d_A, d_ipiv, _ = CUSOLVER.sytrf!(uplo, d_A)
            d_ipiv = CuVector{Int64}(d_ipiv)
            A, ipiv, _ = LAPACK.sytrf!(uplo, A)
            CUSOLVER.sytrs!(uplo, d_A, d_ipiv, d_B)
            LAPACK.sytrs!(uplo, A, ipiv, B)
            @test B ≈ collect(d_B)
        end
    end

    @testset "trtri!" begin
        for uplo in ('L', 'U')
            for diag in ('N', 'U')
                A = rand(elty,n,n)
                A = uplo == 'L' ? tril(A) : triu(A)
                A = diag == 'N' ? A : A - Diagonal(A) + I
                d_A = CuMatrix(A)
                d_B = copy(d_A)
                CUSOLVER.trtri!(uplo, diag, d_B)
                @test collect(d_A * d_B) ≈ I
            end
        end
    end

    @testset "potrf! -- potrs!" begin
        for uplo in ('L', 'U')
            A    = rand(elty,n,n)
            A    = A*A' + I
            B    = rand(elty,n,p)
            d_A  = CuMatrix(A)
            d_B  = CuMatrix(B)

            CUSOLVER.Xpotrf!(uplo, d_A)
            CUSOLVER.Xpotrs!(uplo, d_A, d_B)
            LAPACK.potrf!(uplo, A)
            LAPACK.potrs!(uplo, A, B)
            @test B ≈ collect(d_B)
        end
    end

    @testset "getrf! -- getrs!" begin
        for trans in ('N', 'T', 'C')
            A   = rand(elty,n,n)
            B   = rand(elty,n,p)
            d_A = CuMatrix(A)
            d_B = CuMatrix(B)

            d_A, d_ipiv, _ = CUSOLVER.Xgetrf!(d_A)
            CUSOLVER.Xgetrs!(trans, d_A, d_ipiv, d_B)
            A, ipiv, _ = LAPACK.getrf!(A)
            LAPACK.getrs!(trans, A, ipiv, B)
            @test B ≈ collect(d_B)
        end
    end

    @testset "geqrf! -- omgqr!" begin
        A = rand(elty,m,n)
        d_A = CuMatrix(A)
        d_A, tau = CUSOLVER.Xgeqrf!(d_A)
        CUSOLVER.orgqr!(d_A, tau)
        @test collect(d_A' * d_A) ≈ I
    end

    @testset "syevd!" begin
        for uplo in ('L', 'U')
            A = rand(elty,n,n)
            B = A + A'
            A = uplo == 'L' ? tril(B) : triu(B)
            d_A = CuMatrix(A)
            W, V = CUSOLVER.Xsyevd!('V', uplo, d_A)
            @test B ≈ collect(V * Diagonal(W) * V')

            d_A = CuMatrix(A)
            d_W = CUSOLVER.Xsyevd!('N', uplo, d_A)
        end
    end

    @testset "gesvd!" begin
        A = rand(elty,m,n)
        d_A = CuMatrix(A)
        U, Σ, Vt = CUSOLVER.Xgesvd!('A', 'A', d_A)
        @test A ≈ collect(U[:,1:n] * Diagonal(Σ) * Vt)
    end

    # @testset "syevdx!" begin
    #     if elty <: Real
    #         A = [3.5 0.5 0.0; 0.5 3.5 0.0; 0.0 0.0 2.0]
    #         d_A = CuMatrix(A)
    #         d_W, d_V, neig = CUSOLVER.Xsyevdx!('V', 'L', 'A', d_A)
    #         @test neig = 3
    #         W = collect(d_W)
    #         V = collect(d_V)
    #         @test W ≈ elty[2.0, 3.0, 4.0]
    #         test A * V ≈ V * Diagonal(W)
    #     end
    # end
end
