using CUDA.CUSOLVER
using LinearAlgebra

m = 15
n = 10
p = 5

@testset "cusolver -- generic API -- $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    if CUSOLVER.version() >= v"11.7.1"
        @testset "geev!" begin
            A = rand(elty,n,n)
            d_A = CuMatrix(A)
            d_B = copy(d_A)
            W, VL, VR = CUSOLVER.Xgeev!('N', 'V', d_A)
            if elty <: Complex
                @test d_B * VR ≈ VR * Diagonal(W)
            else
                h_W = collect(W)
                i = 1
                while i <= n
                    if h_W[i].im ≈ zero(elty)
                        @test d_B * VR[:,i] ≈ h_W[i].re * VR[:,i]
                        i = i + 1
                    else
                        V1 = VR[:,i] + im * VR[:,i+1]
                        @test d_B * V1 ≈ h_W[i] * V1
                        V2 = VR[:,i] - im * VR[:,i+1]
                        @test d_B * V2 ≈ h_W[i+1] * V2
                        i = i + 2
                    end
                end
            end
        end

        @testset "syevBatched!" begin
            batch_size = 5
            for uplo in ('L', 'U')
                (CUSOLVER.version() < v"11.7.2") && (uplo == 'L') && (elty == ComplexF32) && continue

                A = rand(elty, n, n * batch_size)
                B = rand(elty, n, n * batch_size)
                for i = 1:batch_size
                    S = rand(elty,n,n)
                    S = S * S' + I
                    B[:,(i-1)*n+1:i*n] .= S
                    S = uplo == 'L' ? tril(S) : triu(S)
                    A[:,(i-1)*n+1:i*n] .= S
                end
                d_A = CuMatrix(A)
                d_W, d_V = CUSOLVER.XsyevBatched!('V', uplo, d_A)
                W = collect(d_W)
                V = collect(d_V)
                for i = 1:batch_size
                    Bᵢ = B[:,(i-1)*n+1:i*n]
                    Wᵢ = Diagonal(W[(i-1)*n+1:i*n])
                    Vᵢ = V[:,(i-1)*n+1:i*n]
                    @test Bᵢ * Vᵢ ≈ Vᵢ * Diagonal(Wᵢ)
                end

                d_A = CuMatrix(A)
                d_W = CUSOLVER.XsyevBatched!('N', uplo, d_A)
            end
        end
    end

    if CUSOLVER.version() >= v"11.6.0"
        @testset "larft!" begin
            @testset "direct = $direct" for direct in ('F', 'B')
                direct == 'B' && continue
                A = rand(elty,m,n)
                t = rand(elty,n,n)

                dA = CuMatrix(A)
                dA, dτ = CUSOLVER.geqrf!(dA)
                hI = Matrix{elty}(I, m, m)
                dI = CuArray(hI)
                dH = CUSOLVER.ormqr!('L', 'N', dA, dτ, copy(dI))

                v = Array(dA)
                for j = 1:n
                    v[j,j] = one(elty)
                    for i = 1:j-1
                        v[i,j] = zero(elty)
                    end
                end
                dv = CuArray(v)
                dt = CuMatrix(t)
                dt = CUSOLVER.larft!(direct, 'C', dv, dτ, dt)
                @test dI - dv * dt * dv' ≈ dH
            end
        end
    end

    @testset "sytrs!" begin
        @testset "uplo = $uplo" for uplo in ('L', 'U')
            A = rand(elty,n,n)
            B = rand(elty,n,p)
            A = A + transpose(A)
            X = A \ B
            d_A = CuMatrix(A)
            d_B = CuMatrix(B)
            @testset "pivoting = $pivoting" for pivoting in (false, true)
                !pivoting && (CUSOLVER.version() < v"11.7.2") && continue
                if pivoting
                    d_A, d_ipiv, _ = CUSOLVER.sytrf!(uplo, d_A; pivoting)
                    d_ipiv = CuVector{Int64}(d_ipiv)
                    A, ipiv, _ = LAPACK.sytrf!(uplo, A)
                    CUSOLVER.sytrs!(uplo, d_A, d_ipiv, d_B)
                    LAPACK.sytrs!(uplo, A, ipiv, B)
                    @test B ≈ collect(d_B)
                else
                    d_A, _ = CUSOLVER.sytrf!(uplo, d_A; pivoting)
                    CUSOLVER.sytrs!(uplo, d_A, d_B)
                    @test X ≈ collect(d_B)
                end
            end
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

        for jobu in ('A', 'S', 'N', 'O')
            for jobvt in ('A', 'S', 'N', 'O')
                (jobu == 'A') && (jobvt == 'A') && continue
                (jobu == 'O') && (jobvt == 'O') && continue
                d_A = CuMatrix(A)
                U2, Σ2, Vt2 = CUSOLVER.Xgesvd!(jobu, jobvt, d_A)
                @test Σ ≈ Σ2
            end
        end
    end

    @testset "gesvdp!" begin
        A = rand(elty,m,n)
        d_A = CuMatrix(A)
        U, Σ, V, err_sigma = CUSOLVER.Xgesvdp!('V', 0, d_A)
        @test A ≈ collect(U[:,1:n]) * Diagonal(collect(Σ)) * collect(V)'

        d_A = CuMatrix(A)
        U, Σ, V, err_sigma = CUSOLVER.Xgesvdp!('V', 1, d_A)
        @test A ≈ collect(U) * Diagonal(collect(Σ)) * collect(V)'

        A = rand(elty,n,m)
        d_A = CuMatrix(A)
        U, Σ, V, err_sigma = CUSOLVER.Xgesvdp!('V', 0, d_A)
        @test A ≈ collect(U) * Diagonal(collect(Σ)) * collect(V[:,1:n])'

        d_A = CuMatrix(A)
        U, Σ, V, err_sigma = CUSOLVER.Xgesvdp!('V', 1, d_A)
        @test A ≈ collect(U) * Diagonal(collect(Σ)) * collect(V)'
    end

    @testset "gesvdr!" begin
        R = real(elty)
        tol = R == Float32 ? 1e-2 : 1e-5
        ℓ = min(m, n)

        B = rand(elty,m,m)
        FB = qr(B)
        C = rand(elty,n,n)
        FC = qr(C)
        Σ = zeros(R,m,n)
        for i = 1:ℓ
            Σ[i,i] = (10-i+1)*one(R)
        end
        A = FB.Q * Σ * FC.Q
        d_A = CuMatrix(A)

        d_U, d_Σ, d_V = CUSOLVER.Xgesvdr!('N', 'N', d_A, 3)
        @test norm(diag(Σ)[1:3] - collect(d_Σ[1:3])) ≤ tol

        d_U, d_Σ, d_V = CUSOLVER.Xgesvdr!('S', 'S', d_A, ℓ)
        @test norm(diag(Σ) - collect(d_Σ)) ≤ tol
    end

    @testset "syevdx!" begin
        R = real(elty)
        Σ = [i*one(R) for i = 1:10]
        B = rand(elty, 10, 10)
        F = qr(B)
        A = F.Q * Diagonal(Σ) * F.Q'
        for uplo in ('L', 'U')
            h_A = uplo == 'L' ? tril(A) : triu(A)
            d_A = CuMatrix{elty}(h_A)

            d_W, d_V, neig = CUSOLVER.Xsyevdx!('V', 'A', uplo, d_A, vl=3.5, vu= 7.5, il=1, iu=3)
            @test neig == 10
            @test collect(d_W) ≈ Σ
            @test A ≈ collect(d_V * Diagonal(d_W) * d_V')

            d_W, neig = CUSOLVER.Xsyevdx!('N', 'I', uplo, d_A, vl=3.5, vu= 7.5, il=1, iu=3)
            @test neig == 3

            d_W, neig = CUSOLVER.Xsyevdx!('N', 'V', uplo, d_A, vl=3.5, vu= 7.5, il=1, iu=3)
        end
    end
end
