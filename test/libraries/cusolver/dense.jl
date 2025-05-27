using CUDA.CUSOLVER

using LinearAlgebra
using LinearAlgebra: BlasInt

m = 15
n = 10
p = 5
l = 13
k = 1

@testset "elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    @testset "gesv!" begin
        @testset "irs_precision = AUTO" begin
            A = rand(elty, n, n)
            X = zeros(elty, n, p)
            B = rand(elty, n, p)
            dA = CuArray(A)
            dX = CuArray(X)
            dB = CuArray(B)
            CUSOLVER.gesv!(dX, dA, dB)
            tol = real(elty) |> eps |> sqrt
            dR = dB - dA * dX
            @test norm(dR) <= tol
        end
        @testset "irs_precision = $elty" begin
            irs_precision = elty <: Real ? "R_" : "C_"
            irs_precision *= string(sizeof(real(elty)) * 8) * "F"
            A = rand(elty, n, n)
            X = zeros(elty, n, p)
            B = rand(elty, n, p)
            dA = CuArray(A)
            dX = CuArray(X)
            dB = CuArray(B)
            CUSOLVER.gesv!(dX, dA, dB; irs_precision=irs_precision)
            tol = real(elty) |> eps |> sqrt
            dR = dB - dA * dX
            @test norm(dR) <= tol
        end
        @testset "IRSParameters" begin
            params = CUSOLVER.CuSolverIRSParameters()
            max_iter = 10
            CUSOLVER.cusolverDnIRSParamsSetMaxIters(params, max_iter)
            @test CUSOLVER.get_info(params, :maxiters) == max_iter
            @test_throws ErrorException("The information fake is incorrect.") CUSOLVER.get_info(params, :fake)
            A = rand(elty, n, n)
            X = zeros(elty, n, p)
            B = rand(elty, n, p)
            dA = CuArray(A)
            dX = CuArray(X)
            dB = CuArray(B)
            dX, info = CUSOLVER.gesv!(dX, dA, dB; maxiters=max_iter)
            @test CUSOLVER.get_info(info, :maxiters) == max_iter
            @test CUSOLVER.get_info(info, :niters) <= max_iter
            @test CUSOLVER.get_info(info, :outer_niters) <= max_iter
            @test_throws ErrorException("The information fake is incorrect.") CUSOLVER.get_info(info, :fake)
        end
    end

    @testset "gels!" begin
        @testset "irs_precision = AUTO" begin
            A = rand(elty, m, n)
            X = zeros(elty, n, p)
            B = A * rand(elty, n, p)  # ensure that AX = B is consistent
            dA = CuArray(A)
            dX = CuArray(X)
            dB = CuArray(B)
            CUSOLVER.gels!(dX, dA, dB)
            tol = real(elty) |> eps |> sqrt
            dR = dB - dA * dX
        end
        @testset "irs_precision = $elty" begin
            irs_precision = elty <: Real ? "R_" : "C_"
            irs_precision *= string(sizeof(real(elty)) * 8) * "F"
            A = rand(elty, m, n)
            X = zeros(elty, n, p)
            B = A * rand(elty, n, p)  # ensure that AX = B is consistent
            dA = CuArray(A)
            dX = CuArray(X)
            dB = CuArray(B)
            CUSOLVER.gels!(dX, dA, dB; irs_precision=irs_precision)
            tol = real(elty) |> eps |> sqrt
            dR = dB - dA * dX
        end
    end

    @testset "geqrf! -- orgqr!" begin
        A = rand(elty, m, n)
        dA = CuArray(A)
        dA, τ = CUSOLVER.geqrf!(dA)
        CUSOLVER.orgqr!(dA, τ)
        @test dA' * dA ≈ I
        dB = CuArray(A)
        dB, τ_b = LAPACK.geqrf!(dB)
        LAPACK.orgqr!(dB, τ_b)
        @test dB ≈ dA
        @test τ ≈ τ_b
        dB, τ_b = LAPACK.geqrf!(dB, similar(τ_b))
        LAPACK.orgqr!(dB, τ_b)
        @test dB ≈ dA
    end

    @testset "ormqr!" begin
        @testset "side = $side" for side in ['L', 'R']
            @testset "trans = $trans" for (trans, op) in [('N', identity), ('T', transpose), ('C', adjoint)]
                (elty <: Complex) && (trans == 'T') && continue
                A = rand(elty, m, n)
                dA = CuArray(A)
                dA, dτ = CUSOLVER.geqrf!(dA)

                hI = Matrix{elty}(I, m, m)
                dI = CuArray(hI)
                dH = CUSOLVER.ormqr!(side, 'N', dA, dτ, dI)
                @test dH' * dH ≈ I

                C = side == 'L' ? rand(elty, m, n) : rand(elty, n, m)
                dC = CuArray(C)
                dD = side == 'L' ? op(dH) * dC : dC * op(dH)

                CUSOLVER.ormqr!(side, trans, dA, dτ, dC)
                @test dC ≈ dD
            end
        end
    end

    @testset "inv -- unsymmetric" begin
        A = rand(elty,n,n)
        dA = CuArray(A)
        dA⁻¹ = inv(dA)
        dI = dA * dA⁻¹
        @test Array(dI) ≈ I
    end

    @testset "inv -- symmetric" begin
        A = rand(elty,n,n)
        A = A + transpose(A)
        dA = Symmetric(CuArray(A))
        dA⁻¹ = inv(dA)
        dI = dA.data * dA⁻¹
        @test Array(dI) ≈ I
    end

    @testset "inv -- triangular" begin
        for (triangle, uplo, diag) in ((LowerTriangular, 'L', 'N'), (UnitLowerTriangular, 'L', 'U'),
                                       (UpperTriangular, 'U', 'N'), (UnitUpperTriangular, 'U', 'U'))
            A = rand(elty,n,n)
            A = uplo == 'L' ? tril(A) : triu(A)
            A = diag == 'N' ? A : A - Diagonal(A) + I
            dA = triangle(view(CuArray(A), 1:2:n, 1:2:n)) # without this view, we are hitting the CUBLAS method!
            dA⁻¹ = inv(dA)
            dI = CuArray(dA) * CuArray(dA⁻¹)
            @test Array(dI) ≈ I
        end
    end

    @testset "Cholesky (po)" begin
        A    = rand(elty,n,n)
        A    = A*A'+I #posdef
        B    = rand(elty,n,n)
        d_A  = CuArray(A)
        d_B  = CuArray(B)

        pivot = NoPivot()
        d_F  = cholesky(d_A, pivot)
        F    = cholesky(A, pivot)
        @test F.U   ≈ collect(d_F.U)
        @test F\(A'B) ≈ collect(d_F\(d_A'd_B))

        d_F  = cholesky(Hermitian(d_A, :L), pivot)
        F    = cholesky(Hermitian(A, :L), pivot)
        @test F.L   ≈ collect(d_F.L)
        @test F\(A'B) ≈ collect(d_F\(d_A'd_B))

        @test_throws DimensionMismatch LinearAlgebra.LAPACK.potrs!('U',d_A,CUDA.rand(elty,m,m))

        A    = rand(elty,m,n)
        d_A  = CuArray(A)
        @test_throws DimensionMismatch cholesky(d_A)
        @test_throws DimensionMismatch LinearAlgebra.LAPACK.potrs!('U',d_A,d_B)

        A    = zeros(elty,n,n)
        d_A  = CuArray(A)
        @test_throws LinearAlgebra.PosDefException cholesky(d_A)
    end

    @testset "Cholesky inverse (potri)" begin
        # test lower
        A    = rand(elty,n,n)
        A    = A*A'+I #posdef
        d_A  = CuArray(A)

        LinearAlgebra.LAPACK.potrf!('L', A)
        LinearAlgebra.LAPACK.potrf!('L', d_A)

        LinearAlgebra.LAPACK.potri!('L', A)
        LinearAlgebra.LAPACK.potri!('L', d_A)
        @test A  ≈ collect(d_A)

        # test upper
        A    = rand(elty,n,n)
        A    = A*A'+I #posdef
        d_A  = CuArray(A)

        LinearAlgebra.LAPACK.potrf!('U', A)
        LinearAlgebra.LAPACK.potrf!('U', d_A)
        LinearAlgebra.LAPACK.potri!('U', A)
        LinearAlgebra.LAPACK.potri!('U', d_A)
        @test A  ≈ collect(d_A)
    end

    @testset "getrf!" begin
        A = rand(elty,m,n)
        d_A = CuArray(A)
        d_A,d_ipiv,info = CUSOLVER.getrf!(d_A)
        LinearAlgebra.checknonsingular(info)
        h_A = collect(d_A)
        h_ipiv = collect(d_ipiv)
        alu = LinearAlgebra.LU(h_A, convert(Vector{BlasInt},h_ipiv), zero(BlasInt))
        @test A ≈ Array(alu)
        d_B = CuArray(A)
        d_B,d_ipiv,info = LAPACK.getrf!(d_B)
        @test d_B ≈ d_A
        d_B = CuArray(A)
        d_B,d_ipiv,info = LAPACK.getrf!(d_B, similar(d_ipiv))
        @test d_B ≈ d_A

        d_A,d_ipiv,info = CUSOLVER.getrf!(CUDA.zeros(elty,n,n))
        @test_throws LinearAlgebra.SingularException LinearAlgebra.checknonsingular(info)
    end

    @testset "getrs!" begin
        A          = rand(elty,n,n)
        d_A        = CuArray(A)
        d_A,d_ipiv = CUSOLVER.getrf!(d_A)
        B          = rand(elty,n,n)
        d_B        = CuArray(B)
        d_B        = CUSOLVER.getrs!('N',d_A,d_ipiv,d_B)
        d_C        = CuArray(B)
        d_C        = LAPACK.getrs!('N',d_A,d_ipiv,d_C)
        @test d_C  ≈ d_B
        h_B        = collect(d_B)
        @test h_B  ≈ A\B
        A          = rand(elty,m,n)
        d_A        = CuArray(A)
        @test_throws DimensionMismatch CUSOLVER.getrs!('N',d_A,d_ipiv,d_B)
        A          = rand(elty,n,n)
        d_A        = CuArray(A)
        B          = rand(elty,m,n)
        d_B        = CuArray(B)
        @test_throws DimensionMismatch CUSOLVER.getrs!('N',d_A,d_ipiv,d_B)
    end

    @testset "sytrf!" begin
        A = rand(elty,n,n)
        A = A + A' #symmetric
        d_A = CuArray(A)
        d_B = CuArray(A)
        d_C = CuArray(A)
        d_A,d_ipiv,info = CUSOLVER.sytrf!('U',d_A)
        LinearAlgebra.checknonsingular(info)
        h_A = collect(d_A)
        h_ipiv = collect(d_ipiv)
        A, ipiv = LAPACK.sytrf!('U',A)
        @test ipiv == h_ipiv
        @test A ≈ h_A
        d_B, ipiv_b = LAPACK.sytrf!('U',d_B)
        @test d_B ≈ d_A
        d_C, ipiv_b = LAPACK.sytrf!('U',d_C, similar(ipiv_b))
        @test d_C ≈ d_A
        A    = rand(elty,m,n)
        d_A  = CuArray(A)
        @test_throws DimensionMismatch CUSOLVER.sytrf!('U',d_A)

        d_A,d_ipiv,info = CUSOLVER.sytrf!('U',CUDA.zeros(elty,n,n))
        @test_throws LinearAlgebra.SingularException LinearAlgebra.checknonsingular(info)
    end

    @testset "gebrd!" begin
        A                             = rand(elty,m,n)
        d_A                           = CuArray(A)
        d_B                           = CuArray(A)
        d_A, d_D, d_E, d_TAUQ, d_TAUP = CUSOLVER.gebrd!(d_A)
        h_A                           = collect(d_A)
        h_D                           = collect(d_D)
        h_E                           = collect(d_E)
        h_TAUQ                        = collect(d_TAUQ)
        h_TAUP                        = collect(d_TAUP)
        A,d,e,q,p                     = LAPACK.gebrd!(A)
        @test A ≈ h_A
        @test d ≈ h_D
        @test e[min(m,n)-1] ≈ h_E[min(m,n)-1]
        @test q ≈ h_TAUQ
        @test p ≈ h_TAUP
        d_B, d_D, d_E, d_TAUQ, d_TAUP = LAPACK.gebrd!(d_B)
        @test d_B ≈ d_A
    end

    @testset "gesvd!" begin
        A = rand(elty,m,n)
        d_A = CuMatrix(A)
        U, Σ, Vt = CUSOLVER.gesvd!('A', 'A', d_A)
        @test A ≈ collect(U[:,1:n] * Diagonal(Σ) * Vt)

        for jobu in ('A', 'S', 'N', 'O')
            for jobvt in ('A', 'S', 'N', 'O')
                (jobu == 'A') && (jobvt == 'A') && continue
                (jobu == 'O') && (jobvt == 'O') && continue
                d_A = CuMatrix(A)
                U2, Σ2, Vt2 = CUSOLVER.gesvd!(jobu, jobvt, d_A)
                @test Σ ≈ Σ2
                d_A = CuMatrix(A)
                U2, Σ2, Vt2 = LAPACK.gesvd!(jobu, jobvt, d_A)
                @test Σ ≈ Σ2
            end
        end
    end

    @testset "geev!" begin
        A              = rand(elty,m,m)
        d_A            = CuArray(A)
        local d_W, d_V
        d_W, d_V       = CUSOLVER.Xgeev!('N','V', d_A)
        d_W_b, d_V_b   = LAPACK.geev!('N','V', CuArray(A))
        @test d_W ≈ d_W_b
        @test d_V ≈ d_V_b
        h_W            = collect(d_W)
        h_V            = collect(d_V)
        h_V⁻¹          = inv(h_V)
        Eig            = eigen(A)
        @test Eig.values ≈ h_W
        @test abs.(Eig.vectors*h_V⁻¹) ≈ I
        d_A            = CuArray(A)
        d_W            = CUSOLVER.Xgeev!('N','N', d_A)
        h_W            = collect(d_W)
        @test Eig.values ≈ h_W

        A              = rand(elty,m,m)
        d_A            = CuArray(A)
        Eig            = eigen(A)
        d_eig          = eigen(d_A)
        @test Eig.values ≈ collect(d_eig.values)
        h_V            = collect(d_eig.vectors)
        h_V⁻¹          = inv(h_V)
        @test abs.(Eig.vectors* h_V⁻¹) ≈ I
    end

    @testset "syevd!" begin
        A              = rand(elty,m,m)
        A             += A'
        d_A            = CuArray(A)
        local d_W, d_V
        if( elty <: Complex )
            d_W, d_V   = CUSOLVER.heevd!('V','U', d_A)
            d_W_b, d_V_b  = LAPACK.syev!('V','U', CuArray(A))
            @test d_W ≈ d_W_b
            @test d_V ≈ d_V_b
            d_W_c, d_V_c  = LAPACK.syevd!('V','U', CuArray(A))
            @test d_W ≈ d_W_c
            @test d_V ≈ d_V_c
        else
            d_W, d_V   = CUSOLVER.syevd!('V','U', d_A)
            d_W_b, d_V_b  = LAPACK.syev!('V','U', CuArray(A))
            @test d_W ≈ d_W_b
            @test d_V ≈ d_V_b
            d_W_c, d_V_c  = LAPACK.syevd!('V','U', CuArray(A))
            @test d_W ≈ d_W_c
            @test d_V ≈ d_V_c
        end
        h_W            = collect(d_W)
        h_V            = collect(d_V)
        Eig            = eigen(A)
        @test Eig.values ≈ h_W
        @test abs.(Eig.vectors'*h_V) ≈ I
        d_A            = CuArray(A)
        if( elty <: Complex )
            d_W   = CUSOLVER.heevd!('N','U', d_A)
        else
            d_W   = CUSOLVER.syevd!('N','U', d_A)
        end
        h_W            = collect(d_W)
        @test Eig.values ≈ h_W

        A              = rand(elty,m,m)
        A             += A'
        d_A            = CuArray(A)
        Eig            = eigen(LinearAlgebra.Hermitian(A))
        d_eig          = eigen(d_A)
        @test Eig.values ≈ collect(d_eig.values)
        d_eig          = eigen(LinearAlgebra.Hermitian(d_A))
        @test Eig.values ≈ collect(d_eig.values)
        h_V            = collect(d_eig.vectors)
        @test abs.(Eig.vectors'*h_V) ≈ I
        if elty <: Real
            Eig            = eigen(LinearAlgebra.Symmetric(A))
            d_eig          = eigen(LinearAlgebra.Symmetric(d_A))
            @test Eig.values ≈ collect(d_eig.values)
            h_V            = collect(d_eig.vectors)
            @test abs.(Eig.vectors'*h_V) ≈ I
        end

    end

    @testset "sygvd!" begin
        A              = rand(elty,m,m)
        B              = rand(elty,m,m)
        A              = A*A'+I # posdef
        B              = B*B'+I # posdef
        d_A            = CuArray(A)
        d_B            = CuArray(B)
        local d_W, d_VA, d_VB
        if( elty <: Complex )
            d_W, d_VA, d_VB = CUSOLVER.hegvd!(1, 'V','U', d_A, d_B)
            d_W2, d_VA2, d_VB2 = LAPACK.sygvd!(1, 'V','U', CuArray(A), CuArray(B))
            @test d_W2 ≈ d_W
            @test d_VA2 ≈ d_VA
            @test d_VB2 ≈ d_VB
        else
            d_W, d_VA, d_VB = CUSOLVER.sygvd!(1, 'V','U', d_A, d_B)
            d_W2, d_VA2, d_VB2 = LAPACK.sygvd!(1, 'V','U', CuArray(A), CuArray(B))
            @test d_W2 ≈ d_W
            @test d_VA2 ≈ d_VA
            @test d_VB2 ≈ d_VB
        end
        h_W            = collect(d_W)
        h_VA           = collect(d_VA)
        h_VB           = collect(d_VB)
        Eig            = eigen(Hermitian(A), Hermitian(B))
        @test Eig.values ≈ h_W
        @test A*h_VA ≈ B*h_VA*Diagonal(h_W) rtol=1e-4
        # test normalization condition for eigtype 1
        @test abs.(h_VA'B*h_VA) ≈ Matrix(one(elty)*I, m, m)
        d_A            = CuArray(A)
        d_B            = CuArray(B)
        if( elty <: Complex )
            d_W   = CUSOLVER.hegvd!(1, 'N','U', d_A, d_B)
        else
            d_W   = CUSOLVER.sygvd!(1, 'N','U', d_A, d_B)
        end
        h_W            = collect(d_W)
        @test Eig.values ≈ h_W
        d_B            = CUDA.rand(elty, m+1, m+1)
        if( elty <: Complex )
            @test_throws DimensionMismatch CUSOLVER.hegvd!(1, 'N','U', d_A, d_B)
        else
            @test_throws DimensionMismatch CUSOLVER.sygvd!(1, 'N','U', d_A, d_B)
        end
    end

    @testset "syevj!" begin
        A              = rand(elty,m,m)
        B              = rand(elty,m,m)
        A              = A*A'+I # posdef
        B              = B*B'+I # posdef
        d_A            = CuArray(A)
        d_B            = CuArray(B)
        local d_W, d_VA, d_VB
        if( elty <: Complex )
            d_W, d_VA, d_VB = CUSOLVER.hegvj!(1, 'V','U', d_A, d_B)
        else
            d_W, d_VA, d_VB = CUSOLVER.sygvj!(1, 'V','U', d_A, d_B)
        end
        h_W            = collect(d_W)
        h_VA           = collect(d_VA)
        h_VB           = collect(d_VB)
        Eig            = eigen(Hermitian(A), Hermitian(B))
        @test Eig.values ≈ h_W
        @test A*h_VA ≈ B*h_VA*Diagonal(h_W) rtol=1e-4
        # test normalization condition for eigtype 1
        @test abs.(h_VA'B*h_VA) ≈ Matrix(one(elty)*I, m, m)
        d_A            = CuArray(A)
        d_B            = CuArray(B)
        if( elty <: Complex )
            d_W   = CUSOLVER.hegvj!(1, 'N','U', d_A, d_B)
        else
            d_W   = CUSOLVER.sygvj!(1, 'N','U', d_A, d_B)
        end
        h_W            = collect(d_W)
        @test Eig.values ≈ h_W
    end

    @testset "elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        @testset "syevjBatched!" begin
            # Generate a random symmetric/hermitian matrix
            A = rand(elty, m,m,n)
            A += permutedims(A, (2,1,3))
            d_A = CuArray(A)

            # Run the solver
            local d_W, d_V
            if( elty <: Complex )
                d_W, d_V   = CUSOLVER.heevjBatched!('V','U', d_A)
            else
                d_W, d_V   = CUSOLVER.syevjBatched!('V','U', d_A)
            end

            # Pull it back to hardware
            h_W   = collect(d_W)
            h_V   = collect(d_V)

            # Use non-GPU blas to estimate the eigenvalues as well
            for i = 1:n
                # Get our eigenvalues
                Eig = eigen(LinearAlgebra.Hermitian(A[:,:,i]))

                # Compare to the actual ones
                @test Eig.values ≈ h_W[:,i]
                @test abs.(Eig.vectors'*h_V[:,:,i]) ≈ I
            end

            # Do it all again, but with the option to not compute eigenvectors
            d_A = CuArray(A)

            # Run the solver
            local d_W
            if( elty <: Complex )
                d_W   = CUSOLVER.heevjBatched!('N','U', d_A)
            else
                d_W   = CUSOLVER.syevjBatched!('N','U', d_A)
            end

            # Pull it back to hardware
            h_W   = collect(d_W)

            # Use non-GPU blas to estimate the eigenvalues as well
            for i = 1:n
                # Get the reference results
                Eig = eigen(LinearAlgebra.Hermitian(A[:,:,i]))

                # Compare to the actual ones
                @test Eig.values ≈ h_W[:,i]
            end
        end
    end

    @testset "$svd_f with $alg algorithm" for
        svd_f in (svd, svd!),
        alg in (CUSOLVER.QRAlgorithm(), CUSOLVER.JacobiAlgorithm()),
        (_m, _n) in ((m, n), (n, m))

        A              = rand(elty, _m, _n)
        U, S, V        = svd(A; full=true)
        d_A            = CuArray(A)

        if _m > _n || alg == CUSOLVER.JacobiAlgorithm()
            d_U, d_S, d_V  = svd_f(copy(d_A); full=true, alg=alg)
            h_S            = collect(d_S)
            h_U            = collect(d_U)
            h_V            = collect(d_V)
            @test abs.(h_U'h_U) ≈ I
            @test abs.(h_U[:,1:min(_m,_n)]'U[:,1:min(_m,_n)]) ≈ I
            @test collect(svdvals(d_A; alg=alg)) ≈ svdvals(A)
            @test svdvals!(copy(d_A); alg=alg) == svdvals(d_A; alg=alg)
            @test abs.(h_V'*h_V) ≈ I
            @test abs.(h_V[:,1:min(_m,_n)]'*V[:,1:min(_m,_n)]) ≈ I
            @test collect(d_U'*d_A*d_V) ≈ U'*A*V
            @test collect(svd(d_A; alg=alg).V') == h_V[:,1:min(_m,_n)]'
        else
            @test_throws ArgumentError svd(d_A; alg=alg)
        end
    end
    # Check that constant propagation works
    _svd(A) = svd(A; alg=CUSOLVER.QRAlgorithm())
    @inferred _svd(CUDA.rand(Float32, 4, 4))

    @testset "batched $svd_f with $alg algorithm" for
        svd_f in (svd, svd!),
        alg in (CUSOLVER.JacobiAlgorithm(), CUSOLVER.ApproximateAlgorithm()),
        (_m, _n, _b) in ((m, n, n), (n, m, n), (33,33,1))

        A              = rand(elty, _m, _n, _b)
        d_A            = CuArray(A)
        r = min(_m, _n)

        if (_m >= _n && alg == CUSOLVER.ApproximateAlgorithm()) || (_m <= 32 && _n <= 32 && alg == CUSOLVER.JacobiAlgorithm())
            d_U, d_S, d_V  = svd_f(copy(d_A); full=true, alg=alg)
            h_S            = collect(d_S)
            h_U            = collect(d_U)
            h_V            = collect(d_V)
            for i=1:_b
                U, S, V = svd(A[:,:,i]; full=true)
                @test abs.(h_U[:,:,i]'*h_U[:,:,i]) ≈ I
                @test abs.(h_U[:,1:min(_m,_n),i]'U[:,1:min(_m,_n)]) ≈ I
                @test collect(svdvals(d_A; alg=alg))[:,i] ≈ svdvals(A[:,:,i])
                @test abs.(h_V[:,:,i]'*h_V[:,:,i]) ≈ I
                @test collect(d_U[:,:,i]'*d_A[:,:,i]*d_V[:,:,i])[1:r,1:r] ≈ (U'*A[:,:,i]*V)[1:r,1:r]
            end
        else
            @test_throws ArgumentError svd(d_A; alg=alg)
        end
    end

    @testset "2-opnorm($sz x $elty)" for sz in [(2, 0), (2, 3)]
        A = rand(elty, sz)
        d_A = CuArray(A)
        @test opnorm(A, 2) ≈ opnorm(d_A, 2)
    end

    @testset "qr" begin
        tol = min(m, n)*eps(real(elty))*(1 + (elty <: Complex))

        A              = rand(elty, m, n)
        F              = qr(A)

        d_A            = CuArray(A)
        d_F            = qr(d_A)

        d_RR           = d_F.Q'*d_A
        @test collect(d_RR[1:n,:]) ≈ collect(d_F.R) atol=tol*norm(A)
        @test norm(d_RR[n+1:end,:]) < tol*norm(A)

        d_RRt          = d_A'*d_F.Q
        @test collect(d_RRt[:,1:n]) ≈ collect(d_F.R') atol=tol*norm(A)
        @test norm(d_RRt[:,n+1:end]) < tol*norm(A)

        @test size(d_F) == size(A)
        @test size(d_F.Q) == (m,m)
        @test size(d_F.R) == (n,n)
        @test size(d_RR) == size(d_A)
        @test size(d_RRt) == size(d_A')

        @test CuArray(d_F) ≈ d_A

        d_I = CuMatrix{elty}(I, size(d_F.Q))
        @test det(d_F.Q) ≈ det(collect(d_F.Q * CuMatrix{elty}(I, size(d_F.Q)))) atol=tol*norm(A)
        @test collect((d_F.Q'd_I) * d_F.Q) ≈ collect(d_I)
        @test collect(d_F.Q * (d_I * d_F.Q')) ≈ collect(d_I)

        d_I = CuMatrix{elty}(I, size(d_F.R))
        @test collect(d_F.R * d_I) ≈ collect(d_F.R)
        @test collect(d_I * d_F.R) ≈ collect(d_F.R)

        CUDA.@allowscalar begin
            qval = d_F.Q[1, 1]
            @test qval ≈ F.Q[1, 1]
            qrstr = sprint(show, MIME"text/plain"(), d_F)
            @test qrstr == "$(typeof(d_F))\nQ factor: $(sprint(show, MIME"text/plain"(), d_F.Q))\nR factor:\n$(sprint(show, MIME"text/plain"(), d_F.R))"
        end

        Q, R = F
        dQ, dR = d_F
        @test collect(dQ*dR) ≈ A
        @test collect(dR * dQ') ≈ (R * Q')
        A              = rand(elty, n, m)
        d_A            = CuArray(A)
        d_F            = qr(d_A)
        @test det(d_F.Q) ≈ det(collect(d_F.Q * CuMatrix{elty}(I, size(d_F.Q)))) atol=tol*norm(A)
        A              = rand(elty, m, n)
        d_A            = CuArray(A)
        d_q, d_r       = qr(d_A)
        q, r           = qr(A)
        @test Array(d_q) ≈ Array(q)
        @test collect(CuArray{elty}(d_q)) ≈ Array(q)
        @test collect(CuArray(d_q)) ≈ Array(q)
        @test Array(d_r) ≈ Array(r)
        @test CuArray(d_q) ≈ convert(typeof(d_A), d_q)
        A              = rand(elty, n, m)
        d_A            = CuArray(A)
        d_q, d_r       = qr(d_A)
        q, r           = qr(A)
        @test Array(d_q) ≈ Array(q)
        @test Array(d_r) ≈ Array(r)
        A              = rand(elty, n)  # A and B are vectors
        d_A            = CuArray(A)
        M              = qr(A)
        d_M            = qr(d_A)
        B              = rand(elty, n)
        d_B            = CuArray(B)
        @test collect(d_M \ d_B) ≈ M \ B
        @test_throws DimensionMismatch("arguments must have the same number of rows") d_M \ CUDA.ones(elty, n+1)
        A              = rand(elty, m, n)  # A is a matrix and B,C is a vector
        d_A            = CuArray(A)
        M              = qr(A)
        d_M            = qr(d_A)
        B              = rand(elty, m)
        d_B            = CuArray(B)
        C              = rand(elty, n)
        d_C            = CuArray(C)
        @test collect(d_M \ d_B) ≈ M \ B
        @test collect(d_M.Q * d_B) ≈ (M.Q * B)
        @test collect(d_M.Q' * d_B) ≈ (M.Q' * B)
        @test collect(d_B' * d_M.Q) ≈ (B' * M.Q)
        @test collect(d_B' * d_M.Q') ≈ (B' * M.Q')
        @test collect(d_M.R * d_C) ≈ (M.R * C)
        @test collect(d_M.R' * d_C) ≈ (M.R' * C)
        @test collect(d_C' * d_M.R) ≈ (C' * M.R)
        @test collect(d_C' * d_M.R') ≈ (C' * M.R')
        A              = rand(elty, m, n)  # A and B,C are matrices
        d_A            = CuArray(A)
        M              = qr(A)
        d_M            = qr(d_A)
        B              = rand(elty, m, l) #different second dimension to verify whether dimensions agree
        d_B            = CuArray(B)
        C              = rand(elty, n, l) #different second dimension to verify whether dimensions agree
        d_C            = CuArray(C)
        @test collect(d_M \ d_B) ≈ (M \ B)
        @test collect(d_M.Q * d_B) ≈ (M.Q * B)
        @test collect(d_M.Q' * d_B) ≈ (M.Q' * B)
        @test collect(d_B' * d_M.Q) ≈ (B' * M.Q)
        @test collect(d_B' * d_M.Q') ≈ (B' * M.Q')
        @test collect(d_M.R * d_C) ≈ (M.R * C)
        @test collect(d_M.R' * d_C) ≈ (M.R' * C)
        @test collect(d_C' * d_M.R) ≈ (C' * M.R)
        @test collect(d_C' * d_M.R') ≈ (C' * M.R')
    end

    @testset "potrsBatched!" begin
        @testset "elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            # Test lower
            bA = [rand(elty, m, m) for i in 1:n]
            bA = [bA[i]*bA[i]' for i in 1:n]
            bB = [rand(elty, m) for i in 1:n]

            # move to device
            bd_A = CuArray{elty, 2}[]
            bd_B = CuArray{elty, 1}[]
            for i in 1:length(bA)
                push!(bd_A, CuArray(bA[i]))
                push!(bd_B, CuArray(bB[i]))
            end

            bd_X = CUSOLVER.potrsBatched!('L', bd_A, bd_B)
            bh_X = [collect(bd_X[i]) for i in 1:n]

            for i = 1:n
                LinearAlgebra.LAPACK.potrs!('L', bA[i], bB[i])
                @test bB[i] ≈ bh_X[i]
            end

            # Test upper
            bA = [rand(elty, m, m) for i in 1:n]
            bA = [bA[i]*bA[i]' for i in 1:n]
            bB = [rand(elty, m) for i in 1:n]

            # move to device
            bd_A = CuArray{elty, 2}[]
            bd_B = CuArray{elty, 1}[]
            for i in 1:length(bA)
                push!(bd_A, CuArray(bA[i]))
                push!(bd_B, CuArray(bB[i]))
            end

            bd_X = CUSOLVER.potrsBatched!('U', bd_A, bd_B)
            bh_X = [collect(bd_X[i]) for i in 1:n]

            for i = 1:n
                LinearAlgebra.LAPACK.potrs!('U', bA[i], bB[i])
                @test bB[i] ≈ bh_X[i]
            end
            # error throwing tests
            bA = [rand(elty, m, m) for i in 1:n]
            bA = [bA[i]*bA[i]' for i in 1:n]
            bB = [rand(elty, m) for i in 1:n+1]

            # move to device
            bd_A = CuArray{elty, 2}[]
            bd_B = CuArray{elty, 1}[]
            for i in 1:length(bA)
                push!(bd_A, CuArray(bA[i]))
                push!(bd_B, CuArray(bB[i]))
            end
            push!(bd_B, CuArray(bB[end]))

            @test_throws DimensionMismatch CUSOLVER.potrsBatched!('L', bd_A, bd_B)

            bA = [rand(elty, m, m) for i in 1:n]
            bA = [bA[i]*bA[i]' for i in 1:n]
            bB = [rand(elty, m) for i in 1:n]
            bB[1] = rand(elty, m+1)
            # move to device
            bd_A = CuArray{elty, 2}[]
            bd_B = CuArray{elty, 1}[]
            for i in 1:length(bA)
                push!(bd_A, CuArray(bA[i]))
                push!(bd_B, CuArray(bB[i]))
            end

            @test_throws DimensionMismatch CUSOLVER.potrsBatched!('L', bd_A, bd_B)

            bA = [rand(elty, m, m) for i in 1:n]
            bA = [bA[i]*bA[i]' for i in 1:n]
            bB = [rand(elty, m, m) for i in 1:n]
            # move to device
            bd_A = CuArray{elty, 2}[]
            bd_B = CuArray{elty, 2}[]
            for i in 1:length(bA)
                push!(bd_A, CuArray(bA[i]))
                push!(bd_B, CuArray(bB[i]))
            end

            @test_throws ArgumentError CUSOLVER.potrsBatched!('L', bd_A, bd_B)
        end
    end

    @testset "potrfBatched!" begin
        @testset "elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            # Test lower
            bA = [rand(elty, m, m) for i in 1:n]
            bA = [bA[i]*bA[i]' for i in 1:n]

            # move to device
            bd_A = CuArray{elty, 2}[]
            for i in 1:length(bA)
                push!(bd_A, CuArray(bA[i]))
            end

            bd_A, info = CUSOLVER.potrfBatched!('L', bd_A)
            bh_A = [collect(bd_A[i]) for i in 1:n]

            for i = 1:n
                LinearAlgebra.LAPACK.potrf!('L', bA[i])
                @test bA[i] ≈ bh_A[i]
            end

            # Test upper
            bA = [rand(elty, m, m) for i in 1:n]
            bA = [bA[i]*bA[i]' for i in 1:n]

            # move to device
            bd_A = CuArray{elty, 2}[]
            for i in 1:length(bA)
                push!(bd_A, CuArray(bA[i]))
            end

            bd_A, info = CUSOLVER.potrfBatched!('U', bd_A)
            bh_A = [collect(bd_A[i]) for i in 1:n]

            for i = 1:n
                LinearAlgebra.LAPACK.potrf!('U', bA[i])
                # cuSOLVER seems to return symmetric/hermitian matrix when using 'U'
                @test Hermitian(bA[i]) ≈ bh_A[i]
            end
        end
    end

    @testset "ldiv!" begin
        @testset "elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            @test testf(rand(elty, m, m), rand(elty, m)) do A, x
                ldiv!(qr(A), x)
                x
            end

            @test testf(rand(elty, m, m), rand(elty, m), rand(elty, m)) do A, x, y
                ldiv!(y, qr(A), x)
                y
            end
        end
    end

    @testset "lu" begin
        @testset "elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            A = CuArray(rand(elty, m, m))
            F = lu(A)
            @test F.L*F.U ≈ A[F.p, :]

            @test_throws LinearAlgebra.SingularException lu(CUDA.zeros(elty,n,n))
        end
        @testset "lu ldiv! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            A = rand(elty, m, m)
            B = rand(elty, m, m)
            A_d = CuArray(A)
            B_d = CuArray(B)
            lu_cpu = lu(A)
            lu_gpu = lu(A_d)
            @test ldiv!(lu_cpu, B) ≈ collect(ldiv!(lu_gpu, B_d))
        end

        A = CuMatrix(rand(1024, 1024))
        lua = lu(A)
        @test Matrix(lua.L) * Matrix(lua.U) ≈ Matrix(lua.P) * Matrix(A)

        A = CuMatrix(rand(1024, 512))
        lua = lu(A)
        @test Matrix(lua.L) * Matrix(lua.U) ≈ Matrix(lua.P) * Matrix(A)

        A = CuMatrix(rand(512, 1024))
        lua = lu(A)
        @test Matrix(lua.L) * Matrix(lua.U) ≈ Matrix(lua.P) * Matrix(A)
    end
end

@testset "Promotion from elty = $elty" for elty in [Float16, ComplexF16, Int32, Int64, Complex{Int32}, Complex{Int64}]

    @testset "svd with $alg algorithm" for
        alg in (CUSOLVER.QRAlgorithm(), CUSOLVER.JacobiAlgorithm()),
        (_m, _n) in ((m, n), (n, m))

        d_A = CuArray(rand(elty, _m, _n))
        d_Af = promote_type(Float32, elty).(d_A)

        if _m > _n || alg == CUSOLVER.JacobiAlgorithm()
            @test svd(d_A; alg=alg) == svd(d_Af; alg=alg)
            @test svdvals(d_A; alg=alg) == svdvals(d_Af; alg=alg)
        else
            @test_throws ArgumentError svd(d_A; alg=alg)
            @test_throws ArgumentError svdvals(d_A; alg=alg)
        end
    end

end

@testset "Matrix division $elty1 \\ $elty2" for elty1 in [
    Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64, Int32, Int64, Complex{Int32}, Complex{Int64}
], elty2 in [
    Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64, Int32, Int64, Complex{Int32}, Complex{Int64}
]
    @testset "Symmetric linear systems" begin
        A = rand(elty1,n,n)
        A = A + transpose(A)
        B = rand(elty2,n,5)
        b = rand(elty2,n)
        d_A = CuArray(A)
        d_B = CuArray(B)
        d_b = CuArray(b)
        cublasfloat = promote_type(Float32, promote_type(elty1, elty2))
        Af = Symmetric(cublasfloat.(A))
        Bf = cublasfloat.(B)
        bf = cublasfloat.(b)
        @test Array(d_A \ d_B) ≈ (Af \ Bf)
        @test Array(Symmetric(d_A) \ d_B) ≈ (Af \ Bf)
        @test Array(d_A \ d_b) ≈ (Af \ bf)
        @test Array(Symmetric(d_A) \ d_b) ≈ (Af \ bf)
        @inferred d_A \ d_B
        @inferred d_A \ d_b
    end

    @testset "Square and unsymmetric linear systems" begin
        A = rand(elty1,n,n)
        B = rand(elty2,n,5)
        b = rand(elty2,n)
        d_A = CuArray(A)
        d_B = CuArray(B)
        d_b = CuArray(b)
        cublasfloat = promote_type(Float32, promote_type(elty1, elty2))
        Af = cublasfloat.(A)
        Bf = cublasfloat.(B)
        bf = cublasfloat.(b)
        @test Array(d_A \ d_B) ≈ (Af \ Bf)
        @test Array(d_A \ d_b) ≈ (Af \ bf)
        @inferred d_A \ d_B
        @inferred d_A \ d_b
    end

    @testset "Overdetermined linear systems" begin
        A = rand(elty1,m,n)
        B = rand(elty2,m,5)
        b = rand(elty2,m)
        d_A = CuArray(A)
        d_B = CuArray(B)
        d_b = CuArray(b)
        cublasfloat = promote_type(Float32, promote_type(elty1, elty2))
        Af = cublasfloat.(A)
        Bf = cublasfloat.(B)
        bf = cublasfloat.(b)
        @test Array(d_A \ d_B) ≈ (Af \ Bf)
        @test Array(d_A \ d_b) ≈ (Af \ bf)
        @inferred d_A \ d_B
        @inferred d_A \ d_b
    end

    @testset "Underdetermined linear systems" begin
        A = rand(elty1,n,m)
        B = rand(elty2,n,5)
        b = rand(elty2,n)
        d_A = CuArray(A)
        d_B = CuArray(B)
        d_b = CuArray(b)
        cublasfloat = promote_type(Float32, promote_type(elty1, elty2))
        Af = cublasfloat.(A)
        Bf = cublasfloat.(B)
        bf = cublasfloat.(b)
        @test Array(d_A \ d_B) ≈ (Af \ Bf)
        @test Array(d_A \ d_b) ≈ (Af \ bf)
        @inferred d_A \ d_B
        @inferred d_A \ d_b
    end
end
