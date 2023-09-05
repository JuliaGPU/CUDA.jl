using CUDA.CUSOLVER

using LinearAlgebra
using LinearAlgebra: BlasInt

m = 15
n = 10
l = 13
k = 1

m_sub_length = 7
n_sub_length = 3
l_sub_length= 11
m_sub_start = 4
n_sub_start = 2
l_sub_start = 1
m_subrange = (1:m_sub_length) .+ (m_sub_start-1)
n_subrange = (1:n_sub_length) .+ (n_sub_start -1)
l_subrange = (1:l_sub_length) .+ (l_sub_start -1)

m_large=50
n_large=30
l_large=20
m_range = (1:m) .+ (m_sub_start-1)
n_range = (1:n) .+ (n_sub_start -1)
l_range = (1:l) .+ (l_sub_start -1)

@testset "elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
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

    CUDA.CUSOLVER.version() >= v"10.1" && @testset "Cholesky inverse (potri)" begin
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
        d_A,d_ipiv,info = CUSOLVER.sytrf!('U',d_A)
        LinearAlgebra.checknonsingular(info)
        h_A = collect(d_A)
        h_ipiv = collect(d_ipiv)
        A, ipiv = LAPACK.sytrf!('U',A)
        @test ipiv == h_ipiv
        @test A ≈ h_A
        A    = rand(elty,m,n)
        d_A  = CuArray(A)
        @test_throws DimensionMismatch CUSOLVER.sytrf!('U',d_A)

        d_A,d_ipiv,info = CUSOLVER.sytrf!('U',CUDA.zeros(elty,n,n))
        @test_throws LinearAlgebra.SingularException LinearAlgebra.checknonsingular(info)
    end

    @testset "gebrd!" begin
        A                             = rand(elty,m,n)
        d_A                           = CuArray(A)
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
    end

    @testset "syevd!" begin
        A              = rand(elty,m,m)
        A             += A'
        d_A            = CuArray(A)
        local d_W, d_V
        if( elty <: Complex )
            d_W, d_V   = CUSOLVER.heevd!('V','U', d_A)
        else
            d_W, d_V   = CUSOLVER.syevd!('V','U', d_A)
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
        else
            d_W, d_VA, d_VB = CUSOLVER.sygvd!(1, 'V','U', d_A, d_B)
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

        d_I = CuMatrix{elty}(I, size(d_F.Q))
        @test det(d_F.Q) ≈ det(collect(d_F.Q * CuMatrix{elty}(I, size(d_F.Q)))) atol=tol*norm(A)
        if VERSION >= v"1.10-"
            @test collect((d_F.Q'd_I) * d_F.Q) ≈ collect(d_I)
            @test collect(d_F.Q * (d_I * d_F.Q')) ≈ collect(d_I)
        else
            @test collect(d_F.Q * d_I) ≈ collect(d_F.Q)
            @test collect(d_I * d_F.Q) ≈ collect(d_F.Q)
        end

        d_I = CuMatrix{elty}(I, size(d_F.R))
        @test collect(d_F.R * d_I) ≈ collect(d_F.R)
        @test collect(d_I * d_F.R) ≈ collect(d_F.R)

        CUDA.@allowscalar begin
            qval = d_F.Q[1, 1]
            @test qval ≈ F.Q[1, 1]
            qrstr = sprint(show, MIME"text/plain"(), d_F)
            if VERSION >= v"1.10-"
                @test qrstr == "$(typeof(d_F))\nQ factor: $(sprint(show, MIME"text/plain"(), d_F.Q))\nR factor:\n$(sprint(show, MIME"text/plain"(), d_F.R))"
            else
                @test qrstr == "$(typeof(d_F))\nQ factor:\n$(sprint(show, MIME"text/plain"(), d_F.Q))\nR factor:\n$(sprint(show, MIME"text/plain"(), d_F.R))"
            end
        end

        A_view         = view(A, m_subrange, n_subrange)
        F              = qr(A_view)

        d_A_view       = view(d_A, m_subrange, n_subrange)
        d_F            = qr(d_A_view)
        
        d_RR           = d_F.Q'*d_A_view
        @test collect(d_RR[1:n_sub_length,:]) ≈ collect(d_F.R) atol=tol*norm(A_view)
        @test norm(d_RR[n_sub_length+1:end,:]) < tol*norm(A_view)

        d_RRt          = d_A_view'*d_F.Q
        @test collect(d_RRt[:,1:n_sub_length]) ≈ collect(d_F.R') atol=tol*norm(A_view)
        @test norm(d_RRt[:,n_sub_length+1:end]) < tol*norm(A_view)

        @test size(d_F) == size(A_view)
        @test size(d_F.Q) == (m_sub_length,m_sub_length)
        @test size(d_F.R) == (n_sub_length,n_sub_length)
        @test size(d_RR) == size(d_A_view)
        @test size(d_RRt) == size(d_A_view')

        d_I = CuMatrix{elty}(I, size(d_F.Q))
        @test det(d_F.Q) ≈ det(collect(d_F.Q * CuMatrix{elty}(I, size(d_F.Q)))) atol=tol*norm(A_view)
        @test collect(d_F.Q * d_I) ≈ collect(d_F.Q)
        @test collect(d_I * d_F.Q) ≈ collect(d_F.Q)

        d_I = CuMatrix{elty}(I, size(d_F.R))
        @test collect(d_F.R * d_I) ≈ collect(d_F.R)
        @test collect(d_I * d_F.R) ≈ collect(d_F.R)

        CUDA.@allowscalar begin
            qval = d_F.Q[1, 1]
            @test qval ≈ F.Q[1, 1]
            qrstr = sprint(show, MIME"text/plain"(), d_F)
            if VERSION >= v"1.8-"
                @test qrstr == "$(typeof(d_F))\nQ factor:\n$(sprint(show, MIME"text/plain"(), d_F.Q))\nR factor:\n$(sprint(show, MIME"text/plain"(), d_F.R))"
            else
                @test qrstr == "$(typeof(d_F)) with factors Q and R:\n$(sprint(show, d_F.Q))\n$(sprint(show, d_F.R))"
            end
        end

        A              = rand(elty, n, m)
        F              = qr(A)
        d_A            = CuArray(A)
        d_F            = qr(d_A)
        Q, R           = F
        dQ, dR         = d_F
        @test collect(dQ*dR) ≈ A
        @test collect(dR' * dQ') ≈ A'
        @test det(d_F.Q) ≈ det(collect(d_F.Q * CuMatrix{elty}(I, size(d_F.Q)))) atol=tol*norm(A)
        A_view          = view(A, n_subrange, m_subrange)
        F               = qr(A_view)
        d_A_view        = view(d_A, n_subrange, m_subrange)
        d_F             = qr(d_A_view)
        Q, R            = F
        dQ, dR          = d_F
        @test collect(dQ*dR) ≈ A_view
        @test collect(dR' * dQ') ≈ A_view' 
        @test det(d_F.Q) ≈ det(collect(d_F.Q * CuMatrix{elty}(I, size(d_F.Q)))) atol=tol*norm(A_view)

        A              = rand(elty, m, n)
        d_A            = CuArray(A)
        d_q, d_r       = qr(d_A)
        q, r           = qr(A)
        @test Array(d_q) ≈ Array(q)
        @test collect(CuArray(d_q)) ≈ Array(q)
        @test Array(d_r) ≈ Array(r)
        @test CuArray(d_q) ≈ convert(typeof(d_A), d_q)
        A_view         = view(A, m_subrange, n_subrange)
        d_A_view       = view(d_A, m_subrange, n_subrange)
        d_q, d_r       = qr(d_A_view)
        q, r           = qr(A_view)
        @test Array(d_q) ≈ Array(q)
        @test collect(CuArray(d_q)) ≈ Array(q)
        @test Array(d_r) ≈ Array(r)
        @test CuArray(d_q) ≈ convert(typeof(d_A), d_q)

        A              = rand(elty, n, m)
        d_A            = CuArray(A)
        d_q, d_r       = qr(d_A)
        q, r           = qr(A)
        @test Array(d_q) ≈ Array(q)
        @test Array(d_r) ≈ Array(r)
        A_view         = view(A, m_subrange, n_subrange)
        d_A_view       = view(d_A, m_subrange, n_subrange)
        d_q, d_r       = qr(d_A_view)
        q, r           = qr(A_view)
        @test Array(d_q) ≈ Array(q)
        @test Array(d_r) ≈ Array(r)

        
        A              = rand(elty, n, m)
        d_A            = CuArray(A)
        d_q, d_r       = qr!(d_A)
        q, r           = qr!(A)
        @test collect(d_q) ≈ Array(q)
        @test collect(d_r) ≈ Array(r)
        A              = rand(elty, n, m)
        d_A            = CuArray(A)
        A_view         = view(A, m_subrange, n_subrange)
        d_A_view       = view(d_A, m_subrange, n_subrange)
        d_q, d_r       = qr!(d_A_view)
        q, r           = qr!(A_view)
        @test collect(d_q) ≈ collect(q)
        @test collect(d_r) ≈ collect(r)

        A              = rand(elty, n)  # A and B are vectors
        d_A            = CuArray(A)
        M              = qr(A)
        d_M            = qr(d_A)
        B              = rand(elty, n)
        d_B            = CuArray(B)
        @test collect(d_M \ d_B) ≈ M \ B
        A_view         = view(A, n_subrange) 
        d_A_view       = view(d_A, n_subrange)
        M_view         = qr(A_view)
        d_M_view       = qr(d_A_view)
        B_view         = view(B, n_subrange)
        d_B_view       = view(d_B, n_subrange)
        @test collect(d_M_view \ d_B_view) ≈ M_view \ B_view
        B_large        = rand(elty, n_large) 
        B              = view(B_large, n_range)
        d_B_large      = CuArray(B_large)
        d_B            = view(d_B_large, n_range)
        @test collect(d_M \ d_B) ≈ M \ B

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
        A_view         = view(A, m_subrange, n_subrange)
        d_A_view       = view(d_A, m_subrange, n_subrange)
        M_view         = qr(A_view)
        d_M_view       = qr(d_A_view)
        B_view         = view(B, m_subrange)
        d_B_view       = view(d_B, m_subrange)
        C_view         = view(C, n_subrange)
        d_C_view       = view(d_C, n_subrange)
        @test collect(d_M_view \ d_B_view) ≈ M_view \ B_view
        @test collect(d_M_view.Q * d_B_view) ≈ (M_view.Q * B_view)
        @test collect(d_M_view.Q' * d_B_view) ≈ (M_view.Q' * B_view)
        @test collect(d_B_view' * d_M_view.Q) ≈ (B_view' * M_view.Q)
        @test collect(d_B_view' * d_M_view.Q') ≈ (B_view' * M_view.Q')
        @test collect(d_M_view.R * d_C_view) ≈ (M_view.R * C_view)
        @test collect(d_M_view.R' * d_C_view) ≈ (M_view.R' * C_view)
        @test collect(d_C_view' * d_M_view.R) ≈ (C_view' * M_view.R)
        @test collect(d_C_view' * d_M_view.R') ≈ (C_view' * M_view.R')
        B_large        = rand(elty, m_large) 
        B              = view(B_large, m_range)
        d_B_large      = CuArray(B_large)
        d_B            = view(d_B_large, m_range)
        C_large        = rand(elty, n_large) 
        C              = view(C_large, n_range)
        d_C_large      = CuArray(C_large)
        d_C            = view(d_C_large, n_range)
        @test collect(d_M \ d_B) ≈ M \ B
        @test collect(d_M.Q * d_B) ≈ (M.Q * B)
        @test collect(d_M.Q' * d_B) ≈ (M.Q' * B)
        @test collect(d_B' * d_M.Q) ≈ (B' * M.Q)
        @test collect(d_B' * d_M.Q') ≈ (B' * M.Q')
        @test collect(d_M.R * d_C) ≈ (M.R * C)
        @test collect(d_M.R' * d_C) ≈ (M.R' * C)
        @test collect(d_C' * d_M.R) ≈ (C' * M.R)
        @test collect(d_C' * d_M.R') ≈ (C' * M.R')

        A              = rand(elty, m, n)  # A is a matrix and B,C is a vector
        d_A            = CuArray(A)
        M              = qr!(A)
        d_M            = qr!(d_A)
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
        A              = rand(elty, m, n) 
        d_A            = CuArray(A)
        A_view         = view(A, m_subrange, n_subrange)
        d_A_view       = view(d_A, m_subrange, n_subrange)
        M_view         = qr!(A_view)
        d_M_view       = qr!(d_A_view)
        B_view         = view(B, m_subrange)
        d_B_view       = view(d_B, m_subrange)
        C_view         = view(C, n_subrange)
        d_C_view       = view(d_C, n_subrange)
        @test collect(d_M_view \ d_B_view) ≈ M_view \ B_view
        @test collect(d_M_view.Q * d_B_view) ≈ (M_view.Q * B_view)
        @test collect(d_M_view.Q' * d_B_view) ≈ (M_view.Q' * B_view)
        @test collect(d_B_view' * d_M_view.Q) ≈ (B_view' * M_view.Q)
        @test collect(d_B_view' * d_M_view.Q') ≈ (B_view' * M_view.Q')
        @test collect(d_M_view.R * d_C_view) ≈ (M_view.R * C_view)
        @test collect(d_M_view.R' * d_C_view) ≈ (M_view.R' * C_view)
        @test collect(d_C_view' * d_M_view.R) ≈ (C_view' * M_view.R)
        @test collect(d_C_view' * d_M_view.R') ≈ (C_view' * M_view.R')
        B_large        = rand(elty, m_large) 
        B              = view(B_large, m_range)
        d_B_large      = CuArray(B_large)
        d_B            = view(d_B_large, m_range)
        C_large        = rand(elty, n_large) 
        C              = view(C_large, n_range)
        d_C_large      = CuArray(C_large)
        d_C            = view(d_C_large, n_range)
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
        A_view         = view(A, m_subrange, n_subrange)
        d_A_view       = view(d_A, m_subrange, n_subrange)
        M_view         = qr(A_view)
        d_M_view       = qr(d_A_view)
        B_view         = view(B, m_subrange, l_subrange)
        d_B_view       = view(d_B, m_subrange, l_subrange)
        C_view         = view(C, n_subrange, l_subrange)
        d_C_view       = view(d_C, n_subrange, l_subrange)
        @test collect(d_M_view \ d_B_view) ≈ M_view \ B_view
        @test collect(d_M_view.Q * d_B_view) ≈ (M_view.Q * B_view)
        @test collect(d_M_view.Q' * d_B_view) ≈ (M_view.Q' * B_view)
        @test collect(d_B_view' * d_M_view.Q) ≈ (B_view' * M_view.Q)
        @test collect(d_B_view' * d_M_view.Q') ≈ (B_view' * M_view.Q')
        @test collect(d_M_view.R * d_C_view) ≈ (M_view.R * C_view)
        @test collect(d_M_view.R' * d_C_view) ≈ (M_view.R' * C_view)
        @test collect(d_C_view' * d_M_view.R) ≈ (C_view' * M_view.R)
        @test collect(d_C_view' * d_M_view.R') ≈ (C_view' * M_view.R')
        B_large        = rand(elty, m_large, l_large) 
        B              = view(B_large, m_range, l_range)
        d_B_large      = CuArray(B_large)
        d_B            = view(d_B_large, m_range, l_range)
        C_large        = rand(elty, n_large, l_large) 
        C              = view(C_large, n_range, l_range)
        d_C_large      = CuArray(C_large)
        d_C            = view(d_C_large, n_range, l_range)
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
        M              = qr!(A)
        d_M            = qr!(d_A)
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
        A              = rand(elty, m, n)  
        d_A            = CuArray(A)
        A_view         = view(A, m_subrange, n_subrange)
        d_A_view       = view(d_A, m_subrange, n_subrange)
        M_view         = qr!(A_view)
        d_M_view       = qr!(d_A_view)
        B_view         = view(B, m_subrange, l_subrange)
        d_B_view       = view(d_B, m_subrange, l_subrange)
        C_view         = view(C, n_subrange, l_subrange)
        d_C_view       = view(d_C, n_subrange, l_subrange)
        @test collect(d_M_view \ d_B_view) ≈ M_view \ B_view
        @test collect(d_M_view.Q * d_B_view) ≈ (M_view.Q * B_view)
        @test collect(d_M_view.Q' * d_B_view) ≈ (M_view.Q' * B_view)
        @test collect(d_B_view' * d_M_view.Q) ≈ (B_view' * M_view.Q)
        @test collect(d_B_view' * d_M_view.Q') ≈ (B_view' * M_view.Q')
        @test collect(d_M_view.R * d_C_view) ≈ (M_view.R * C_view)
        @test collect(d_M_view.R' * d_C_view) ≈ (M_view.R' * C_view)
        @test collect(d_C_view' * d_M_view.R) ≈ (C_view' * M_view.R)
        @test collect(d_C_view' * d_M_view.R') ≈ (C_view' * M_view.R')
        B_large        = rand(elty, m_large, l_large) 
        B              = view(B_large, m_range, l_range)
        d_B_large      = CuArray(B_large)
        d_B            = view(d_B_large, m_range, l_range)
        C_large        = rand(elty, n_large, l_large) 
        C              = view(C_large, n_range, l_range)
        d_C_large      = CuArray(C_large)
        d_C            = view(d_C_large, n_range, l_range)
        @test collect(d_M \ d_B) ≈ M \ B
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
    @testset "Square linear systems" begin
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
