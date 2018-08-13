using CuArrays
using CuArrays.CUSOLVER
using Test

@testset "cuSolver" begin

m = 15
n = 10
l = 13
k = 1

@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    @testset "potrf!" begin
        A    = rand(elty,n,n)
        A    = A*A' #posdef
        d_A  = CuArray(A)
        d_A  = CUSOLVER.potrf!('U',d_A)
        h_A  = collect(d_A)
        cA,_ = LAPACK.potrf!('U',A)
        @test h_A ≈ cA
        A    = rand(elty,m,n)
        d_A  = CuArray(A)
        @test_throws DimensionMismatch CUSOLVER.potrf!('U',d_A)
        A    = zeros(elty,n,n)
        d_A  = CuArray(A)
        @test_throws LinearAlgebra.SingularException CUSOLVER.potrf!('U',d_A)
    end
    @testset "potrs!" begin
        A     = rand(elty,n,n)
        A     = A*A' #posdef
        d_A   = CuArray(A)
        d_A   = CUSOLVER.potrf!('U',d_A)
        h_A   = collect(d_A)
        B     = rand(elty,n,n)
        d_B   = CuArray(B)
        d_B   = CUSOLVER.potrs!('U',d_A,d_B)
        h_B   = collect(d_B)
        @test h_B ≈ LAPACK.potrs!('U',h_A,B)
        A    = rand(elty,m,n)
        d_A  = CuArray(A)
        @test_throws DimensionMismatch CUSOLVER.potrs!('U',d_A,d_B)
        A    = rand(elty,n,n)
        d_A  = CuArray(A)
        B     = rand(elty,m,m)
        d_B   = CuArray(B)
        @test_throws DimensionMismatch CUSOLVER.potrs!('U',d_A,d_B)
    end

    @testset "getrf!" begin
        A          = rand(elty,m,n)
        d_A        = CuArray(A)
        d_A,d_ipiv = CUSOLVER.getrf!(d_A)
        h_A        = collect(d_A)
        h_ipiv     = collect(d_ipiv)
        alu        = LinearAlgebra.LU(h_A, convert(Vector{Int},h_ipiv), zero(Int))
        @test A ≈ Array(alu)
        A    = zeros(elty,n,n)
        d_A  = CuArray(A)
        @test_throws LinearAlgebra.SingularException CUSOLVER.getrf!(d_A)
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

    @testset "geqrf!" begin
        A         = rand(elty,m,n)
        d_A       = CuArray(A)
        d_A,d_tau = CUSOLVER.geqrf!(d_A)
        h_A       = collect(d_A)
        h_tau     = collect(d_tau)
        qra       = LinearAlgebra.QR(h_A, h_tau)
        @test A ≈ Array(qra)
    end

    @testset "ormqr!" begin
        A          = rand(elty, m, n)
        d_A        = CuArray(A)
        d_A, d_tau = CUSOLVER.geqrf!(d_A)
        B          = rand(elty, n, l)
        d_B        = CuArray(B)
        d_B        = CUSOLVER.ormqr!('L', 'N', d_A, d_tau, d_B)
        h_B        = collect(d_B)
        F          = qr!(A)
        @test h_B  ≈ Array(F.Q)*B
        A          = rand(elty, n, m)
        d_A        = CuArray(A)
        d_A, d_tau = CUSOLVER.geqrf!(d_A)
        B          = rand(elty, n, l)
        d_B        = CuArray(B)
        d_B        = CUSOLVER.ormqr!('L', 'N', d_A, d_tau, d_B)
        h_B        = collect(d_B)
        F          = qr!(A)
        @test h_B  ≈ Array(F.Q)*B
        A          = rand(elty, m, n)
        d_A        = CuArray(A)
        d_A, d_tau = CUSOLVER.geqrf!(d_A)
        B          = rand(elty, l, m)
        d_B        = CuArray(B)
        d_B        = CUSOLVER.ormqr!('R', 'N', d_A, d_tau, d_B)
        h_B        = collect(d_B)
        F          = qr!(A)
        @test h_B  ≈ B*Array(F.Q)
        A          = rand(elty, n, m)
        d_A        = CuArray(A)
        d_A, d_tau = CUSOLVER.geqrf!(d_A)
        B          = rand(elty, l, n)
        d_B        = CuArray(B)
        d_B        = CUSOLVER.ormqr!('R', 'N', d_A, d_tau, d_B)
        h_B        = collect(d_B)
        F          = qr!(A)
        @test h_B  ≈ B*Array(F.Q)
    end

    @testset "orgqr!" begin
        A         = rand(elty,n,m)
        d_A       = CuArray(A)
        d_A,d_tau = CUSOLVER.geqrf!(d_A)
        d_Q       = CUSOLVER.orgqr!(d_A, d_tau)
        h_Q       = collect(d_Q)
        F         = qr!(A)
        @test h_Q ≈ Array(F.Q)
        A         = rand(elty,m,n)
        d_A       = CuArray(A)
        d_A,d_tau = CUSOLVER.geqrf!(d_A)
        d_Q       = CUSOLVER.orgqr!(d_A, d_tau)
        h_Q       = collect(d_Q)
        F         = qr!(A)
        @test h_Q ≈ Array(F.Q)
    end

    @testset "sytrf!" begin
        A          = rand(elty,n,n)
        A          = A + A' #symmetric
        d_A        = CuArray(A)
        d_A,d_ipiv = CUSOLVER.sytrf!('U',d_A)
        h_A        = collect(d_A)
        h_ipiv     = collect(d_ipiv)
        A, ipiv    = LAPACK.sytrf!('U',A)
        @test ipiv == h_ipiv
        @test A ≈ h_A
        A    = rand(elty,m,n)
        d_A  = CuArray(A)
        @test_throws DimensionMismatch CUSOLVER.sytrf!('U',d_A)
        A    = zeros(elty,n,n)
        d_A  = CuArray(A)
        @test_throws LinearAlgebra.SingularException CUSOLVER.sytrf!('U',d_A)
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
        #@test A ≈ h_A
        @test d ≈ h_D
        @test e ≈ h_E
        @test q ≈ h_TAUQ
        @test p ≈ h_TAUP
    end

    @testset "gesvd!" begin
        A              = rand(elty,m,n)
        d_A            = CuArray(A)
        d_U, d_S, d_Vt = CUSOLVER.gesvd!('A','A',d_A)
        h_S            = collect(d_S)
        h_U            = collect(d_U)
        h_Vt           = collect(d_Vt)
        svda           = svdfact(A,thin=false)
        @test abs.(h_U'h_U) ≈ eye(elty, m)
        @test abs.(h_U[:,1:n]'svda[:U][:,1:10]) ≈ eye(elty, n)
        @test h_S ≈ svdvals(A)
        @test abs.(h_Vt*svda[:Vt]') ≈ eye(elty, n)
    end

    @testset "qr" begin
        A              = rand(elty, m, n)
        d_A            = CuArray(A)
        F              = qr(d_A)
        @test F.Q'*A ≈ F.R
        A              = rand(elty, n, m)
        d_A            = CuArray(A)
        F              = qr(d_A)
        @test F.Q'*A ≈ F.R
        CuArrays.allowscalar(true)
        A              = rand(elty, m, n)
        d_A            = CuArray(A)
        d_q, d_r       = qr(d_A)
        h_q, h_r       = collect(d_q), collect(d_r)
        q, r           = qr(A)
        @test h_q ≈ q
        @test h_r ≈ r
        A              = rand(elty, n, m)
        d_A            = CuArray(A)
        d_q, d_r       = qr(d_A)
        q, r           = qr(A)
        @test collect(d_q) ≈ collect(q)
        @test collect(d_r) ≈ collect(r)
        CuArrays.allowscalar(false)
    end

end

end
