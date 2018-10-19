@testset "cuSOLVER" begin

using CuArrays.CUSOLVER
using LinearAlgebra

m = 15
n = 10
l = 13
k = 1

@testset "elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    @testset "Cholesky (po)" begin
        A    = rand(elty,n,n)
        A    = A*A' #posdef
        B    = rand(elty,n,n)
        d_A  = CuArray(A)
        d_B  = CuArray(B)

        d_F  = cholesky(d_A, Val(false))
        F    = cholesky(A, Val(false))
        @test F.U   ≈ collect(d_F.U)
        @test F\(A'B) ≈ collect(d_F\(d_A'd_B))

        d_F  = cholesky(Hermitian(d_A, :L), Val(false))
        F    = cholesky(Hermitian(A, :L), Val(false))
        @test F.L   ≈ collect(d_F.L)
        @test F\(A'B) ≈ collect(d_F\(d_A'd_B))

        @test_throws DimensionMismatch LinearAlgebra.LAPACK.potrs!('U',d_A,CuArray(rand(elty,m,m)))

        A    = rand(elty,m,n)
        d_A  = CuArray(A)
        @test_throws DimensionMismatch cholesky(d_A)
        @test_throws DimensionMismatch LinearAlgebra.LAPACK.potrs!('U',d_A,d_B)

        A    = zeros(elty,n,n)
        d_A  = CuArray(A)
        @test_throws LinearAlgebra.PosDefException cholesky(d_A)
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
        F              = svd(A, full=true)
        @test abs.(h_U'h_U) ≈ Matrix(one(elty)*I, m, m)
        @test abs.(h_U[:,1:n]'F.U[:,1:n]) ≈ Matrix(one(elty)*I, n, n)
        @test h_S ≈ svdvals(A)
        @test abs.(h_Vt*F.Vt') ≈ Matrix(one(elty)*I, n, n)
    end


    @testset "svd!" begin
        A              = rand(elty,m,n)
        d_A            = CuArray(A)
        d_U, d_S, d_Vt = svd!(d_A)
        h_S            = collect(d_S)
        h_U            = collect(d_U)
        h_Vt           = collect(d_Vt)
        F              = svd(A, full=true)
        @test abs.(h_U'h_U) ≈ Matrix(one(elty)*I, m, m)
        @test abs.(h_U[:,1:n]'F.U[:,1:n]) ≈ Matrix(one(elty)*I, n, n)
        @test h_S ≈ svdvals(A)
        @test abs.(h_Vt*F.Vt') ≈ Matrix(one(elty)*I, n, n)
    end


    @testset "qr" begin
        tol = min(m, n)*eps(real(elty))*(1 + (elty <: Complex))

        A              = rand(elty, m, n)
        d_A            = CuArray(A)
        d_F            = qr(d_A)
        d_RR           = d_F.Q'*d_A
        @test d_RR[1:n,:] ≈ d_F.R atol=tol*norm(A)
        @test norm(d_RR[n+1:end,:]) < tol*norm(A)
        A              = rand(elty, n, m)
        d_A            = CuArray(A)
        d_F            = qr(d_A)
        @test d_F.Q'*d_A ≈ d_F.R atol=tol*norm(A)
        A              = rand(elty, m, n)
        d_A            = CuArray(A)
        h_q, h_r       = qr(d_A)
        q, r           = qr(A)
        @test Array(h_q) ≈ Array(q)
        @test Array(h_r) ≈ Array(r)
        A              = rand(elty, n, m)
        d_A            = CuArray(A)
        h_q, h_r       = qr(d_A) # FixMe! Use iteration protocol when implemented
        q, r           = qr(A)
        @test Array(h_q) ≈ Array(q)
        @test Array(h_r) ≈ Array(r)
    end

end

@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    @testset "csrlsvlu!" begin
        A = sparse(rand(elty,n,n))
        b = rand(elty,n)
        x = zeros(elty,n)
        tol = convert(real(elty),1e-6)
        x = CUSOLVER.csrlsvlu!(A,b,x,tol,one(Cint),'O')
        @test x ≈ full(A)\b
        A = sparse(rand(elty,m,n))
        @test_throws DimensionMismatch CUSOLVER.csrlsvlu!(A,b,x,tol,one(Cint),'O')
        A = sparse(rand(elty,n,n))
        b = rand(elty,m)
        x = zeros(elty,n)
        @test_throws DimensionMismatch CUSOLVER.csrlsvlu!(A,b,x,tol,one(Cint),'O')
        b = rand(elty,n)
        x = zeros(elty,m)
        @test_throws DimensionMismatch CUSOLVER.csrlsvlu!(A,b,x,tol,one(Cint),'O')
    end

    @testset "csrlsvqr!" begin
        A     = sparse(rand(elty,n,n))
        d_A   = CudaSparseMatrixCSR(A)
        b     = rand(elty,n)
        d_b   = CudaArray(b)
        x     = zeros(elty,n)
        d_x   = CudaArray(x)
        tol   = convert(real(elty),1e-4)
        d_x   = CUSOLVER.csrlsvqr!(d_A,d_b,d_x,tol,one(Cint),'O')
        h_x   = to_host(d_x)
        @test h_x ≈ full(A)\b
        A     = sparse(rand(elty,m,n))
        d_A   = CudaSparseMatrixCSR(A)
        @test_throws DimensionMismatch CUSOLVER.csrlsvqr!(d_A,d_b,d_x,tol,one(Cint),'O')
        A = sparse(rand(elty,n,n))
        b = rand(elty,m)
        x = zeros(elty,n)
        @test_throws DimensionMismatch CUSOLVER.csrlsvqr!(d_A,d_b,d_x,tol,one(Cint),'O')
        b = rand(elty,n)
        x = zeros(elty,m)
        @test_throws DimensionMismatch CUSOLVER.csrlsvqr!(d_A,d_b,d_x,tol,one(Cint),'O')
    end

    @testset "csrlsvchol!" begin
        A     = rand(elty,n,n)
        A     = sparse(A*A') #posdef
        d_A   = CudaSparseMatrixCSR(A)
        b     = rand(elty,n)
        d_b   = CudaArray(b)
        x     = zeros(elty,n)
        d_x   = CudaArray(x)
        tol   = 10^2*eps(real(elty))
        d_x   = CUSOLVER.csrlsvchol!(d_A,d_b,d_x,tol,zero(Cint),'O')
        h_x   = to_host(d_x)
        @test h_x ≈ full(A)\b
        b     = rand(elty,m)
        d_b   = CudaArray(b)
        @test_throws DimensionMismatch CUSOLVER.csrlsvchol!(d_A,d_b,d_x,tol,zero(Cint),'O')
        b     = rand(elty,n)
        d_b   = CudaArray(b)
        x     = rand(elty,m)
        d_x   = CudaArray(x)
        @test_throws DimensionMismatch CUSOLVER.csrlsvchol!(d_A,d_b,d_x,tol,zero(Cint),'O')
        A     = sparse(rand(elty,m,n))
        d_A   = CudaSparseMatrixCSR(A)
        @test_throws DimensionMismatch CUSOLVER.csrlsvchol!(d_A,d_b,d_x,tol,zero(Cint),'O')
    end

    @testset "csreigvsi" begin
        A     = sparse(rand(elty,n,n))
        d_A   = CudaSparseMatrixCSR(A)
        evs   = eigvals(full(A))
        x_0   = CudaArray(rand(elty,n))
        μ,x   = CUSOLVER.csreigvsi(d_A,convert(elty,evs[1]),x_0,convert(real(elty),1e-6),convert(Cint,1000),'O')
        @test μ ≈ evs[1]
        A     = sparse(rand(elty,m,n))
        d_A   = CudaSparseMatrixCSR(A)
        @test_throws DimensionMismatch CUSOLVER.csreigvsi(d_A,convert(elty,evs[1]),x_0,convert(real(elty),1e-6),convert(Cint,1000),'O')
        A     = sparse(rand(elty,n,n))
        d_A   = CudaSparseMatrixCSR(A)
        x_0   = CudaArray(rand(elty,m))
        @test_throws DimensionMismatch CUSOLVER.csreigvsi(d_A,convert(elty,evs[1]),x_0,convert(real(elty),1e-6),convert(Cint,1000),'O')
    end
    @testset "csreigs" begin
        celty = complex(elty)
        A   = rand(real(elty),n,n)
        A   = sparse(A + A')
        num = CUSOLVER.csreigs(A,convert(celty,complex(-100,-100)),convert(celty,complex(100,100)),'O')
        @test num <= n
        A     = sparse(rand(celty,m,n))
        d_A   = CudaSparseMatrixCSR(A)
        @test_throws DimensionMismatch CUSOLVER.csreigs(A,convert(celty,complex(-100,-100)),convert(celty,complex(100,100)),'O')
    end
    @testset "csrlsqvqr!" begin
        A = sparse(rand(elty,n,n))
        b = rand(elty,n)
        x = zeros(elty,n)
        tol = convert(real(elty),1e-4)
        x = CUSOLVER.csrlsqvqr!(A,b,x,tol,'O')
        @test x[1] ≈ full(A)\b
        A = sparse(rand(elty,n,m))
        x = zeros(elty,n)
        @test_throws DimensionMismatch CUSOLVER.csrlsqvqr!(A,b,x,tol,'O')
        A = sparse(rand(elty,n,n))
        b = rand(elty,m)
        x = zeros(elty,n)
        @test_throws DimensionMismatch CUSOLVER.csrlsqvqr!(A,b,x,tol,'O')
        b = rand(elty,n)
        x = zeros(elty,m)
        @test_throws DimensionMismatch CUSOLVER.csrlsqvqr!(A,b,x,tol,'O')
    end
end

end
