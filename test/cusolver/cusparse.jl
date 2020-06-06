using CUDA.CUSOLVER
using CUDA.CUSPARSE

using LinearAlgebra
using SparseArrays

m = 15
n = 10
l = 13
k = 1

@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    @testset "csrlsvlu!" begin
        A = sparse(rand(elty,n,n))
        b = rand(elty,n)
        x = zeros(elty,n)
        tol = convert(real(elty),1e-6)
        x = CUSOLVER.csrlsvlu!(A,b,x,tol,one(Cint),'O')
        @test x ≈ Array(A)\b
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
        d_A   = CuSparseMatrixCSR(A)
        b     = rand(elty,n)
        d_b   = CuArray(b)
        x     = zeros(elty,n)
        d_x   = CuArray(x)
        tol   = convert(real(elty),1e-4)
        d_x   = CUSOLVER.csrlsvqr!(d_A,d_b,d_x,tol,one(Cint),'O')
        h_x   = collect(d_x)
        @test h_x ≈ Array(A)\b
        A     = sparse(rand(elty,m,n))
        d_A   = CuSparseMatrixCSR(A)
        @test_throws DimensionMismatch CUSOLVER.csrlsvqr!(d_A,d_b,d_x,tol,one(Cint),'O')
        A = sparse(rand(elty,n,n))
        b = rand(elty,m)
        x = zeros(elty,n)
        d_A   = CuSparseMatrixCSR(A)
        d_b   = CuArray(b)
        d_x   = CuArray(x)
        @test_throws DimensionMismatch CUSOLVER.csrlsvqr!(d_A,d_b,d_x,tol,one(Cint),'O')
        b = rand(elty,n)
        x = zeros(elty,m)
        d_A   = CuSparseMatrixCSR(A)
        d_b   = CuArray(b)
        d_x   = CuArray(x)
        @test_throws DimensionMismatch CUSOLVER.csrlsvqr!(d_A,d_b,d_x,tol,one(Cint),'O')
        dA    = diagm(0=>rand(elty, n))
        dA[1,1] = zero(elty)
        A     = sparse(dA)
        d_A   = CuSparseMatrixCSR(A)
        b     = rand(elty,n)
        d_b   = CuArray(b)
        x     = zeros(elty,n)
        d_x   = CuArray(x)
        @test_throws SingularException CUSOLVER.csrlsvqr!(d_A,d_b,d_x,tol,one(Cint),'O')
    end

    @testset "csrlsvchol!" begin
        A     = rand(elty,n,n)
        A     = sparse(A*A') #posdef
        d_A   = CuSparseMatrixCSR(A)
        b     = rand(elty,n)
        d_b   = CuArray(b)
        x     = zeros(elty,n)
        d_x   = CuArray(x)
        tol   = 10^2*eps(real(elty))
        d_x   = CUSOLVER.csrlsvchol!(d_A,d_b,d_x,tol,zero(Cint),'O')
        h_x   = collect(d_x)
        @test h_x ≈ Array(A)\b
        b     = rand(elty,m)
        d_b   = CuArray(b)
        @test_throws DimensionMismatch CUSOLVER.csrlsvchol!(d_A,d_b,d_x,tol,zero(Cint),'O')
        b     = rand(elty,n)
        d_b   = CuArray(b)
        x     = rand(elty,m)
        d_x   = CuArray(x)
        @test_throws DimensionMismatch CUSOLVER.csrlsvchol!(d_A,d_b,d_x,tol,zero(Cint),'O')
        A     = sparse(rand(elty,m,n))
        d_A   = CuSparseMatrixCSR(A)
        @test_throws DimensionMismatch CUSOLVER.csrlsvchol!(d_A,d_b,d_x,tol,zero(Cint),'O')
        dA    = diagm(0=>rand(elty, n))
        dA[1,1] = zero(elty)
        A     = sparse(dA)
        d_A   = CuSparseMatrixCSR(A)
        b     = rand(elty,n)
        d_b   = CuArray(b)
        x     = zeros(elty,n)
        d_x   = CuArray(x)
        @test_throws SingularException CUSOLVER.csrlsvchol!(d_A,d_b,d_x,tol,one(Cint),'O')
    end

    @testset "csreigvsi" begin
        A     = sparse(rand(elty,n,n))
        A     = A + A'
        d_A   = CuSparseMatrixCSR(A)
        evs   = eigvals(Array(A))
        x_0   = CUDA.rand(elty,n)
        μ,x   = CUSOLVER.csreigvsi(d_A,convert(elty,evs[1]),x_0,convert(real(elty),1e-6),convert(Cint,1000),'O')
        @test μ ≈ evs[1]
        A     = sparse(rand(elty,m,n))
        d_A   = CuSparseMatrixCSR(A)
        @test_throws DimensionMismatch CUSOLVER.csreigvsi(d_A,convert(elty,evs[1]),x_0,convert(real(elty),1e-6),convert(Cint,1000),'O')
        A     = sparse(rand(elty,n,n))
        d_A   = CuSparseMatrixCSR(A)
        x_0   = CUDA.rand(elty,m)
        @test_throws DimensionMismatch CUSOLVER.csreigvsi(d_A,convert(elty,evs[1]),x_0,convert(real(elty),1e-6),convert(Cint,1000),'O')
    end
    @testset "csreigs" begin
        celty = complex(elty)
        A   = rand(real(elty),n,n)
        A   = sparse(A + A')
        num = CUSOLVER.csreigs(A,convert(celty,complex(-100,-100)),convert(celty,complex(100,100)),'O')
        @test num <= n
        A     = sparse(rand(celty,m,n))
        d_A   = CuSparseMatrixCSR(A)
        @test_throws DimensionMismatch CUSOLVER.csreigs(A,convert(celty,complex(-100,-100)),convert(celty,complex(100,100)),'O')
    end
    @testset "csrlsqvqr!" begin
        A = sparse(rand(elty,n,n))
        b = rand(elty,n)
        x = zeros(elty,n)
        tol = convert(real(elty),1e-4)
        x = CUSOLVER.csrlsqvqr!(A,b,x,tol,'O')
        @test x[1] ≈ Array(A)\b
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
