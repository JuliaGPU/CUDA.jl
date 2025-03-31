using CUDA.CUSPARSE
using LinearAlgebra, SparseArrays

@testset "opnorm and norm $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
    S = sprand(T, 10, 10, 0.1)
    dS_csc = CuSparseMatrixCSC(S)
    dS_csr = CuSparseMatrixCSR(S)
    @test opnorm(S, Inf) ≈ opnorm(dS_csc, Inf)
    @test opnorm(S, Inf) ≈ opnorm(dS_csr, Inf)
    @test opnorm(S, 1) ≈ opnorm(dS_csc, 1)
    @test opnorm(S, 1) ≈ opnorm(dS_csr, 1)
    @test_throws ArgumentError("p=2 is not supported") opnorm(dS_csr, 2)
    @test norm(S, 0) ≈ norm(dS_csc, 0)
    @test norm(S, 0) ≈ norm(dS_csr, 0)
    @test norm(S, 1) ≈ norm(dS_csc, 1)
    @test norm(S, 1) ≈ norm(dS_csr, 1)
    @test norm(S, 2) ≈ norm(dS_csc, 2)
    @test norm(S, 2) ≈ norm(dS_csr, 2)
    @test norm(S, Inf) ≈ norm(dS_csc, Inf)
    @test norm(S, Inf) ≈ norm(dS_csr, Inf)
    @test norm(S, -Inf) ≈ norm(dS_csc, -Inf)
    @test norm(S, -Inf) ≈ norm(dS_csr, -Inf)
end

@testset "triu tril exp $typ" for
    typ in [CuSparseMatrixCSR, CuSparseMatrixCSC]

    a = sprand(ComplexF32, 100, 100, 0.02)
    A = typ(a)
    @test Array(triu(A)) ≈ triu(a)
    @test Array(triu(A, 1)) ≈ triu(a, 1)
    @test Array(tril(A)) ≈ tril(a)
    @test Array(tril(A, 1)) ≈ tril(a, 1)
    if CUSPARSE.version() > v"11.4.1"
        @test Array(exp(A)) ≈ exp(collect(a))
    end
end

@testset "$typ kronecker product" for
    typ in [CuSparseMatrixCSR, CuSparseMatrixCSC]

    a = sprand(ComplexF32, 100, 100, 0.1)
    b = sprand(ComplexF32, 100, 100, 0.1)
    A = typ(a)
    B = typ(b)
    za = spzeros(ComplexF32, 100, 100)
    ZA = typ(za)
    @test collect(kron(A, B)) ≈ kron(a, b)
    @test collect(kron(transpose(A), B)) ≈ kron(transpose(a), b)
    @test collect(kron(A, transpose(B))) ≈ kron(a, transpose(b))
    @test collect(kron(transpose(A), transpose(B))) ≈ kron(transpose(a), transpose(b))
    @test collect(kron(A', B)) ≈ kron(a', b)
    @test collect(kron(A, B')) ≈ kron(a, b')
    @test collect(kron(A', B')) ≈ kron(a', b')
    
    @test collect(kron(ZA, B)) ≈ kron(za, b)
    @test collect(kron(transpose(ZA), B)) ≈ kron(transpose(za), b)
    @test collect(kron(ZA, transpose(B))) ≈ kron(za, transpose(b))
    @test collect(kron(transpose(ZA), transpose(B))) ≈ kron(transpose(za), transpose(b))
    @test collect(kron(ZA', B)) ≈ kron(za', b)
    @test collect(kron(ZA, B')) ≈ kron(za, b')
    @test collect(kron(ZA', B')) ≈ kron(za', b')

    C = I(50)
    @test collect(kron(A, C)) ≈ kron(a, C)
    @test collect(kron(C, A)) ≈ kron(C, a)
    @test collect(kron(transpose(A), C)) ≈ kron(transpose(a), C)
    @test collect(kron(C, transpose(A))) ≈ kron(C, transpose(a))
    @test collect(kron(adjoint(A), C)) ≈ kron(adjoint(a), C)
    @test collect(kron(C, adjoint(A))) ≈ kron(C, adjoint(a))
    @test collect(kron(A', C)) ≈ kron(a', C)
    @test collect(kron(C, A')) ≈ kron(C, a')
end

@testset "Reshape $typ (100,100) -> (20, 500) and droptol" for
    typ in [CuSparseMatrixCSR, CuSparseMatrixCSC]

    a = sprand(ComplexF32, 100, 100, 0.1)
    dims = (20, 500)
    A = typ(a)
    @test Array(reshape(A, dims)) ≈ reshape(a, dims)
    droptol!(a, 0.4)
    droptol!(A, 0.4)
    @test collect(A) ≈ a
end

@testset "Generalized dot product for $typ and $elty" for
    typ in [CuSparseMatrixCSR, CuSparseMatrixCSC], elty in [Int64, Float32, Float64, ComplexF64]

    N1 = 1000*2
    N2 = 1000*3
    x = rand(elty, N1)
    y = rand(elty, N2)
    A = sprand(elty, N1, N2, 1/N1)

    x2 = CuArray(x)
    y2 = CuArray(y)
    A2 = typ(A)

    @test dot(x, A, y) ≈ dot(x2, A2, y2)
    @test_throws DimensionMismatch("dimensions must match") dot(CUDA.rand(elty, N1+1), A2, y2)
end
