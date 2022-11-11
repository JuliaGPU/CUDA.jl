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
    @test collect(kron(A, B)) ≈ kron(a, b)
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
