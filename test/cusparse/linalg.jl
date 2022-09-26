using CUDA.CUSPARSE
using LinearAlgebra, SparseArrays

@testset "opnorm and norm" for T in [Float32, Float64, ComplexF32, ComplexF64]
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
    else
        @test_throws CUSPARSEError Array(exp(A)) ≈ exp(collect(a))
    end
end