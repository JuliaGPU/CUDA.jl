using Test
using Random
using CUDA
using CUDA.CUSPARSE, SparseArrays

Random.seed!(123)

@testset "mm algo=$algo" for algo in [
    CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT,
    CUSPARSE.CUSPARSE_SPMM_CSR_ALG1,
    CUSPARSE.CUSPARSE_SPMM_CSR_ALG2,
    CUSPARSE.CUSPARSE_SPMM_CSR_ALG3,
]
    @testset "mm $T" T in [Float32, Float64, ComplexF32, ComplexF64]
        A = sprand(T, 10, 10, 0.1)
        B = rand(T, 10, 2)
        C = rand(T, 10, 2)
        dA = CuSparseMatrixCSR(A)
        dB = CuArray(B)
        dC = CuArray(C)

        alpha = 1.2
        beta = 1.3
        mm!('N', 'N', alpha, dA, dB, beta, dC, 'O', algo)
        @test alpha * A * B + beta * C ≈ collect(dC)
    end
end

@testset "mv algo=$algo" for algo in [
    CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT,
    CUSPARSE.CUSPARSE_SPMV_CSR_ALG1,
    CUSPARSE.CUSPARSE_SPMV_CSR_ALG2,
]
    @testset "mv $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
        A = sprand(T, 10, 10, 0.1)
        B = rand(T, 10)
        C = rand(T, 10)
        dA = CuSparseMatrixCSR(A)
        dB = CuArray(B)
        dC = CuArray(C)

        alpha = 1.2
        beta = 1.3
        mv!('N', alpha, dA, dB, beta, dC, 'O', algo)
        @test alpha * A * B + beta * C ≈ collect(dC)
    end
end
