using SparseMatricesCSR
using SparseArrays
using CUDA
using Test

@testset "SparseMatricesCSR" begin
    A = sprand(10, 10, 0.1)
    A_csr = SparseMatrixCSR(A)
    A_gpu = CUSPARSE.CuSparseMatrixCSR(A_csr)

    @test size(A_gpu) == size(A_csr)
    @test CUSPARSE.nnz(A_gpu) == nnz(A_csr)
    @test SparseMatrixCSR(A_gpu) ≈ A_csr
    @test A_csr |> cu isa CUSPARSE.CuSparseMatrixCSR

    # convert from CSR to CuCSC
    A_csc_gpu = CUSPARSE.CuSparseMatrixCSC(A_csr)
    @test size(A_csc_gpu) == size(A)
    @test CUSPARSE.nnz(A_csc_gpu) == nnz(A)
    @test SparseMatrixCSC(A_csc_gpu) ≈ A
end
