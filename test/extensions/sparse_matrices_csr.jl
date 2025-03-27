using SparseMatricesCSR
using SparseArrays
using CUDA
using Test 

@testset "SparseMatricesCSR -> CuSparseMatrixCSR" begin
    A = sprand(10, 10, 0.1)
    A_csr = SparseMatrixCSR(A)
    A_gpu = CUSPARSE.CuSparseMatrixCSR(A_csr)

    @test size(A_gpu) == size(A_csr)
    @test CUSPARSE.nnz(A_gpu) == nnz(A_csr)
    @test SparseMatrixCSR(A_gpu) ≈ A_csr
end
