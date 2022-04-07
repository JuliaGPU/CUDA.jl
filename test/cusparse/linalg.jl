using CUDA.CUSPARSE
using LinearAlgebra, SparseArrays

@testset "opnorm" begin
    S = sprand(100, 100, 0.1)
    dS_csc = CuSparseMatrixCSC(S)
    dS_csr = CuSparseMatrixCSR(S)
    @test opnorm(S, Inf) ≈ opnorm(dS_csc, Inf)
    @test opnorm(S, Inf) ≈ opnorm(dS_csr, Inf)
end
