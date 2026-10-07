using SparseMatricesCSR
using SparseArrays
using CUDACore
using cuSPARSE
using Test

@testset "SparseMatricesCSRExt" begin

    # conversions between the GPU formats themselves are tested in conversion.jl
    for (n, bd, p) in [(100, 5, 0.02)]
        @testset "conversions between CuSparseMatrices (n, bd, p) = ($n, $bd, $p)" begin
            _A = sprand(n, n, p)
            A = SparseMatrixCSR(_A)
            blockdim = bd
            for CuSparseMatrixType1 in (CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO, CuSparseMatrixBSR)
                dA1 = CuSparseMatrixType1 == CuSparseMatrixBSR ? CuSparseMatrixType1(A, blockdim) : CuSparseMatrixType1(A)
                @testset "conversion $CuSparseMatrixType1 --> SparseMatrixCSR" begin
                    @test SparseMatrixCSR(dA1) ≈ A
                end
            end
        end
    end
end
