using SparseMatricesCSR
using SparseArrays
using CUDA
using CUDA.CUSPARSE
using Test

@testset "SparseMatricesCSRExt" begin

    for (n, bd, p) in [(100, 5, 0.02), (5, 1, 0.8), (4, 2, 0.5)]
        v"12.0" <= CUSPARSE.version() < v"12.1" && n == 4 && continue
        @testset "conversions between CuSparseMatrices (n, bd, p) = ($n, $bd, $p)" begin
            _A = sprand(n, n, p)
            A = SparseMatrixCSR(_A)
            blockdim = bd
            for CuSparseMatrixType1 in (CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO, CuSparseMatrixBSR)
                dA1 = CuSparseMatrixType1 == CuSparseMatrixBSR ? CuSparseMatrixType1(A, blockdim) : CuSparseMatrixType1(A)
                @testset "conversion $CuSparseMatrixType1 --> SparseMatrixCSR" begin
                    @test SparseMatrixCSR(dA1) ≈ A
                end
                for CuSparseMatrixType2 in (CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO, CuSparseMatrixBSR)
                    CuSparseMatrixType1 == CuSparseMatrixType2 && continue
                    dA2 = CuSparseMatrixType2 == CuSparseMatrixBSR ? CuSparseMatrixType2(dA1, blockdim) : CuSparseMatrixType2(dA1)
                    @testset "conversion $CuSparseMatrixType1 --> $CuSparseMatrixType2" begin
                        @test collect(dA1) ≈ collect(dA2)
                    end
                end
            end
        end
    end
end
