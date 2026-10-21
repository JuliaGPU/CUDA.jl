using SparseMatricesCSR
using SparseArrays
using CUDACore
using cuSPARSE
using Test
using GPUArrays

@testset "explicit SparseMatricesCSR integer widths" begin
    host=SparseMatrixCSR(sparse(Float32[1 0 2;0 3 0]))
    for S in (CuSparseMatrixCSR,CuSparseMatrixCSC,CuSparseMatrixCOO)
        A=S{Float32,Int64}(host)
        @test SparseArrays.indtype(A)===Int64
        S===CuSparseMatrixCOO || @test eltype(GPUArrays.compressed_ptr(A))===Int64
        @test Array(A)==Matrix(host)
    end
    # A valid wide coordinate with a tiny stored pattern must not first narrow to Cint.
    wide=Int(typemax(Int32))+1
    host=SparseMatrixCSR{1}(1,wide,Int64[1,2],Int64[wide],Float32[3])
    A=CuSparseMatrixCOO{Float32,Int64}(host)
    @test size(A)==(1,wide)
    @test Array(A.colInd)==Int64[wide]
    @test Array(nonzeros(A))==Float32[3]
    @test_throws ArgumentError CuSparseMatrixCOO{Float32,Int32}(host)
end

@testset "SparseMatricesCSRExt" begin

    for (n, bd, p) in [(100, 5, 0.02), (5, 1, 0.8), (4, 2, 0.5)]
        v"12.0" <= cuSPARSE.version() < v"12.1" && n == 4 && continue
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
                    dense_conversion(Float64, CuSparseMatrixType1) &&
                        dense_conversion(Float64, CuSparseMatrixType2) || continue
                    @testset "conversion $CuSparseMatrixType1 --> $CuSparseMatrixType2" begin
                        @test collect(dA1) ≈ collect(dA2)
                    end
                end
            end
        end
    end
end
