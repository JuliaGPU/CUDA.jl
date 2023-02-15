using LinearAlgebra
using Adapt
using CUDA.CUSPARSE
using SparseArrays
using CUDA

@testset "sparse" begin
    n, m = 4, 4
    I = [1,2,3] |> cu
    J = [2,3,4] |> cu
    V = Float32[1,2,3] |> cu

    dense = rand(3,3) |> cu

    # check defaults
    @test sparse(I, J, V) isa CuSparseMatrixCSC
    @test sparse(dense) isa CuSparseMatrixCSC

    for (fmt, T) in  [(:coo, CuSparseMatrixCOO),
                      (:csc, CuSparseMatrixCSC),
                      (:csr, CuSparseMatrixCSR),
                      (:bsr, CuSparseMatrixBSR)
                     ]
        @testset "sparse $T" begin
            if fmt != :bsr # bsr not supported
                x = sparse(I, J, V; fmt=fmt)
                @test x isa T{Float32}
                @test size(x) == (3, 4)

                x = sparse(I, J, V, m, n; fmt=fmt)
                @test x isa T{Float32}
                @test  size(x) == (4, 4)
            end

            x = sparse(dense; fmt=fmt)
            @test x isa T{Float32}
            @test collect(x) == collect(dense)
        end
    end
end

@testset "unsorted sparse (CUDA.jl#1407)" begin
    I = [1, 1, 2, 3, 3, 4, 5, 4, 6, 4, 5, 6, 6, 6]
    J = [4, 6, 4, 5, 6, 6, 6, 1, 1, 2, 3, 3, 4, 5]

    for typ in (Float16, Float32)
        V = rand(typ, length(I))
        A = sparse(I, J, V, 6, 6)
        for format ∈ (:coo, :csr, :csc)
            Agpu = sparse(I |> cu, J |> cu, V |> cu, 6, 6, fmt=format)
            @test Array(Agpu) == A
        end
    end
end

@testset "CuSparseMatrix(::Diagonal)" begin
    X = Diagonal(rand(10))
    dX = cu(X)
    dY = CuSparseMatrixCSC{Float64, Int32}(dX)
    dZ = CuSparseMatrixCSR{Float64, Int32}(dX)
    @test SparseMatrixCSC(dY) ≈ SparseMatrixCSC(dZ)
    @test SparseMatrixCSC(CuSparseMatrixCSC(X)) ≈ SparseMatrixCSC(CuSparseMatrixCSR(X))
end

@testset "prune" begin
    for SparseMatrixType in (CuSparseMatrixCSC, CuSparseMatrixCSR)
        for T in (Float32, Float64)
            A = sprand(T, 20, 10, 0.7)
            threshold = T(0.5)
            dA = SparseMatrixType(A)
            dC = CUSPARSE.prune(dA, threshold, 'O')
            @test droptol!(A, threshold) ≈ collect(dC)
        end
    end
end

@testset "conversions between CuSparseMatrices" begin
    for (n, bd, p) in [(100, 5, 0.02), (4, 2, 0.5)]
    A = sprand(n, n, p)
    blockdim = bd
    for CuSparseMatrixType1 in (CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO, CuSparseMatrixBSR)
        if CuSparseMatrixType1 == CuSparseMatrixBSR
            dA1 = CuSparseMatrixType1(A, blockdim)
        else
            dA1 = CuSparseMatrixType1(A)
        end
        @testset "conversion SparseMatrixCSC --> $CuSparseMatrixType1" begin
            @test collect(dA1) ≈ A
        end
        for CuSparseMatrixType2 in (CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO, CuSparseMatrixBSR)
            CuSparseMatrixType1 == CuSparseMatrixType2 && continue
            @testset "conversion $CuSparseMatrixType1 --> $CuSparseMatrixType2" begin
                if CuSparseMatrixType2 == CuSparseMatrixBSR
                    dA2 = CuSparseMatrixType2(dA1, blockdim)
                else
                    dA2 = CuSparseMatrixType2(dA1)
                end
                @test collect(dA1) ≈ collect(dA2)
            end
        end
    end
end

@testset "sort CuSparseMatrix" begin
    #     [5 7 0]
    # A = [8 0 6]
    #     [0 4 9]
    @testset "sort_coo" begin
        rows = [3, 1, 2, 3, 2, 1] |> cu
        cols = [3, 2, 1, 2, 3, 1] |> cu
        vals = [9, 7, 8, 4, 6, 5] |> cu
        coo = CuSparseMatrixCOO(rows, cols, vals, (3,3))

        sorted_coo_R = sort_coo(coo, 'R')
        @test collect(sorted_coo_R.rowInd) ≈ [1, 1, 2, 2, 3, 3]
        @test collect(sorted_coo_R.colInd) ≈ [1, 2, 1, 3, 2, 3]
        @test collect(sorted_coo_R.nzVal)  ≈ [5, 7, 8, 6, 4, 9]

        sorted_coo_C = sort_coo(coo, 'C')
        @test collect(sorted_coo_C.rowInd) ≈ [1, 2, 1, 3, 2, 3]
        @test collect(sorted_coo_C.colInd) ≈ [1, 1, 2, 2, 3, 3]
        @test collect(sorted_coo_C.nzVal)  ≈ [5, 8, 7, 4, 6, 9]
    end
    @testset "sort_csc" begin
        rows = [2, 1, 3, 1, 2, 3] |> cu
        ccols = [1, 3, 5, 7] |> cu
        vals = [8, 5, 4, 7, 6, 9] |> cu
        csc = CuSparseMatrixCSC(ccols, rows, vals, (3,3))

        sorted_csc = sort_csc(csc)
        @test collect(sorted_csc.colPtr) ≈ [1, 3, 5, 7]
        @test collect(sorted_csc.rowVal) ≈ [1, 2, 1, 3, 2, 3]
        @test collect(sorted_csc.nzVal)  ≈ [5, 8, 7, 4, 6, 9]
    end
    @testset "sort_csr" begin
        crows = [1, 3, 5, 7] |> cu
        cols = [2, 1, 1, 2, 3, 2] |> cu
        vals = [7, 5, 8, 6, 9, 4] |> cu
        csr = CuSparseMatrixCSR(crows, cols, vals, (3,3))

        sorted_csr = sort_csr(csr)
        @test collect(sorted_csr.rowPtr) ≈ [1, 3, 5, 7]
        @test collect(sorted_csr.colVal) ≈ [1, 2, 1, 2, 2, 3]
        @test collect(sorted_csr.nzVal)  ≈ [5, 7, 8, 6, 4, 9]
    end
end
