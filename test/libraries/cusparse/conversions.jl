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

    dense = rand(Float32, 3,3) |> cu

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

@testset "sparse -- empty matrix" begin
    n, m = 0, 0
    I = [] |> cu
    J = [] |> cu
    V = Float64[] |> cu

    dense = rand(m,n) |> cu

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
                @test x isa T{Float64}
                @test size(x) == (0, 0)

                x = sparse(I, J, V, m, n; fmt=fmt)
                @test x isa T{Float64}
                @test  size(x) == (0, 0)

                if fmt == :csc
                    @test collect(x.colPtr) == [1]
                end
                if fmt == :csr
                    @test collect(x.rowPtr) == [1]
                end
            end

            x = sparse(dense; fmt=fmt)
            @test x isa T{Float64}
            @test collect(x) == collect(dense)
            if fmt == :csc
                @test collect(x.colPtr) == [1]
            end
            if fmt == :csr
                @test collect(x.rowPtr) == [1]
            end
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

if !(v"12.0" <= CUSPARSE.version() < v"12.1")
    x = [0.0; 1.0; 2.0; 0.0; 3.0] |> SparseVector |> CuSparseVector
    A = Matrix{Float64}(undef, 5, 1)
    A[:, 1] .= [0.0; 1.0; 2.0; 0.0; 3.0]
    A = SparseMatrixCSC(A)
    for CuSparseMatrixType in (CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO)
        @testset "conversion CuSparseVector --> $CuSparseMatrixType" begin
            B = CuSparseMatrixType(x)
            @test collect(B)[:] ≈ collect(x)
        end
        @testset "conversion $CuSparseMatrixType --> CuSparseVector" begin
            B = CuSparseMatrixType(A)
            y = CuSparseVector(B)
            @test collect(B)[:] ≈ collect(y)
        end
    end
end

for (n, bd, p) in [(100, 5, 0.02), (5, 1, 0.8), (4, 2, 0.5)]
    v"12.0" <= CUSPARSE.version() < v"12.1" && n == 4 && continue
    @testset "conversions between CuSparseMatrices (n, bd, p) = ($n, $bd, $p)" begin
        A = sprand(n, n, p)
        blockdim = bd
        for CuSparseMatrixType1 in (CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO, CuSparseMatrixBSR)
            dA1 = CuSparseMatrixType1 == CuSparseMatrixBSR ? CuSparseMatrixType1(A, blockdim) : CuSparseMatrixType1(A)
            @testset "conversion $CuSparseMatrixType1 --> SparseMatrixCSC" begin
                @test SparseMatrixCSC(dA1) ≈ A
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
        cols = [2, 1, 1, 3, 3, 2] |> cu
        vals = [7, 5, 8, 6, 9, 4] |> cu
        csr = CuSparseMatrixCSR(crows, cols, vals, (3,3))

        sorted_csr = sort_csr(csr)
        @test collect(sorted_csr.rowPtr) ≈ [1, 3, 5, 7]
        @test collect(sorted_csr.colVal) ≈ [1, 2, 1, 3, 2, 3]
        @test collect(sorted_csr.nzVal)  ≈ [5, 7, 8, 6, 4, 9]
    end
end

if !(v"12.0" <= CUSPARSE.version() < v"12.1")
    @testset "conversion CuSparseMatrix" begin
        # A = [ 0 0 5 ]
        #     [ 0 6 7 ]
        @testset "1-based indexing" begin
            # COO format
            rows_O = [1, 2, 2] |> cu
            cols_O = [3, 2, 3] |> cu
            vals_O = [5, 6, 7] |> cu
            coo_O  = CuSparseMatrixCOO{Float64}(rows_O, cols_O, vals_O, (2,3))

            # CSC format
            rows_O  = [2, 1, 2]    |> cu
            ccols_O = [1, 1, 2, 4] |> cu
            vals_O  = [6, 5, 7]    |> cu
            csc_O   = CuSparseMatrixCSC{Float64}(ccols_O, rows_O, vals_O, (2,3))

            # CSR format
            crows_O = [1, 2, 4] |> cu
            cols_O  = [3, 2, 3] |> cu
            vals_O  = [5, 6, 7] |> cu
            csr_O   = CuSparseMatrixCSR{Float64}(crows_O, cols_O, vals_O, (2,3))

            csr_to_csc_O = CuSparseMatrixCSC{Float64}(csr_O, index='O')
            @test csr_to_csc_O.colPtr ≈ csc_O.colPtr
            @test csr_to_csc_O.rowVal ≈ csc_O.rowVal
            @test csr_to_csc_O.nzVal  ≈ csc_O.nzVal

            csc_to_csr_O = CuSparseMatrixCSR{Float64}(csc_O, index='O')
            @test csc_to_csr_O.rowPtr ≈ csr_O.rowPtr
            @test csc_to_csr_O.colVal ≈ csr_O.colVal
            @test csc_to_csr_O.nzVal  ≈ csr_O.nzVal

            csr_to_coo_O = CuSparseMatrixCOO{Float64}(csr_O, index='O')
            @test csr_to_coo_O.rowInd ≈ coo_O.rowInd
            @test csr_to_coo_O.colInd ≈ coo_O.colInd
            @test csr_to_coo_O.nzVal  ≈ coo_O.nzVal

            coo_to_csr_O = CuSparseMatrixCSR{Float64}(coo_O, index='O')
            @test coo_to_csr_O.rowPtr ≈ csr_O.rowPtr
            @test coo_to_csr_O.colVal ≈ csr_O.colVal
            @test coo_to_csr_O.nzVal  ≈ csr_O.nzVal

            csc_to_coo_O = CuSparseMatrixCOO{Float64}(csc_O, index='O')
            @test csc_to_coo_O.rowInd ≈ coo_O.rowInd
            @test csc_to_coo_O.colInd ≈ coo_O.colInd
            @test csc_to_coo_O.nzVal  ≈ coo_O.nzVal

            coo_to_csc_O = CuSparseMatrixCSC{Float64}(coo_O, index='O')
            @test coo_to_csc_O.colPtr ≈ csc_O.colPtr
            @test coo_to_csc_O.rowVal ≈ csc_O.rowVal
            @test coo_to_csc_O.nzVal  ≈ csc_O.nzVal
        end

        @testset "0-based indexing" begin
            # COO format
            rows_Z = [0, 1, 1] |> cu
            cols_Z = [2, 1, 2] |> cu
            vals_Z = [5, 6, 7] |> cu
            coo_Z  = CuSparseMatrixCOO{Float64}(rows_Z, cols_Z, vals_Z, (2,3))

            # CSC format
            rows_Z  = [1, 0, 1]    |> cu
            ccols_Z = [0, 0, 1, 3] |> cu
            vals_Z  = [6, 5, 7]    |> cu
            csc_Z   = CuSparseMatrixCSC{Float64}(ccols_Z, rows_Z, vals_Z, (2,3))

            # CSR format
            crows_Z = [0, 1, 3] |> cu
            cols_Z  = [2, 1, 2] |> cu
            vals_Z  = [5, 6, 7] |> cu
            csr_Z   = CuSparseMatrixCSR{Float64}(crows_Z, cols_Z, vals_Z, (2,3))

            csr_to_csc_Z = CuSparseMatrixCSC{Float64}(csr_Z, index='Z')
            @test csr_to_csc_Z.colPtr ≈ csc_Z.colPtr
            @test csr_to_csc_Z.rowVal ≈ csc_Z.rowVal
            @test csr_to_csc_Z.nzVal  ≈ csc_Z.nzVal

            csc_to_csr_Z = CuSparseMatrixCSR{Float64}(csc_Z, index='Z')
            @test csc_to_csr_Z.rowPtr ≈ csr_Z.rowPtr
            @test csc_to_csr_Z.colVal ≈ csr_Z.colVal
            @test csc_to_csr_Z.nzVal  ≈ csr_Z.nzVal

            csr_to_coo_Z = CuSparseMatrixCOO{Float64}(csr_Z, index='Z')
            @test csr_to_coo_Z.rowInd ≈ coo_Z.rowInd
            @test csr_to_coo_Z.colInd ≈ coo_Z.colInd
            @test csr_to_coo_Z.nzVal  ≈ coo_Z.nzVal

            coo_to_csr_Z = CuSparseMatrixCSR{Float64}(coo_Z, index='Z')
            @test coo_to_csr_Z.rowPtr ≈ csr_Z.rowPtr
            @test coo_to_csr_Z.colVal ≈ csr_Z.colVal
            @test coo_to_csr_Z.nzVal  ≈ csr_Z.nzVal

            csc_to_coo_Z = CuSparseMatrixCOO{Float64}(csc_Z, index='Z')
            @test csc_to_coo_Z.rowInd ≈ coo_Z.rowInd
            @test csc_to_coo_Z.colInd ≈ coo_Z.colInd
            @test csc_to_coo_Z.nzVal  ≈ coo_Z.nzVal

            coo_to_csc_Z = CuSparseMatrixCSC{Float64}(coo_Z, index='Z')
            @test coo_to_csc_Z.colPtr ≈ csc_Z.colPtr
            @test coo_to_csc_Z.rowVal ≈ csc_Z.rowVal
            @test coo_to_csc_Z.nzVal  ≈ csc_Z.nzVal
        end
    end
end
