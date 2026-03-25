@testset "duplicate entries" begin
    # already sorted
    let
        I = [1, 3, 4, 4]
        J = [1, 2, 3, 3]
        V = [1f0, 2f0, 3f0, 10f0]
        coo = sparse(cu(I), cu(J), cu(V); fmt=:coo)
        @test Array(coo.rowInd) == [1, 3, 4]
        @test Array(coo.colInd) == [1, 2, 3]
        @test Array(coo.nzVal) == [1f0, 2f0, 13f0]
    end

    # out of order
    let
        I = [4, 1, 3, 4]
        J = [3, 1, 2, 3]
        V = [10f0, 1f0, 2f0, 3f0]
        coo = sparse(cu(I), cu(J), cu(V); fmt=:coo)
        @test Array(coo.rowInd) == [1, 3, 4]
        @test Array(coo.colInd) == [1, 2, 3]
        @test Array(coo.nzVal) == [1f0, 2f0, 13f0]
    end

    # JuliaGPU/CUDA.jl#2494
    let
        I = [1, 2, 1]
        J = [1, 2, 1]
        V = [10f0, 1f0, 2f0]
        coo = sparse(cu(I), cu(J), cu(V); fmt=:coo)
        @test Array(coo.rowInd) == [1, 2]
        @test Array(coo.colInd) == [1, 2]
        @test Array(coo.nzVal) == [12f0, 1f0]
    end
end

@testset "Utility type conversions" begin
    @test convert(cuSPARSE.cusparseIndexType_t, Int32) == cuSPARSE.CUSPARSE_INDEX_32I
    @test convert(cuSPARSE.cusparseIndexType_t, Int64) == cuSPARSE.CUSPARSE_INDEX_64I
    @test_throws ArgumentError("CUSPARSE type equivalent for index type Int8 does not exist!") convert(cuSPARSE.cusparseIndexType_t, Int8)
    @test convert(Type, cuSPARSE.CUSPARSE_INDEX_32I) == Int32
    @test convert(Type, cuSPARSE.CUSPARSE_INDEX_64I) == Int64

    @test convert(cuSPARSE.cusparseIndexBase_t, 0) == cuSPARSE.CUSPARSE_INDEX_BASE_ZERO
    @test convert(cuSPARSE.cusparseIndexBase_t, 1) == cuSPARSE.CUSPARSE_INDEX_BASE_ONE
    @test_throws ArgumentError("CUSPARSE does not support index base 2!") convert(cuSPARSE.cusparseIndexBase_t, 2)
    @test convert(Int8, cuSPARSE.CUSPARSE_INDEX_BASE_ZERO) == zero(Int8)
    @test convert(Int8, cuSPARSE.CUSPARSE_INDEX_BASE_ONE)  == one(Int8)

    @test_throws ArgumentError("Unknown operation X") convert(cuSPARSE.cusparseOperation_t, cuSPARSE.SparseChar('X'))

    @test convert(cuSPARSE.cusparseMatrixType_t, cuSPARSE.SparseChar('G')) == cuSPARSE.CUSPARSE_MATRIX_TYPE_GENERAL
    @test convert(cuSPARSE.cusparseMatrixType_t, cuSPARSE.SparseChar('T')) == cuSPARSE.CUSPARSE_MATRIX_TYPE_TRIANGULAR
    @test convert(cuSPARSE.cusparseMatrixType_t, cuSPARSE.SparseChar('S')) == cuSPARSE.CUSPARSE_MATRIX_TYPE_SYMMETRIC
    @test convert(cuSPARSE.cusparseMatrixType_t, cuSPARSE.SparseChar('H')) == cuSPARSE.CUSPARSE_MATRIX_TYPE_HERMITIAN
    @test_throws ArgumentError("Unknown matrix type X") convert(cuSPARSE.cusparseMatrixType_t, cuSPARSE.SparseChar('X'))

    @test_throws ArgumentError("Unknown attribute X") convert(cuSPARSE.cusparseSpMatAttribute_t, cuSPARSE.SparseChar('X'))
    @test_throws ArgumentError("Unknown fill mode X") convert(cuSPARSE.cusparseFillMode_t, cuSPARSE.SparseChar('X'))
    @test_throws ArgumentError("Unknown diag type X") convert(cuSPARSE.cusparseDiagType_t, cuSPARSE.SparseChar('X'))
    @test_throws ArgumentError("Unknown index base X") convert(cuSPARSE.cusparseIndexBase_t, cuSPARSE.SparseChar('X'))

    @test convert(cuSPARSE.cusparseDirection_t, cuSPARSE.SparseChar('R')) == cuSPARSE.CUSPARSE_DIRECTION_ROW
    @test convert(cuSPARSE.cusparseDirection_t, cuSPARSE.SparseChar('C')) == cuSPARSE.CUSPARSE_DIRECTION_COLUMN
    @test_throws ArgumentError("Unknown direction X") convert(cuSPARSE.cusparseDirection_t, cuSPARSE.SparseChar('X'))

    @test convert(cuSPARSE.cusparseOrder_t, cuSPARSE.SparseChar('R')) == cuSPARSE.CUSPARSE_ORDER_ROW
    @test convert(cuSPARSE.cusparseOrder_t, cuSPARSE.SparseChar('C')) == cuSPARSE.CUSPARSE_ORDER_COL
    @test_throws ArgumentError("Unknown order X") convert(cuSPARSE.cusparseOrder_t, cuSPARSE.SparseChar('X'))

    @test convert(cuSPARSE.cusparseSpSVUpdate_t, cuSPARSE.SparseChar('G')) == cuSPARSE.CUSPARSE_SPSV_UPDATE_GENERAL
    @test convert(cuSPARSE.cusparseSpSVUpdate_t, cuSPARSE.SparseChar('D')) == cuSPARSE.CUSPARSE_SPSV_UPDATE_DIAGONAL
    @test_throws ArgumentError("Unknown update X") convert(cuSPARSE.cusparseSpSVUpdate_t, cuSPARSE.SparseChar('X'))

    @test convert(cuSPARSE.cusparseSpSMUpdate_t, cuSPARSE.SparseChar('G')) == cuSPARSE.CUSPARSE_SPSV_UPDATE_GENERAL
    @test convert(cuSPARSE.cusparseSpSMUpdate_t, cuSPARSE.SparseChar('D')) == cuSPARSE.CUSPARSE_SPSV_UPDATE_DIAGONAL
    @test_throws ArgumentError("Unknown update X") convert(cuSPARSE.cusparseSpSMUpdate_t, cuSPARSE.SparseChar('X'))
end

@testset "CuSparseArrayCSR" begin
    x = sprand(n, m, 0.2)
    d_x = CuSparseArrayCSR(CuArray(x.colptr), CuArray(x.rowval), CuArray(x.nzval), (m, n))
    @test d_x isa CuSparseArrayCSR
    @test length(d_x) == m*n
    @test CuSparseArrayCSR(d_x) === d_x
    @test size(similar(d_x)) == size(d_x)
    @test size(d_x, 3) == 1
    @test_throws ArgumentError("dimension must be ≥ 1, got 0") size(d_x, 0)
    CUDACore.@allowscalar begin
        @test d_x[1, 2] == x[2, 1]
    end
end
