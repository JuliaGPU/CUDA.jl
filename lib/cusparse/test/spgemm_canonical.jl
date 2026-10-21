using GPUArrays
using SparseArrays
using LinearAlgebra

if cuSPARSE.version() >= v"11"
    algorithms = [cuSPARSE.CUSPARSE_SPGEMM_DEFAULT]
    if cuSPARSE.version() >= v"12"
        append!(algorithms, (cuSPARSE.CUSPARSE_SPGEMM_ALG1,
                             cuSPARSE.CUSPARSE_SPGEMM_ALG2,
                             cuSPARSE.CUSPARSE_SPGEMM_ALG3))
    end
    @testset "canonical SpGEMM output" for format in (CuSparseMatrixCSR, CuSparseMatrixCSC), algo in algorithms
        # Many contributing paths per coordinate, cancellation, and empty major slices.
        A = sparse(Float32[1 1 1 1; 0 0 0 0; 2 -2 2 -2; 3 0 0 0])
        B = sparse(Float32[1 0 2 0 3; -1 0 2 0 3; 1 0 2 0 3; -1 0 2 0 3])
        for (a, b) in ((A,B), (spzeros(Float32,4,4),B),
                       (A,spzeros(Float32,4,5)))
            da, db = format(a), format(b)
            C = cuSPARSE.gemm('N','N',1f0,da,db,'O',algo)
            @test GPUArrays.check_structure(C) === C
            @test Array(C) ≈ Matrix(a*b)
            @test eltype(GPUArrays.compressed_ptr(C)) === Int32
            @test SparseArrays.indtype(C) === Int32
            @test !Base.mightalias(C,da)
            @test !Base.mightalias(C,db)
            GC.gc()
            @test Array(C) ≈ Matrix(a*b)
        end
    end
end
