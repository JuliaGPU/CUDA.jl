using CUDA.CUSPARSE, SparseArrays

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
        if fmt != :bsr # bsr not supported
            x = sparse(I, J, V; fmt=fmt)
            @test x isa T{Float32}
            @test size(x) == (3, 4)

            x = sparse(I, J, V, m, n; fmt=fmt)
            @test x isa T{Float32}
            @test  size(x) == (4, 4)
        end

        if fmt != :coo # dense to COO not implemented
            x = sparse(dense; fmt=fmt)
            @test x isa T{Float32}
            @test collect(x) == collect(dense)
        end
    end
end
