using LinearAlgebra
using CUDA.CUSPARSE, SparseArrays
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

@testset "unsorted sparse (CUDA.jl#1407)" begin
    I = [1, 1, 2, 3, 3, 4, 5, 4, 6, 4, 5, 6, 6, 6]
    J = [4, 6, 4, 5, 6, 6, 6, 1, 1, 2, 3, 3, 4, 5]

    # ensure we cover both the CUSPARSE-based and native COO row sort
    for typ in (Float16, Float32)
        A = sparse(I, J, ones(typ, length(I)), 6, 6)
        Agpu = sparse(I |> cu, J |> cu, ones(typ, length(I)) |> cu, 6, 6)
        @test Array(Agpu) == A
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