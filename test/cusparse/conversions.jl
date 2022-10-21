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
        V = rand(typ, length(I))
        A = sparse(I, J, V, 6, 6)
        Agpu = sparse(I |> cu, J |> cu, V |> cu, 6, 6)
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
