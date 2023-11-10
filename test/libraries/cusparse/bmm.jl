using CUDA, CUDA.CUSPARSE

using LinearAlgebra
using SparseArrays

m = 5
n = 15
k = 25
p = 0.5

@testset "Sparse-Dense $elty bmm!" for elty in (Float64, Float32)
    α = rand(elty) 
    β = rand(elty) 

    @testset "C = αAB + βC" begin
        A1 = CuSparseMatrixCSR{elty}(sprand(elty, m, k, p))
        A2 = copy(A1)
        A2.nzVal = CUDA.rand(elty, size(A2.nzVal)...)
        A = cat(A1, A2; dims=3)

        B = CUDA.rand(elty, k, n, 2)
        C = CUDA.rand(elty, m, n, 2)
        D = copy(C)

        CUSPARSE.bmm!('N', 'N', α, A, B, β, C, 'O') 

        D[:,:,1] = α * A1 * B[:,:,1] + β * D[:,:,1]
        D[:,:,2] = α * A2 * B[:,:,2] + β * D[:,:,2]

        @test D ≈ C
    end
    elty isa Int32 && break

    @testset "C = αAᵀB + βC" begin
        A1 = CuSparseMatrixCSR{elty}(sprand(elty, k, m, p))
        A2 = copy(A1)
        A2.nzVal = CUDA.rand(elty, size(A2.nzVal)...)
        A = cat(A1, A2; dims=3)

        B = CUDA.rand(elty, k, n, 2)
        C = CUDA.rand(elty, m, n, 2)
        D = copy(C)

        CUSPARSE.bmm!('T', 'N', α, A, B, β, C, 'O') 

        D[:,:,1] = α * A1' * B[:,:,1] + β * D[:,:,1]
        D[:,:,2] = α * A2' * B[:,:,2] + β * D[:,:,2]

        @test D ≈ C
    end


    @testset "C = αABᵀ + βC" begin
        A1 = CuSparseMatrixCSR{elty}(sprand(elty, m, k, p))
        A2 = copy(A1)
        A2.nzVal = CUDA.rand(elty, size(A2.nzVal)...)
        A = cat(A1, A2; dims=3)

        B = CUDA.rand(elty, n, k, 2)
        C = CUDA.rand(elty, m, n, 2)
        D = copy(C)

        CUSPARSE.bmm!('N', 'T', α, A, B, β, C, 'O') 

        D[:,:,1] = α * A1 * B[:,:,1]' + β * D[:,:,1]
        D[:,:,2] = α * A2 * B[:,:,2]' + β * D[:,:,2]

        @test D ≈ C
    end


    @testset "C = αAᵀBᵀ + βC" begin
        A1 = CuSparseMatrixCSR{elty}(sprand(elty, k, m, p))
        A2 = copy(A1)
        A2.nzVal = CUDA.rand(elty, size(A2.nzVal)...)
        A = cat(A1, A2; dims=3)

        B = CUDA.rand(elty, n, k, 2)
        C = CUDA.rand(elty, m, n, 2)
        D = copy(C)

        CUSPARSE.bmm!('T', 'T', α, A, B, β, C, 'O') 

        D[:,:,1] = α * A1' * B[:,:,1]' + β * D[:,:,1]
        D[:,:,2] = α * A2' * B[:,:,2]' + β * D[:,:,2]

        @test D ≈ C
    end

    @testset "extended batch-dims" begin
        A1 = CuSparseMatrixCSR{elty}(sprand(elty, m, k, p))
        A2 = copy(A1)
        A2.nzVal = CUDA.rand(elty, size(A2.nzVal)...)
        A3 = cat(A1, A2; dims=3)

        A4 = copy(A3)
        A4.nzVal = CUDA.rand(elty, size(A3.nzVal)...)

        A5 = copy(A3)
        A5.nzVal = CUDA.rand(elty, size(A3.nzVal)...)

        A = cat(A3, A4, A5; dims=4)

        B = CUDA.rand(elty, k, n, 2, 3)
        C = CUDA.rand(elty, m, n, 2, 3)
        D = copy(C)

        CUSPARSE.bmm!('N', 'N', α, A, B, β, C, 'O') 

        for c in CartesianIndices((2,3))
            CUDA.@allowscalar D[:,:,c] = α * A[:,:,c.I...] * B[:,:,c] + β*D[:,:,c]
        end

        @test D ≈ C
    end
end
