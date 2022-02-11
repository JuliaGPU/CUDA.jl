using CUDA.CUSPARSE, SparseArrays

m,n = 2,3
p = 0.5

for elty in [Float32]
    @testset "CuSparseVector" begin
        x = sprand(elty, m*n, p)
        dx = CuSparseVector(x)

        # zero-preserving
        y = x .* 1
        dy = dx .* 1
        @test dy isa CuSparseVector{elty}
        @test y == SparseVector(dy)

        # not zero-preserving
        y = x .+ 1
        dy = dx .+ 1
        @test dy isa CuVector{elty}
        @test y == Array(dy)

        # involving something dense
        y = x .* ones(m*n)
        dy = dx .* CUDA.ones(m*n)
        @test dy isa CuVector{elty}
        @test y == Array(dy)
    end

    # TODO: BSR
    @testset "$typ" for typ in [CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO]
        x = sprand(elty, m, n, p)
        dx = typ(x)

        # zero-preserving
        y = x .* 1
        dy = dx .* 1
        @test dy isa typ{elty}
        @test y == SparseMatrixCSC(dy)

        # not zero-preserving
        y = x .+ 1
        dy = dx .+ 1
        @test dy isa CuArray{elty}
        @test y == Array(dy)

        # involving something dense
        y = x .* ones(m, n)
        dy = dx .* CUDA.ones(m, n)
        @test dy isa CuArray{elty}
        @test y == Array(dy)
    end
end
