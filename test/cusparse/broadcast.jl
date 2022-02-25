using CUDA.CUSPARSE, SparseArrays

m,n = 2,3
p = 0.5

# we rely on CUSPARSE conversions, so only test supported types
for elty in [Float32, Float64]
    @testset "$typ" for typ in [CuSparseMatrixCSR, CuSparseMatrixCSC]
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

        # multiple inputs
        y = sprand(elty, m, n, p)
        dy = typ(y)
        z = x .* y .* 2
        dz = dx .* dy .* 2
        @test dz isa typ{elty}
        @test z == SparseMatrixCSC(dz)
    end
end

@testset "bug: type conversions" begin
    x = CuSparseMatrixCSR(sparse([1, 2], [2, 1], [5.0, 5.0]))
    y = Int.(x)
    @test eltype(y) == Int
end
