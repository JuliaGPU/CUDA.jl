using CUDA.CUSPARSE, SparseArrays

m,n = 2,3
p = 0.5

for elty in [Int32, Int64, Float32, Float64]
    @testset "$typ($elty)" for typ in [CuSparseMatrixCSR, CuSparseMatrixCSC]
        x = sprand(elty, m, n, p)
        dx = typ(x)

        # zero-preserving
        y = x .* elty(1)
        dy = dx .* elty(1)
        @test dy isa typ{elty}
        @test y == SparseMatrixCSC(dy)

        # not zero-preserving
        y = x .+ elty(1)
        dy = dx .+ elty(1)
        @test dy isa CuArray{elty}
        @test y == Array(dy)

        # involving something dense
        y = x .* ones(elty, m, n)
        dy = dx .* CUDA.ones(elty, m, n)
        @test dy isa CuArray{elty}
        @test y == Array(dy)

        # multiple inputs
        y = sprand(elty, m, n, p)
        dy = typ(y)
        z = x .* y .* elty(2)
        dz = dx .* dy .* elty(2)
        @test dz isa typ{elty}
        @test z == SparseMatrixCSC(dz)
    end
end

@testset "bug: type conversions" begin
    x = CuSparseMatrixCSR(sparse([1, 2], [2, 1], [5.0, 5.0]))
    y = Int.(x)
    @test eltype(y) == Int
end
