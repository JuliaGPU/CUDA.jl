using CUDA.CUSPARSE, SparseArrays

m,n = 5,6
p = 0.5

for elty in [Int32, Int64, Float32, Float64]
    @testset "$typ($elty)" for typ in [CuSparseMatrixCSR, CuSparseMatrixCSC]
        x = sprand(elty, m, n, p)
        dx = typ(x)

        # dim=:
        y = sum(x)
        dy = sum(dx)
        @test y ≈ dy

        # dim=1
        y = sum(x, dims=1)
        dy = sum(dx, dims=1)
        @test y ≈ Array(dy)

        # dim=2
        y = sum(x, dims=2)
        dy = sum(dx, dims=2)
        @test y ≈ Array(dy)
        if elty in (Float32, Float64)
            dy = mapreduce(abs, +, dx; init=zero(elty))
            y  = mapreduce(abs, +, x; init=zero(elty))
            @test y ≈ dy
        end
    end
end
