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
        
        y = mapreduce(exp, +, x)
        dy = mapreduce(exp, +, dx)
        @test y ≈ dy

        # dim=1
        y = sum(x, dims=1)
        dy = sum(dx, dims=1)
        @test y ≈ Array(dy)
        
        y = mapreduce(exp, +, x, dims=1)
        dy = mapreduce(exp, +, dx, dims=1)
        @test y ≈ Array(dy)

        # dim=2
        y = sum(x, dims=2)
        dy = sum(dx, dims=2)
        @test y ≈ Array(dy)
        
        y = mapreduce(exp, +, x, dims=2)
        dy = mapreduce(exp, +, dx, dims=2)
        @test y ≈ Array(dy)
        if elty in (Float32, Float64)
            dy = mapreduce(abs, +, dx; init=zero(elty))
            y  = mapreduce(abs, +, x; init=zero(elty))
            @test y ≈ dy
        end
        
        # test with a matrix with fully empty rows
        x = zeros(elty, m, n)
        x[2, :] .= -one(elty)
        x[2, end] = -elty(16)
        dx = typ(sparse(x))
        y  = mapreduce(abs, max, x)
        dy = mapreduce(abs, max, dx)
        @test y ≈ dy
    end
end
