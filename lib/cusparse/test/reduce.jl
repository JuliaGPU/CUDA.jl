using cuSPARSE, SparseArrays

for elty in [Int32, Int64, Float32, Float64]
    @testset "$typ($elty)" for typ in [CuSparseMatrixCSR, CuSparseMatrixCSC]
        m, n = 5, 6
        p = 0.5
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
        x2 = zeros(elty, m, n)
        x2[2, :] .= -one(elty)
        x2[2, end] = -elty(16)
        dx2 = typ(sparse(x2))
        y   = mapreduce(abs, max, x2)
        dy  = mapreduce(abs, max, dx2)
        @test y ≈ dy
    end
end
