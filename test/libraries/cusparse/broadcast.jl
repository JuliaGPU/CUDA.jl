using CUDA.CUSPARSE, SparseArrays

for elty in [Int32, Int64, Float32, Float64]
   @testset "$typ($elty)" for typ in [CuSparseMatrixCSR, CuSparseMatrixCSC]
        m,n = 5,6
        p = 0.5
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
    @testset "$typ($elty)" for typ in [CuSparseVector,]
        m = 64 
        p = 0.5
        x = sprand(elty, m, p)
        dx = typ(x)

        # zero-preserving
        y = x .* elty(1)
        dy = dx .* elty(1)
        @test dy isa typ{elty}
        @test y == SparseVector(dy)

        # not zero-preserving
        y = x .+ elty(1)
        dy = dx .+ elty(1)
        @test dy isa CuArray{elty}
        @test y == Array(dy)

        # involving something dense - broken for now
        y = x .+ ones(elty, m)
        dy = dx .+ CUDA.ones(elty, m)
        @test dy isa CuArray{elty}
        @test y == Array(dy)
        
        # sparse to sparse 
        y = sprand(elty, m, p)
        dy = typ(y)
        dx = typ(x)
        z  = x .* y
        dz = dx .* dy
        @test dz isa typ{elty}
        @test z == SparseVector(dz)

        # multiple inputs
        #=y = sprand(elty, m, p)
        w = sprand(elty, m, p)
        dy = typ(y)
        dx = typ(x)
        dw = typ(w)
        z  = @. x * y * w
        dz = @. dx * dy * w
        @test dz isa typ{elty}
        @test z == SparseVector(dz)=#

        # broken due to llvm IR
        y = sprand(elty, m, p)
        dy = typ(y)
        dx = typ(x)
        z  = x .* y .* elty(2)
        dz = dx .* dy .* elty(2)
        @test dz isa typ{elty}
        @test z == SparseVector(dz)
    end
end

@testset "bug: type conversions" begin
    x = CuSparseMatrixCSR(sparse([1, 2], [2, 1], [5.0, 5.0]))
    y = Int.(x)
    @test eltype(y) == Int
end
