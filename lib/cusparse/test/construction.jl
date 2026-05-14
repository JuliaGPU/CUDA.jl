@testset "construction" begin
    @testset for elty in [Int32, Int64, Float32, Float64, ComplexF32, ComplexF64]
        @testset "vector" begin
            x = sprand(elty,m, 0.2)
            d_x = CuSparseVector(x)
            @test collect(d_x) == collect(x)
            @test similar(d_x) isa CuSparseVector{elty}
            @test similar(d_x, Float32) isa CuSparseVector{Float32}
        end

        @testset "CSC" begin
            x = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCSC(x)
            @test collect(d_x) == collect(x)
            @test similar(d_x) isa CuSparseMatrixCSC{elty}
            @test similar(d_x, (3, 4)) isa CuSparseMatrixCSC{elty}
            @test size(similar(d_x, (3, 4))) == (3, 4)
            @test similar(d_x, Float32) isa CuSparseMatrixCSC{Float32}
            @test similar(d_x, Float32, n, m) isa CuSparseMatrixCSC{Float32}
            @test similar(d_x, Float32, (n, m)) isa CuSparseMatrixCSC{Float32}
        end

        @testset "CSR" begin
            x = sprand(elty,m,n, 0.2)
            d_x  = CuSparseMatrixCSR(x)
            @test collect(d_x) == collect(x)
            @test similar(d_x) isa CuSparseMatrixCSR{elty}
            @test similar(d_x, (3, 4)) isa CuSparseMatrixCSR{elty}
            @test size(similar(d_x, (3, 4))) == (3, 4)
            @test similar(d_x, Float32) isa CuSparseMatrixCSR{Float32}
            @test similar(d_x, Float32, n, m) isa CuSparseMatrixCSR{Float32}
            @test similar(d_x, Float32, (n, m)) isa CuSparseMatrixCSR{Float32}
        end

        if elty <: Union{Float32, Float64, ComplexF32, ComplexF64}
            @testset "COO" begin
                x = sprand(elty,m,n, 0.2)
                d_x  = CuSparseMatrixCOO(x)
                @test collect(d_x) == collect(x)
                @test similar(d_x) isa CuSparseMatrixCOO{elty}
                @test similar(d_x, (3, 4)) isa CuSparseMatrixCOO{elty}
                @test size(similar(d_x, (3, 4))) == (3, 4)
                @test size(similar(d_x, Float64, (3, 4))) == (3, 4)
                @test similar(d_x, Float32) isa CuSparseMatrixCOO{Float32}
                @test CuSparseMatrixCOO(d_x) === d_x
            end
        end

        @testset "BSR" begin
            x   = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixBSR(x, blockdim)
            @test collect(d_x) == collect(x)
            @test similar(d_x) isa CuSparseMatrixBSR{elty}
            @test similar(d_x, Float32) isa CuSparseMatrixBSR{Float32}
        end

        @testset "COO" begin
            x   = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCOO(x)
            @test collect(d_x) == collect(x)
            @test similar(d_x) isa CuSparseMatrixCOO{elty}
            @test similar(d_x, Float32) isa CuSparseMatrixCOO{Float32}
            @test similar(d_x, Float32, n, m) isa CuSparseMatrixCOO{Float32}
        end
    end

    @testset "#1641: too strictly-typed constructors" begin
        rows = CuVector{Int32}([3, 1, 2, 3, 2, 1])
        cols = CuVector{Int32}([3, 2, 1, 2, 3, 1])
        vals = CuVector{Float32}([9, 7, 8, 4, 6, 5])
        @test sparse(rows, cols, vals, fmt=:coo) isa CuSparseMatrixCOO{Float32}
    end
end

if capability(device()) >= v"5.3"
@testset "construction f16" begin
    @testset for elty in [Float16, ComplexF16]
        @testset "CSC" begin
            x = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCSC(x)
            @test collect(d_x) == collect(x)
        end

        @testset "CSR" begin
            x = sprand(elty,m,n, 0.2)
            d_x  = CuSparseMatrixCSR(x)
            @test collect(d_x) == collect(x)
        end
    end
end
end
