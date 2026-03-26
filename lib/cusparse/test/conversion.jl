@testset "conversion" begin
    @testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
        @testset "CSC(::CSR)" begin
            x = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCSR(x)
            d_x = CuSparseMatrixCSC(d_x)
            @test collect(d_x) == collect(x)
        end

        @testset "CSR(::CSC)" begin
            x = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCSC(x)
            d_x = CuSparseMatrixCSR(d_x)
            @test collect(d_x) == collect(x)
        end

        @testset "BSR(::CSR)" begin
            x = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCSR(x)
            d_x = CuSparseMatrixBSR(d_x, blockdim)
            @test collect(d_x) == collect(x)
        end
        # CSR(::BSR) already covered by the non-direct collect

        @testset "BSR(::Dense)" begin
            x = rand(elty,m,n)
            d_x = CuArray(x)
            d_x = CuSparseMatrixBSR(d_x, blockdim)
            @test collect(d_x) ≈ x
        end

        @testset "COO(::CSR)" begin
            x = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCSR(x)
            d_x = CuSparseMatrixCOO(d_x)
            @test collect(d_x) == collect(x)
        end
        # CSR(::COO) already covered by the non-direct collect

        @testset "Dense(::CSR)" begin
            x = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCSR(x)
            h_x = collect(d_x)
            @test h_x ≈ Array(x)
        end

        @testset "Dense(::CSC)" begin
            x = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCSC(x)
            h_x = collect(d_x)
            @test h_x ≈ Array(x)
        end

        @testset "CSC(::Dense)" begin
            x = rand(elty,m,n)
            d_x = CuArray(x)
            d_x = CuSparseMatrixCSC(d_x)
            h_x = collect(d_x)
            @test h_x ≈ sparse(x)

            d_x_dense = CuMatrix(d_x)
            @test h_x == collect(d_x_dense)
            h_x_dense = Array(d_x)
            @test h_x == h_x_dense
        end

        @testset "CSR(::Dense)" begin
            x = rand(elty,m,n)
            d_x = CuArray(x)
            d_x = CuSparseMatrixCSR(d_x)
            h_x = collect(d_x)
            @test h_x ≈ sparse(x)

            d_x_dense = CuMatrix(d_x)
            @test h_x == collect(d_x_dense)
            h_x_dense = Array(d_x)
            @test h_x == h_x_dense
        end
    end
end

if capability(device()) >= v"5.3"
@testset "conversion f16" begin
    @testset for elty in [Float16, ComplexF16]
        @testset "CSC(::CSR)" begin
            x = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCSR(x)
            d_x = CuSparseMatrixCSC(d_x)
            @test collect(d_x) == collect(x)
        end

        @testset "CSR(::CSC)" begin
            x = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCSC(x)
            d_x = CuSparseMatrixCSR(d_x)
            @test collect(d_x) == collect(x)
        end

        @testset "Dense(::CSR)" begin
            x = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCSR(x)
            h_x = collect(d_x)
            @test h_x ≈ Array(x)
        end

        @testset "Dense(::CSC)" begin
            x = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCSC(x)
            h_x = collect(d_x)
            @test h_x ≈ Array(x)
        end

        @testset "CSC(::Dense)" begin
            x = rand(elty,m,n)
            d_x = CuArray(x)
            d_x = CuSparseMatrixCSC(d_x)
            h_x = collect(d_x)
            @test h_x ≈ sparse(x)
        end

        @testset "CSR(::Dense)" begin
            x = rand(elty,m,n)
            d_x = CuArray(x)
            d_x = CuSparseMatrixCSR(d_x)
            h_x = collect(d_x)
            @test h_x ≈ sparse(x)
        end
    end
end
end
