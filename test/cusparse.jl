using CUDA.CUSPARSE

using LinearAlgebra
using SparseArrays
using SparseArrays: nonzeroinds

@test CUSPARSE.version() isa VersionNumber

m = 25
n = 35
k = 10
blockdim = 5

@testset "array" begin
    x = sprand(m,0.2)
    d_x = CuSparseVector(x)
    @test length(d_x) == m
    @test size(d_x)   == (m, 1)
    @test size(d_x,1) == m
    @test size(d_x,2) == 1
    @test ndims(d_x)  == 1
    CUDA.@allowscalar begin
        @test Array(d_x[:])        == x[:]
        @test d_x[firstindex(d_x)] == x[firstindex(x)]
        @test d_x[div(end, 2)]     == x[div(end, 2)]
        @test d_x[end]             == x[end]
        @test Array(d_x[firstindex(d_x):end]) == x[firstindex(x):end]
    end
    @test_throws BoundsError d_x[firstindex(d_x) - 1]
    @test_throws BoundsError d_x[end + 1]
    @test nnz(d_x)    == nnz(x)
    @test Array(nonzeros(d_x)) == nonzeros(x)
    @test Array(nonzeroinds(d_x)) == nonzeroinds(x)
    @test nnz(d_x)    == length(nonzeros(d_x))
    x = sprand(m,n,0.2)
    d_x = CuSparseMatrixCSC(x)
    @test length(d_x) == m*n
    @test size(d_x)   == (m,n)
    @test size(d_x,1) == m
    @test size(d_x,2) == n
    @test size(d_x,3) == 1
    @test ndims(d_x)  == 2
    CUDA.@allowscalar begin
        @test Array(d_x[:])        == x[:]
        @test d_x[firstindex(d_x)] == x[firstindex(x)]
        @test d_x[div(end, 2)]     == x[div(end, 2)]
        @test d_x[end]             == x[end]
        @test d_x[firstindex(d_x), firstindex(d_x)] == x[firstindex(x), firstindex(x)]
        @test d_x[div(end, 2), div(end, 2)]         == x[div(end, 2), div(end, 2)]
        @test d_x[end, end]        == x[end, end]
        @test Array(d_x[firstindex(d_x):end, firstindex(d_x):end]) == x[:, :]
        for i in 1:size(x, 2)
            @test Array(d_x[:, i]) == x[:, i]
        end
    end
    @test_throws BoundsError d_x[firstindex(d_x) - 1]
    @test_throws BoundsError d_x[end + 1]
    @test_throws BoundsError d_x[firstindex(d_x) - 1, firstindex(d_x) - 1]
    @test_throws BoundsError d_x[end + 1, end + 1]
    @test_throws BoundsError d_x[firstindex(d_x) - 1:end + 1, :]
    @test_throws BoundsError d_x[firstindex(d_x) - 1, :]
    @test_throws BoundsError d_x[end + 1, :]
    @test_throws BoundsError d_x[:, firstindex(d_x) - 1:end + 1]
    @test_throws BoundsError d_x[:, firstindex(d_x) - 1]
    @test_throws BoundsError d_x[:, end + 1]
    @test nnz(d_x)    == nnz(x)
    @test Array(nonzeros(d_x)) == nonzeros(x)
    @test nnz(d_x)    == length(nonzeros(d_x))
    @test !issymmetric(d_x)
    @test !ishermitian(d_x)
    @test_throws ArgumentError size(d_x,0)
    @test_throws ArgumentError CUSPARSE.CuSparseVector(x)
    y = sprand(k,n,0.2)
    d_y = CuSparseMatrixCSC(y)
    @test_throws ArgumentError copyto!(d_y,d_x)
    d_y = CuSparseMatrixCSR(d_y)
    d_x = CuSparseMatrixCSR(d_x)
    @test_throws ArgumentError copyto!(d_y,d_x)
    CUDA.@allowscalar begin
        for i in 1:size(y, 1)
          @test d_y[i, :] ≈ y[i, :]
        end
    end
    d_y = CuSparseMatrixBSR(d_y, blockdim)
    d_x = CuSparseMatrixBSR(d_x, blockdim)
    @test_throws ArgumentError copyto!(d_y,d_x)
    x = sprand(m,0.2)
    d_x = CuSparseVector(x)
    @test size(d_x, 1) == m
    @test size(d_x, 2) == 1
    @test_throws ArgumentError size(d_x, 0)
    y = sprand(n,0.2)
    d_y = CuSparseVector(y)
    @test_throws ArgumentError copyto!(d_y,d_x)
    x = sprand(m,m,0.2)
    d_x = Symmetric(CuSparseMatrixCSC(x + transpose(x)))
    @test issymmetric(d_x)
    x = sprand(ComplexF64, m, m, 0.2)
    d_x = Hermitian(CuSparseMatrixCSC(x + x'))
    @test ishermitian(d_x)
    x = sprand(m,m,0.2)
    d_x = UpperTriangular(CuSparseMatrixCSC(x))
    @test istriu(d_x)
    @test !istril(d_x)
    d_x = LowerTriangular(CuSparseMatrixCSC(x))
    @test !istriu(d_x)
    @test istril(d_x)
end

@testset "construction" begin
    @testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
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

        @testset "BSR" begin
            x = sprand(elty,m,n, 0.2)
            d_x  = CuSparseMatrixBSR(x, blockdim)
            @test collect(d_x) == collect(x)
        end

        @testset "BSR" begin
            x = sprand(elty,m,n, 0.2)
            d_x  = CuSparseMatrixCOO(x)
            @test collect(d_x) == collect(x)
        end
    end
end

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
            d_x = CuSparseMatrixBSR(d_x)
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

@testset "bsric02" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = rand(elty, m, m)
        A += adjoint(A)
        A += m * Diagonal{elty}(I, m)

        @testset "bsric02!" begin
            d_A = CuSparseMatrixCSR(sparse(tril(A)))
            d_A = CuSparseMatrixBSR(d_A, blockdim)
            d_A = CUSPARSE.ic02!(d_A,'O')
            h_A = SparseMatrixCSC(CuSparseMatrixCSR(d_A))
            Ac = sparse(Array(cholesky(Hermitian(A))))
            h_A = transpose(h_A) * h_A
            @test rowvals(h_A) ≈ rowvals(Ac)
            @test reduce(&, isfinite.(nonzeros(h_A)))
            d_A = CuSparseMatrixCSR(sparse(tril(rand(elty,m,n))))
            d_A = CuSparseMatrixBSR(d_A, blockdim)
            @test_throws DimensionMismatch CUSPARSE.ic02!(d_A,'O')
        end

        @testset "bsric02" begin
            d_A = CuSparseMatrixCSR(sparse(tril(A)))
            d_A = CuSparseMatrixBSR(d_A, blockdim)
            d_B = CUSPARSE.ic02(d_A,'O')
            h_A = SparseMatrixCSC(CuSparseMatrixCSR(d_B))
            Ac = sparse(Array(cholesky(Hermitian(A))))
            h_A = transpose(h_A) * h_A
            @test rowvals(h_A) ≈ rowvals(Ac)
            @test reduce(&, isfinite.(nonzeros(h_A)))
        end
    end
end

@testset "bsrilu02" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = rand(elty,m,m)
        A += transpose(A)
        A += m * Diagonal{elty}(I, m)
        @testset "bsrilu02!" begin
            d_A = CuSparseMatrixCSR(sparse(A))
            d_A = CuSparseMatrixBSR(d_A, blockdim)
            d_A = CUSPARSE.ilu02!(d_A,'O')
            h_A = SparseMatrixCSC(CuSparseMatrixCSR(d_A))
            Alu = lu(Array(A), Val(false))
            Ac = sparse(Alu.L*Alu.U)
            h_A = adjoint(h_A) * h_A
            @test rowvals(h_A) ≈ rowvals(Ac)
            @test reduce(&, isfinite.(nonzeros(h_A)))
            d_A = CuSparseMatrixCSR(sparse(rand(elty,m,n)))
            d_A = CuSparseMatrixBSR(d_A, blockdim)
            @test_throws DimensionMismatch CUSPARSE.ilu02!(d_A,'O')
        end

        @testset "bsrilu02" begin
            d_A = CuSparseMatrixCSR(sparse(A))
            d_A = CuSparseMatrixBSR(d_A, blockdim)
            d_B = CUSPARSE.ilu02(d_A,'O')
            h_A = SparseMatrixCSC(CuSparseMatrixCSR(d_B))
            Alu = lu(Array(A),Val(false))
            Ac = sparse(Alu.L*Alu.U)
            h_A = adjoint(h_A) * h_A
            @test rowvals(h_A) ≈ rowvals(Ac)
            @test reduce(&, isfinite.(nonzeros(h_A)))
        end
    end
end

@testset "bsrsv2" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        @testset "bsrsv2!" begin
            for unit_diag ∈ (false, true)
                diag = unit_diag ? 'U' : 'N'
                A = rand(elty,m,m)
                A = (unit_diag ? triu(A, 1) + I : triu(A))
                X = rand(elty,m)
                alpha = rand(elty)
                d_X = CuArray(X)
                d_A = CuSparseMatrixCSR(sparse(A))
                d_A = CuSparseMatrixBSR(d_A, blockdim)
                d_X = CUSPARSE.sv2!('N','U',diag,alpha,d_A,d_X,'O')
                h_Y = collect(d_X)
                Y = A\(alpha * X)
                @test Y ≈ h_Y
                d_X = CUDA.rand(elty,n)
                @test_throws DimensionMismatch CUSPARSE.sv2!('N','U',diag,alpha,d_A,d_X,'O')
                A = sparse(rand(elty,m,n))
                d_A = CuSparseMatrixCSR(A)
                d_A = CuSparseMatrixBSR(d_A, blockdim)
                @test_throws DimensionMismatch CUSPARSE.sv2!('N','U',diag,alpha,d_A,d_X,'O')
            end
        end

        @testset "bsrsv2" begin
            for unit_diag ∈ (false, true)
                diag = unit_diag ? 'U' : 'N'
                A = rand(elty,m,m)
                A = (unit_diag ? triu(A, 1) + I : triu(A))
                Al = (unit_diag ? tril(A, -1) + I : tril(A))
                X = rand(elty,m)
                alpha = rand(elty)
                d_X = CuArray(X)
                d_Al = CuSparseMatrixCSR(sparse(Al))
                d_Al = CuSparseMatrixBSR(d_Al, blockdim)
                d_Y = CUSPARSE.sv2('N','L',diag,alpha,d_Al,d_X,'O')
                h_Y = collect(d_Y)
                Y = Al\(alpha * X)
                @test Y ≈ h_Y
                d_A = CuSparseMatrixCSR(sparse(A))
                d_A = CuSparseMatrixBSR(d_A, blockdim)
                d_Y = CUSPARSE.sv2('N','U',diag,alpha,d_A,d_X,'O')
                h_Y = collect(d_Y)
                Y = A\(alpha * X)
                @test Y ≈ h_Y
                UA = (unit_diag ? UnitUpperTriangular(d_A) : UpperTriangular(d_A))
                d_Y = UA\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ A\X
                #=d_Y = UpperTriangular(d_A)'\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ A'\X=#
                d_Y = transpose(UA)\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ transpose(A)\X
                LA = (unit_diag ? UnitLowerTriangular(d_A) : LowerTriangular(d_A))
                d_Y = LA\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ Al\X
                #=d_Y = LowerTriangular(d_A)'\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ A'\X=#
                d_Y = transpose(LA)\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ transpose(Al)\X
                A = sparse(rand(elty,m,n))
                d_A = CuSparseMatrixCSR(A)
                d_A = CuSparseMatrixBSR(d_A, blockdim)
                @test_throws DimensionMismatch CUSPARSE.sv2('N','U',diag,alpha,d_A,d_X,'O')
            end
        end
    end
end

@testset "bsrsm2" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        @testset "bsrsm2!" begin
            for unit_diag ∈ (false, true)
                diag = unit_diag ? 'U' : 'N'
                A = rand(elty,m,m)
                A = (unit_diag ? triu(A, 1) + I : triu(A))
                X = rand(elty,m,m)
                alpha = rand(elty)
                d_X = CuArray(X)
                d_A = CuSparseMatrixCSR(sparse(A))
                d_A = CuSparseMatrixBSR(d_A, blockdim)
                d_X = CUSPARSE.sm2!('N','N','U',diag,alpha,d_A,d_X,'O')
                h_Y = collect(d_X)
                Y = A\(alpha * X)
                @test Y ≈ h_Y
                d_X = CUDA.rand(elty,n,n)
                @test_throws DimensionMismatch CUSPARSE.sm2!('N','N','U',diag,alpha,d_A,d_X,'O')
                A = sparse(rand(elty,m,n))
                d_A = CuSparseMatrixCSR(A)
                d_A = CuSparseMatrixBSR(d_A, blockdim)
                @test_throws DimensionMismatch CUSPARSE.sm2!('N','N','U',diag,alpha,d_A,d_X,'O')
            end
        end

        @testset "bsrsm2" begin
            for unit_diag ∈ (false, true)
                diag = unit_diag ? 'U' : 'N'
                A = rand(elty,m,m)
                A = (unit_diag ? triu(A, 1) + I : triu(A))
                Al = (unit_diag ? tril(A, -1) + I : tril(A))
                X = rand(elty,m, m)
                alpha = rand(elty)
                d_X = CuArray(X)
                d_Al = CuSparseMatrixCSR(sparse(Al))
                d_Al = CuSparseMatrixBSR(d_Al, blockdim)
                d_Y = CUSPARSE.sm2('N','N','L',diag,alpha,d_Al,d_X,'O')
                h_Y = collect(d_Y)
                Y = Al\(alpha * X)
                @test Y ≈ h_Y
                d_A = CuSparseMatrixCSR(sparse(A))
                d_A = CuSparseMatrixBSR(d_A, blockdim)
                d_Y = CUSPARSE.sm2('N','N','U',diag,alpha,d_A,d_X,'O')
                h_Y = collect(d_Y)
                Y = A\(alpha * X)
                @test Y ≈ h_Y
                UA = (unit_diag ? UnitUpperTriangular(d_A) : UpperTriangular(d_A))
                d_Y = UA\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ A\X
                #=d_Y = UpperTriangular(d_A)'\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ A'\X=#
                d_Y = transpose(UA)\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ transpose(A)\X
                LA = (unit_diag ? UnitLowerTriangular(d_A) : LowerTriangular(d_A))
                d_Y = LA\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ Al\X
                #=d_Y = LowerTriangular(d_A)'\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ A'\X=#
                d_Y = transpose(LA)\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ transpose(Al)\X
                A = sparse(rand(elty,m,n))
                d_A = CuSparseMatrixCSR(A)
                d_A = CuSparseMatrixBSR(d_A, blockdim)
                @test_throws DimensionMismatch CUSPARSE.sm2('N','N','U',diag,alpha,d_A,d_X,'O')
            end
        end
    end
end

@testset "ilu02" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        @testset "csr" begin
            A = rand(elty,m,m)
            A += transpose(A)
            A += m * Diagonal{elty}(I, m)
            d_A = CuSparseMatrixCSR(sparse(A))
            d_B = CUSPARSE.ilu02(d_A,'O')
            h_A = SparseMatrixCSC(d_B)
            Alu = lu(Array(A),Val(false))
            Ac = sparse(Alu.L*Alu.U)
            h_A = adjoint(h_A) * h_A
            @test rowvals(h_A) ≈ rowvals(Ac)
            @test reduce(&, isfinite.(nonzeros(h_A)))
        end
        @testset "csc" begin
            A = rand(elty,m,m)
            A += transpose(A)
            A += m * Diagonal{elty}(I, m)
            d_A = CuSparseMatrixCSC(sparse(A))
            d_B = CUSPARSE.ilu02(d_A,'O')
            h_A = SparseMatrixCSC(d_B)
            Alu = lu(Array(A),Val(false))
            Ac = sparse(Alu.L*Alu.U)
            h_A = adjoint(h_A) * h_A
            @test rowvals(h_A) ≈ rowvals(Ac)
            @test reduce(&, isfinite.(nonzeros(h_A)))
        end
    end
end

@testset "ic2" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        @testset "csr" begin
            A   = rand(elty, m, m)
            A  += adjoint(A)
            A  += m * Diagonal{elty}(I, m)
            d_A = CuSparseMatrixCSR(sparse(tril(A)))
            d_B = CUSPARSE.ic02(d_A, 'O')
            h_A = SparseMatrixCSC(d_B)
            Ac  = sparse(Array(cholesky(Hermitian(A))))
            h_A = transpose(h_A) * h_A
            @test rowvals(h_A) ≈ rowvals(Ac)
            @test reduce(&, isfinite.(nonzeros(h_A)))
            A   = rand(elty,m,n)
            d_A = CuSparseMatrixCSR(sparse(tril(A)))
            @test_throws DimensionMismatch CUSPARSE.ic02(d_A, 'O')
        end
        @testset "csc" begin
            A   = rand(elty, m, m)
            A  += adjoint(A)
            A  += m * Diagonal{elty}(I, m)
            d_A = CuSparseMatrixCSC(sparse(tril(A)))
            d_B = CUSPARSE.ic02(d_A, 'O')
            h_A = SparseMatrixCSC(d_B)
            Ac  = sparse(Array(cholesky(Hermitian(A))))
            h_A = transpose(h_A) * h_A
            @test rowvals(h_A) ≈ rowvals(Ac)
            @test reduce(&, isfinite.(nonzeros(h_A)))
            A   = rand(elty,m,n)
            d_A = CuSparseMatrixCSC(sparse(tril(A)))
            @test_throws DimensionMismatch CUSPARSE.ic02(d_A, 'O')
        end
    end
end

@testset "cssv" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        @testset "csrsv2" begin
            for unit_diag ∈ (false, true)
                diag = unit_diag ? 'U' : 'N'
                A = rand(elty,m,m)
                A = (unit_diag ? triu(A, 1) + I : triu(A))
                Al = (unit_diag ? tril(A, -1) + I : tril(A))
                X = rand(elty,m)
                alpha = rand(elty)
                d_X = CuArray(X)
                d_Al = CuSparseMatrixCSR(sparse(Al))
                d_Y = CUSPARSE.sv2('N','L',diag,alpha,d_Al,d_X,'O')
                h_Y = collect(d_Y)
                Y = Al\(alpha * X)
                @test Y ≈ h_Y
                d_Y = CUSPARSE.sv2('T','L',diag,alpha,d_Al,d_X,'O')
                h_Y = collect(d_Y)
                Y = transpose(Al)\(alpha * X)
                @test Y ≈ h_Y
                d_A = CuSparseMatrixCSR(sparse(A))
                d_Y = CUSPARSE.sv2('N','U',diag,alpha,d_A,d_X,'O')
                h_Y = collect(d_Y)
                Y = A\(alpha * X)
                @test Y ≈ h_Y
                d_Y = CUSPARSE.sv2('T','U',diag,alpha,d_A,d_X,'O')
                h_Y = collect(d_Y)
                Y = transpose(A)\(alpha * X)
                UA = (unit_diag ? UnitUpperTriangular(d_A) : UpperTriangular(d_A))
                d_y = UA\d_X
                h_y = collect(d_y)
                y = A\X
                @test y ≈ h_y
                d_y = transpose(UA)\d_X
                h_y = collect(d_y)
                y = transpose(A)\X
                @test y ≈ h_y
                #=d_y = UpperTriangular(d_A)'\d_X
                h_y = collect(d_y)
                y = A'\X
                @test y ≈ h_y=#
                LA = (unit_diag ? UnitLowerTriangular(d_A) : LowerTriangular(d_A))
                d_y = LA\d_X
                h_y = collect(d_y)
                y = Al\X
                @test y ≈ h_y
                d_y = transpose(LA)\d_X
                h_y = collect(d_y)
                y = transpose(Al)\X
                @test y ≈ h_y
                #=d_y = LowerTriangular(d_A)'\d_X
                h_y = collect(d_y)
                y = A'\X
                @test y ≈ h_y=#
                A = sparse(rand(elty,m,n))
                d_A = CuSparseMatrixCSR(A)
                @test_throws DimensionMismatch CUSPARSE.sv2('N','U',diag,alpha,d_A,d_X,'O')
            end
        end

        @testset "cscsv2" begin
            for unit_diag ∈ (false, true)
                diag = unit_diag ? 'U' : 'N'
                A = rand(elty,m,m)
                A = (unit_diag ? triu(A, 1) + I : triu(A))
                Al = (unit_diag ? tril(A, -1) + I : tril(A))
                X = rand(elty,m)
                alpha = rand(elty)
                d_X = CuArray(X)
                d_Al = CuSparseMatrixCSC(sparse(Al))
                d_Y = CUSPARSE.sv2('N','L',diag,alpha,d_Al,d_X,'O')
                h_Y = collect(d_Y)
                Y = Al\(alpha * X)
                @test Y ≈ h_Y
                d_Y = CUSPARSE.sv2('T','L',diag,alpha,d_Al,d_X,'O')
                h_Y = collect(d_Y)
                Y = transpose(Al)\(alpha * X)
                @test Y ≈ h_Y
                d_A = CuSparseMatrixCSC(sparse(A))
                d_Y = CUSPARSE.sv2('N','U',diag,alpha,d_A,d_X,'O')
                h_Y = collect(d_Y)
                Y = A\(alpha * X)
                @test Y ≈ h_Y
                d_Y = CUSPARSE.sv2('T','U',diag,alpha,d_A,d_X,'O')
                h_Y = collect(d_Y)
                Y = transpose(A)\(alpha * X)
                @test Y ≈ h_Y
                UA = (unit_diag ? UnitUpperTriangular(d_A) : UpperTriangular(d_A))
                d_y = UA\d_X
                h_y = collect(d_y)
                y = A\X
                @test y ≈ h_y
                d_y = transpose(UA)\d_X
                h_y = collect(d_y)
                y = transpose(A)\X
                @test y ≈ h_y
                LA = (unit_diag ? UnitLowerTriangular(d_A) : LowerTriangular(d_A))
                d_y = LA\d_X
                h_y = collect(d_y)
                y = Al\X
                @test y ≈ h_y
                d_y = transpose(LA)\d_X
                h_y = collect(d_y)
                y = transpose(Al)\X
                @test y ≈ h_y
                #=d_y = UpperTriangular(d_A)'\d_X
                h_y = collect(d_y)
                y = A'\X
                @test y ≈ h_y=#
                # shouldn't work for now bc sv2 has no way to do conj...
                A = sparse(rand(elty,m,n))
                d_A = CuSparseMatrixCSC(A)
                @test_throws DimensionMismatch CUSPARSE.sv2('N','U',diag,alpha,d_A,d_X,'O')
            end
        end
    end
end

@testset "cssm" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        @testset "csrsm2" begin
            for unit_diag ∈ (false, true)
                diag = unit_diag ? 'U' : 'N'
                A = rand(elty,m,m)
                A = (unit_diag ? triu(A, 1) + I : triu(A))
                Al = (unit_diag ? tril(A, -1) + I : tril(A))
                X = rand(elty,m,m)
                alpha = rand(elty)
                d_X = CuArray(X)
                d_Al = CuSparseMatrixCSR(sparse(Al))
                d_Y = CUSPARSE.sm2('N','N','L',diag,alpha,d_Al,d_X,'O')
                h_Y = collect(d_Y)
                Y = Al\(alpha * X)
                @test Y ≈ h_Y
                d_Y = CUSPARSE.sm2('T','N','L',diag,alpha,d_Al,d_X,'O')
                h_Y = collect(d_Y)
                Y = transpose(Al)\(alpha * X)
                @test Y ≈ h_Y
                d_A = CuSparseMatrixCSR(sparse(A))
                d_Y = CUSPARSE.sm2('N','N','U',diag,alpha,d_A,d_X,'O')
                h_Y = collect(d_Y)
                Y = A\(alpha * X)
                @test Y ≈ h_Y
                d_Y = CUSPARSE.sm2('T','N','U',diag,alpha,d_A,d_X,'O')
                h_Y = collect(d_Y)
                Y = transpose(A)\(alpha * X)
                @test Y ≈ h_Y
                UA = (unit_diag ? UnitUpperTriangular(d_A) : UpperTriangular(d_A))
                d_y = UA\d_X
                h_y = collect(d_y)
                y = A\X
                @test y ≈ h_y
                d_y = transpose(UA)\d_X
                h_y = collect(d_y)
                y = transpose(A)\X
                @test y ≈ h_y
                #=d_y = UpperTriangular(d_A)'\d_X
                h_y = collect(d_y)
                y = A'\X
                @test y ≈ h_y=#
                LA = (unit_diag ? UnitLowerTriangular(d_A) : LowerTriangular(d_A))
                d_y = LA\d_X
                h_y = collect(d_y)
                y = Al\X
                @test y ≈ h_y
                d_y = transpose(LA)\d_X
                h_y = collect(d_y)
                y = transpose(Al)\X
                @test y ≈ h_y
                #=d_y = LowerTriangular(d_A)'\d_X
                h_y = collect(d_y)
                y = A'\X
                @test y ≈ h_y=#
                A = sparse(rand(elty,m,n))
                d_A = CuSparseMatrixCSR(A)
                @test_throws DimensionMismatch CUSPARSE.sm2('N','N','U',diag,alpha,d_A,d_X,'O')
            end
        end

        @testset "cscsm2" begin
            for unit_diag ∈ (false, true)
                diag = unit_diag ? 'U' : 'N'
                A = rand(elty,m,m)
                A = (unit_diag ? triu(A, 1) + I : triu(A))
                Al = (unit_diag ? tril(A, -1) + I : tril(A))
                X = rand(elty,m,m)
                alpha = rand(elty)
                d_X = CuArray(X)
                d_Al = CuSparseMatrixCSC(sparse(Al))
                d_Y = CUSPARSE.sm2('N','N','L',diag,alpha,d_Al,d_X,'O')
                h_Y = collect(d_Y)
                Y = Al\(alpha * X)
                @test Y ≈ h_Y
                d_Y = CUSPARSE.sm2('T','N','L',diag,alpha,d_Al,d_X,'O')
                h_Y = collect(d_Y)
                Y = transpose(Al)\(alpha * X)
                @test Y ≈ h_Y
                d_A = CuSparseMatrixCSC(sparse(A))
                d_Y = CUSPARSE.sm2('N','N','U',diag,alpha,d_A,d_X,'O')
                h_Y = collect(d_Y)
                Y = A\(alpha * X)
                @test Y ≈ h_Y
                d_Y = CUSPARSE.sm2('T','N','U',diag,alpha,d_A,d_X,'O')
                h_Y = collect(d_Y)
                Y = transpose(A)\(alpha * X)
                @test Y ≈ h_Y
                UA = (unit_diag ? UnitUpperTriangular(d_A) : UpperTriangular(d_A))
                d_y = UA\d_X
                h_y = collect(d_y)
                y = A\X
                @test y ≈ h_y
                d_y = transpose(UA)\d_X
                h_y = collect(d_y)
                y = transpose(A)\X
                @test y ≈ h_y
                LA = (unit_diag ? UnitLowerTriangular(d_A) : LowerTriangular(d_A))
                d_y = LA\d_X
                h_y = collect(d_y)
                y = Al\X
                @test y ≈ h_y
                d_y = transpose(LA)\d_X
                h_y = collect(d_y)
                y = transpose(Al)\X
                @test y ≈ h_y
                #=d_y = UpperTriangular(d_A)'\d_X
                h_y = collect(d_y)
                y = A'\X
                @test y ≈ h_y=#
                # shouldn't work for now bc sv2 has no way to do conj...
                A = sparse(rand(elty,m,n))
                d_A = CuSparseMatrixCSC(A)
                @test_throws DimensionMismatch CUSPARSE.sm2('N','N','U',diag,alpha,d_A,d_X,'O')
            end
        end
    end
end

@testset "axpyi" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        @testset "axpyi!" begin
            x = sparsevec(rand(1:m,k), rand(elty,k), m)
            y = rand(elty,m)
            d_x = CuSparseVector(x)
            d_y = CuArray(y)
            alpha = rand(elty)
            d_y = CUSPARSE.axpyi!(alpha,d_x,d_y,'O')
            #compare
            h_y = collect(d_y)
            y[nonzeroinds(x)] += alpha * nonzeros(x)
            @test h_y ≈ y
        end

        @testset "axpyi" begin
            x = sparsevec(rand(1:m,k), rand(elty,k), m)
            y = rand(elty,m)
            d_x = CuSparseVector(x)
            d_y = CuArray(y)
            alpha = rand(elty)
            d_z = CUSPARSE.axpyi(alpha,d_x,d_y,'O')
            #compare
            h_z = collect(d_z)
            z = copy(y)
            z[nonzeroinds(x)] += alpha * nonzeros(x)
            @test h_z ≈ z
            d_z = CUSPARSE.axpyi(d_x,d_y,'O')
            #compare
            h_z = collect(d_z)
            z = copy(y)
            z[nonzeroinds(x)] += nonzeros(x)
            @test h_z ≈ z
        end
    end
end

@testset "gthr and gthrz" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        x = sparsevec(rand(1:m,k), rand(elty,k), m)
        y = rand(elty,m)
        @testset "gthr!" begin
            d_x = CuSparseVector(x)
            d_y = CuArray(y)
            d_y = CUSPARSE.gthr!(d_x,d_y,'O')
            h_x = collect(d_x)
            @test h_x ≈ SparseVector(m,nonzeroinds(x),y[nonzeroinds(x)])
        end

        @testset "gthr" begin
            d_x = CuSparseVector(x)
            d_y = CuArray(y)
            d_z = CUSPARSE.gthr(d_x,d_y,'O')
            h_z = collect(d_z)
            @test h_z ≈ SparseVector(m,nonzeroinds(x),y[nonzeroinds(x)])
        end

        @testset "gthrz!" begin
            d_x = CuSparseVector(x)
            d_y = CuArray(y)
            d_x,d_y = CUSPARSE.gthrz!(d_x,d_y,'O')
            h_x = collect(d_x)
            h_y = collect(d_y)
            @test h_x ≈ SparseVector(m,nonzeroinds(x),y[nonzeroinds(x)])
            #y[nonzeroinds(x)] = zero(elty)
            #@test h_y ≈ y
        end

        @testset "gthrz" begin
            d_x = CuSparseVector(x)
            d_y = CuArray(y)
            d_z,d_w = CUSPARSE.gthrz(d_x,d_y,'O')
            h_w = collect(d_w)
            h_z = collect(d_z)
            @test h_z ≈ SparseVector(m,nonzeroinds(x),y[nonzeroinds(x)])
            #y[nonzeroinds(x)] = zero(elty)
            #@test h_w ≈ y
        end
    end
end

@testset "mv!" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = sparse(rand(elty,m,n))
        x = rand(elty,n)
        y = rand(elty,m)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "$(typeof(d_A))" for d_A in [CuSparseMatrixCSR(A),
                                              CuSparseMatrixCSC(A),
                                              CuSparseMatrixBSR(A, blockdim)]
            d_x = CuArray(x)
            d_y = CuArray(y)
            @test_throws DimensionMismatch CUSPARSE.mv!('T',alpha,d_A,d_x,beta,d_y,'O')
            @test_throws DimensionMismatch CUSPARSE.mv!('N',alpha,d_A,d_y,beta,d_x,'O')
            CUSPARSE.mv!('N',alpha,d_A,d_x,beta,d_y,'O')
            h_z = collect(d_y)
            z = alpha * A * x + beta * y
            @test z ≈ h_z
            if d_A isa CuSparseMatrixCSR
                @test d_y' * (d_A * d_x) ≈ (d_y' * d_A) * d_x
            end
        end
    end
end

@testset "mm!" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = sparse(rand(elty,m,k))
        B = rand(elty,k,n)
        C = rand(elty,m,n)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "$(typeof(d_A))" for d_A in [CuSparseMatrixCSR(A),
                                              CuSparseMatrixCSC(A),
                                              CuSparseMatrixBSR(A, blockdim)]
            d_B = CuArray(B)
            d_C = CuArray(C)
            mm! = if CUSPARSE.version() < v"10.3.1" && d_A isa CuSparseMatrixCSR
                CUSPARSE.mm2!
            else
                CUSPARSE.mm!
            end
            @test_throws DimensionMismatch mm!('N','T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch mm!('T','N',alpha,d_A,d_B,beta,d_C,'O')
            if CUSPARSE.version() < v"10.3.1" && d_A isa CuSparseMatrixCSR
                # A^T is not supported with B^T
                @test_throws ArgumentError mm!('T','T',alpha,d_A,d_B,beta,d_C,'O')
            else
                @test_throws DimensionMismatch mm!('T','T',alpha,d_A,d_B,beta,d_C,'O')
            end
            @test_throws DimensionMismatch mm!('N','N',alpha,d_A,d_B,beta,d_B,'O')
            mm!('N','N',alpha,d_A,d_B,beta,d_C,'O')
            h_D = collect(d_C)
            D = alpha * A * B + beta * C
            @test D ≈ h_D
        end
    end

    @testset "issue 493" begin
        x = cu(rand(20))
        cu(sprand(Float32,10,10,0.1)) * @view(x[1:10])
    end
end

@testset "sctr" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        x = sparsevec(rand(1:m,k), rand(elty,k), m)
        y = zeros(elty,m)
        @testset "sctr!" begin
            d_x = CuSparseVector(x)
            d_y = CuArray(y)
            d_y = CUSPARSE.sctr!(d_x,d_y,'O')
            h_y = collect(d_y)
            y[nonzeroinds(x)]  += nonzeros(x)
            @test h_y ≈ y
        end
        y = zeros(elty,m)

        @testset "sctr" begin
            d_x = CuSparseVector(x)
            d_y = CUSPARSE.sctr(d_x,'O')
            h_y = collect(d_y)
            y = zeros(elty,m)
            y[nonzeroinds(x)]  += nonzeros(x)
            @test h_y ≈ y
        end
    end
end

@testset "roti" begin
    @testset for elty in [Float32,Float64]
        x = sparsevec(rand(1:m,k), rand(elty,k), m)
        y = rand(elty,m)
        @testset "roti!" begin
            d_x = CuSparseVector(x)
            d_y = CuArray(y)
            angle = rand(elty)
            d_x,d_y = CUSPARSE.roti!(d_x,d_y,cos(angle),sin(angle),'O')
            h_x = collect(d_x)
            h_y = collect(d_y)
            z = copy(x)
            w = copy(y)
            y[nonzeroinds(x)] = cos(angle)*w[nonzeroinds(z)] - sin(angle)*nonzeros(z)
            @test h_x ≈ SparseVector(m,nonzeroinds(x),cos(angle)*nonzeros(z) + sin(angle)*w[nonzeroinds(z)])
            @test h_y ≈ y
        end

        @testset "roti" begin
            d_x = CuSparseVector(x)
            d_y = CuArray(y)
            angle = rand(elty)
            d_z,d_w = CUSPARSE.roti(d_x,d_y,cos(angle),sin(angle),'O')
            h_w = collect(d_w)
            h_z = collect(d_z)
            z = copy(x)
            w = copy(y)
            w[nonzeroinds(z)] = cos(angle)*y[nonzeroinds(x)] - sin(angle)*nonzeros(x)
            @test h_z ≈ SparseVector(m,nonzeroinds(z), cos(angle)*nonzeros(x) + sin(angle)*y[nonzeroinds(x)])
            @test h_w ≈ w
        end
    end
end
