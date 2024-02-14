using CUDA.CUSPARSE

using LinearAlgebra
using SparseArrays
using SparseArrays: nonzeroinds, getcolptr

@test CUSPARSE.version() isa VersionNumber

m = 25
n = 35
k = 10
p = 5
blockdim = 5

@testset "array" begin
    x = sprand(m,0.2)
    d_x = CuSparseVector(x)
    @test length(d_x) == m
    @test size(d_x)   == (m,)
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

    A = sprand(n, n, 0.2)
    d_A = CuSparseMatrixCSC(A)
    @test Array(getcolptr(d_A)) == getcolptr(A)
    i, j, v = findnz(A)
    d_i, d_j, d_v = findnz(d_A)
    @test Array(d_i) == i && Array(d_j) == j && Array(d_v) == v
    i = unique(sort(rand(1:n, 10)))
    vals = rand(length(i))
    d_i = CuArray(i)
    d_vals = CuArray(vals)
    v = sparsevec(i, vals, n)
    d_v = sparsevec(d_i, d_vals, n)
    @test Array(d_v.iPtr) == v.nzind
    @test Array(d_v.nzVal) == v.nzval
    @test d_v.len == v.n
end

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
            @test similar(d_x, Float32) isa CuSparseMatrixCSC{Float32}
        end

        @testset "CSR" begin
            x = sprand(elty,m,n, 0.2)
            d_x  = CuSparseMatrixCSR(x)
            @test collect(d_x) == collect(x)
            @test similar(d_x) isa CuSparseMatrixCSR{elty}
            @test similar(d_x, Float32) isa CuSparseMatrixCSR{Float32}
        end

        @testset "BSR" begin
            x = sprand(elty,m,n, 0.2)
            d_x  = CuSparseMatrixBSR(x, blockdim)
            @test collect(d_x) == collect(x)
            @test similar(d_x) isa CuSparseMatrixBSR{elty}
            @test similar(d_x, Float32) isa CuSparseMatrixBSR{Float32}
        end

        @testset "BSR" begin
            x = sprand(elty,m,n, 0.2)
            d_x  = CuSparseMatrixCOO(x)
            @test collect(d_x) == collect(x)
            @test similar(d_x) isa CuSparseMatrixCOO{elty}
            @test similar(d_x, Float32) isa CuSparseMatrixCOO{Float32}
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


@testset "bsric02" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = rand(elty, m, m)
        A += adjoint(A)
        A += m * Diagonal{elty}(I, m)

        @testset "bsric02!" begin
            d_A = CuSparseMatrixCSR(sparse(tril(A)))
            d_A = CuSparseMatrixBSR(d_A, blockdim)
            d_A = CUSPARSE.ic02!(d_A)
            h_A = SparseMatrixCSC(CuSparseMatrixCSR(d_A))
            Ac = sparse(Array(cholesky(Hermitian(A))))
            h_A = transpose(h_A) * h_A
            @test rowvals(h_A) ≈ rowvals(Ac)
            @test reduce(&, isfinite.(nonzeros(h_A)))
            d_A = CuSparseMatrixCSR(sparse(tril(rand(elty,m,n))))
            d_A = CuSparseMatrixBSR(d_A, blockdim)
            @test_throws DimensionMismatch CUSPARSE.ic02!(d_A)
        end

        @testset "bsric02" begin
            d_A = CuSparseMatrixCSR(sparse(tril(A)))
            d_A = CuSparseMatrixBSR(d_A, blockdim)
            d_B = CUSPARSE.ic02(d_A)
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
            d_A = CUSPARSE.ilu02!(d_A)
            h_A = SparseMatrixCSC(CuSparseMatrixCSR(d_A))
            pivot = NoPivot()
            Alu = lu(Array(A), pivot)
            Ac = sparse(Alu.L*Alu.U)
            h_A = adjoint(h_A) * h_A
            @test rowvals(h_A) ≈ rowvals(Ac)
            @test reduce(&, isfinite.(nonzeros(h_A)))
            d_A = CuSparseMatrixCSR(sparse(rand(elty,m,n)))
            d_A = CuSparseMatrixBSR(d_A, blockdim)
            @test_throws DimensionMismatch CUSPARSE.ilu02!(d_A)
        end

        @testset "bsrilu02" begin
            d_A = CuSparseMatrixCSR(sparse(A))
            d_A = CuSparseMatrixBSR(d_A, blockdim)
            d_B = CUSPARSE.ilu02(d_A)
            h_A = SparseMatrixCSC(CuSparseMatrixCSR(d_B))
            pivot = NoPivot()
            Alu = lu(Array(A),pivot)
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
            d_B = CUSPARSE.ilu02(d_A)
            h_A = SparseMatrixCSC(d_B)
            pivot = NoPivot()
            Alu = lu(Array(A),pivot)
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
            d_B = CUSPARSE.ilu02(d_A)
            h_A = SparseMatrixCSC(d_B)
            pivot = NoPivot()
            Alu = lu(Array(A),pivot)
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
            d_B = CUSPARSE.ic02(d_A)
            h_A = SparseMatrixCSC(d_B)
            Ac  = sparse(Array(cholesky(Hermitian(A))))
            h_A = transpose(h_A) * h_A
            @test rowvals(h_A) ≈ rowvals(Ac)
            @test reduce(&, isfinite.(nonzeros(h_A)))
            A   = rand(elty,m,n)
            d_A = CuSparseMatrixCSR(sparse(tril(A)))
            @test_throws DimensionMismatch CUSPARSE.ic02(d_A)
        end
        @testset "csc" begin
            A   = rand(elty, m, m)
            A  += adjoint(A)
            A  += m * Diagonal{elty}(I, m)
            d_A = CuSparseMatrixCSC(sparse(tril(A)))
            d_B = CUSPARSE.ic02(d_A)
            h_A = SparseMatrixCSC(d_B)
            Ac  = sparse(Array(cholesky(Hermitian(A))))
            h_A = transpose(h_A) * h_A
            @test rowvals(h_A) ≈ rowvals(Ac)
            @test reduce(&, isfinite.(nonzeros(h_A)))
            A   = rand(elty,m,n)
            d_A = CuSparseMatrixCSC(sparse(tril(A)))
            @test_throws DimensionMismatch CUSPARSE.ic02(d_A)
        end
    end
end

if CUSPARSE.version() < v"12.0"
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
            #if d_A isa CuSparseMatrixCSR
            #    @test d_y' * (d_A * d_x) ≈ (d_y' * d_A) * d_x
            #end
        end
    end

    if capability(device()) >= v"5.3"
    @testset for elty in [Float16,ComplexF16]
        A = sparse(rand(elty,m,n))
        x = rand(elty,n)
        y = rand(elty,m)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "$(typeof(d_A))" for d_A in [CuSparseMatrixCSR(A),
                                              CuSparseMatrixCSC(A)]
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
            @test_throws DimensionMismatch CUSPARSE.mm!('N','T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm!('T','N',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm!('T','T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm!('N','N',alpha,d_A,d_B,beta,d_B,'O')
            CUSPARSE.mm!('N','N',alpha,d_A,d_B,beta,d_C,'O')
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

@testset "gemvi!" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
            elty <: Complex && transa == 'C' && continue
            A = transa == 'N' ? rand(elty,m,n) : rand(elty,n,m)
            x = sparsevec(rand(1:n,k), rand(elty,k), n)
            y = rand(elty,m)
            dA = CuArray(A)
            dx = CuSparseVector(x)
            dy = CuArray(y)
            alpha = rand(elty)
            beta = rand(elty)
            gemvi!(transa,alpha,dA,dx,beta,dy,'O')
            z = alpha * opa(A) * x + beta * y
            @test z ≈ collect(dy)
        end
    end
end

for SparseMatrixType in [CuSparseMatrixCSC, CuSparseMatrixCSR]
    @testset "$SparseMatrixType -- color" begin
        @testset "color $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
            A = sprand(T, 50, 50, 0.03)
            A = A + A'
            dA = SparseMatrixType(A)
            ncolors, coloring, reordering = color(dA, 'O')
            @test 1 ≤ ncolors ≤ 50
            @test maximum(coloring) == ncolors
            @test minimum(reordering) == 1
            @test maximum(reordering) == 50
            @test CUDA.@allowscalar isperm(reordering)
        end

        # The routine color returns the wrong result with small sparse matrices.
        # NVIDIA investigates the issue.
        if false
            A = [ 1 1 0 0 0 ;
                  1 1 1 1 0 ;
                  0 1 1 0 1 ;
                  0 1 0 1 0 ;
                  0 0 1 0 1 ]
            A = sparse(A)
            # The adjacency graph of A has at least two colors, one color for
            # {1, 3, 4} and another one for {2, 5}.
            @testset "5x5 example -- color $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
                dA = SparseMatrixType{T}(A)
                ncolors, coloring, reordering = color(dA, 'O')
                @test ncolors == 2
                @test minimum(reordering) == 1
                @test maximum(reordering) == 5
                CUDA.allowscalar() do
                    @test isperm(reordering)
                    @test coloring[1] == coloring[3]  == coloring[4]
                    @test coloring[2] == coloring[5]
                    @test coloring[1] != coloring[2]
                end
            end
        end
    end
end

@testset "gtsv2" begin
    dl1 = [0; 1; 3]
    d1 = [1; 1; 4]
    du1 = [1; 2; 0]
    B1 = [1 0 0; 0 1 0; 0 0 1]
    X1 = [1/3 2/3 -1/3; 2/3 -2/3 1/3; -1/2 1/2 0]

    dl2 = [0; 1; 1; 1; 1; 1; 0]
    d2 = [6; 4; 4; 4; 4; 4; 6]
    du2 = [0; 1; 1; 1; 1; 1; 0]
    B2 = [0; 1; 2; -6; 2; 1; 0]
    X2 = [0; 0; 1; -2; 1; 0; 0]

    dl3 = [0; 1; 1; 7; 6; 3; 8; 6; 5; 4]
    d3 = [2; 3; 3; 2; 2; 4; 1; 2; 4; 5]
    du3 = [1; 2; 1; 6; 1; 3; 5; 7; 3; 0]
    B3 = [1; 2; 6; 34; 10; 1; 4; 22; 25; 3]
    X3 = [1; -1; 2; 1; 3; -2; 0; 4; 2; -1]
    for pivoting ∈ (false, true)
        @testset "gtsv2 with pivoting=$pivoting -- $elty" for elty in [Float32,Float64,ComplexF32,ComplexF64]
            @testset "example 1" begin
                dl1_d = CuVector{elty}(dl1)
                d1_d = CuVector{elty}(d1)
                du1_d = CuVector{elty}(du1)
                B1_d = CuArray{elty}(B1)
                X1_d = gtsv2(dl1_d, d1_d, du1_d, B1_d; pivoting)
                @test collect(X1_d) ≈ X1
                gtsv2!(dl1_d, d1_d, du1_d, B1_d; pivoting)
                @test collect(B1_d) ≈ X1
            end
            @testset "example 2" begin
                dl2_d = CuVector{elty}(dl2)
                d2_d = CuVector{elty}(d2)
                du2_d = CuVector{elty}(du2)
                B2_d = CuArray{elty}(B2)
                X2_d = gtsv2(dl2_d, d2_d, du2_d, B2_d; pivoting)
                @test collect(X2_d) ≈ X2
                gtsv2!(dl2_d, d2_d, du2_d, B2_d; pivoting)
                @test collect(B2_d) ≈ X2
            end
            @testset "example 3" begin
                dl3_d = CuVector{elty}(dl3)
                d3_d = CuVector{elty}(d3)
                du3_d = CuVector{elty}(du3)
                B3_d = CuArray{elty}(B3)
                X3_d = gtsv2(dl3_d, d3_d, du3_d, B3_d; pivoting)
                @test collect(X3_d) ≈ X3
                gtsv2!(dl3_d, d3_d, du3_d, B3_d; pivoting)
                @test collect(B3_d) ≈ X3
            end
        end
    end
end

@testset "Triangular solves" begin
    @testset "$SparseMatrixType" for SparseMatrixType in (CuSparseMatrixCSR, CuSparseMatrixCSC, CuSparseMatrixBSR)
        (SparseMatrixType ∈ (CuSparseMatrixCSR, CuSparseMatrixCSC)) && (CUSPARSE.version() ≥ v"12.0") && continue
        @testset "y = T \\ x -- $elty" for elty in (Float32, Float64, ComplexF32, ComplexF64)
            for (trans, op) in (('N', identity), ('T', transpose), ('C', adjoint))
                (SparseMatrixType == CuSparseMatrixCSC) && (trans == 'C') && (elty <: Complex) && continue
                for uplo in ('L', 'U')
                    for diag in ('N', 'U')
                        @testset "trans = $trans | uplo = $uplo | diag = $diag" begin
                            T = rand(elty,n,n)
                            T = uplo == 'L' ? tril(T) : triu(T)
                            T = diag == 'N' ? T : T - Diagonal(T) + I
                            T = sparse(T)
                            d_T = SparseMatrixType == CuSparseMatrixBSR ? SparseMatrixType(CuSparseMatrixCSR(T), blockdim) : SparseMatrixType(T)
                            x = rand(elty,n)
                            d_x = CuVector{elty}(x)
                            d_y = CUSPARSE.sv2(trans, uplo, diag, d_T, d_x, 'O')
                            y = op(T) \ x
                            @test collect(d_y) ≈ y
                        end
                    end
                end
            end
        end

        @testset "Y = T \\ X -- $elty" for elty in (Float32, Float64, ComplexF32, ComplexF64)
            for (transT, opT) in (('N', identity), ('T', transpose), ('C', adjoint))
                (SparseMatrixType == CuSparseMatrixCSC) && (transT == 'C') && (elty <: Complex) && continue
                for (transX, opX) in (('N', identity), ('T', transpose))
                    for uplo in ('L', 'U')
                        for diag in ('N', 'U')
                            @testset "transT = $transT | transX = $transX | uplo = $uplo | diag = $diag" begin
                                T = rand(elty,n,n)
                                T = uplo == 'L' ? tril(T) : triu(T)
                                T = diag == 'N' ? T : T - Diagonal(T) + I
                                T = sparse(T)
                                d_T = SparseMatrixType == CuSparseMatrixBSR ? SparseMatrixType(CuSparseMatrixCSR(T), blockdim) : SparseMatrixType(T)
                                X = transX == 'N' ? rand(elty,n,p) : rand(elty,p,n)
                                d_X = CuMatrix{elty}(X)
                                d_Y = CUSPARSE.sm2(transT, transX, uplo, diag, d_T, d_X, 'O')
                                Y = opT(T) \ opX(X)
                                @test collect(d_Y) ≈ (transX == 'N' ? Y : transpose(Y))
                            end
                        end
                    end
                end
            end
        end
    end
end

@testset "duplicate entries" begin
    # already sorted
    let
        I = [1, 3, 4, 4]
        J = [1, 2, 3, 3]
        V = [1f0, 2f0, 3f0, 10f0]
        coo = sparse(cu(I), cu(J), cu(V); fmt=:coo)
        @test Array(coo.rowInd) == [1, 3, 4]
        @test Array(coo.colInd) == [1, 2, 3]
        @test Array(coo.nzVal) == [1f0, 2f0, 13f0]
    end

    # out of order
    let
        I = [4, 1, 3, 4]
        J = [3, 1, 2, 3]
        V = [10f0, 1f0, 2f0, 3f0]
        coo = sparse(cu(I), cu(J), cu(V); fmt=:coo)
        @test Array(coo.rowInd) == [1, 3, 4]
        @test Array(coo.colInd) == [1, 2, 3]
        @test Array(coo.nzVal) == [1f0, 2f0, 13f0]
    end
end
