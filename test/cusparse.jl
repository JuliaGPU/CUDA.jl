using CUDA.CUSPARSE

using LinearAlgebra
using SparseArrays

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
    @test Array(SparseArrays.nonzeroinds(d_x)) == SparseArrays.nonzeroinds(x)
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
    d_y = CUSPARSE.switch2csr(d_y)
    d_x = CUSPARSE.switch2csr(d_x)
    @test_throws ArgumentError copyto!(d_y,d_x)
    d_y = CUSPARSE.switch2bsr(d_y,convert(Cint,blockdim))
    d_x = CUSPARSE.switch2bsr(d_x,convert(Cint,blockdim))
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

@testset "conversion" begin
    @testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
        @testset "make_csc" begin
            x = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCSC(x)
            h_x = collect(d_x)
            @test h_x == x
            @test eltype(d_x) == elty
        end

        @testset "make_csr" begin
            x = sprand(elty,m,n, 0.2)
            d_xc = CuSparseMatrixCSC(x)
            d_x  = CuSparseMatrixCSR(x)
            h_x = collect(d_x)
            @test h_x == x
        end

        @testset "convert_r2c" begin
            x = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCSR(x)
            d_x = CUSPARSE.switch2csc(d_x)
            h_x = collect(d_x)
            @test h_x.rowval == x.rowval
            @test h_x.nzval ≈ x.nzval
        end

        @testset "convert_r2b" begin
            x = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCSR(x)
            d_x = CUSPARSE.switch2bsr(d_x,convert(Cint,blockdim))
            d_x = CUSPARSE.switch2csr(d_x)
            h_x = collect(d_x)
            @test h_x ≈ x
        end

        @testset "convert_c2b" begin
            x = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCSC(x)
            d_x = CUSPARSE.switch2bsr(d_x,convert(Cint,blockdim))
            d_x = CUSPARSE.switch2csc(d_x)
            h_x = collect(d_x)
            @test h_x ≈ x
        end

        @testset "convert_d2b" begin
            x = rand(elty,m,n)
            d_x = CuArray(x)
            d_x = CUSPARSE.sparse(d_x,'B')
            d_y = Array(d_x)
            h_x = collect(d_y)
            @test h_x ≈ x
        end

        @testset "convert_c2r" begin
            x = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCSC(x)
            d_x = CUSPARSE.switch2csr(d_x)
            h_x = collect(d_x)
            @test h_x.rowval == x.rowval
            @test h_x.nzval ≈ x.nzval
        end

        @testset "convert_r2d" begin
            x = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCSR(x)
            d_x = Array(d_x)
            h_x = collect(d_x)
            @test h_x ≈ Array(x)
        end

        @testset "convert_c2d" begin
            x = sprand(elty,m,n, 0.2)
            d_x = CuSparseMatrixCSC(x)
            d_x = Array(d_x)
            h_x = collect(d_x)
            @test h_x ≈ Array(x)
        end

        @testset "convert_d2c" begin
            x = rand(elty,m,n)
            d_x = CuArray(x)
            d_x = CUSPARSE.sparse(d_x,'C')
            h_x = collect(d_x)
            @test h_x ≈ sparse(x)
        end

        @testset "convert_d2r" begin
            x = rand(elty,m,n)
            d_x = CuArray(x)
            d_x = CUSPARSE.sparse(d_x)
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
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            d_A = CUSPARSE.ic02!(d_A,'O')
            h_A = collect(CUSPARSE.switch2csr(d_A))
            Ac = sparse(Array(cholesky(Hermitian(A))))
            h_A = transpose(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
            d_A = CuSparseMatrixCSR(sparse(tril(rand(elty,m,n))))
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            @test_throws DimensionMismatch CUSPARSE.ic02!(d_A,'O')
        end

        @testset "bsric02" begin
            d_A = CuSparseMatrixCSR(sparse(tril(A)))
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            d_B = CUSPARSE.ic02(d_A,'O')
            h_A = collect(CUSPARSE.switch2csr(d_B))
            Ac = sparse(Array(cholesky(Hermitian(A))))
            h_A = transpose(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
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
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            d_A = CUSPARSE.ilu02!(d_A,'O')
            h_A = collect(CUSPARSE.switch2csr(d_A))
            Alu = lu(Array(A), Val(false))
            Ac = sparse(Alu.L*Alu.U)
            h_A = adjoint(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
            d_A = CuSparseMatrixCSR(sparse(rand(elty,m,n)))
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            @test_throws DimensionMismatch CUSPARSE.ilu02!(d_A,'O')
        end

        @testset "bsrilu02" begin
            d_A = CuSparseMatrixCSR(sparse(A))
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            d_B = CUSPARSE.ilu02(d_A,'O')
            h_A = collect(CUSPARSE.switch2csr(d_B))
            Alu = lu(Array(A),Val(false))
            Ac = sparse(Alu.L*Alu.U)
            h_A = adjoint(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
        end
    end
end

@testset "bsrsm2" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        @testset "bsrsm2!" begin
            A = rand(elty,m,m)
            A = triu(A)
            X = rand(elty,m,n)
            alpha = rand(elty)
            d_X = CuArray(X)
            d_A = CuSparseMatrixCSR(sparse(A))
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            d_X = CUSPARSE.bsrsm2!('N','N',alpha,d_A,d_X,'O')
            h_Y = collect(d_X)
            Y = A\(alpha * X)
            @test Y ≈ h_Y
            d_X = CUDA.rand(elty,n,n)
            @test_throws DimensionMismatch CUSPARSE.bsrsm2!('N','N',alpha,d_A,d_X,'O')
            @test_throws DimensionMismatch CUSPARSE.bsrsm2!('N','T',alpha,d_A,d_X,'O')
            A = sparse(rand(elty,m,n))
            d_A = CuSparseMatrixCSR(A)
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            @test_throws DimensionMismatch CUSPARSE.bsrsm2!('N','N',alpha,d_A,d_X,'O')
        end

        @testset "bsrsm2" begin
            A = rand(elty,m,m)
            A = triu(A)
            X = rand(elty,m,n)
            alpha = rand(elty)
            d_X = CuArray(X)
            d_A = CuSparseMatrixCSR(sparse(A))
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            d_Y = CUSPARSE.bsrsm2('N','N',alpha,d_A,d_X,'O')
            h_Y = collect(d_Y)
            Y = A\(alpha * X)
            @test Y ≈ h_Y
            A = sparse(rand(elty,m,n))
            d_A = CuSparseMatrixCSR(A)
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            @test_throws DimensionMismatch CUSPARSE.bsrsm2('N','N',alpha,d_A,d_X,'O')
        end
    end
end
@testset "bsrsv2" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        @testset "bsrsv2!" begin
            A = rand(elty,m,m)
            A = triu(A)
            X = rand(elty,m)
            alpha = rand(elty)
            d_X = CuArray(X)
            d_A = CuSparseMatrixCSR(sparse(A))
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            d_X = CUSPARSE.sv2!('N','U',alpha,d_A,d_X,'O')
            h_Y = collect(d_X)
            Y = A\(alpha * X)
            @test Y ≈ h_Y
            d_X = CUDA.rand(elty,n)
            @test_throws DimensionMismatch CUSPARSE.sv2!('N','U',alpha,d_A,d_X,'O')
            A = sparse(rand(elty,m,n))
            d_A = CuSparseMatrixCSR(A)
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            @test_throws DimensionMismatch CUSPARSE.sv2!('N','U',alpha,d_A,d_X,'O')
        end

        @testset "bsrsv2" begin
            A = rand(elty,m,m)
            A = triu(A)
            Al = tril(A)
            X = rand(elty,m)
            alpha = rand(elty)
            d_X = CuArray(X)
            d_A = CuSparseMatrixCSR(sparse(A))
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            d_Y = CUSPARSE.sv2('N','U',alpha,d_A,d_X,'O')
            h_Y = collect(d_Y)
            Y = A\(alpha * X)
            @test Y ≈ h_Y
            d_Y = UpperTriangular(d_A)\d_X
            h_Y = collect(d_Y)
            @test h_Y ≈ A\X
            #=d_Y = UpperTriangular(d_A)'\d_X
            h_Y = collect(d_Y)
            @test h_Y ≈ A'\X=#
            d_Y = transpose(UpperTriangular(d_A))\d_X
            h_Y = collect(d_Y)
            @test h_Y ≈ transpose(A)\X
            d_Y = LowerTriangular(d_A)\d_X
            h_Y = collect(d_Y)
            @test h_Y ≈ Al\X
            #=d_Y = LowerTriangular(d_A)'\d_X
            h_Y = collect(d_Y)
            @test h_Y ≈ A'\X=#
            d_Y = transpose(LowerTriangular(d_A))\d_X
            h_Y = collect(d_Y)
            @test h_Y ≈ transpose(Al)\X
            A = sparse(rand(elty,m,n))
            d_A = CuSparseMatrixCSR(A)
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            @test_throws DimensionMismatch CUSPARSE.sv2('N','U',alpha,d_A,d_X,'O')
        end
    end
end

@testset "csrsm2" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        @testset "csrsm2!" begin
            A = rand(elty,m,m)
            A = triu(A)
            X = rand(elty,m,n)
            alpha = rand(elty)
            d_X = CuArray(X)
            d_A = CuSparseMatrixCSR(sparse(A))
            d_X = CUSPARSE.csrsm2!('N','N',alpha,d_A,d_X,'O')
            h_Y = collect(d_X)
            Y = A\(alpha * X)
            @test Y ≈ h_Y
            d_X = CUDA.rand(elty,n,n)
            @test_throws DimensionMismatch CUSPARSE.csrsm2!('N','N',alpha,d_A,d_X,'O')
            @test_throws DimensionMismatch CUSPARSE.csrsm2!('N','T',alpha,d_A,d_X,'O')
            A = sprand(elty,m,n,0.7)
            d_A = CuSparseMatrixCSR(A)
            @test_throws DimensionMismatch CUSPARSE.csrsm2!('N','N',alpha,d_A,d_X,'O')
        end

        @testset "csrsm2" begin
            A = rand(elty,m,m)
            A = triu(A)
            X = rand(elty,m,n)
            alpha = rand(elty)
            d_X = CuArray(X)
            d_A = CuSparseMatrixCSR(sparse(A))
            d_Y = CUSPARSE.csrsm2('N','N',alpha,d_A,d_X,'O')
            h_Y = collect(d_Y)
            Y = A\(alpha * X)
            @test Y ≈ h_Y
            A = sprand(elty,m,n,0.7)
            d_A = CuSparseMatrixCSR(A)
            @test_throws DimensionMismatch CUSPARSE.csrsm2('N','N',alpha,d_A,d_X,'O')
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
            h_A = collect(d_B)
            Alu = lu(Array(A),Val(false))
            Ac = sparse(Alu.L*Alu.U)
            h_A = adjoint(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
        end
        @testset "csc" begin
            A = rand(elty,m,m)
            A += transpose(A)
            A += m * Diagonal{elty}(I, m)
            d_A = CuSparseMatrixCSC(sparse(A))
            d_B = CUSPARSE.ilu02(d_A,'O')
            h_A = collect(d_B)
            Alu = lu(Array(A),Val(false))
            Ac = sparse(Alu.L*Alu.U)
            h_A = adjoint(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
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
            h_A = collect(d_B)
            Ac  = sparse(Array(cholesky(Hermitian(A))))
            h_A = transpose(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
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
            h_A = collect(d_B)
            Ac  = sparse(Array(cholesky(Hermitian(A))))
            h_A = transpose(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
            A   = rand(elty,m,n)
            d_A = CuSparseMatrixCSC(sparse(tril(A)))
            @test_throws DimensionMismatch CUSPARSE.ic02(d_A, 'O')
        end
    end
end

@testset "cssv" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        @testset "csrsv2" begin
            A = rand(elty,m,m)
            A = triu(A)
            Al = tril(A)
            X = rand(elty,m)
            alpha = rand(elty)
            d_X = CuArray(X)
            d_A = CuSparseMatrixCSR(sparse(A))
            d_Y = CUSPARSE.sv2('N','U',alpha,d_A,d_X,'O')
            h_Y = collect(d_Y)
            Y = A\(alpha * X)
            @test Y ≈ h_Y
            d_y = UpperTriangular(d_A)\d_X
            h_y = collect(d_y)
            y = A\X
            @test y ≈ h_y
            d_y = transpose(UpperTriangular(d_A))\d_X
            h_y = collect(d_y)
            y = transpose(A)\X
            @test y ≈ h_y
            #=d_y = UpperTriangular(d_A)'\d_X
            h_y = collect(d_y)
            y = A'\X
            @test y ≈ h_y=#
            d_y = LowerTriangular(d_A)\d_X
            h_y = collect(d_y)
            y = Al\X
            @test y ≈ h_y
            d_y = transpose(LowerTriangular(d_A))\d_X
            h_y = collect(d_y)
            y = transpose(Al)\X
            @test y ≈ h_y
            #=d_y = LowerTriangular(d_A)'\d_X
            h_y = collect(d_y)
            y = A'\X
            @test y ≈ h_y=#
            A = sparse(rand(elty,m,n))
            d_A = CuSparseMatrixCSR(A)
            @test_throws DimensionMismatch CUSPARSE.sv2('N','U',alpha,d_A,d_X,'O')
        end

        @testset "cscsv2" begin
            A = rand(elty,m,m)
            A = triu(A)
            Al = tril(A)
            X = rand(elty,m)
            alpha = rand(elty)
            d_X = CuArray(X)
            d_A = CuSparseMatrixCSC(sparse(A))
            d_Y = CUSPARSE.sv2('N','U',alpha,d_A,d_X,'O')
            h_Y = collect(d_Y)
            Y = A\(alpha * X)
            @test Y ≈ h_Y
            d_Y = CUSPARSE.sv2('T','U',alpha,d_A,d_X,'O')
            h_Y = collect(d_Y)
            Y = transpose(A)\(alpha * X)
            @test Y ≈ h_Y
            d_y = UpperTriangular(d_A)\d_X
            h_y = collect(d_y)
            y = A\X
            @test y ≈ h_y
            d_y = transpose(UpperTriangular(d_A))\d_X
            h_y = collect(d_y)
            y = transpose(A)\X
            @test y ≈ h_y
            d_y = LowerTriangular(d_A)\d_X
            h_y = collect(d_y)
            y = Al\X
            @test y ≈ h_y
            d_y = transpose(LowerTriangular(d_A))\d_X
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
            @test_throws DimensionMismatch CUSPARSE.sv2('N','U',alpha,d_A,d_X,'O')
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
            y[x.nzind] += alpha * x.nzval
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
            z[x.nzind] += alpha * x.nzval
            @test h_z ≈ z
            d_z = CUSPARSE.axpyi(d_x,d_y,'O')
            #compare
            h_z = collect(d_z)
            z = copy(y)
            z[x.nzind] += x.nzval
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
            @test h_x ≈ SparseVector(m,x.nzind,y[x.nzind])
        end

        @testset "gthr" begin
            d_x = CuSparseVector(x)
            d_y = CuArray(y)
            d_z = CUSPARSE.gthr(d_x,d_y,'O')
            h_z = collect(d_z)
            @test h_z ≈ SparseVector(m,x.nzind,y[x.nzind])
        end

        @testset "gthrz!" begin
            d_x = CuSparseVector(x)
            d_y = CuArray(y)
            d_x,d_y = CUSPARSE.gthrz!(d_x,d_y,'O')
            h_x = collect(d_x)
            h_y = collect(d_y)
            @test h_x ≈ SparseVector(m,x.nzind,y[x.nzind])
            #y[x.nzind] = zero(elty)
            #@test h_y ≈ y
        end

        @testset "gthrz" begin
            d_x = CuSparseVector(x)
            d_y = CuArray(y)
            d_z,d_w = CUSPARSE.gthrz(d_x,d_y,'O')
            h_w = collect(d_w)
            h_z = collect(d_z)
            @test h_z ≈ SparseVector(m,x.nzind,y[x.nzind])
            #y[x.nzind] = zero(elty)
            #@test h_w ≈ y
        end
    end
end

@testset "bsrmm2" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = sparse(rand(elty,m,k))
        B = rand(elty,k,n)
        C = rand(elty,m,n)
        alpha = rand(elty)
        beta = rand(elty)
        d_B = CuArray(B)
        d_C = CuArray(C)
        d_A = CuSparseMatrixCSR(A)
        d_A = CUSPARSE.switch2bsr(d_A,convert(Cint,blockdim))
        @test_throws DimensionMismatch CUSPARSE.mm2('N','T',alpha,d_A,d_B,beta,d_C,'O')
        @test_throws DimensionMismatch CUSPARSE.mm2('T','N',alpha,d_A,d_B,beta,d_C,'O')
        @test_throws DimensionMismatch CUSPARSE.mm2('T','T',alpha,d_A,d_B,beta,d_C,'O')
        @test_throws DimensionMismatch CUSPARSE.mm2('N','N',alpha,d_A,d_B,beta,d_B,'O')
        d_D = CUSPARSE.mm2('N','N',alpha,d_A,d_B,beta,d_C,'O')
        h_D = collect(d_D)
        D = alpha * A * B + beta * C
        @test D ≈ h_D
        d_D = CUSPARSE.mm2('N','N',d_A,d_B,beta,d_C,'O')
        h_D = collect(d_D)
        D = A * B + beta * C
        @test D ≈ h_D
        d_D = CUSPARSE.mm2('N','N',d_A,d_B,d_C,'O')
        h_D = collect(d_D)
        D = A * B + C
        @test D ≈ h_D
        d_D = CUSPARSE.mm2('N','N',alpha,d_A,d_B,'O')
        h_D = collect(d_D)
        D = alpha * A * B
        @test D ≈ h_D
        d_D = CUSPARSE.mm2('N','N',d_A,d_B,'O')
        h_D = collect(d_D)
        D = A * B
        @test D ≈ h_D
    end
end
@testset "bsrmm2!" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = sparse(rand(elty,m,k))
        B = rand(elty,k,n)
        C = rand(elty,m,n)
        alpha = rand(elty)
        beta = rand(elty)
        d_B = CuArray(B)
        d_C = CuArray(C)
        d_A = CuSparseMatrixCSR(A)
        d_A = CUSPARSE.switch2bsr(d_A,convert(Cint,blockdim))
        @test_throws DimensionMismatch CUSPARSE.mm2!('N','T',alpha,d_A,d_B,beta,d_C,'O')
        @test_throws DimensionMismatch CUSPARSE.mm2!('T','N',alpha,d_A,d_B,beta,d_C,'O')
        @test_throws DimensionMismatch CUSPARSE.mm2!('T','T',alpha,d_A,d_B,beta,d_C,'O')
        @test_throws DimensionMismatch CUSPARSE.mm2!('N','N',alpha,d_A,d_B,beta,d_B,'O')
        CUSPARSE.mm2!('N','N',alpha,d_A,d_B,beta,d_C,'O')
        h_D = collect(d_C)
        D = alpha * A * B + beta * C
        @test D ≈ h_D
        d_C = CuArray(C)
        mul!(d_C, d_A, d_B)
        h_C = collect(d_C)
        D = A * B
        @test D ≈ h_C
    end
end
@testset "mv!" begin
    for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = sparse(rand(elty,m,n))
        x = rand(elty,n)
        y = rand(elty,m)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "mv!" begin
            @testset "$mattype" for (mattype, d_A) in [
                ("csr", CuSparseMatrixCSR(A)),
                ("bsr", CUSPARSE.switch2bsr(CuSparseMatrixCSR(A),convert(Cint,blockdim)))
            ]
                d_x = CuArray(x)
                d_y = CuArray(y)
                @test_throws DimensionMismatch CUSPARSE.mv!('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv!('N',alpha,d_A,d_y,beta,d_x,'O')
                CUSPARSE.mv!('N',alpha,d_A,d_x,beta,d_y,'O')
                h_z = collect(d_y)
                z = alpha * A * x + beta * y
                @test z ≈ h_z
                mul!(d_y, d_A, d_x)
                h_y = collect(d_y)
                z = A * x
                @test z ≈ h_y
                if mattype=="csr"
                    @test d_y' * (d_A * d_x) ≈ (d_y' * d_A) * d_x
                end
            end
        end
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
            y[x.nzind]  += x.nzval
            @test h_y ≈ y
        end
        y = zeros(elty,m)

        @testset "sctr" begin
            d_x = CuSparseVector(x)
            d_y = CUSPARSE.sctr(d_x,'O')
            h_y = collect(d_y)
            y = zeros(elty,m)
            y[x.nzind]  += x.nzval
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
            y[x.nzind] = cos(angle)*w[z.nzind] - sin(angle)*z.nzval
            @test h_x ≈ SparseVector(m,x.nzind,cos(angle)*z.nzval + sin(angle)*w[z.nzind])
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
            w[z.nzind] = cos(angle)*y[x.nzind] - sin(angle)*x.nzval
            @test h_z ≈ SparseVector(m,z.nzind, cos(angle)*x.nzval + sin(angle)*y[x.nzind])
            @test h_w ≈ w
        end
    end
end

