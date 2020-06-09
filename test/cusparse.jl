using CUDA.CUSPARSE

using LinearAlgebra
using SparseArrays

m = 25
n = 35
k = 10
blockdim = 5

@testset "util" begin
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

@testset "char" begin
    @test_throws ArgumentError CUSPARSE.cusparseop('Z')
    @test_throws ArgumentError CUSPARSE.cusparsetype('Z')
    @test_throws ArgumentError CUSPARSE.cusparsefill('Z')
    @test_throws ArgumentError CUSPARSE.cusparsediag('Z')
    @test_throws ArgumentError CUSPARSE.cusparsedir('Z')
    @test_throws ArgumentError CUSPARSE.cusparseindex('A')

    @test CUSPARSE.cusparseop('N') == CUSPARSE.CUSPARSE_OPERATION_NON_TRANSPOSE
    @test CUSPARSE.cusparseop('C') == CUSPARSE.CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
    @test CUSPARSE.cusparseop('T') == CUSPARSE.CUSPARSE_OPERATION_TRANSPOSE
    @test CUSPARSE.cusparsefill('U') == CUSPARSE.CUSPARSE_FILL_MODE_UPPER
    @test CUSPARSE.cusparsefill('L') == CUSPARSE.CUSPARSE_FILL_MODE_LOWER
    @test CUSPARSE.cusparsediag('U') == CUSPARSE.CUSPARSE_DIAG_TYPE_UNIT
    @test CUSPARSE.cusparsediag('N') == CUSPARSE.CUSPARSE_DIAG_TYPE_NON_UNIT
    @test CUSPARSE.cusparseindex('Z') == CUSPARSE.CUSPARSE_INDEX_BASE_ZERO
    @test CUSPARSE.cusparseindex('O') == CUSPARSE.CUSPARSE_INDEX_BASE_ONE
end

@testset "conversion" begin
    @testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
        @testset "make_csc" begin
            x = sparse(rand(elty,m,n))
            d_x = CuSparseMatrixCSC(x)
            h_x = collect(d_x)
            @test h_x == x
            @test eltype(d_x) == elty
        end

        @testset "make_csr" begin
            x = sparse(rand(elty,m,n))
            d_x = CuSparseMatrixCSR(x)
            h_x = collect(d_x)
            @test h_x == x
        end

        @testset "convert_r2c" begin
            x = sparse(rand(elty,m,n))
            d_x = CuSparseMatrixCSR(x)
            d_x = CUSPARSE.switch2csc(d_x)
            h_x = collect(d_x)
            @test h_x.rowval == x.rowval
            @test h_x.nzval ≈ x.nzval
        end

        @testset "convert_r2b" begin
            x = sparse(rand(elty,m,n))
            d_x = CuSparseMatrixCSR(x)
            d_x = CUSPARSE.switch2bsr(d_x,convert(Cint,blockdim))
            d_x = CUSPARSE.switch2csr(d_x)
            h_x = collect(d_x)
            @test h_x ≈ x
        end

        @testset "convert_c2b" begin
            x = sparse(rand(elty,m,n))
            d_x = CuSparseMatrixCSC(x)
            d_x = CUSPARSE.switch2bsr(d_x,convert(Cint,blockdim))
            d_x = CUSPARSE.switch2csc(d_x)
            h_x = collect(d_x)
            @test h_x ≈ x
        end

        @testset "convert_c2h" begin
            x = sparse(rand(elty,m,n))
            d_x = CuSparseMatrixCSC(x)
            d_x = CUSPARSE.switch2hyb(d_x)
            d_y = CUSPARSE.switch2csc(d_x)
            CUSPARSE.cusparseDestroyHybMat(d_x.Mat)
            h_x = collect(d_y)
            @test h_x.rowval == x.rowval
            @test h_x.nzval ≈ x.nzval
        end

        @testset "convert_r2h" begin
            x = sparse(rand(elty,m,n))
            d_x = CuSparseMatrixCSR(x)
            d_x = CUSPARSE.switch2hyb(d_x)
            d_y = CUSPARSE.switch2csr(d_x)
            CUSPARSE.cusparseDestroyHybMat(d_x.Mat)
            h_x = collect(d_y)
            @test h_x.rowval == x.rowval
            @test h_x.nzval ≈ x.nzval
        end

        @testset "convert_d2h" begin
            x = rand(elty,m,n)
            d_x = CuArray(x)
            d_x = CUSPARSE.sparse(d_x,'H')
            d_y = Array(d_x)
            CUSPARSE.cusparseDestroyHybMat(d_x.Mat)
            h_x = collect(d_y)
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
            x = sparse(rand(elty,m,n))
            d_x = CuSparseMatrixCSC(x)
            d_x = CUSPARSE.switch2csr(d_x)
            h_x = collect(d_x)
            @test h_x.rowval == x.rowval
            @test h_x.nzval ≈ x.nzval
        end

        @testset "convert_r2d" begin
            x = sparse(rand(elty,m,n))
            d_x = CuSparseMatrixCSR(x)
            d_x = Array(d_x)
            h_x = collect(d_x)
            @test h_x ≈ Array(x)
        end

        @testset "convert_c2d" begin
            x = sparse(rand(elty,m,n))
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

@testset "ilu0" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = rand(elty,m,m)
        @testset "csr" begin
            d_A = CuSparseMatrixCSR(sparse(A))
            info = CUSPARSE.sv_analysis('N','G','U',d_A,'O')
            d_B = CUSPARSE.ilu0('N',d_A,info,'O')
            h_B = collect(d_B)
            Alu = lu(Array(A),Val(false))
            Ac = sparse(Alu.L*Alu.U)
            h_A = adjoint(h_B) * h_B
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
        end
        @testset "csc" begin
            d_A = CuSparseMatrixCSC(sparse(A))
            info = CUSPARSE.sv_analysis('N','G','U',d_A,'O')
            d_B = CUSPARSE.ilu0('N',d_A,info,'O')
            h_B = collect(d_B)
            Alu = lu(Array(A),Val(false))
            Ac = sparse(Alu.L*Alu.U)
            h_A = adjoint(h_B) * h_B
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
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

@testset "ic0" begin
    @testset for elty in [Float32,Float64]
        @testset "csr" begin
            A    = rand(elty, m ,m)
            A    = A + transpose(A)
            A   += m * Diagonal{elty}(I, m)
            d_A  = Symmetric(CuSparseMatrixCSR(sparse(triu(A))))
            info = CUSPARSE.sv_analysis('N', 'S', 'U', d_A, 'O')
            d_B  = CUSPARSE.ic0('N', 'S', d_A, info, 'O')
            h_A  = collect(d_B)
            Ac   = sparse(Array(cholesky(A)))
            h_A  = transpose(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
        end
        @testset "csc" begin
            A    = rand(elty, m ,m)
            A    = A + transpose(A)
            A   += m * Diagonal{elty}(I, m)
            d_A  = Symmetric(CuSparseMatrixCSC(sparse(triu(A))))
            info = CUSPARSE.sv_analysis('N', 'S', 'U', d_A, 'O')
            d_B  = CUSPARSE.ic0('N', 'S', d_A, info, 'O')
            h_A  = collect(d_B)
            Ac   = sparse(Array(cholesky(A)))
            h_A  = transpose(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
        end
    end
    @testset for elty in [ComplexF32,ComplexF64]
        @testset "csr" begin
            A    = rand(elty,m,m)
            A    = A + adjoint(A)
            A   += m * Diagonal{elty}(I, m)
            d_A  = Hermitian(CuSparseMatrixCSR(sparse(triu(A))))
            info = CUSPARSE.sv_analysis('N', 'H', 'U', d_A, 'O')
            d_B  = CUSPARSE.ic0('N', 'H', d_A, info, 'O')
            h_A  = collect(d_B)
            Ac   = sparse(Array(cholesky(A)))
            h_A  = adjoint(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
        end
        @testset "csc" begin
            A    = rand(elty,m,m)
            A    = A + adjoint(A)
            A   += m * Diagonal{elty}(I, m)
            d_A  = Hermitian(CuSparseMatrixCSC(sparse(triu(A))))
            info = CUSPARSE.sv_analysis('N', 'H', 'U', d_A, 'O')
            d_B  = CUSPARSE.ic0('N', 'H', d_A, info, 'O')
            h_A  = collect(d_B)
            Ac   = sparse(Array(cholesky(A)))
            h_A  = adjoint(h_A) * h_A
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

@testset "csrsm" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        @testset "csr" begin
            A = rand(elty,m,m)
            A = triu(A)
            X = rand(elty,m,n)
            alpha = rand(elty)
            d_X = CuArray(X)
            d_A = CuSparseMatrixCSR(sparse(A))
            info = CUSPARSE.sm_analysis('N','U',d_A,'O')
            d_Y = CUSPARSE.sm_solve('N','U',alpha,d_A,d_X,info,'O')
            h_Y = collect(d_Y)
            Y = A\(alpha * X)
            @test Y ≈ h_Y
            d_y = UpperTriangular(d_A)\d_X
            h_y = collect(d_y)
            y = A\X
            @test y ≈ h_y
            #=d_y = UpperTriangular(d_A)'\d_X
            h_y = collect(d_y)
            y = A'\X=#
            @test y ≈ h_y
            d_y = transpose(UpperTriangular(d_A))\d_X
            h_y = collect(d_y)
            y = transpose(A)\X
            @test y ≈ h_y
            d_X = CUDA.rand(elty,n,n)
            @test_throws DimensionMismatch CUSPARSE.sm_solve('N','U',alpha,d_A,d_X,info,'O')
            A = sparse(rand(elty,m,n))
            d_A = CuSparseMatrixCSR(A)
            @test_throws DimensionMismatch CUSPARSE.sm_analysis('T','U',d_A,'O')
            CUSPARSE.cusparseDestroySolveAnalysisInfo(info)
        end
        @testset "csc" begin
            A = rand(elty,m,m)
            A = triu(A)
            X = rand(elty,m,n)
            alpha = rand(elty)
            d_X = CuArray(X)
            d_A = CuSparseMatrixCSC(sparse(A))
            info = CUSPARSE.sm_analysis('N','U',d_A,'O')
            d_Y = CUSPARSE.sm_solve('N','U',alpha,d_A,d_X,info,'O')
            h_Y = collect(d_Y)
            Y = A\(alpha * X)
            @test Y ≈ h_Y
            d_y = UpperTriangular(d_A)\d_X
            h_y = collect(d_y)
            y = A\X
            @test y ≈ h_y
            if elty <: Real
                d_y = UpperTriangular(d_A)'\d_X
                h_y = collect(d_y)
                y = A'\X
                @test y ≈ h_y
            end
            d_y = transpose(UpperTriangular(d_A))\d_X
            h_y = collect(d_y)
            y = transpose(A)\X
            @test y ≈ h_y
            d_X = CUDA.rand(elty,n,n)
            @test_throws DimensionMismatch CUSPARSE.sm_solve('N','U',alpha,d_A,d_X,info,'O')
            A = sparse(rand(elty,m,n))
            d_A = CuSparseMatrixCSC(A)
            @test_throws DimensionMismatch CUSPARSE.sm_analysis('T','U',d_A,'O')
            CUSPARSE.cusparseDestroySolveAnalysisInfo(info)
        end
    end
end

@testset "cssv" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        @testset "csrsv" begin
            A = rand(elty,m,m)
            A = triu(A)
            x = rand(elty,m)
            alpha = rand(elty)
            d_x = CuArray(x)
            d_A = CuSparseMatrixCSR(sparse(A))
            d_y = CUSPARSE.sv('N','T','U',alpha,d_A,d_x,'O')
            h_y = collect(d_y)
            y = A\(alpha * x)
            @test y ≈ h_y
            d_A = CuSparseMatrixCSR(sparse(A))
            d_y = CUSPARSE.sv('N','T','U',d_A,d_x,'O')
            h_y = collect(d_y)
            y = A\x
            @test y ≈ h_y
            d_A = UpperTriangular(CuSparseMatrixCSR(sparse(A)))
            d_y = CUSPARSE.sv('N',d_A,d_x,'O')
            h_y = collect(d_y)
            y = A\x
            @test y ≈ h_y
            d_A = CuSparseMatrixCSR(sparse(A))
            x = rand(elty,n)
            d_x = CuArray(x)
            info = CUSPARSE.sv_analysis('N','T','U',d_A,'O')
            @test_throws DimensionMismatch CUSPARSE.sv_solve('N','U',alpha,d_A,d_x,info,'O')
            A = sparse(rand(elty,m,n))
            d_A = CuSparseMatrixCSR(A)
            @test_throws DimensionMismatch CUSPARSE.sv_analysis('T','T','U',d_A,'O')
            CUSPARSE.cusparseDestroySolveAnalysisInfo(info)
        end

        @testset "cscsv" begin
            A = rand(elty,m,m)
            A = triu(A)
            x = rand(elty,m)
            alpha = rand(elty)
            d_x = CuArray(x)
            d_A = CuSparseMatrixCSC(sparse(A))
            d_y = CUSPARSE.sv('N','T','U',alpha,d_A,d_x,'O')
            h_y = collect(d_y)
            y = A\(alpha * x)
            @test y ≈ h_y
            d_x = CuArray(x)
            d_A = CuSparseMatrixCSC(sparse(A))
            d_y = CUSPARSE.sv('N','T','U',d_A,d_x,'O')
            h_y = collect(d_y)
            y = A\x
            @test y ≈ h_y
            d_x = CuArray(x)
            d_A = UpperTriangular(CuSparseMatrixCSC(sparse(A)))
            d_y = CUSPARSE.sv('N',d_A,d_x,'O')
            h_y = collect(d_y)
            y = A\x
            @test y ≈ h_y
            d_A = CuSparseMatrixCSC(sparse(A))
            x = rand(elty,n)
            d_x = CuArray(x)
            info = CUSPARSE.sv_analysis('N','T','U',d_A,'O')
            @test_throws DimensionMismatch CUSPARSE.sv_solve('N','U',alpha,d_A,d_x,info,'O')
            A = sparse(rand(elty,m,n))
            d_A = CuSparseMatrixCSC(A)
            @test_throws DimensionMismatch CUSPARSE.sv_analysis('T','T','U',d_A,'O')
            CUSPARSE.cusparseDestroySolveAnalysisInfo(info)
        end

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
            d_y = CUSPARSE.sv2('N','U',d_A,d_X,'O')
            h_y= collect(d_y)
            Y = A\X
            @test Y ≈ h_Y
            d_y = CUSPARSE.sv2('N','U',alpha,UpperTriangular(d_A),d_X,'O')
            h_y= collect(d_y)
            Y = A\(alpha*X)
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
            A = sparse(rand(elty,m,m))
            d_A = CuSparseMatrixCSR(A)
            X = rand(elty,n)
            d_X = CuArray(X)
            @test_throws DimensionMismatch CUSPARSE.sv2('N','U',d_A,d_X,'O')
            A = rand(elty,m,m)
            A[1, :] .= zero(elty)
            A[:, 1] .= zero(elty)
            A = triu(A)
            X = rand(elty,m)
            d_X = CuArray(X)
            d_A = CuSparseMatrixCSR(sparse(A))
            @test_throws ErrorException CUSPARSE.sv2!('N','U',d_A,d_X,'O')
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
            d_y = CUSPARSE.sv2('N','U',d_A,d_X,'O')
            h_y= collect(d_y)
            Y = A\X
            @test Y ≈ h_Y
            d_y = CUSPARSE.sv2('N','U',alpha,UpperTriangular(d_A),d_X,'O')
            h_y= collect(d_y)
            Y = A\(alpha*X)
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
            A = sparse(rand(elty,m,m))
            d_A = CuSparseMatrixCSR(A)
            X = rand(elty,n)
            d_X = CuArray(X)
            @test_throws DimensionMismatch CUSPARSE.sv2('N','U',d_A,d_X,'O')
            A = rand(elty,m,m)
            A[1, :] .= zero(elty)
            A = triu(A)
            X = rand(elty,m)
            d_X = CuArray(X)
            d_A = CuSparseMatrixCSC(sparse(A))
            @test_throws ErrorException CUSPARSE.sv2!('N','U',d_A,d_X,'O')
        end
    end
end

@testset "cssm" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        @testset "csrsm" begin
            A = rand(elty,m, m)
            A = triu(A)
            x = rand(elty,m, n)
            alpha = rand(elty)
            d_x = CuArray(x)
            d_A = CuSparseMatrixCSR(sparse(A))
            d_y = CUSPARSE.sm('N','U',alpha,d_A,d_x,'O')
            h_y = collect(d_y)
            y = A\(alpha * x)
            @test y ≈ h_y
            d_A = CuSparseMatrixCSR(sparse(A))
            d_y = CUSPARSE.sm('N','U',d_A,d_x,'O')
            h_y = collect(d_y)
            y = A\x
            @test y ≈ h_y
            d_A = UpperTriangular(CuSparseMatrixCSR(sparse(A)))
            d_y = CUSPARSE.sm('N',alpha,d_A,d_x,'O')
            h_y = collect(d_y)
            y = A\(alpha*x)
            @test y ≈ h_y
        end
        @testset "cscsm" begin
            A = rand(elty,m, m)
            A = triu(A)
            x = rand(elty,m, n)
            alpha = rand(elty)
            d_x = CuArray(x)
            d_A = CuSparseMatrixCSC(sparse(A))
            d_y = CUSPARSE.sm('N','U',alpha,d_A,d_x,'O')
            h_y = collect(d_y)
            y = A\(alpha * x)
            @test y ≈ h_y
            d_A = CuSparseMatrixCSC(sparse(A))
            d_y = CUSPARSE.sm('N','U',d_A,d_x,'O')
            h_y = collect(d_y)
            y = A\x
            @test y ≈ h_y
            d_A = UpperTriangular(CuSparseMatrixCSC(sparse(A)))
            d_y = CUSPARSE.sm('N',alpha,d_A,d_x,'O')
            h_y = collect(d_y)
            y = A\(alpha*x)
            @test y ≈ h_y
        end
    end
end
@testset "doti" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        x = sparsevec(rand(1:m,k), rand(elty,k), m)
        y = rand(elty,m)
        d_x = CuSparseVector(x)
        d_y = CuArray(y)
        ddot = CUSPARSE.doti(d_x,d_y,'O')
        #compare
        dot = zero(elty)
        for i in 1:length(x.nzval)
            dot += x.nzval[i] * y[x.nzind[i]]
        end
        @test ddot ≈ dot
    end
end

@testset "dotci" begin
    @testset for elty in [ComplexF32,ComplexF64]
        x = sparsevec(rand(1:m,k), rand(elty,k), m)
        y = rand(elty,m)
        d_x = CuSparseVector(x)
        d_y = CuArray(y)
        ddot = CUSPARSE.dotci(d_x,d_y,'O')
        #compare
        dot = zero(elty)
        for i in 1:length(x.nzval)
            dot += conj(x.nzval[i]) * y[x.nzind[i]]
        end
        @test ddot ≈ dot
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

@testset "geam" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = sparse(rand(elty,m,n))
        B = sparse(rand(elty,m,n))
        alpha = rand(elty)
        beta = rand(elty)
        @testset "csr" begin
            d_A = CuSparseMatrixCSR(A)
            d_B = CuSparseMatrixCSR(B)
            C = alpha * A + beta * B
            d_C = CUSPARSE.geam(alpha,d_A,beta,d_B,'O','O','O')
            h_C = collect(d_C)
            @test C ≈ h_C
            d_C = CUSPARSE.geam(d_A,beta,d_B,'O','O','O')
            h_C = collect(d_C)
            C = A + beta * B
            @test C ≈ h_C
            d_C = CUSPARSE.geam(alpha,d_A,d_B,'O','O','O')
            h_C = collect(d_C)
            C = alpha * A + B
            @test C ≈ h_C
            d_C = CUSPARSE.geam(d_A,d_B,'O','O','O')
            h_C = collect(d_C)
            C = A + B
            @test C ≈ h_C
            d_C = d_A + d_B
            h_C = collect(d_C)
            C = A + B
            @test C ≈ h_C
            d_C = d_A - d_B
            h_C = collect(d_C)
            C = A - B
            @test C ≈ h_C
            B_ = sparse(rand(elty,k,n))
            d_B = CuSparseMatrixCSR(B_)
            @test_throws DimensionMismatch CUSPARSE.geam(d_B,d_A,'O','O','O')
        end
        @testset "csc" begin
            d_A = CuSparseMatrixCSC(A)
            #=d_B = CuSparseMatrixCSC(B)
            C = alpha * A + beta * B
            d_C = CUSPARSE.geam(alpha,d_A,beta,d_B,'O','O','O')
            h_C = transpose!(collect(d_C))
            @test C ≈ h_C
            d_C = CUSPARSE.geam(d_A,beta,d_B,'O','O','O')
            h_C = transpose!(collect(d_C))
            C = A + beta * B
            @test C ≈ h_C
            d_C = CUSPARSE.geam(alpha,d_A,d_B,'O','O','O')
            h_C = transpose!(collect(d_C))
            C = alpha * A + B
            @test C ≈ h_C
            d_C = CUSPARSE.geam(d_A,d_B,'O','O','O')
            h_C = transpose!(collect(d_C))
            C = A + B
            @test C ≈ h_C
            d_C = d_A + d_B
            h_C = transpose!(collect(d_C))
            C = A + B
            @test C ≈ h_C
            d_C = d_A - d_B
            h_C = transpose!(collect(d_C))
            C = A - B
            @test C ≈ h_C=#
            B_ = sparse(rand(elty,k,n))
            d_B = CuSparseMatrixCSC(B_)
            @test_throws DimensionMismatch CUSPARSE.geam(d_B,d_A,'O','O','O')
        end
        @testset "mixed" begin
            A = spdiagm(0=>rand(elty,m))
            B = spdiagm(0=>rand(elty,m)) + sparse(diagm(1=>rand(elty,m-1)))
            d_A = CuSparseMatrixCSR(A)
            d_B = CuSparseMatrixCSC(B)
            d_C = d_B + d_A
            h_C = collect(d_C)
            @test h_C ≈ A + B
            d_C = d_A + d_B
            h_C = collect(d_C)
            @test h_C ≈ A + B
            d_C = d_A - d_B
            h_C = collect(d_C)
            @test h_C ≈ A - B
            d_C = d_B - d_A
            h_C = collect(d_C)
            @test h_C ≈ B - A
        end
    end
end

@testset "gemm" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        # CSR
        A = sparse(rand(elty,m,k)); d_A = CuSparseMatrixCSR(A)
        B = sparse(rand(elty,k,n)); d_B = CuSparseMatrixCSR(B)
        d_C = CUSPARSE.gemm('N','N',d_A,d_B,'O','O','O')
        @test A * B ≈ collect(d_C)
        #
        @test_throws DimensionMismatch CUSPARSE.gemm('N','T',d_A,d_B,'O','O','O')
        @test_throws DimensionMismatch CUSPARSE.gemm('T','T',d_A,d_B,'O','O','O')
        @test_throws DimensionMismatch CUSPARSE.gemm('T','N',d_A,d_B,'O','O','O')
        @test_throws DimensionMismatch CUSPARSE.gemm('N','N',d_B,d_A,'O','O','O')
        #
        A = sparse(rand(elty,m,k)); d_A = CuSparseMatrixCSR(A)
        B = sparse(rand(elty,n,k)); d_B = CuSparseMatrixCSR(B)
        d_C = CUSPARSE.gemm('N','T',d_A,d_B,'O','O','O')
        @test A * transpose(B) ≈ collect(d_C)

        # CSC
        A = sparse(rand(elty,m,k)); d_A = CuSparseMatrixCSC(A)
        B = sparse(rand(elty,k,n)); d_B = CuSparseMatrixCSC(B)
        d_C = CUSPARSE.gemm('N','N',d_A,d_B,'O','O','O')
        @test A * B ≈ collect(d_C)
        #
        @test_throws(DimensionMismatch,CUSPARSE.gemm('N','T',d_A,d_B,'O','O','O'))
        @test_throws(DimensionMismatch,CUSPARSE.gemm('T','T',d_A,d_B,'O','O','O'))
        @test_throws(DimensionMismatch,CUSPARSE.gemm('T','N',d_A,d_B,'O','O','O'))
        @test_throws(DimensionMismatch,CUSPARSE.gemm('N','N',d_B,d_A,'O','O','O'))
        #
        A = sparse(rand(elty,m,k)); d_A = CuSparseMatrixCSC(A)
        B = sparse(rand(elty,n,k)); d_B = CuSparseMatrixCSC(B)
        d_C = CUSPARSE.gemm('N','T',d_A,d_B,'O','O','O')
        @test A * transpose(B) ≈ collect(d_C)
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

@testset "gtsv" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        @testset "gtsv!" begin
            dl = rand(elty,m-1)
            du = rand(elty,m-1)
            d = rand(elty,m)
            B = rand(elty,m,n)
            d_dl = CuArray(vcat([0],dl))
            d_du = CuArray(vcat(du,[0]))
            d_d = CuArray(d)
            d_B = CuArray(B)
            d_B = CUSPARSE.gtsv!(d_dl,d_d,d_du,d_B)
            C = diagm(0=>d) + diagm(1=>du) + diagm(-1=>dl)
            h_B = collect(d_B)
            @test h_B ≈ C\B
        end

        @testset "gtsv" begin
            dl = rand(elty,m-1)
            du = rand(elty,m-1)
            d = rand(elty,m)
            B = rand(elty,m,n)
            d_dl = CuArray(vcat([0],dl))
            d_du = CuArray(vcat(du,[0]))
            d_d = CuArray(d)
            d_B = CuArray(B)
            d_C = CUSPARSE.gtsv(d_dl,d_d,d_du,d_B)
            C = diagm(0=>d) + diagm(1=>du) + diagm(-1=>dl)
            h_C = collect(d_C)
            @test h_C ≈ C\B
        end

        @testset "gtsv_nopivot!" begin
            dl = rand(elty,m-1)
            du = rand(elty,m-1)
            d = rand(elty,m)
            B = rand(elty,m,n)
            d_dl = CuArray(vcat([0],dl))
            d_du = CuArray(vcat(du,[0]))
            d_d = CuArray(d)
            d_B = CuArray(B)
            d_B = CUSPARSE.gtsv_nopivot!(d_dl,d_d,d_du,d_B)
            C = diagm(0=>d) + diagm(1=>du) + diagm(-1=>dl)
            h_B = collect(d_B)
            @test h_B ≈ C\B
        end

        @testset "gtsv_nopivot" begin
            dl = rand(elty,m-1)
            du = rand(elty,m-1)
            d = rand(elty,m)
            B = rand(elty,m,n)
            d_dl = CuArray(vcat([0],dl))
            d_du = CuArray(vcat(du,[0]))
            d_d = CuArray(d)
            d_B = CuArray(B)
            d_C = CUSPARSE.gtsv_nopivot(d_dl,d_d,d_du,d_B)
            C = diagm(0=>d) + diagm(1=>du) + diagm(-1=>dl)
            h_C = collect(d_C)
            @test h_C ≈ C\B
        end

        @testset "gtsvStridedBatch!" begin
            dla = rand(elty,m-1)
            dua = rand(elty,m-1)
            da = rand(elty,m)
            dlb = rand(elty,m-1)
            dub = rand(elty,m-1)
            db = rand(elty,m)
            xa = rand(elty,m)
            xb = rand(elty,m)
            d_dl = CuArray(vcat([0],dla,[0],dlb))
            d_du = CuArray(vcat(dua,[0],dub,[0]))
            d_d = CuArray(vcat(da,db))
            d_x = CuArray(vcat(xa,xb))
            d_x = CUSPARSE.gtsvStridedBatch!(d_dl,d_d,d_du,d_x,2,m)
            Ca = diagm(0=>da) + diagm(1=>dua) + diagm(-1=>dla)
            Cb = diagm(0=>db) + diagm(1=>dub) + diagm(-1=>dlb)
            h_x = collect(d_x)
            @test h_x[1:m] ≈ Ca\xa
            @test h_x[m+1:2*m] ≈ Cb\xb
        end

        @testset "gtsvStridedBatch" begin
            dla = rand(elty,m-1)
            dua = rand(elty,m-1)
            da = rand(elty,m)
            dlb = rand(elty,m-1)
            dub = rand(elty,m-1)
            db = rand(elty,m)
            xa = rand(elty,m)
            xb = rand(elty,m)
            d_dl = CuArray(vcat([0],dla,[0],dlb))
            d_du = CuArray(vcat(dua,[0],dub,[0]))
            d_d = CuArray(vcat(da,db))
            d_x = CuArray(vcat(xa,xb))
            d_y = CUSPARSE.gtsvStridedBatch(d_dl,d_d,d_du,d_x,2,m)
            Ca = diagm(0=>da) + diagm(1=>dua) + diagm(-1=>dla)
            Cb = diagm(0=>db) + diagm(1=>dub) + diagm(-1=>dlb)
            h_y = collect(d_y)
            @test h_y[1:m] ≈ Ca\xa
            @test h_y[m+1:2*m] ≈ Cb\xb
        end
    end
end

@testset "hybsv" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = rand(elty,m,m)
        A = triu(A)
        x = rand(elty,m)
        alpha = rand(elty)
        beta = rand(elty)
        d_x = CuArray(x)
        d_A = CuSparseMatrixCSR(sparse(A))
        d_A = CUSPARSE.switch2hyb(d_A)
        d_y = CUSPARSE.sv('N','T','U',alpha,d_A,d_x,'O')
        h_y = collect(d_y)
        y = A\(alpha * x)
        @test y ≈ h_y

        #=d_y = UpperTriangular(d_A) \ d_x
        h_y = collect(d_y)
        @test h_y ≈ A\x=#

        d_x = CUDA.rand(elty,n)
        info = CUSPARSE.sv_analysis('N','T','U',d_A,'O')
        @test_throws DimensionMismatch CUSPARSE.sv_solve('N','U',alpha,d_A,d_x,info,'O')
        A = sparse(rand(elty,m,n))
        d_A = CuSparseMatrixCSR(A)
        d_A = CUSPARSE.switch2hyb(d_A)
        @test_throws DimensionMismatch CUSPARSE.sv_analysis('T','T','U',d_A,'O')
        CUSPARSE.cusparseDestroySolveAnalysisInfo(info)
    end
end

@testset "mm" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = sparse(rand(elty,m,k))
        B = rand(elty,k,n)
        C = rand(elty,m,n)
        Ah = sparse(rand(elty,k,k))
        Ah += Ah'
        alpha = rand(elty)
        beta = rand(elty)
        @testset "csr" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = CuSparseMatrixCSR(A)
            @test_throws DimensionMismatch CUSPARSE.mm!('T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_B,'O')
            d_D = CUSPARSE.mm('N',alpha,d_A,d_B,beta,d_C,'O')
            h_D = collect(d_D)
            D = alpha * A * B + beta * C
            @test D ≈ h_D
            d_D = CUSPARSE.mm('N',d_A,d_B,beta,d_C,'O')
            h_D = collect(d_D)
            D = A * B + beta * C
            @test D ≈ h_D
            d_D = CUSPARSE.mm('N',d_A,d_B,d_C,'O')
            h_D = collect(d_D)
            D = A * B + C
            @test D ≈ h_D
            d_D = CUSPARSE.mm('N',alpha,d_A,d_B,'O')
            h_D = collect(d_D)
            D = alpha * A * B
            @test D ≈ h_D
            d_D = CUSPARSE.mm('N',d_A,d_B,'O')
            h_D = collect(d_D)
            D = A * B
            @test D ≈ h_D
            d_D = d_A*d_B
            h_D = collect(d_D)
            D = A * B
            @test D ≈ h_D
            d_A = Hermitian(CuSparseMatrixCSR(Ah))
            d_D = CUSPARSE.mm('N',d_A,d_B,'O')
            h_D = collect(d_D)
            D = Ah * B
            @test D ≈ h_D
        end
        @testset "csc" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = CuSparseMatrixCSC(A)
            @test_throws DimensionMismatch CUSPARSE.mm!('T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_B,'O')
            d_D = CUSPARSE.mm('N',alpha,d_A,d_B,beta,d_C,'O')
            h_D = collect(d_D)
            D = alpha * A * B + beta * C
            @test D ≈ h_D
            d_D = CUSPARSE.mm('N',d_A,d_B,beta,d_C,'O')
            h_D = collect(d_D)
            D = A * B + beta * C
            @test D ≈ h_D
            d_D = CUSPARSE.mm('N',d_A,d_B,d_C,'O')
            h_D = collect(d_D)
            D = A * B + C
            @test D ≈ h_D
            d_D = CUSPARSE.mm('N',alpha,d_A,d_B,'O')
            h_D = collect(d_D)
            D = alpha * A * B
            @test D ≈ h_D
            d_D = CUSPARSE.mm('N',d_A,d_B,'O')
            h_D = collect(d_D)
            D = A * B
            @test D ≈ h_D
            d_D = d_A*d_B
            h_D = collect(d_D)
            D = A * B
            @test D ≈ h_D
        end
    end
end

@testset "mm_symm" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = sparse(rand(elty,m,m))
        A = A + transpose(A)
        B = rand(elty,m,n)
        C = rand(elty,m,n)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "csr" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = Symmetric(CuSparseMatrixCSR(A))
            d_C = CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_C,'O')
            h_C = collect(d_C)
            D = alpha * A * B + beta * C
            @test D ≈ h_C
            d_C = transpose(d_A) * d_B
            h_C = collect(d_C)
            D = transpose(A) * B
            @test D ≈ h_C
            d_B = CUDA.rand(elty,k,n)
            @test_throws DimensionMismatch CUSPARSE.mm!('T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_B,'O')
        end
        @testset "csc" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = Symmetric(CuSparseMatrixCSC(A))
            d_C = CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_C,'O')
            h_C = collect(d_C)
            D = alpha * A * B + beta * C
            @test D ≈ h_C
            d_C = transpose(d_A) * d_B
            h_C = collect(d_C)
            D = transpose(A) * B
            @test D ≈ h_C
            d_B = CUDA.rand(elty,k,n)
            @test_throws DimensionMismatch CUSPARSE.mm!('T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_B,'O')
        end
    end
end
@testset "mm_herm" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = sparse(rand(elty,m,m))
        A = A + A'
        B = rand(elty,m,n)
        C = rand(elty,m,n)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "csr" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = Hermitian(CuSparseMatrixCSR(A))
            d_C = CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_C,'O')
            h_C = collect(d_C)
            D = alpha * A * B + beta * C
            @test D ≈ h_C
            d_B = CUDA.rand(elty,k,n)
            @test_throws DimensionMismatch CUSPARSE.mm!('T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_B,'O')
        end
        @testset "csc" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = Hermitian(CuSparseMatrixCSC(A))
            d_C = CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_C,'O')
            h_C = collect(d_C)
            D = alpha * A * B + beta * C
            @test D ≈ h_C
            d_B = CUDA.rand(elty,k,n)
            @test_throws DimensionMismatch CUSPARSE.mm!('T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_B,'O')
        end
    end
end

@testset "mm2" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = sparse(rand(elty,m,k))
        B = rand(elty,k,n)
        C = rand(elty,m,n)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "csr" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = CuSparseMatrixCSR(A)
            @test_throws DimensionMismatch CUSPARSE.mm2!('N','T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm2!('T','N',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm2!('T','T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm2!('N','N',alpha,d_A,d_B,beta,d_B,'O')
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
        @testset "csc" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = CuSparseMatrixCSC(A)
            @test_throws DimensionMismatch CUSPARSE.mm2!('N','T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm2!('T','N',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm2!('T','T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm2!('N','N',alpha,d_A,d_B,beta,d_B,'O')
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

@testset "mv" begin
    for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = sparse(rand(elty,m,m))
        x = rand(elty,m)
        y = rand(elty,m)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "mv_symm" begin
            A_s = A + transpose(A)
            @testset "csr" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = Symmetric(CuSparseMatrixCSR(A_s))
                d_y = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
                h_y = collect(d_y)
                z = alpha * A_s * x + beta * y
                @test z ≈ h_y
                x_  = rand(elty,n)
                d_x = CuArray(x_)
                @test_throws DimensionMismatch CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O')
            end
            @testset "csc" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = Symmetric(CuSparseMatrixCSC(A_s))
                d_y = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
                h_y = collect(d_y)
                z = alpha * A_s * x + beta * y
                @test z ≈ h_y
                x_  = rand(elty,n)
                d_x = CuArray(x_)
                @test_throws DimensionMismatch CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O')
            end
        end

        @testset "mv_herm" begin
            A_h = A + adjoint(A)
            @testset "csr" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = Hermitian(CuSparseMatrixCSR(A_h))
                d_y = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
                h_y = collect(d_y)
                z = alpha * A_h * x + beta * y
                @test z ≈ h_y
                x_  = rand(elty,n)
                d_x = CuArray(x_)
                @test_throws DimensionMismatch CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O')
            end
            @testset "csc" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = Hermitian(CuSparseMatrixCSC(A_h))
                d_y = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
                h_y = collect(d_y)
                z = alpha * A_h * x + beta * y
                @test z ≈ h_y
                x_  = rand(elty,n)
                d_x = CuArray(x_)
                @test_throws DimensionMismatch CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O')
            end
        end
        A = sparse(rand(elty,m,n))
        x = rand(elty,n)
        y = rand(elty,m)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "mv" begin
            @testset "csr" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = CuSparseMatrixCSR(A)
                @test_throws DimensionMismatch CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O')
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
                h_z = collect(d_z)
                z = alpha * A * x + beta * y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,beta,d_y,'O')
                h_z = collect(d_z)
                z = A * x + beta * y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,d_y,'O')
                h_z = collect(d_z)
                z = A * x + y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,d_y,'O')
                h_z = collect(d_z)
                z = alpha * A * x + y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,'O')
                h_z = collect(d_z)
                z = alpha * A * x
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,'O')
                h_z = collect(d_z)
                z = A * x
                @test z ≈ h_z
                d_z = d_A*d_x
                h_z = collect(d_z)
                z = A * x
                @test z ≈ h_z
            end
            @testset "csc" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = CuSparseMatrixCSC(A)
                @test_throws DimensionMismatch CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O')
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
                h_z = collect(d_z)
                z = alpha * A * x + beta * y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,beta,d_y,'O')
                h_z = collect(d_z)
                z = A * x + beta * y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,d_y,'O')
                h_z = collect(d_z)
                z = A * x + y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,d_y,'O')
                h_z = collect(d_z)
                z = alpha * A * x + y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,'O')
                h_z = collect(d_z)
                z = alpha * A * x
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,'O')
                h_z = collect(d_z)
                z = A * x
                @test z ≈ h_z
                d_z = d_A*d_x
                h_z = collect(d_z)
                z = A * x
                @test z ≈ h_z
            end
            @testset "bsr" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = CuSparseMatrixCSR(A)
                d_A = CUSPARSE.switch2bsr(d_A,convert(Cint,blockdim))
                @test_throws DimensionMismatch CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O')
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
                h_z = collect(d_z)
                z = alpha * A * x + beta * y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,beta,d_y,'O')
                h_z = collect(d_z)
                z = A * x + beta * y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,d_y,'O')
                h_z = collect(d_z)
                z = A * x + y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,d_y,'O')
                h_z = collect(d_z)
                z = alpha * A * x + y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,'O')
                h_z = collect(d_z)
                z = alpha * A * x
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,'O')
                h_z = collect(d_z)
                z = A * x
                @test z ≈ h_z
                d_z = d_A*d_x
                h_z = collect(d_z)
                z = A * x
                @test z ≈ h_z
                @test_throws DimensionMismatch CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O')
            end
            @testset "hyb" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = CuSparseMatrixCSR(A)
                d_A = CUSPARSE.switch2hyb(d_A)
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
                h_z = collect(d_z)
                z = alpha * A * x + beta * y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,beta,d_y,'O')
                h_z = collect(d_z)
                z = A * x + beta * y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,d_y,'O')
                h_z = collect(d_z)
                z = A * x + y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,d_y,'O')
                h_z = collect(d_z)
                z = alpha * A * x + y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,'O')
                h_z = collect(d_z)
                z = alpha * A * x
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,'O')
                h_z = collect(d_z)
                z = A * x
                @test z ≈ h_z
                d_z = d_A*d_x
                h_z = collect(d_z)
                z = A * x
                @test z ≈ h_z
                @test_throws DimensionMismatch CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O')
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
        @testset "csr" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = CuSparseMatrixCSR(A)
            CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_C,'O')
            h_D = collect(d_C)
            D = alpha * A * B + beta * C
            @test D ≈ h_D
            d_C = CuArray(C)
            mul!(d_C, d_A, d_B)
            h_C = collect(d_C)
            D = A * B
            @test D ≈ h_C
        end
        @testset "csc" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = CuSparseMatrixCSC(A)
            CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_C,'O')
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
end
@testset "mm_symm!" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = sparse(rand(elty,m,m))
        A = A + transpose(A)
        B = rand(elty,m,n)
        C = rand(elty,m,n)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "csr" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = Symmetric(CuSparseMatrixCSR(A))
            CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_C,'O')
            h_C = collect(d_C)
            D = alpha * A * B + beta * C
            @test D ≈ h_C
            d_C = CuArray(C)
            mul!(d_C, d_A, d_B)
            h_C = collect(d_C)
            D = A * B
            @test D ≈ h_C
        end
        @testset "csc" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = Symmetric(CuSparseMatrixCSC(A))
            CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_C,'O')
            h_C = collect(d_C)
            D = alpha * A * B + beta * C
            @test D ≈ h_C
            d_C = CuArray(C)
            mul!(d_C, d_A, d_B)
            h_C = collect(d_C)
            D = A * B
            @test D ≈ h_C
        end
    end
end
@testset "mm_herm!" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = sparse(rand(elty,m,m))
        A = A + A'
        B = rand(elty,m,n)
        C = rand(elty,m,n)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "csr" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = Hermitian(CuSparseMatrixCSR(A))
            CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_C,'O')
            h_C = collect(d_C)
            D = alpha * A * B + beta * C
            @test D ≈ h_C
            d_C = CuArray(C)
            mul!(d_C, d_A, d_B)
            h_C = collect(d_C)
            D = A * B
            @test D ≈ h_C
        end
        @testset "csc" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = Hermitian(CuSparseMatrixCSC(A))
            CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_C,'O')
            h_C = collect(d_C)
            D = alpha * A * B + beta * C
            @test D ≈ h_C
            d_C = CuArray(C)
            mul!(d_C, d_A, d_B)
            h_C = collect(d_C)
            D = A * B
            @test D ≈ h_C
        end
    end
end
@testset "mm2!" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = sparse(rand(elty,m,k))
        B = rand(elty,k,n)
        C = rand(elty,m,n)
        Bᵀ = collect(transpose(B))
        alpha = rand(elty)
        beta = rand(elty)
        @testset "csr" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = CuSparseMatrixCSR(A)
            CUSPARSE.mm2!('N','N',alpha,d_A,d_B,beta,d_C,'O')
            h_D = collect(d_C)
            D = alpha * A * B + beta * C
            @test D ≈ h_D
            d_C = CuArray(C)
            mul!(d_C, d_A, d_B)
            h_C = collect(d_C)
            D = A * B
            @test D ≈ h_C
            d_Bᵀ = CuArray(Bᵀ)
            d_C = CuArray(C)
            mul!(d_C, d_A, transpose(d_Bᵀ))
            h_C = collect(d_C)
            D = A * transpose(Bᵀ)
            @test D ≈ h_C
        end
        @testset "csc" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = CuSparseMatrixCSC(A)
            CUSPARSE.mm2!('N','N',alpha,d_A,d_B,beta,d_C,'O')
            h_D = collect(d_C)
            D = alpha * A * B + beta * C
            @test D ≈ h_D
            d_C = CuArray(C)
            mul!(d_C, d_A, d_B)
            h_C = collect(d_C)
            D = A * B
            @test D ≈ h_C
            d_Bᵀ = CuArray(Bᵀ)
            d_C = CuArray(C)
            @test_throws CUSPARSEError mul!(d_C, d_A, transpose(d_Bᵀ))
        end
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
        A = sparse(rand(elty,m,m))
        x = rand(elty,m)
        y = rand(elty,m)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "mv_symm!" begin
            A_s = A + transpose(A)
            @testset "csr" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = Symmetric(CuSparseMatrixCSR(A_s))
                CUSPARSE.mv!('N',alpha,d_A,d_x,beta,d_y,'O')
                h_y = collect(d_y)
                z = alpha * A_s * x + beta * y
                @test z ≈ h_y
                x_  = rand(elty,n)
                d_x = CuArray(x_)
                @test_throws DimensionMismatch CUSPARSE.mv!('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv!('N',alpha,d_A,d_y,beta,d_x,'O')
                d_x = CuArray(x)
                mul!(d_y, d_A, d_x)
                h_y = collect(d_y)
                z = A_s * x
                @test z ≈ h_y
            end
            @testset "csc" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = Symmetric(CuSparseMatrixCSC(A_s))
                mv!('N',alpha,d_A,d_x,beta,d_y,'O')
                h_y = collect(d_y)
                z = alpha * A_s * x + beta * y
                @test z ≈ h_y
                x_  = rand(elty,n)
                d_x = CuArray(x_)
                @test_throws DimensionMismatch CUSPARSE.mv!('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv!('N',alpha,d_A,d_y,beta,d_x,'O')
                d_x = CuArray(x)
                mul!(d_y, d_A, d_x)
                h_y = collect(d_y)
                z = A_s * x
                @test z ≈ h_y
            end
        end

        @testset "mv_herm!" begin
            A_h = A + adjoint(A)
            @testset "csr" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = Hermitian(CuSparseMatrixCSR(A_h))
                CUSPARSE.mv!('N',alpha,d_A,d_x,beta,d_y,'O')
                h_y = collect(d_y)
                z = alpha * A_h * x + beta * y
                @test z ≈ h_y
                x_  = rand(elty,n)
                d_x = CuArray(x_)
                @test_throws DimensionMismatch CUSPARSE.mv!('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv!('N',alpha,d_A,d_y,beta,d_x,'O')
                d_x = CuArray(x)
                mul!(d_y, d_A, d_x)
                h_y = collect(d_y)
                z = A_h * x
                @test z ≈ h_y
            end
            @testset "csc" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = Hermitian(CuSparseMatrixCSC(A_h))
                CUSPARSE.mv!('N',alpha,d_A,d_x,beta,d_y,'O')
                h_y = collect(d_y)
                z = alpha * A_h * x + beta * y
                @test z ≈ h_y
                x_  = rand(elty,n)
                d_x = CuArray(x_)
                @test_throws DimensionMismatch CUSPARSE.mv!('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv!('N',alpha,d_A,d_y,beta,d_x,'O')
                d_x = CuArray(x)
                mul!(d_y, d_A, d_x)
                h_y = collect(d_y)
                z = A_h * x
                @test z ≈ h_y
            end
        end
        A = sparse(rand(elty,m,n))
        x = rand(elty,n)
        y = rand(elty,m)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "mv!" begin
            @testset "csr" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = CuSparseMatrixCSR(A)
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
            end
            @testset "csc" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = CuSparseMatrixCSC(A)
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
            end
            @testset "bsr" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = CuSparseMatrixCSR(A)
                d_A = CUSPARSE.switch2bsr(d_A,convert(Cint,blockdim))
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
            end
            @testset "hyb" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = CuSparseMatrixCSR(A)
                d_A = CUSPARSE.switch2hyb(d_A)
                CUSPARSE.mv!('N',alpha,d_A,d_x,beta,d_y,'O')
                h_z = collect(d_y)
                z = alpha * A * x + beta * y
                @test z ≈ h_z
                @test_throws DimensionMismatch CUSPARSE.mv!('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv!('N',alpha,d_A,d_y,beta,d_x,'O')
                mul!(d_y, d_A, d_x)
                h_y = collect(d_y)
                z = A * x
                @test z ≈ h_y
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

@testset "mul!" begin
    for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = sparse(rand(elty,m,m))
        x = rand(elty,m)
        y = rand(elty,m)
        @testset "csr -- $elty" begin
            d_x  = CuArray(x)
            d_y  = CuArray(y)
            d_A  = CuSparseMatrixCSR(A)
            d_Aᵀ = transpose(d_A)
            d_Aᴴ = adjoint(d_A)
            CUSPARSE.mul!(d_y, d_A, d_x)
            h_y = collect(d_y)
            z = A * x
            @test z ≈ h_y
            CUSPARSE.mul!(d_y, d_Aᵀ, d_x)
            h_y = collect(d_y)
            z = transpose(A) * x
            @test z ≈ h_y
            CUSPARSE.mul!(d_y, d_Aᴴ, d_x)
            h_y = collect(d_y)
            z = adjoint(A) * x
            @test z ≈ h_y
        end
        @testset "csc -- $elty" begin
            d_x = CuArray(x)
            d_y = CuArray(y)
            d_A = CuSparseMatrixCSC(A)
            d_Aᵀ = transpose(d_A)
            d_Aᴴ = adjoint(d_A)
            CUSPARSE.mul!(d_y, d_A, d_x)
            h_y = collect(d_y)
            z = A * x
            @test z ≈ h_y
            CUSPARSE.mul!(d_y, d_Aᵀ, d_x)
            h_y = collect(d_y)
            z = transpose(A) * x
            @test z ≈ h_y
            CUSPARSE.mul!(d_y, d_Aᴴ, d_x)
            h_y = collect(d_y)
            z = adjoint(A) * x
            @test z ≈ h_y
        end
    end
end
