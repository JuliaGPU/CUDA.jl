@testset "bsric02" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = rand(elty, m, m)
        A += adjoint(A)
        A += m * Diagonal{elty}(I, m)

        @testset "bsric02!" begin
            d_A = CuSparseMatrixCSR(sparse(tril(A)))
            d_A = CuSparseMatrixBSR(d_A, blockdim)
            d_A = cuSPARSE.ic02!(d_A)
            h_A = SparseMatrixCSC(CuSparseMatrixCSR(d_A))
            Ac = sparse(Array(cholesky(Hermitian(A))))
            h_A = transpose(h_A) * h_A
            @test rowvals(h_A) ≈ rowvals(Ac)
            @test reduce(&, isfinite.(nonzeros(h_A)))
            d_A = CuSparseMatrixCSR(sparse(tril(rand(elty,m,n))))
            d_A = CuSparseMatrixBSR(d_A, blockdim)
            @test_throws DimensionMismatch cuSPARSE.ic02!(d_A)
        end

        @testset "bsric02" begin
            d_A = CuSparseMatrixCSR(sparse(tril(A)))
            d_A = CuSparseMatrixBSR(d_A, blockdim)
            d_B = cuSPARSE.ic02(d_A)
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
            d_A = cuSPARSE.ilu02!(d_A)
            h_A = SparseMatrixCSC(CuSparseMatrixCSR(d_A))
            pivot = NoPivot()
            Alu = lu(Array(A), pivot)
            Ac = sparse(Alu.L*Alu.U)
            h_A = adjoint(h_A) * h_A
            @test rowvals(h_A) ≈ rowvals(Ac)
            @test reduce(&, isfinite.(nonzeros(h_A)))
            d_A = CuSparseMatrixCSR(sparse(rand(elty,m,n)))
            d_A = CuSparseMatrixBSR(d_A, blockdim)
            @test_throws DimensionMismatch cuSPARSE.ilu02!(d_A)
        end

        @testset "bsrilu02" begin
            d_A = CuSparseMatrixCSR(sparse(A))
            d_A = CuSparseMatrixBSR(d_A, blockdim)
            d_B = cuSPARSE.ilu02(d_A)
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

@testset "ilu02" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        @testset "csr" begin
            A = rand(elty,m,m)
            A += transpose(A)
            A += m * Diagonal{elty}(I, m)
            d_A = CuSparseMatrixCSR(sparse(A))
            d_B = cuSPARSE.ilu02(d_A)
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
            d_B = cuSPARSE.ilu02(d_A)
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
            d_B = cuSPARSE.ic02(d_A)
            h_A = SparseMatrixCSC(d_B)
            Ac  = sparse(Array(cholesky(Hermitian(A))))
            h_A = transpose(h_A) * h_A
            @test rowvals(h_A) ≈ rowvals(Ac)
            @test reduce(&, isfinite.(nonzeros(h_A)))
            A   = rand(elty,m,n)
            d_A = CuSparseMatrixCSR(sparse(tril(A)))
            @test_throws DimensionMismatch cuSPARSE.ic02(d_A)
        end
        @testset "csc" begin
            A   = rand(elty, m, m)
            A  += adjoint(A)
            A  += m * Diagonal{elty}(I, m)
            d_A = CuSparseMatrixCSC(sparse(tril(A)))
            d_B = cuSPARSE.ic02(d_A)
            h_A = SparseMatrixCSC(d_B)
            Ac  = sparse(Array(cholesky(Hermitian(A))))
            h_A = transpose(h_A) * h_A
            @test rowvals(h_A) ≈ rowvals(Ac)
            @test reduce(&, isfinite.(nonzeros(h_A)))
            A   = rand(elty,m,n)
            d_A = CuSparseMatrixCSC(sparse(tril(A)))
            @test_throws DimensionMismatch cuSPARSE.ic02(d_A)
        end
    end
end
