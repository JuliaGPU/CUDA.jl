using CUDA.CUSPARSE
using LinearAlgebra, SparseArrays
using FillArrays

@testset "FillArrays - Kronecker product" begin
    @testset "T = $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
        m, n = 100, 80
        A = sprand(T, m, n, 0.2)
        
        # Test with Ones
        @testset "Ones - size $diag_size" for diag_size in [5, 10]
            C_ones = Diagonal(Ones{T}(diag_size))
            dA = CuSparseMatrixCSR(A)
            
            # Test kron(sparse, diagonal)
            result_gpu = collect(kron(dA, C_ones))
            result_cpu = kron(A, collect(C_ones))
            @test result_gpu ≈ result_cpu
            
            # Test kron(diagonal, sparse)
            result_gpu = collect(kron(C_ones, dA))
            result_cpu = kron(collect(C_ones), A)
            @test result_gpu ≈ result_cpu
        end
        
        # Test with Fill (constant value)
        @testset "Fill - value $val, size $diag_size" for val in [T(2.5), T(-1.0)], diag_size in [5, 10]
            C_fill = Diagonal(Fill(val, diag_size))
            dA = CuSparseMatrixCSR(A)
            
            # Test kron(sparse, diagonal)
            result_gpu = collect(kron(dA, C_fill))
            result_cpu = kron(A, collect(C_fill))
            @test result_gpu ≈ result_cpu
            
            # Test kron(diagonal, sparse)
            result_gpu = collect(kron(C_fill, dA))
            result_cpu = kron(collect(C_fill), A)
            @test result_gpu ≈ result_cpu
        end
        
        # Test with Zeros
        @testset "Zeros - size $diag_size" for diag_size in [5, 10]
            C_zeros = Diagonal(Zeros{T}(diag_size))
            dA = CuSparseMatrixCSR(A)
            
            # Test kron(sparse, diagonal)
            result_gpu = kron(dA, C_zeros)
            result_cpu = kron(A, collect(C_zeros))
            @test SparseMatrixCSC(result_gpu) ≈ result_cpu
            @test nnz(result_gpu) == 0
            
            # Test kron(diagonal, sparse)
            result_gpu = kron(C_zeros, dA)
            result_cpu = kron(collect(C_zeros), A)
            @test SparseMatrixCSC(result_gpu) ≈ result_cpu
            @test nnz(result_gpu) == 0
        end
        
        # Test with transpose and adjoint wrappers
        @testset "Transpose/Adjoint with Fill" begin
            diag_size = 5
            val = T(3.0)
            C_fill = Diagonal(Fill(val, diag_size))
            
            @testset "opa = $opa" for opa in (identity, transpose, adjoint)
                dA = CuSparseMatrixCSR(A)
                
                # Test kron(opa(sparse), diagonal)
                result_gpu = collect(kron(opa(dA), C_fill))
                result_cpu = kron(opa(A), collect(C_fill))
                @test result_gpu ≈ result_cpu
                
                # Test kron(diagonal, opa(sparse))
                result_gpu = collect(kron(C_fill, opa(dA)))
                result_cpu = kron(collect(C_fill), opa(A))
                @test result_gpu ≈ result_cpu
            end
        end
        
        # Test type promotion
        @testset "Type promotion" begin
            @testset "T1 = $T1, T2 = $T2" for (T1, T2) in [(Float64, Float32), (ComplexF64, Float64), (ComplexF32, Float32)]
                A_T1 = sprand(T1, m, n, 0.2)
                dA_T1 = CuSparseMatrixCSR(A_T1)
                C_fill_T2 = Diagonal(Fill(T2(2.5), 5))
                
                # Test kron(sparse_T1, diagonal_T2)
                result_gpu = kron(dA_T1, C_fill_T2)
                result_cpu = kron(A_T1, collect(C_fill_T2))
                @test eltype(result_gpu) == promote_type(T1, T2)
                @test collect(result_gpu) ≈ result_cpu
                
                # Test kron(diagonal_T2, sparse_T1)
                result_gpu = kron(C_fill_T2, dA_T1)
                result_cpu = kron(collect(C_fill_T2), A_T1)
                @test eltype(result_gpu) == promote_type(T1, T2)
                @test collect(result_gpu) ≈ result_cpu
            end
        end
    end
end
