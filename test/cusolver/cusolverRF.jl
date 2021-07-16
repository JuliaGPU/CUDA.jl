using LinearAlgebra, SparseArrays, Test
using CUDA
using CUDA.CUSOLVER
using CUDA.CUSPARSE

n = 512

@testset "cusolverRF" begin
    @testset "RfLU factorization" begin
        A = sprand(n, n, .2)
        A += A'
        b = rand(n)
        # Compute solution with UMFPACK
        solution = A \ b

        d_A = CuSparseMatrixCSR(A)
        d_b = CuVector{Float64}(b)
        d_x = CUDA.zeros(Float64, n)

        rflu = CUSOLVER.RfLU(d_A)
        @test isa(rflu, LinearAlgebra.Factorization)

        copyto!(d_x, d_b)
        CUSOLVER.rf_solve!(rflu, d_x)
        res = Array(d_x)
        @test isapprox(res, solution)
        # Test refactoring
        scale = 2.0
        d_A.nzVal .*= scale
        CUSOLVER.rf_refactor!(rflu, d_A)
        copyto!(d_x, d_b)
        CUSOLVER.rf_solve!(rflu, d_x)
        res = Array(d_x)
        @test isapprox(res, solution ./ scale)

        # Test LinearAlgebra's overloading
        d_A = CuSparseMatrixCSR(A)
        rflu = lu(d_A)
        ldiv!(d_x, rflu, d_b)
        res = Array(d_x)
        @test isapprox(res, solution)
        d_A.nzVal .*= scale
        lu!(rflu, d_A)
        ldiv!(d_x, rflu, d_b)
        res = Array(d_x)
        @test isapprox(res, solution ./ scale)
    end

    @testset "RfBatchLU factorization" begin
        # One matrix, multiple RHS
        A = sprand(n, n, .2)
        A += A'
        nbatch = 32
        B = rand(n, nbatch)
        # Compute solution with UMFPACK
        solution = A \ B

        d_A = CuSparseMatrixCSR(A)
        d_B = CuMatrix{Float64}(B)
        d_X = CUDA.zeros(Float64, n, nbatch)
        rflu = CUSOLVER.RfBatchLU(d_A, nbatch)
        @test isa(rflu, LinearAlgebra.Factorization)

        copyto!(d_X, d_B)
        CUSOLVER.rf_batch_solve!(rflu, d_X)
        res = Array(d_X)
        @test isapprox(res, solution)

        # Refactoring
        scale = 2.0
        d_A.nzVal .*= scale
        CUSOLVER.rf_batch_refactor!(rflu, d_A)
        copyto!(d_X, d_B)
        CUSOLVER.rf_batch_solve!(rflu, d_X)
        res = Array(d_X)
        @test isapprox(res, solution ./ scale)

        # Test LinearAlgebra's overloading
        d_A = CuSparseMatrixCSR(A)
        rflu = CUSOLVER.RfBatchLU(d_A, nbatch)
        ldiv!(d_X, rflu, d_B)
        res = Array(d_X)
        @test isapprox(res, solution)
        d_A.nzVal .*= scale
        lu!(rflu, d_A)
        ldiv!(d_X, rflu, d_B)
        res = Array(d_X)
        @test isapprox(res, solution ./ scale)

        # Parallel refactoring
        # Matrices should have the same sparsity pattern
        I, J, V = findnz(A)
        nnzA = length(V)
        # Create a batch of matrices
        As_batch = [sparse(I, J, randn(nnzA)) for i in 1:nbatch]
        d_As_batch = [CuSparseMatrixCSR(Ab) for Ab in As_batch]
        CUSOLVER.rf_batch_refactor!(rflu, d_As_batch)
        ldiv!(d_X, rflu, d_B)
        res = Array(d_X)
        for i in 1:nbatch
            solution = As_batch[i] \ B[:, i]
            @test isapprox(res[:, i], solution)
        end
    end
end

