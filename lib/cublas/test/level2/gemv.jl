using cuBLAS
using LinearAlgebra

using StaticArrays

@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    m = 20
    n = 35

    @testset "gemv" begin
        alpha = rand(elty)
        @test testf(*, rand(elty, m, n), rand(elty, n))
        @test testf(*, transpose(rand(elty, m, n)), rand(elty, m))
        @test testf(*, rand(elty, m, n)', rand(elty, m))

        x = rand(elty, m)
        dx = CuArray(x)
        dA = CuArray(rand(elty, m, m + 1))
        dy = CuArray(rand(elty, n))
        @test_throws DimensionMismatch mul!(dy, dA, dx)
        dA = CuArray(rand(elty, m + 1, m))
        @test_throws DimensionMismatch mul!(dy, dA, dx)

        A = rand(elty, n, m)
        y = rand(elty, n)
        dA = CuArray(A)

        dr = cuBLAS.gemv('N', alpha, dA, dx)
        @test Array(dr) ≈ alpha * A * x

        dr = cuBLAS.gemv('N', dA, dx)
        @test Array(dr) ≈ A * x

        dy = CuArray(y)
        dr = cuBLAS.gemv(elty <: Real ? 'T' : 'C', alpha, dA, dy)
        @test collect(dr) ≈ alpha * A' * y
    end

    @testset "gemv_batched" begin
        alpha = rand(elty)
        beta = rand(elty)
        x = [rand(elty, m) for i=1:10]
        A = [rand(elty, n, m) for i=1:10]
        y = [rand(elty, n) for i=1:10]
        dx = CuArray{elty, 1}[]
        dA = CuArray{elty, 2}[]
        dy = CuArray{elty, 1}[]
        dbad = CuArray{elty, 1}[]
        dx_bad = CuArray{elty, 1}[]
        dA_bad = CuArray{elty, 2}[]
        for i=1:length(A)
            push!(dA, CuArray(A[i]))
            push!(dx, CuArray(x[i]))
            push!(dy, CuArray(y[i]))
            if i < length(A) - 2
                push!(dbad, CuArray(dx[i]))
                push!(dx_bad, CuArray(dx[i]))
                push!(dA_bad, CuArray(A[i]))
            else
                push!(dx_bad, CuArray(rand(elty, m+1)))
                push!(dA_bad, CuArray(rand(elty, n+1, m+1)))
            end
        end
        @test_throws DimensionMismatch cuBLAS.gemv_batched!('N', alpha, dA, dx, beta, dbad)
        @test_throws DimensionMismatch cuBLAS.gemv_batched!('N', alpha, dA, dx_bad, beta, dy)
        @test_throws DimensionMismatch cuBLAS.gemv_batched!('N', alpha, dA_bad, dx, beta, dy)
        cuBLAS.gemv_batched!('N', alpha, dA, dx, beta, dy)
        for i in 1:length(A)
            y[i] = alpha * A[i] * x[i] + beta * y[i]
            @test y[i] ≈ collect(dy[i])
        end

        dy = CuArray{elty, 1}[CuArray(y[i]) for i=1:length(A)]
        cuBLAS.gemv_batched!(elty <: Real ? 'T' : 'C', alpha, dA, dy, beta, dx)
        for i in 1:length(A)
            x[i] = alpha * A[i]' * y[i] + beta * x[i]
            @test x[i] ≈ collect(dx[i])
        end
    end

    @testset "gemv_strided_batched" begin
        alpha = rand(elty)
        beta = rand(elty)
        x = rand(elty, m, 10)
        A = rand(elty, n, m, 10)
        y = rand(elty, n, 10)
        dx = CuArray(x)
        dA = CuArray(A)
        dy = CuArray(y)

        dbad = CuArray(rand(elty, m, 10))
        @test_throws DimensionMismatch cuBLAS.gemv_strided_batched!('N', alpha, dA, dx, beta, dbad)
        dbad = CuArray(rand(elty, n, 2))
        @test_throws DimensionMismatch cuBLAS.gemv_strided_batched!('N', alpha, dA, dx, beta, dbad)

        cuBLAS.gemv_strided_batched!('N', alpha, dA, dx, beta, dy)
        for i in 1:size(A, 3)
            y[:, i] = alpha * A[:, :, i] * x[:, i] + beta * y[:, i]
            @test y[:, i] ≈ collect(dy[:, i])
        end
        dy = CuArray(y)
        cuBLAS.gemv_strided_batched!(elty <: Real ? 'T' : 'C', alpha, dA, dy, beta, dx)
        for i in 1:size(A, 3)
            x[:, i] = alpha * A[:, :, i]' * y[:, i] + beta * x[:, i]
            @test x[:, i] ≈ collect(dx[:, i])
        end
    end

    @testset "mul! y = $f(A) * x * $Ts(a) + y * $Ts(b)" for f in (identity, transpose, adjoint),
                                                            Ts in (Int, elty)
        y, A, x = rand(elty, 5), rand(elty, 5, 5), rand(elty, 5)
        dy, dA, dx = CuArray(y), CuArray(A), CuArray(x)
        mul!(dy, f(dA), dx, Ts(1), Ts(2))
        mul!(y, f(A), x, Ts(1), Ts(2))
        @test Array(dy) ≈ y
    end

    @testset "hermitian" begin
        y, A, x = rand(elty, 5), Hermitian(rand(elty, 5, 5)), rand(elty, 5)
        dy, dA, dx = CuArray(y), Hermitian(CuArray(A)), CuArray(x)
        mul!(dy, dA, dx)
        mul!(y, A, x)
        @test Array(dy) ≈ y
    end
end

@testset "gemv! with strided inputs" begin  # JuliaGPU/CUDA.jl#445
    testf(rand(16), rand(4)) do p, b
        W = @view p[reshape(1:16, 4, 4)]
        W * b
    end
end

@testset "StaticArray eltype" begin
    A = CuArray(rand(SVector{2, Float64}, 3, 3))
    B = CuArray(rand(Float64, 3, 1))
    @test Array(A * B) ≈ Array(A) * Array(B)
end
