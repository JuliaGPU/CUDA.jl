using CUDA.CUBLAS

using LinearAlgebra

using BFloat16s
using StaticArrays

@test CUBLAS.version() isa VersionNumber
@test CUBLAS.version().major == CUBLAS.cublasGetProperty(CUDA.MAJOR_VERSION)
@test CUBLAS.version().minor == CUBLAS.cublasGetProperty(CUDA.MINOR_VERSION)
@test CUBLAS.version().patch == CUBLAS.cublasGetProperty(CUDA.PATCH_LEVEL)

m = 20
n = 35
k = 13

@testset "level 1" begin
    @testset for T in [Float32, Float64, ComplexF32, ComplexF64]
        A = CUDA.rand(T, m)
        B = CuArray{T}(undef, m)
        CUBLAS.copy!(m,A,B)
        @test Array(A) == Array(B)

        @test testf(rmul!, rand(T, 6, 9, 3), Ref(rand()))
        @test testf(dot, rand(T, m), rand(T, m))
        @test testf(*, transpose(rand(T, m)), rand(T, m))
        @test testf(*, rand(T, m)', rand(T, m))
        @test testf(norm, rand(T, m))
        @test testf(BLAS.asum, rand(T, m))
        @test testf(axpy!, Ref(rand()), rand(T, m), rand(T, m))
        @test testf(axpby!, Ref(rand()), rand(T, m), Ref(rand()), rand(T, m))

        if T <: Complex
            @test testf(dot, rand(T, m), rand(T, m))
            x = rand(T, m)
            y = rand(T, m)
            dx = CuArray(x)
            dy = CuArray(y)
            dz = dot(dx, dy)
            z = dot(x, y)
            @test dz ≈ z
        end

        @testset "rotate!" begin
            @test testf(rotate!, rand(T, m), rand(T, m), rand(real(T)), rand(real(T)))
            @test testf(rotate!, rand(T, m), rand(T, m), rand(real(T)), rand(T))
        end
        @testset "reflect!" begin
            @test testf(reflect!, rand(T, m), rand(T, m), rand(real(T)), rand(real(T)))
            @test testf(reflect!, rand(T, m), rand(T, m), rand(real(T)), rand(T))
        end

        @testset "swap!" begin
            # swap is an extension
            x = rand(T, m)
            y = rand(T, m)
            dx = CuArray(x)
            dy = CuArray(y)
            CUBLAS.swap!(m, dx, dy)
            h_x = collect(dx)
            h_y = collect(dy)
            @test h_x ≈ y
            @test h_y ≈ x
        end

        @testset "iamax/iamin" begin
            a = convert.(T, [1.0, 2.0, -0.8, 5.0, 3.0])
            ca = CuArray(a)
            @test BLAS.iamax(a) == CUBLAS.iamax(ca)
            @test CUBLAS.iamin(ca) == 3
        end
    end # level 1 testset
    @testset for T in [Float16, ComplexF16]
        A = CuVector(rand(T, m)) # CUDA.rand doesn't work with 16 bit types yet
        B = CuArray{T}(undef, m)
        CUBLAS.copy!(m,A,B)
        @test Array(A) == Array(B)

        @test testf(dot, rand(T, m), rand(T, m))
        @test testf(*, transpose(rand(T, m)), rand(T, m))
        @test testf(*, rand(T, m)', rand(T, m))
        @test testf(norm, rand(T, m))
        @test testf(axpy!, Ref(rand()), rand(T, m), rand(T, m))
        @test testf(LinearAlgebra.axpby!, Ref(rand()), rand(T, m), Ref(rand()), rand(T, m))

        if T <: Complex
            @test testf(dot, rand(T, m), rand(T, m))
            x = rand(T, m)
            y = rand(T, m)
            dx = CuArray(x)
            dy = CuArray(y)
            dz = dot(dx, dy)
            z = dot(x, y)
            @test dz ≈ z
        end
    end # level 1 testset
end
