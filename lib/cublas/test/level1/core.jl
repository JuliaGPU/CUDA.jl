using cuBLAS
using LinearAlgebra

@testset for T in [Float32, Float64, ComplexF32, ComplexF64]
    m = 20

    @testset "copy!" begin
        A = CuArray(rand(T, m))
        B = CuArray{T}(undef, m)
        cuBLAS.copy!(m, A, B)
        @test Array(A) == Array(B)
    end

    @test testf(rmul!, rand(T, 6, 9, 3), rand())
    @test testf(dot, rand(T, m), rand(T, m))
    @test testf(dot, rand(T, 0), rand(T, 0))
    @test testf(*, transpose(rand(T, m)), rand(T, m))
    @test testf(*, rand(T, m)', rand(T, m))
    @test testf(norm, rand(T, m))
    @test testf(norm, rand(T, 0))
    @test testf(LinearAlgebra.norm2, rand(T, m))
    @test testf(BLAS.asum, rand(T, m))
    @test testf(BLAS.asum, rand(T, 0))

    @test testf(axpy!, rand(), rand(T, m), rand(T, m))
    @test testf(LinearAlgebra.axpby!, rand(), rand(T, m), rand(), rand(T, m))

    if T <: Complex
        @test testf(dot, rand(T, m), rand(T, m))
        x = rand(T, m)
        y = rand(T, m)
        dx = CuArray(x)
        dy = CuArray(y)
        @test dot(dx, dy) ≈ dot(x, y)
    end

    @testset "rmul! strong zero" begin
        @test testf(rmul!, fill(T(NaN), 3), false)
        @test testf(rmul!, rand(T, 3), false)
        @test testf(rmul!, rand(T, 3), true)
    end

    @testset "swap!" begin
        x = rand(T, m)
        y = rand(T, m)
        dx = CuArray(x)
        dy = CuArray(y)
        cuBLAS.swap!(m, dx, dy)
        @test collect(dx) ≈ y
        @test collect(dy) ≈ x
    end

    @testset "iamax/iamin" begin
        a = convert.(T, [1.0, 2.0, -0.8, 5.0, 3.0])
        ca = CuArray(a)
        @test BLAS.iamax(a) == cuBLAS.iamax(ca)
        @test cuBLAS.iamin(ca) == 3
        result = CuRef{Int64}(0)
        cuBLAS.iamax(ca, result)
        @test BLAS.iamax(a) == result[]
    end

    @testset "nrm2 with result" begin
        x = rand(T, m)
        dx = CuArray(x)
        result = CuRef{real(T)}(zero(real(T)))
        cuBLAS.nrm2(dx, result)
        @test norm(x) ≈ result[]
    end

    @testset "norm of Diagonal" begin
        x = rand(T, m)
        dDx = Diagonal(CuArray(x))
        Dx = Diagonal(x)
        @test norm(dDx, 1) ≈ norm(Dx, 1)
        @test norm(dDx, 2) ≈ norm(Dx, 2)
        @test norm(dDx, Inf) ≈ norm(Dx, Inf)
    end
end

@testset for T in [Float16, ComplexF16]
    m = 20

    @testset "copy!" begin
        # CUDA.rand doesn't work with 16 bit types yet
        A = CuVector(rand(T, m))
        B = CuArray{T}(undef, m)
        cuBLAS.copy!(m, A, B)
        @test Array(A) == Array(B)
    end

    @test testf(rmul!, rand(T, 6, 9, 3), rand())
    @test testf(dot, rand(T, m), rand(T, m))
    @test testf(dot, rand(T, 0), rand(T, 0))
    @test testf(*, transpose(rand(T, m)), rand(T, m))
    @test testf(*, rand(T, m)', rand(T, m))
    @test testf(norm, rand(T, m))
    @test testf(norm, rand(T, 0))
    @test testf(LinearAlgebra.norm2, rand(T, m))
    @test testf(axpy!, rand(), rand(T, m), rand(T, m))
    @test testf(LinearAlgebra.axpby!, rand(), rand(T, m), rand(), rand(T, m))

    @testset "scal!" begin
        x = rand(T, m)
        d_x = CuArray(x)
        α = rand(Float32)
        d_α = CuArray([α])
        d_x = cuBLAS.scal!(m, d_α, d_x)
        @test Array(d_x) ≈ α * x
    end

    if T <: Complex
        @test testf(dot, rand(T, m), rand(T, m))
        x = rand(T, m)
        y = rand(T, m)
        dx = CuArray(x)
        dy = CuArray(y)
        @test dot(dx, dy) ≈ dot(x, y)
    end
end

@testset "dot with mixed types" begin
    m = 20
    T1 = Float32
    T2 = Float64
    @test testf(dot, rand(T1, m), rand(T2, m))
    @test testf(dot, rand(T1, 0), rand(T2, 0))
end
