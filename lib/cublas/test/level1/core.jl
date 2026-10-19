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
        result_type = cuBLAS.version() >= v"12" ? Int64 : Cint
        result = CuRef{result_type}(0)
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

    @testset "norm of strided views" begin # JuliaGPU/CUDA.jl#2280
        # 1D contiguous view: should hit the cuBLAS nrm2 fast path.
        x = rand(T, m)
        dx = CuArray(x)
        @test norm(@view(dx[2:end-1]), 2) ≈ norm(@view(x[2:end-1]), 2)
        # Multi-dim non-contiguous view: must avoid scalar iteration.
        y = rand(T, 10, 10)
        dy = CuArray(y)
        @test norm(@view(dy[2:end-1, 2:end-1]), 1) ≈ norm(@view(y[2:end-1, 2:end-1]), 1)
        @test norm(@view(dy[2:end-1, 2:end-1]), 2) ≈ norm(@view(y[2:end-1, 2:end-1]), 2)
        @test norm(@view(dy[2:end-1, 2:end-1]), Inf) ≈ norm(@view(y[2:end-1, 2:end-1]), Inf)
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

@testset "views BLAS can't address as a vector" begin
    A = rand(Float32, 3, 4)
    dA = CuArray(A)
    rmul!(view(dA, 1:2, :), 2f0)
    rmul!(view(A, 1:2, :), 2f0)
    @test Array(dA) ≈ A

    # ComplexF16 is computed in ComplexF32 and written back
    x, y = rand(ComplexF16, 20), rand(ComplexF16, 20)
    dx, dy = CuArray(x), CuArray(y)
    axpy!(ComplexF16(2), view(dx, 1:2:20), view(dy, 1:2:20))
    axpy!(ComplexF16(2), view(x, 1:2:20), view(y, 1:2:20))
    @test Array(dy) ≈ y
    cuBLAS.scal!(10, 3f0, view(dx, 1:2:20))
    rmul!(view(x, 1:2:20), 3f0)
    @test Array(dx) ≈ x
end

# BLAS walks a negative increment from the lowest address (JuliaGPU/CUDA.jl#3410)
@testset "negative strides $T" for T in [Float16, ComplexF16, Float32, Float64, ComplexF32, ComplexF64]
    m = 20
    rev(a) = view(a, lastindex(a):-2:1)
    fwd(a) = view(a, 1:2:lastindex(a))
    x, y = rand(T, m), rand(T, m)
    dx, dy = CuArray(x), CuArray(y)

    @test norm(rev(dx)) ≈ norm(rev(x))
    @test dot(rev(dx), fwd(dy)) ≈ dot(rev(x), fwd(y))
    @test dot(rev(dx), rev(dy)) ≈ dot(rev(x), rev(y))
    if T <: cuBLAS.CublasFloat
        @test BLAS.asum(rev(dx)) ≈ BLAS.asum(rev(x))
    end

    for f! in ((x, y) -> T <: cuBLAS.CublasFloat ? rmul!(rev(y), T(3)) : y,
               (x, y) -> axpy!(T(2), rev(x), fwd(y)),
               (x, y) -> axpy!(T(2), fwd(x), rev(y)),
               (x, y) -> axpby!(T(2), rev(x), T(3), rev(y)))
        a, da = copy(y), copy(dy)
        f!(x, a)
        f!(dx, da)
        @test Array(da) ≈ a
    end

    if T <: cuBLAS.CublasFloat
        a, da = copy(y), copy(dy)
        copyto!(fwd(a), rev(x))
        cuBLAS.copy!(m ÷ 2, rev(dx), fwd(da))
        @test Array(da) == a

        a, b, da, db = copy(x), copy(y), copy(dx), copy(dy)
        a[1:2:end], b[end:-2:1] = b[end:-2:1], a[1:2:end]
        cuBLAS.swap!(m ÷ 2, fwd(da), rev(db))
        @test Array(da) == a
        @test Array(db) == b
    end
end

@testset "dot with mixed types" begin
    m = 20
    T1 = Float32
    T2 = Float64
    @test testf(dot, rand(T1, m), rand(T2, m))
    @test testf(dot, rand(T1, 0), rand(T2, 0))
end
