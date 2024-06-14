using CUDA.CUFFT

using CUDA

import FFTW

using LinearAlgebra

@test CUFFT.version() isa VersionNumber

# notes:
#   plan_bfft does not need separate testing since it is used by plan_ifft

N1 = 8
N2 = 32
N3 = 64
N4 = 8

MYRTOL = 1e-5
MYATOL = 1e-8


## complex

function out_of_place(X::AbstractArray{T,N}) where {T <: Complex,N}
    fftw_X = fft(X)
    d_X = CuArray(X)
    p = plan_fft(d_X)
    d_Y = p * d_X
    Y = collect(d_Y)
    @test isapprox(Y, fftw_X, rtol = MYRTOL, atol = MYATOL)

    pinv = plan_ifft(d_Y)
    d_Z = pinv * d_Y
    Z = collect(d_Z)
    @test isapprox(Z, X, rtol = MYRTOL, atol = MYATOL)

    pinv2 = inv(p)
    d_Z = pinv2 * d_Y
    Z = collect(d_Z)
    @test isapprox(Z, X, rtol = MYRTOL, atol = MYATOL)

end

function in_place(X::AbstractArray{T,N}) where {T <: Complex,N}
    fftw_X = fft(X)
    d_X = CuArray(X)
    p = plan_fft!(d_X)
    p * d_X
    Y = collect(d_X)
    @test isapprox(Y, fftw_X, rtol = MYRTOL, atol = MYATOL)

    pinv = plan_ifft!(d_X)
    pinv * d_X
    Z = collect(d_X)
    @test isapprox(Z, X, rtol = MYRTOL, atol = MYATOL)
end

function batched(X::AbstractArray{T,N},region) where {T <: Complex,N}
    fftw_X = fft(X,region)
    d_X = CuArray(X)
    p = plan_fft(d_X,region)
    d_Y = p * d_X
    d_X2 = reshape(d_X, (size(d_X)..., 1))
    @test_throws ArgumentError p * d_X2

    Y = collect(d_Y)
    @test isapprox(Y, fftw_X, rtol = MYRTOL, atol = MYATOL)

    pinv = plan_ifft(d_Y,region)
    d_Z = pinv * d_Y
    Z = collect(d_Z)
    @test isapprox(Z, X, rtol = MYRTOL, atol = MYATOL)

    ldiv!(d_Z, p, d_Y)
    Z = collect(d_Z)
    @test isapprox(Z, X, rtol = MYRTOL, atol = MYATOL)
end

@testset for T in [ComplexF32, ComplexF64]

@testset "simple" begin
    @testset "$(n)D" for n = 1:3
        dims = ntuple(i -> 40, n)
        @test testf(fft!, rand(T, dims))
        @test testf(ifft!, rand(T, dims))

        @test testf(fft, rand(T, dims))
        @test testf(ifft, rand(T, dims))
    end
end

@testset "1D" begin
    dims = (N1,)
    X = rand(T, dims)
    out_of_place(X)
end
@testset "1D inplace" begin
    dims = (N1,)
    X = rand(T, dims)
    in_place(X)
end

@testset "2D" begin
    dims = (N1,N2)
    X = rand(T, dims)
    out_of_place(X)
end
@testset "2D inplace" begin
    dims = (N1,N2)
    X = rand(T, dims)
    in_place(X)
end

@testset "Batch 1D" begin
    dims = (N1,N2)
    X = rand(T, dims)
    batched(X,1)

    dims = (N1,N2)
    X = rand(T, dims)
    batched(X,2)

    dims = (N1,N2)
    X = rand(T, dims)
    batched(X,(1,2))
end

@testset "3D" begin
    dims = (N1,N2,N3)
    X = rand(T, dims)
    out_of_place(X)
end

@testset "3D inplace" begin
    dims = (N1,N2,N3)
    X = rand(T, dims)
    in_place(X)
end

@testset "Batch 2D (in 3D)" begin
    dims = (N1,N2,N3)
    for region in [(1,2),(2,3),(1,3)]
        X = rand(T, dims)
        batched(X,region)
    end

    X = rand(T, dims)
    @test_throws ArgumentError batched(X,(3,1))
end

@testset "Batch 2D (in 4D)" begin
    dims = (N1,N2,N3,N4)
    for region in [(1,2),(1,4),(3,4),(1,3),(2,3),(2,),(3,)]
        X = rand(T, dims)
        batched(X,region)
    end
    for region in [(2,4)]
        X = rand(T, dims)
        @test_throws ArgumentError batched(X,region)
    end

end

end


## real

function out_of_place(X::AbstractArray{T,N}) where {T <: Real,N}
    fftw_X = rfft(X)
    d_X = CuArray(X)
    p = plan_rfft(d_X)
    d_Y = p * d_X
    Y = collect(d_Y)
    @test isapprox(Y, fftw_X, rtol = MYRTOL, atol = MYATOL)

    pinv = plan_irfft(d_Y,size(X,1))
    d_Z = pinv * d_Y
    Z = collect(d_Z)
    @test isapprox(Z, X, rtol = MYRTOL, atol = MYATOL)

    pinv2 = inv(p)
    d_Z = pinv2 * d_Y
    Z = collect(d_Z)
    @test isapprox(Z, X, rtol = MYRTOL, atol = MYATOL)

    pinv3 = inv(pinv)
    d_W = pinv3 * d_X
    W = collect(d_W)
    @test isapprox(W, Y, rtol = MYRTOL, atol = MYATOL)
end

function batched(X::AbstractArray{T,N},region) where {T <: Real,N}
    fftw_X = rfft(X,region)
    d_X = CuArray(X)
    p = plan_rfft(d_X,region)
    d_Y = p * d_X
    Y = collect(d_Y)
    @test isapprox(Y, fftw_X, rtol = MYRTOL, atol = MYATOL)

    pinv = plan_irfft(d_Y,size(X,region[1]),region)
    d_Z = pinv * d_Y
    Z = collect(d_Z)
    @test isapprox(Z, X, rtol = MYRTOL, atol = MYATOL)
end

@testset for T in [Float32, Float64]

@testset "1D" begin
    X = rand(T, N1)
    out_of_place(X)
end

@testset "Batch 1D" begin
    dims = (N1,N2)
    X = rand(T, dims)
    batched(X,1)

    dims = (N1,N2)
    X = rand(T, dims)
    batched(X,2)

    dims = (N1,N2)
    X = rand(T, dims)
    batched(X,(1,2))
end

@testset "2D" begin
    X = rand(T, N1,N2)
    out_of_place(X)
end

@testset "Batch 2D (in 3D)" begin
    dims = (N1,N2,N3)
    for region in [(1,2),(2,3),(1,3)]
        X = rand(T, dims)
        batched(X,region)
    end

    X = rand(T, dims)
    @test_throws ArgumentError batched(X,(3,1))
end

@testset "Batch 2D (in 4D)" begin
    dims = (N1,N2,N3,N4)
    for region in [(1,2),(1,4),(3,4),(1,3),(2,3)]
        X = rand(T, dims)
        batched(X,region)
    end
    for region in [(2,4)]
        X = rand(T, dims)
        @test_throws ArgumentError batched(X,region)
    end
end

@testset "3D" begin
    X = rand(T, N1, N2, N3)
    out_of_place(X)
end

end


## complex integer

function out_of_place(X::AbstractArray{T,N}) where {T <: Complex{<:Integer},N}
    fftw_X = fft(X)
    d_X = CuArray(X)
    p = plan_fft(d_X)
    d_Y = p * d_X
    Y = collect(d_Y)
    @test isapprox(Y, fftw_X, rtol = MYRTOL, atol = MYATOL)

    d_Y = fft(d_X)
    Y = collect(d_Y)
    @test isapprox(Y, fftw_X, rtol = MYRTOL, atol = MYATOL)
end

@testset for T in [Complex{Int32}, Complex{Int64}]

@testset "1D" begin
    dims = (N1,)
    X = rand(T, dims)
    out_of_place(X)
end

end


## real integer

function out_of_place(X::AbstractArray{T,N}) where {T <: Integer,N}
    fftw_X = rfft(X)
    d_X = CuArray(X)
    p = plan_rfft(d_X)
    d_Y = p * d_X
    Y = collect(d_Y)
    @test isapprox(Y, fftw_X, rtol = MYRTOL, atol = MYATOL)

    d_Y = rfft(d_X)
    Y = collect(d_Y)
    @test isapprox(Y, fftw_X, rtol = MYRTOL, atol = MYATOL)
end

@testset for T in [Int32, Int64]

@testset "1D" begin
    X = rand(T, N1)
    out_of_place(X)
end

end


## other

@testset "CUDA.jl#1268" begin
    N=2^20
    v0 = CuArray(ones(N)+im*ones(N))

    v = CuArray(ones(N)+im*ones(N))
    plan = CUFFT.plan_fft!(v,1)
    @test fetch(
        Threads.@spawn begin
            inv(plan)*(plan*v)
            isapprox(v,v0)
        end
    )
end

@testset "CUDA.jl#1311" begin
    x = ones(8, 9)
    p = plan_rfft(x)
    y = similar(p * x)
    mul!(y, p, x)

    dx = CuArray(x)
    dp = plan_rfft(dx)
    dy = similar(dp * dx)
    mul!(dy, dp, dx)

    @test Array(dy) ≈ y
end

@testset "CUDA.jl#2409" begin
    x = CUDA.zeros(ComplexF32, 4)
    p = plan_ifft(x)
    @test p isa AbstractFFTs.ScaledPlan
    # Initialize sz ref to invalid value
    sz = Ref{Csize_t}(typemax(Csize_t))
    # This will call the new convert method for ScaledPlan
    CUFFT.cufftGetSize(p, sz)
    # Make sure the value was modified
    @test sz[] != typemax(Csize_t)
end
