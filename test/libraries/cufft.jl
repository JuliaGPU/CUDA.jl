using CUDA.CUFFT

using CUDA

import FFTW

using AbstractFFTs
using LinearAlgebra

@test CUFFT.version() isa VersionNumber

# FFTW does not support Float16, so we roll our own

function AbstractFFTs.fft!(x::Array{Complex{Float16}}, dims...)
    y = Array{Complex{Float32}}(x)
    fft!(y, dims...)
    x .= y
end
function AbstractFFTs.bfft!(x::Array{Complex{Float16}}, dims...)
    y = Array{Complex{Float32}}(x)
    bfft!(y, dims...)
    x .= y
end
function AbstractFFTs.ifft!(x::Array{Complex{Float16}}, dims...)
    y = Array{Complex{Float32}}(x)
    ifft!(y, dims...)
    x .= y
end

AbstractFFTs.fft(x::Array{Complex{Float16}}, dims...) = Array{Complex{Float16}}(fft(Array{Complex{Float32}}(x), dims...))
AbstractFFTs.bfft(x::Array{Complex{Float16}}, dims...) = Array{Complex{Float16}}(bfft(Array{Complex{Float32}}(x), dims...))
AbstractFFTs.ifft(x::Array{Complex{Float16}}, dims...) = Array{Complex{Float16}}(ifft(Array{Complex{Float32}}(x), dims...))
AbstractFFTs.rfft(x::Array{Float16}, dims...) = Array{Complex{Float16}}(rfft(Array{Float32}(x), dims...))
AbstractFFTs.brfft(x::Array{Complex{Float16}}, dims...) = Array{Float16}(brfft(Array{Complex{Float32}}(x), dims...))
AbstractFFTs.irfft(x::Array{Complex{Float16}}, dims...) = Array{Float16}(irfft(Array{Complex{Float32}}(x), dims...))

struct WrappedFloat16Operator
    op
end
Base.:*(A::WrappedFloat16Operator, b::Array{Float16}) = Array{Float16}(A.op * Array{Float32}(b))
Base.:*(A::WrappedFloat16Operator, b::Array{Complex{Float16}}) = Array{Complex{Float16}}(A.op * Array{Complex{Float32}}(b))
function LinearAlgebra.mul!(C::Array{Float16}, A::WrappedFloat16Operator, B::Array{Float16}, α, β)
    C32 = Array{Float32}(C)
    B32 = Array{Float32}(B)
    mul!(C32, A.op, B32, α, β)
    C .= C32
end
function LinearAlgebra.mul!(C::Array{Complex{Float16}}, A::WrappedFloat16Operator, B::Array{Complex{Float16}}, α, β)
    C32 = Array{Complex{Float32}}(C)
    B32 = Array{Complex{Float32}}(B)
    mul!(C32, A.op, B32, α, β)
    C .= C32
end

function AbstractFFTs.plan_fft!(x::Array{Complex{Float16}}, dims...)
    y = similar(x, Complex{Float32})
    WrappedFloat16Operator(plan_fft!(y, dims...))
end
function AbstractFFTs.plan_bfft!(x::Array{Complex{Float16}}, dims...)
    y = similar(x, Complex{Float32})
    WrappedFloat16Operator(plan_bfft!(y, dims...))
end
function AbstractFFTs.plan_ifft!(x::Array{Complex{Float16}}, dims...)
    y = similar(x, Complex{Float32})
    WrappedFloat16Operator(plan_ifft!(y, dims...))
end

function AbstractFFTs.plan_fft(x::Array{Complex{Float16}}, dims...)
    y = similar(x, Complex{Float32})
    WrappedFloat16Operator(plan_fft(y, dims...))
end
function AbstractFFTs.plan_bfft(x::Array{Complex{Float16}}, dims...)
    y = similar(x, Complex{Float32})
    WrappedFloat16Operator(plan_bfft(y, dims...))
end
function AbstractFFTs.plan_ifft(x::Array{Complex{Float16}}, dims...)
    y = similar(x, Complex{Float32})
    WrappedFloat16Operator(plan_ifft(y, dims...))
end
function AbstractFFTs.plan_rfft(x::Array{Float16}, dims...)
    y = similar(x, Float32)
    WrappedFloat16Operator(plan_rfft(y, dims...))
end
function AbstractFFTs.plan_irfft(x::Array{Complex{Float16}}, dims...)
    y = similar(x, Complex{Float32})
    WrappedFloat16Operator(plan_irfft(y, dims...))
end
function AbstractFFTs.plan_brfft(x::Array{Complex{Float16}}, dims...)
    y = similar(x, Complex{Float32})
    WrappedFloat16Operator(plan_brfft(y, dims...))
end

# notes:
#   plan_bfft does not need separate testing since it is used by plan_ifft

N1 = 8
N2 = 32
N3 = 64
N4 = 8

rtol(::Type{Float16}) = 1e-2
rtol(::Type{Float32}) = 1e-5
rtol(::Type{Float64}) = 1e-12
rtol(::Type{I}) where {I<:Integer} = rtol(float(I))
atol(::Type{Float16}) = 1e-3
atol(::Type{Float32}) = 1e-8
atol(::Type{Float64}) = 1e-15
atol(::Type{I}) where {I<:Integer} = atol(float(I))
rtol(::Type{Complex{T}}) where {T} = rtol(T)
atol(::Type{Complex{T}}) where {T} = atol(T)


## complex

function out_of_place(X::AbstractArray{T,N}) where {T <: Complex,N}
    fftw_X = fft(X)
    d_X = CuArray(X)
    p = plan_fft(d_X)
    d_Y = p * d_X
    Y = collect(d_Y)
    @test isapprox(Y, fftw_X, rtol = rtol(T), atol = atol(T))

    pinv = plan_ifft(d_Y)
    d_Z = pinv * d_Y
    Z = collect(d_Z)
    @test isapprox(Z, X, rtol = rtol(T), atol = atol(T))

    pinv2 = inv(p)
    d_Z = pinv2 * d_Y
    Z = collect(d_Z)
    @test isapprox(Z, X, rtol = rtol(T), atol = atol(T))

end

function in_place(X::AbstractArray{T,N}) where {T <: Complex,N}
    fftw_X = fft(X)
    d_X = CuArray(X)
    p = plan_fft!(d_X)
    p * d_X
    Y = collect(d_X)
    @test isapprox(Y, fftw_X, rtol = rtol(T), atol = atol(T))

    pinv = plan_ifft!(d_X)
    pinv * d_X
    Z = collect(d_X)
    @test isapprox(Z, X, rtol = rtol(T), atol = atol(T))
end

function batched(X::AbstractArray{T,N},region) where {T <: Complex,N}
    fftw_X = fft(X,region)
    d_X = CuArray(X)
    p = plan_fft(d_X,region)
    d_Y = p * d_X
    d_X2 = reshape(d_X, (size(d_X)..., 1))
    @test_throws ArgumentError p * d_X2

    Y = collect(d_Y)
    @test isapprox(Y, fftw_X, rtol = rtol(T), atol = atol(T))

    pinv = plan_ifft(d_Y,region)
    d_Z = pinv * d_Y
    Z = collect(d_Z)
    @test isapprox(Z, X, rtol = rtol(T), atol = atol(T))

    ldiv!(d_Z, p, d_Y)
    Z = collect(d_Z)
    @test isapprox(Z, X, rtol = rtol(T), atol = atol(T))
end

@testset for T in [ComplexF16, ComplexF32, ComplexF64]

@testset "simple" begin
    @testset "$(n)D" for n = 1:3
        # Float16 FFTs must have a length that is a power of 2
        sz = T == ComplexF16 ? 32 : 40
        dims = ntuple(i -> sz, n)
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
    @test isapprox(Y, fftw_X, rtol = rtol(T), atol = atol(T))

    pinv = plan_irfft(d_Y,size(X,1))
    d_Z = pinv * d_Y
    Z = collect(d_Z)
    @test isapprox(Z, X, rtol = rtol(T), atol = atol(T))

    pinv2 = inv(p)
    d_Z = pinv2 * d_Y
    Z = collect(d_Z)
    @test isapprox(Z, X, rtol = rtol(T), atol = atol(T))

    pinv3 = inv(pinv)
    d_W = pinv3 * d_X
    W = collect(d_W)
    @test isapprox(W, Y, rtol = rtol(T), atol = atol(T))
end

function batched(X::AbstractArray{T,N},region) where {T <: Real,N}
    fftw_X = rfft(X,region)
    d_X = CuArray(X)
    p = plan_rfft(d_X,region)
    d_Y = p * d_X
    Y = collect(d_Y)
    @test isapprox(Y, fftw_X, rtol = rtol(T), atol = atol(T))

    pinv = plan_irfft(d_Y,size(X,region[1]),region)
    d_Z = pinv * d_Y
    Z = collect(d_Z)
    @test isapprox(Z, X, rtol = rtol(T), atol = atol(T))
end

@testset for T in [Float16, Float32, Float64]

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
    @test isapprox(Y, fftw_X, rtol = rtol(T), atol = atol(T))

    d_Y = fft(d_X)
    Y = collect(d_Y)
    @test isapprox(Y, fftw_X, rtol = rtol(T), atol = atol(T))
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
    @test isapprox(Y, fftw_X, rtol = rtol(T), atol = atol(T))

    d_Y = rfft(d_X)
    Y = collect(d_Y)
    @test isapprox(Y, fftw_X, rtol = rtol(T), atol = atol(T))
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

@testset "CUDA.jl#2504" begin
    x = CUDA.zeros(Float32, 4)
    p = plan_rfft(x)
    pinv = inv(p)
    @test p isa AbstractFFTs.Plan{Float32}
    @test eltype(p) === Float32
    @test pinv isa AbstractFFTs.Plan{ComplexF32}
    @test eltype(pinv) === ComplexF32
end
