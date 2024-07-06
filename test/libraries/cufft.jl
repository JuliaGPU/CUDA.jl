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
atol(::Type{Float16}) = 1e-3
atol(::Type{Float32}) = 1e-8
atol(::Type{Float64}) = 1e-15
rtol(::Type{Complex{T}}) where {T} = rtol(T)
atol(::Type{Complex{T}}) where {T} = atol(T)


## complex

function out_of_place(X::AbstractArray{T,N}) where {T <: Complex,N}
    fftw_X = fft(X)
    d_X = CuArray(X)
    p = plan_fft(d_X)
    d_Y = p * d_X
    Y = collect(d_Y)
    if ! isapprox(Y, fftw_X, rtol = rtol(T), atol = atol(T))
        @show T length(Y)
        @show norm(Y - fftw_X)
        @show norm(Y) norm(fftw_X)
        @show rtol(T) atol(T)
        @assert false
    end
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
    @show :batched0 region size(X) typeof(X)
    fftw_X = fft(X,region)
    @show :batched1 size(fftw_X) typeof(fftw_X)
    d_X = CuArray(X)
    @show :batched2
    p = plan_fft(d_X,region)
    @show :batched3
    d_Y = p * d_X
    @show :batched4 size(d_Y) typeof(d_Y)
    d_X2 = reshape(d_X, (size(d_X)..., 1))
    @show :batched5 size(d_X2) typeof(d_X2)
    @test_throws ArgumentError p * d_X2
    @show :batched6

    Y = collect(d_Y)
    @show :batched7 size(Y) typeof(Y)
    @test isapprox(Y, fftw_X, rtol = rtol(T), atol = atol(T))
    @show :batched8

    pinv = plan_ifft(d_Y,region)
    @show :batched9
    d_Z = pinv * d_Y
    @show :batched10 size(d_Z)
    Z = collect(d_Z)
    @show :batched11 size(Z)
    @test isapprox(Z, X, rtol = rtol(T), atol = atol(T))
    @show :batched12

    ldiv!(d_Z, p, d_Y)
    @show :batched13
    Z = collect(d_Z)
    @show :batched14 size(Z)
    @test isapprox(Z, X, rtol = rtol(T), atol = atol(T))
    @show :batched15
end

GC.gc(true)
#TODO @testset for T in [ComplexF16, ComplexF32, ComplexF64]
@testset for T in [ComplexF32, ComplexF64]

GC.gc(true)
@testset "simple" begin
    GC.gc(true)
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

GC.gc(true)
@testset "1D" begin
    dims = (N1,)
    X = rand(T, dims)
    out_of_place(X)
end
GC.gc(true)
@testset "1D inplace" begin
    dims = (N1,)
    X = rand(T, dims)
    in_place(X)
end

GC.gc(true)
@testset "2D" begin
    dims = (N1,N2)
    X = rand(T, dims)
    out_of_place(X)
end
GC.gc(true)
@testset "2D inplace" begin
    dims = (N1,N2)
    X = rand(T, dims)
    in_place(X)
end

GC.gc(true)
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

GC.gc(true)
@testset "3D" begin
    dims = (N1,N2,N3)
    X = rand(T, dims)
    out_of_place(X)
end

GC.gc(true)
@testset "3D inplace" begin
    dims = (N1,N2,N3)
    X = rand(T, dims)
    in_place(X)
end

GC.gc(true)
@testset "Batch 2D (in 3D)" begin
    dims = (N1,N2,N3)
    for region in [(1,2),(2,3),(1,3)]
        @show :batch2d3d T dims region
        X = rand(T, dims)
        batched(X,region)
        @show :success
    end

    #TODO @show :batch2d3d T dims
    #TODO X = rand(T, dims)
    #TODO @test_throws ArgumentError batched(X,(3,1))
    #TODO @show :failure
end

GC.gc(true)
@testset "Batch 2D (in 4D)" begin
    dims = (N1,N2,N3,N4)
    for region in [(1,2),(1,4),(3,4),(1,3),(2,3),(2,),(3,)]
        @show :batch2d4d T dims region
        X = rand(T, dims)
        batched(X,region)
        @show :success
    end
    for region in [(2,4)]
        @show :batch2d4d T dims region
        X = rand(T, dims)
        @test_throws ArgumentError batched(X,region)
        @show :failure
    end
end

end


#TODO ## real
#TODO 
#TODO function out_of_place(X::AbstractArray{T,N}) where {T <: Real,N}
#TODO     fftw_X = rfft(X)
#TODO     d_X = CuArray(X)
#TODO     p = plan_rfft(d_X)
#TODO     d_Y = p * d_X
#TODO     Y = collect(d_Y)
#TODO     @test isapprox(Y, fftw_X, rtol = rtol(T), atol = atol(T))
#TODO 
#TODO     pinv = plan_irfft(d_Y,size(X,1))
#TODO     d_Z = pinv * d_Y
#TODO     Z = collect(d_Z)
#TODO     @test isapprox(Z, X, rtol = rtol(T), atol = atol(T))
#TODO 
#TODO     pinv2 = inv(p)
#TODO     d_Z = pinv2 * d_Y
#TODO     Z = collect(d_Z)
#TODO     @test isapprox(Z, X, rtol = rtol(T), atol = atol(T))
#TODO 
#TODO     pinv3 = inv(pinv)
#TODO     d_W = pinv3 * d_X
#TODO     W = collect(d_W)
#TODO     @test isapprox(W, Y, rtol = rtol(T), atol = atol(T))
#TODO end
#TODO 
#TODO function batched(X::AbstractArray{T,N},region) where {T <: Real,N}
#TODO     fftw_X = rfft(X,region)
#TODO     d_X = CuArray(X)
#TODO     p = plan_rfft(d_X,region)
#TODO     d_Y = p * d_X
#TODO     Y = collect(d_Y)
#TODO     @test isapprox(Y, fftw_X, rtol = rtol(T), atol = atol(T))
#TODO 
#TODO     pinv = plan_irfft(d_Y,size(X,region[1]),region)
#TODO     d_Z = pinv * d_Y
#TODO     Z = collect(d_Z)
#TODO     @test isapprox(Z, X, rtol = rtol(T), atol = atol(T))
#TODO end
#TODO 
#TODO @testset for T in [Float16, Float32, Float64]
#TODO 
#TODO @testset "1D" begin
#TODO     X = rand(T, N1)
#TODO     out_of_place(X)
#TODO end
#TODO 
#TODO @testset "Batch 1D" begin
#TODO     dims = (N1,N2)
#TODO     X = rand(T, dims)
#TODO     batched(X,1)
#TODO 
#TODO     dims = (N1,N2)
#TODO     X = rand(T, dims)
#TODO     batched(X,2)
#TODO 
#TODO     dims = (N1,N2)
#TODO     X = rand(T, dims)
#TODO     batched(X,(1,2))
#TODO end
#TODO 
#TODO @testset "2D" begin
#TODO     X = rand(T, N1,N2)
#TODO     out_of_place(X)
#TODO end
#TODO 
#TODO @testset "Batch 2D (in 3D)" begin
#TODO     dims = (N1,N2,N3)
#TODO     for region in [(1,2),(2,3),(1,3)]
#TODO         X = rand(T, dims)
#TODO         batched(X,region)
#TODO     end
#TODO 
#TODO     X = rand(T, dims)
#TODO     @test_throws ArgumentError batched(X,(3,1))
#TODO end
#TODO 
#TODO @testset "Batch 2D (in 4D)" begin
#TODO     dims = (N1,N2,N3,N4)
#TODO     for region in [(1,2),(1,4),(3,4),(1,3),(2,3)]
#TODO         X = rand(T, dims)
#TODO         batched(X,region)
#TODO     end
#TODO     for region in [(2,4)]
#TODO         X = rand(T, dims)
#TODO         @test_throws ArgumentError batched(X,region)
#TODO     end
#TODO end
#TODO 
#TODO @testset "3D" begin
#TODO     X = rand(T, N1, N2, N3)
#TODO     out_of_place(X)
#TODO end
#TODO 
#TODO end
#TODO 
#TODO 
#TODO ## complex integer
#TODO 
#TODO function out_of_place(X::AbstractArray{T,N}) where {T <: Complex{<:Integer},N}
#TODO     fftw_X = fft(X)
#TODO     d_X = CuArray(X)
#TODO     p = plan_fft(d_X)
#TODO     d_Y = p * d_X
#TODO     Y = collect(d_Y)
#TODO     @test isapprox(Y, fftw_X, rtol = rtol(T), atol = atol(T))
#TODO 
#TODO     d_Y = fft(d_X)
#TODO     Y = collect(d_Y)
#TODO     @test isapprox(Y, fftw_X, rtol = rtol(T), atol = atol(T))
#TODO end
#TODO 
#TODO @testset for T in [Complex{Int32}, Complex{Int64}]
#TODO 
#TODO @testset "1D" begin
#TODO     dims = (N1,)
#TODO     X = rand(T, dims)
#TODO     out_of_place(X)
#TODO end
#TODO 
#TODO end
#TODO 
#TODO 
#TODO ## real integer
#TODO 
#TODO function out_of_place(X::AbstractArray{T,N}) where {T <: Integer,N}
#TODO     fftw_X = rfft(X)
#TODO     d_X = CuArray(X)
#TODO     p = plan_rfft(d_X)
#TODO     d_Y = p * d_X
#TODO     Y = collect(d_Y)
#TODO     @test isapprox(Y, fftw_X, rtol = rtol(T), atol = atol(T))
#TODO 
#TODO     d_Y = rfft(d_X)
#TODO     Y = collect(d_Y)
#TODO     @test isapprox(Y, fftw_X, rtol = rtol(T), atol = atol(T))
#TODO end
#TODO 
#TODO @testset for T in [Int32, Int64]
#TODO 
#TODO @testset "1D" begin
#TODO     X = rand(T, N1)
#TODO     out_of_place(X)
#TODO end
#TODO 
#TODO end
#TODO 
#TODO 
#TODO ## other
#TODO 
#TODO @testset "CUDA.jl#1268" begin
#TODO     N=2^20
#TODO     v0 = CuArray(ones(N)+im*ones(N))
#TODO 
#TODO     v = CuArray(ones(N)+im*ones(N))
#TODO     plan = CUFFT.plan_fft!(v,1)
#TODO     @test fetch(
#TODO         Threads.@spawn begin
#TODO             inv(plan)*(plan*v)
#TODO             isapprox(v,v0)
#TODO         end
#TODO     )
#TODO end
#TODO 
#TODO @testset "CUDA.jl#1311" begin
#TODO     x = ones(8, 9)
#TODO     p = plan_rfft(x)
#TODO     y = similar(p * x)
#TODO     mul!(y, p, x)
#TODO 
#TODO     dx = CuArray(x)
#TODO     dp = plan_rfft(dx)
#TODO     dy = similar(dp * dx)
#TODO     mul!(dy, dp, dx)
#TODO 
#TODO     @test Array(dy) ≈ y
#TODO end
#TODO 
#TODO @testset "CUDA.jl#2409" begin
#TODO     x = CUDA.zeros(ComplexF32, 4)
#TODO     p = plan_ifft(x)
#TODO     @test p isa AbstractFFTs.ScaledPlan
#TODO     # Initialize sz ref to invalid value
#TODO     sz = Ref{Csize_t}(typemax(Csize_t))
#TODO     # This will call the new convert method for ScaledPlan
#TODO     CUFFT.cufftGetSize(p, sz)
#TODO     # Make sure the value was modified
#TODO     @test sz[] != typemax(Csize_t)
#TODO end
