using CUDA.CUFFT

using CUDA

import FFTW

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
    Y = collect(d_Y)
    @test_maybe_broken isapprox(Y, fftw_X, rtol = MYRTOL, atol = MYATOL)

    pinv = plan_ifft(d_Y,region)
    d_Z = pinv * d_Y
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
    # @test_throws ArgumentError batched(X,(3,1))
end

@testset "Batch 2D (in 4D)" begin
    dims = (N1,N2,N3,N4)
    for region in [(1,2),(1,4),(3,4)]
        X = rand(T, dims)
        batched(X,region)
    end
    for region in [(1,3),(2,3),(2,4)]
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
    @test_maybe_broken isapprox(Z, X, rtol = MYRTOL, atol = MYATOL)
    # JuliaGPU/CUDA.jl#345, NVIDIA/cuFFT#2714102

    pinv3 = inv(pinv)
    d_W = pinv3 * d_X
    W = collect(d_W)
    @test isapprox(W, Y, rtol = MYRTOL, atol = MYATOL)
    # JuliaGPU/CUDA.jl#345, NVIDIA/cuFFT#2714102
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
    # @test_throws ArgumentError batched(X,(3,1))
end

@testset "Batch 2D (in 4D)" begin
    dims = (N1,N2,N3,N4)
    for region in [(1,2),(1,4),(3,4)]
        X = rand(T, dims)
        batched(X,region)
    end
    for region in [(1,3),(2,3),(2,4)]
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

# stride layout tests
strided_test_2d(::Type{T}, fun, fun′) where T = begin
    @testset "2D" begin
        a = @view CUDA.rand(T,33,32)[1:32,:]
        @test fun(a) ≈ fun′(a)
        @test fun(a,1) ≈ fun′(a,1)
        @test fun(a,2) ≈ fun′(a,2)
    end
end
strided_test_3d(::Type{T}, fun, fun′) where T = begin
    @testset "3D" begin
        a = @view CUDA.rand(T,33,32,32)[1:32,:,:] #view in dim 1
        @test fun(a) ≈ fun′(a)
        @test fun(a,1) ≈ fun′(a,1)
        for r in ((1,2), (2,3), (3,1))
            @test fun(a,r) ≈ fun′(a,r)
        end
        a = @view CUDA.rand(T,32,33,32)[:,1:32,:] #view in dim 2
        @test fun(a) ≈ fun′(a)
        @test fun(a,3) ≈ fun′(a,3)
        for r in ((1,2), (2,3), (3,1))
            @test fun(a,r) ≈ fun′(a,r)
        end
        a = @view CUDA.rand(T,33,33,32)[1:32,1:32,:] #view in dim 1, 2
        @test fun(a) ≈ fun′(a)
        for r in ((1,2), (2,3), (3,1))
            @test fun(a,r) ≈ fun′(a,r)
        end
    end
end
strided_test_3d_throw(::Type{T}, fun) where T = begin
    @testset "3D_throw" begin
        a = @view CUDA.rand(T,33,32,32)[1:32,:,:] #view in dim 1
        @test_throws ArgumentError fun(a,2)
        @test_throws ArgumentError fun(a,3)
        a = @view CUDA.rand(T,32,33,32)[:,1:32,:] #view in dim 2
        @test_throws ArgumentError fun(a,1)
        @test_throws ArgumentError fun(a,2)
        a = @view CUDA.rand(T,33,33,32)[1:32,1:32,:] #view in dim 1, 2
        @test_throws ArgumentError fun(a,1)
        @test_throws ArgumentError fun(a,2)
        @test_throws ArgumentError fun(a,3)
    end
end
strided_test_4d(::Type{T}, fun, fun′) where T = begin
    @testset "4D" begin
        a = @view CUDA.rand(T,17,16,16,16)[1:16,:,:,:] #view in dim 1
        @test fun(a,1) ≈ fun′(a,1)
        @test fun(a,(1,2)) ≈ fun′(a,(2,1))
        @test fun(a,(1,4)) ≈ fun′(a,(4,1))
        for i in 1:4
            r = filter(x -> x != i, (1,2,3,4))
            @test fun(a, r) ≈ fun′(a, reverse(r))
        end
    end
end
strided_test_4d_throw(::Type{T}, fun) where T = begin
    @testset "4D_throw" begin
        #view in dim 1
        a = @view CUDA.rand(T,17,16,16,16)[1:16,:,:,:]
        @test_throws ArgumentError fun(a)
        for i in (2,3,4)
            @test_throws ArgumentError fun(a,i)
        end
        for i in ((1,3),(2,3),(2,4),(3,4))
            @test_throws ArgumentError fun(a,i)
        end
    end
end

#We drop such support in CUDA10.1 and 10.2
if CUFFT.version() >= v"10.2.1"
    @testset for fun in (fft, rfft)
        #out-place (r)fft with strided layout's.
        #Only fft is tested, as bfft and ifft has the same kernel plan
        #brfft need copy, test elsewhere
        @testset for T in (Float32, Float64)
            if fun == fft
                T = complex(T)
            end
            copyfun(x) = fun(copy(x))
            copyfun(x,y) = fun(copy(x),y)
            #compare the results of (r)fft with strided and dense layout
            strided_test_2d(T, fun, copyfun)
            strided_test_3d(T, fun, copyfun)
            strided_test_4d(T, fun, copyfun)
            #unreducable batch should throw an ArgumentError
            strided_test_3d_throw(T, fun)
            strided_test_4d_throw(T, fun)
        end
    end

    # inplace fft is not tested for throw
    @testset for T in (ComplexF32, ComplexF64)
        fft!copy(x) = copy(fft!(x))
        fft!copy(x,y) = copy(fft!(x,y))
        #compare the result of inplace and outplace fft with strided layout
        strided_test_2d(T, fft, fft!copy)
        strided_test_3d(T, fft, fft!copy)
        strided_test_4d(T, fft, fft!copy)
    end
    # bfft 
    @testset "brfft" begin
        a = CUDA.rand(ComplexF32,33,32)
        b = @view a[1:32,:]
        p = plan_brfft(b,63)
        @test p * b ≈ p * copy(b)
        a = CUDA.rand(ComplexF32,34,32)
        b = @view a[2:33,:]
        @test p * b ≈ p * copy(b)
    end
else
    # CUDA 10.x don't support this.
    a = @view CUDA.rand(ComplexF32,10,10)[1:9,:]
    @test_throws ArgumentError fft(a)
end
