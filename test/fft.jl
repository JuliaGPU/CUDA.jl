@testset "cuFFT" begin

# notes:
#   plan_bfft does not need separate testing since it is used by plan_ifft

using CuArrays.CUFFT
using FFTW

N1 = 8
N2 = 32
N3 = 64
N4 = 8

MYRTOL = 1e-5
MYATOL = 1e-8

# out-of-place
function dotest1(X::AbstractArray{T,N}) where {T <: Complex,N}
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

function dotest1(X::AbstractArray{T,N}) where {T <: Real,N}
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

# in-place
function dotest2(X::AbstractArray{T,N}) where {T <: Complex,N}
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

# no inplace rfft for now

# batch transforms
function dotest3(X::AbstractArray{T,N},region) where {T <: Complex,N}
    fftw_X = fft(X,region)
    d_X = CuArray(X)
    p = plan_fft(d_X,region)
    d_Y = p * d_X
    Y = collect(d_Y)
    @test isapprox(Y, fftw_X, rtol = MYRTOL, atol = MYATOL)

    pinv = plan_ifft(d_Y,region)
    d_Z = pinv * d_Y
    Z = collect(d_Z)
    @test isapprox(Z, X, rtol = MYRTOL, atol = MYATOL)
end

function dotest3(X::AbstractArray{T,N},region) where {T <: Real,N}
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


@testset "FFT" for (rtype,ctype) in [(Float32,ComplexF32), (Float64,ComplexF64)]

@testset "1D FFT" begin
    dims = (N1,)
    X = rand(ctype, dims)
    dotest1(X)
end
@testset "1D inplace FFT" begin
    dims = (N1,)
    X = rand(ctype, dims)
    dotest2(X)
end

@testset "2D FFT" begin
    dims = (N1,N2)
    X = rand(ctype, dims)
    dotest1(X)
end
@testset "2D inplace FFT" begin
    dims = (N1,N2)
    X = rand(ctype, dims)
    dotest2(X)
end

@testset "Batch 1D FFT" begin
    dims = (N1,N2)
    X = rand(ctype, dims)
    dotest3(X,1)

    dims = (N1,N2)
    X = rand(ctype, dims)
    dotest3(X,2)

    dims = (N1,N2)
    X = rand(ctype, dims)
    dotest3(X,(1,2))
end

@testset "3D FFT" begin
    dims = (N1,N2,N3)
    X = rand(ctype, dims)
    dotest1(X)
end
@testset "3D inplace FFT" begin
    dims = (N1,N2,N3)
    X = rand(ctype, dims)
    dotest2(X)
end

@testset "Batch 2D FFT (in 3D)" begin
    dims = (N1,N2,N3)
    for region in [(1,2),(2,3),(1,3)]
        X = rand(ctype, dims)
        dotest3(X,region)
    end

    X = rand(ctype, dims)
    @test_throws ArgumentError dotest3(X,(3,1))
end

@testset "Batch 2D FFT (in 4D)" begin
    dims = (N1,N2,N3,N4)
    for region in [(1,2),(1,4),(3,4)]
        X = rand(ctype, dims)
        dotest3(X,region)
    end
    for region in [(1,3),(2,3),(2,4)]
        X = rand(ctype, dims)
        @test_throws ArgumentError dotest3(X,region)
    end

end

@testset "1D real FFT" begin
    X = rand(rtype, N1)
    dotest1(X)
end

@testset "Batch 1D real FFT" begin
    dims = (N1,N2)
    X = rand(rtype, dims)
    dotest3(X,1)

    dims = (N1,N2)
    X = rand(rtype, dims)
    dotest3(X,2)

    dims = (N1,N2)
    X = rand(rtype, dims)
    dotest3(X,(1,2))
end

@testset "2D real FFT" begin
    X = rand(rtype, N1,N2)
    dotest1(X)
end

@testset "Batch 2D real FFT (in 3D)" begin
    dims = (N1,N2,N3)
    for region in [(1,2),(2,3),(1,3)]
        X = rand(rtype, dims)
        dotest3(X,region)
    end

    X = rand(rtype, dims)
    @test_throws ArgumentError dotest3(X,(3,1))
end

@testset "Batch 2D real FFT (in 4D)" begin
    dims = (N1,N2,N3,N4)
    for region in [(1,2),(1,4),(3,4)]
        X = rand(rtype, dims)
        dotest3(X,region)
    end
    for region in [(1,3),(2,3),(2,4)]
        X = rand(rtype, dims)
        @test_throws ArgumentError dotest3(X,region)
    end
end

@testset "3D real FFT" begin
    X = rand(rtype, N1, N2, N3)
    dotest1(X)
end

end # testset FFT

# integer array arguments
function dotest5(X::AbstractArray{T,N}) where {T <: Complex,N}
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

function dotest5(X::AbstractArray{T,N}) where {T <: Real,N}
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

@testset "Int FFT" for (rtype,ctype) in [(Int32,Complex{Int32}), (Int64,Complex{Int64})]

@testset "1D FFT" begin
    dims = (N1,)
    X = rand(ctype, dims)
    dotest5(X)
end

@testset "1D real FFT" begin
    X = rand(rtype, N1)
    dotest5(X)
end


end # testset int FFT

end