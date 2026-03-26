function out_of_place(X::AbstractArray{T,N}) where {T <: Complex,N}
    fftw_X = fft(X)
    d_X = CuArray(X)
    p = @inferred plan_fft(d_X)
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

    pinvb = @inferred plan_bfft(d_Y)
    d_Z = pinvb * d_Y
    Z = collect(d_Z) ./ length(d_Z)
    @test isapprox(Z, X, rtol = rtol(T), atol = atol(T))
end

function in_place(X::AbstractArray{T,N}) where {T <: Complex,N}
    fftw_X = fft(X)
    d_X = CuArray(X)
    p = @inferred plan_fft!(d_X)
    p * d_X
    Y = collect(d_X)
    @test isapprox(Y, fftw_X, rtol = rtol(T), atol = atol(T))

    pinv = plan_ifft!(d_X)
    pinv * d_X
    Z = collect(d_X)
    @test isapprox(Z, X, rtol = rtol(T), atol = atol(T))
    p * d_X

    pinvb = @inferred plan_bfft!(d_X)
    pinvb * d_X
    Z = collect(d_X) ./ length(X)
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
