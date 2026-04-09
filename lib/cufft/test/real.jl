function out_of_place(X::AbstractArray{T,N}) where {T <: Real,N}
    fftw_X = rfft(X)
    d_X = CuArray(X)
    p = @inferred plan_rfft(d_X)
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

    pinvb = @inferred plan_brfft(d_Y,size(X,1))
    d_Z = pinvb * d_Y
    Z = collect(d_Z) ./ length(X)
    @test isapprox(Z, X, rtol = rtol(T), atol = atol(T))
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
    # This should only throw an error for rfft type transforms:
    @test_throws ArgumentError batched(X,(3,1))
end

@testset "Batch 2D (in 4D)" begin
    dims = (N1,N2,N3,N4)
    for region in [(1,2),(1,3),(2,3)]
        X = rand(T, dims)
        batched(X,region)
    end
    for region in [(2,4),(1,4),(3,4)]
        if (T <: Float16)
            # for odd dimensions covering axes of or between transform axes
            # there is a rather unspecific CUFFTError: "CUFFT_INCOMPLETE_PARAMETER_LIST" thrown. This should be caught elsewhere to give more specific hint
            # on how to avoid this. Maybe as of Cuda 13 this issue is removed?
            X = rand(T, dims);
            @test_throws CUFFTError batched(X,region)
        else
            batched(X,region)
        end
    end

    X = rand(T, dims)
    @test_throws ArgumentError batched(X,(3,1))
end

@testset "3D" begin
    X = rand(T, N1, N2, N3)
    out_of_place(X)
end

end
