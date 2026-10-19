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


## complex-to-real promotion

function complex_to_real(X::AbstractArray{T,N}) where {T,N}
    d = 2size(X, 1) - 2
    d_X = CuArray(X)
    for f in (brfft, irfft)
        d_Y = f(d_X, d)
        @test d_Y isa CuArray
        @test isapprox(collect(d_Y), f(X, d), rtol = rtol(T), atol = atol(T))
    end
end

@testset for T in [Complex{Int32}, Complex{Int64}, Float32]

@testset "1D" begin
    X = rand(T, N1)
    complex_to_real(X)
end

end
