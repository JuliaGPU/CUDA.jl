@testset "native generator" begin
    rng = cuRAND.NativeRNG()
    Random.seed!(rng)

    ## in-place

    # uniform
    for T in (Float16, Float32, Float64,
              ComplexF16, ComplexF32, ComplexF64,
              Int8, Int16, Int32, Int64, Int128,
              UInt8, UInt16, UInt32, UInt64, UInt128),
        dims = (0, 2, (2,2), (2,2,2))
        A = CuArray{T}(undef, dims)
        rand!(rng, A)

        B = Array{T}(undef, dims)
        CUDACore.@allowscalar rand!(rng, B)
    end

    # normal
    for T in (Float16, Float32, Float64,
              ComplexF16, ComplexF32, ComplexF64),
        dims = (0, 2, (2,2), (2,2,2))
        A = CuArray{T}(undef, dims)
        randn!(rng, A)

        B = Array{T}(undef, dims)
        CUDACore.@allowscalar rand!(rng, B)
    end

    ## out-of-place

    # uniform
    CUDACore.@allowscalar begin
        @test rand(rng) isa Number
        @test rand(rng, Float32) isa Float32
    end
    for dims in (0, 2, (2,2), (2,2,2))
        @test rand(rng, dims) isa CuArray
        for T in (Float16, Float32, Float64,
                  ComplexF16, ComplexF32, ComplexF64,
                  Int8, Int16, Int32, Int64, Int128,
                  UInt8, UInt16, UInt32, UInt64, UInt128)
            @test rand(rng, T, dims) isa CuArray{T}
        end
    end

    # normal
    CUDACore.@allowscalar begin
        @test randn(rng) isa Number
        @test randn(rng, Float32) isa Float32
    end
    for dims in (0, 2, (2,2), (2,2,2))
        @test randn(rng, dims) isa CuArray
        for T in (Float16, Float32, Float64,
                  ComplexF16, ComplexF32, ComplexF64)
            @test randn(rng, T, dims) isa CuArray{T}
        end
    end

    # #1464: Check that the Box-Muller transform doesn't produce infinities
    # (stemming from zeros in the radial sample).
    @test isfinite(maximum(randn(rng, Float32, 2^26)))

    # #1515: Check that the Box-Muller transform is correctly implemented for
    # complex numbers: the real part should never get too large. The largest
    # possible value for ComplexF32 is sqrt(-log(u)) where u is the smallest
    # nonzero Float32 produced by rand.
    @test maximum(real(randn(rng, ComplexF32, 32))) <= sqrt(-log(2f0^(-33)))
end

@testset "copy NativeRNG" begin
    let r1 = cuRAND.native_rng(), r2 = copy(cuRAND.native_rng())
        @test r2 isa cuRAND.NativeRNG
        @test r1 !== r2
        @test r1 == r2

        rand(r1, 1)
        @test r1 != r2
    end

    # JuliaGPU/CUDA.jl#1575
    let r1 = cuRAND.native_rng(), r2 = copy(cuRAND.native_rng())
        @test rand(r1, 3) == rand(r2, 3)
        @test rand(r1, 30_000) == rand(r2, 30_000)
    end

    let r1 = copy(cuRAND.native_rng()), r2 = copy(cuRAND.native_rng())
        x1 = rand(r1, 30, 10, 100)
        sum(rand(r1, 30) .+ x1 .+ cuRAND.randn(30))  # do some other work
        x2 = rand(r2, 30, 10, 100)
        @test x1 == x2
    end

    let r1 = copy(cuRAND.native_rng()), r2 = copy(cuRAND.native_rng())
        t2 = @async rand(r1, 1)
        t2 = @async rand(r2, 1)
        @test fetch(t2) == fetch(t2)
    end
end
