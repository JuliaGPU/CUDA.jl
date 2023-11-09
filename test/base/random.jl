# NOTE: tests should cover both pow2 and non-pow2 dims

@testset "high-level API" begin
    # in-place
    for (f,T) in ((rand!,Float32),
                  (rand!,Cuint),
                  (randn!,Float32),
                  (rand_logn!,Float32),
                  (rand_poisson!,Cuint)),
        d in (0, 2, (2,2), (2,2,2), 3, (3,3), (3,3,3))
        A = CuArray{T}(undef, d)
        f(A)
    end

    # out-of-place, with implicit type
    for (f,T) in ((CUDA.rand,Float32), (CUDA.randn,Float32),
                  (CUDA.rand_logn,Float32), (CUDA.rand_poisson,Cuint),
                  (rand,Float64), (randn,Float64)),
        args in ((0,), (2,), (2, 2), (3,), (3, 3))
        A = f(args...)
        @test eltype(A) == T
    end

    # out-of-place, with type specified
    for (f,T) in ((CUDA.rand,Float32), (CUDA.randn,Float32), (CUDA.rand_logn,Float32),
                  (CUDA.rand,Float64), (CUDA.randn,Float64), (CUDA.rand_logn,Float64),
                  (CUDA.rand_poisson,Cuint),
                  (rand,Float32), (randn,Float32),
                  (rand,Float64), (randn,Float64)),
        args in ((T, 0), (T, 2), (T, 2, 2), (T, (2, 2)), (T, 3), (T, 3, 3), (T, (3, 3)))
        A = f(args...)
        @test eltype(A) == T
    end

    # unsupported types that fall back to a native generator
    for (f,T) in ((CUDA.rand,Int64), (CUDA.randn,ComplexF64)),
        args in ((T, 0), (T, 2), (T, 2, 2), (T, (2, 2)), (T, 3), (T, 3, 3), (T, (3, 3)))
        A = f(args...)
        @test eltype(A) == T
    end
    for (f,T) in ((rand!,Int64), (randn!,ComplexF64)),
        d in (0, 2, (2,2), (2,2,2), 3, (3,3), (3,3,3))
        A = CuArray{T}(undef, d)
        f(A)
    end

    @test_throws ErrorException rand_logn!(CuArray{Cuint}(undef, 10))
    @test_throws ErrorException rand_poisson!(CuArray{Float64}(undef, 10))

    # seeding of both generators
    CUDA.seed!()
    CUDA.seed!(1)
    ## CUDA CURAND
    CUDA.seed!(1)
    A = CUDA.rand(Float32, 1)
    CUDA.seed!(1)
    B = CUDA.rand(Float32, 1)
    @test all(A .== B)
    ## GPUArrays fallback
    CUDA.seed!(1)
    A = CUDA.rand(Int64, 1)
    CUDA.seed!(1)
    B = CUDA.rand(Int64, 1)
    @test all(A .== B)

    # scalar number generation
    CUDA.@allowscalar let
        CUDA.rand()
        CUDA.rand(Float32)
        CUDA.randn()
        CUDA.randn(Float32)

        CUDA.rand_logn()
        CUDA.rand_logn(Float32)
        CUDA.rand_poisson()
        CUDA.rand_poisson(Cuint)
    end
end

@testset "native generator" begin
    rng = CUDA.RNG()
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
        CUDA.@allowscalar rand!(rng, B)
    end

    # normal
    for T in (Float16, Float32, Float64,
              ComplexF16, ComplexF32, ComplexF64),
        dims = (0, 2, (2,2), (2,2,2))
        A = CuArray{T}(undef, dims)
        randn!(rng, A)

        B = Array{T}(undef, dims)
        CUDA.@allowscalar rand!(rng, B)
    end

    ## out-of-place

    # uniform
    CUDA.@allowscalar begin
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
    CUDA.@allowscalar begin
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

    # #1464: Check that the Box-Muller transform doesn't produce infinities (stemming from
    # zeros in the radial sample). Virtually deterministic for the typical 23-24 bits of
    # entropy; a larger sample would be needed for a higher-entropy algorithm like the one
    # used by CURAND.
    @test isfinite(maximum(randn(rng, Float32, 2^26)))

    # #1515: A quick way to check if the Box-Muller transform is correctly implemented for
    # complex numbers is to check that the real part never gets too large. The largest
    # possible value for ComplexF32 is sqrt(-log(u)) where u is the smallest nonzero Float32
    # that can be produced by rand. Typically u = 2f0^(-23) or u = 2f0^(-24) giving an upper
    # bound of around 4 or 4.1, while CURAND.rand gets down to u = 2f0^(-33) giving an upper
    # bound of around 4.8. In contrast, incorrectly reusing the real Box-Muller transform
    # gives typical real parts in the hundreds.
    @test maximum(real(randn(rng, ComplexF32, 32))) <= sqrt(-log(2f0^(-33)))
end

@testset "seeding idempotency" begin
    t = @async begin
        Random.seed!(1)
        CUDA.seed!(1)
        x = rand()

        Random.seed!(1)
        CUDA.seed!(1)
        y = rand()

        x == y
    end
    @test fetch(t)
end

@testset "copy RNGs" begin
    let r1 = CUDA.default_rng(), r2 = copy(CUDA.default_rng())
        @test r2 isa CUDA.RNG
        @test r1 !== r2
        @test r1 == r2

        rand(r1, 1)
        @test r1 != r2
    end

    # JuliaGPU/CUDA.jl#1575
    let r1 = CUDA.default_rng(), r2 = copy(CUDA.default_rng())
        @test rand(r1, 3) == rand(r2, 3)
        @test rand(r1, 30_000) == rand(r2, 30_000)
    end

    let r1 = copy(CUDA.default_rng()), r2 = copy(CUDA.default_rng())
        x1 = rand(r1, 30, 10, 100)
        sum(rand(r1, 30) .+ x1 .+ CUDA.randn(30))  # do some other work
        x2 = rand(r2, 30, 10, 100)
        @test x1 == x2
    end

    let r1 = copy(CUDA.default_rng()), r2 = copy(CUDA.default_rng())
        t2 = @async rand(r1, 1)
        t2 = @async rand(r2, 1)
        @test fetch(t2) == fetch(t2)
    end
end

