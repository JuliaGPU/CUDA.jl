# NOTE: tests should cover both pow2 and non-pow2 dims

@testset "high-level API" begin
    # in-place
    for (f,T) in ((rand!,Float32),
                  (rand!,Cuint),
                  (randn!,Float32),
                  (rand_logn!,Float32),
                  (rand_poisson!,Cuint)),
        d in (2, (2,2), (2,2,2), 3, (3,3), (3,3,3))
        A = CuArray{T}(undef, d)
        f(A)
    end

    # out-of-place, with implicit type
    for (f,T) in ((CUDA.rand,Float32), (CUDA.randn,Float32),
                  (CUDA.rand_logn,Float32), (CUDA.rand_poisson,Cuint),
                  (rand,Float64), (randn,Float64)),
        args in ((2,), (2, 2), (3,), (3, 3))
        A = f(args...)
        @test eltype(A) == T
    end

    # out-of-place, with type specified
    for (f,T) in ((CUDA.rand,Float32), (CUDA.randn,Float32), (CUDA.rand_logn,Float32),
                  (CUDA.rand,Float64), (CUDA.randn,Float64), (CUDA.rand_logn,Float64),
                  (CUDA.rand_poisson,Cuint),
                  (rand,Float32), (randn,Float32),
                  (rand,Float64), (randn,Float64)),
        args in ((T, 2), (T, 2, 2), (T, (2, 2)), (T, 3), (T, 3, 3), (T, (3, 3)))
        A = f(args...)
        @test eltype(A) == T
    end

    # unsupported types that fall back to a native generator
    for (f,T) in ((CUDA.rand,Int64), (CUDA.randn,ComplexF64)),
        args in ((T, 2), (T, 2, 2), (T, (2, 2)), (T, 3), (T, 3, 3), (T, (3, 3)))
        A = f(args...)
        @test eltype(A) == T
    end
    for (f,T) in ((rand!,Int64), (randn!,ComplexF64)),
        d in (2, (2,2), (2,2,2), 3, (3,3), (3,3,3))
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
              UInt8, UInt16, UInt32, UInt64, UInt128)
        A = CuArray{T}(undef, 2048)
        rand!(rng, A)

        B = Array{T}(undef, 2048)
        CUDA.@allowscalar rand!(rng, B)
    end

    # normal
    for T in (Float16, Float32, Float64,
              ComplexF16, ComplexF32, ComplexF64)
        A = CuArray{T}(undef, 2048)
        randn!(rng, A)

        B = Array{T}(undef, 2048)
        CUDA.@allowscalar rand!(rng, B)
    end

    ## out-of-place

    # uniform
    CUDA.@allowscalar begin
        @test rand(rng) isa Number
        @test rand(rng, Float32) isa Float32
    end
    @test rand(rng, Float32, 1) isa CuArray
    @test rand(rng, 1) isa CuArray

    # normal
    CUDA.@allowscalar begin
        @test randn(rng) isa Number
        @test randn(rng, Float32) isa Float32
    end
    @test randn(rng, Float32, 1) isa CuArray
    @test randn(rng, 1) isa CuArray

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
    r1 = CUDA.default_rng()
    r2 = copy(CUDA.default_rng())
    @test r2 isa CUDA.RNG
    @test r1 == r2
    @test rand(r1, 3) == rand(r2, 3)
    # Before CUDA 3.12.1, size 3 worked, size 30_000 failed:
    @test rand(r1, 30_000) == rand(r2, 30_000)
    @test r1 == r2
    rand(r1, 3)
    @test r1 != r2

    r3 = copy(CUDA.default_rng())
    r4 = copy(CUDA.default_rng())

    x3 = rand(r3, ComplexF32, 30, 10, 100)
    sum(rand(r3, 30) .+ x3 .+ CUDA.randn(30))  # do some other work
    x4 = rand(r4, ComplexF32, 30, 10, 100)
    @test x3 == x4

    t3 = @async rand(r3, ComplexF32, 3, 4)
    t0 = @async rand(r1, 10)
    t4 = @async rand(r4, ComplexF32, 3, 4)
    @test fetch(t0) isa CuArray
    @test_skip fetch(t3) == fetch(t4)
end

