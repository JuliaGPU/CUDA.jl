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
end
