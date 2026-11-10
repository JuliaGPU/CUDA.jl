# NOTE: tests should cover both pow2 and non-pow2 dims

@testset "cuRAND library" begin
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
    for (f,T) in ((cuRAND.rand,Float32), (cuRAND.randn,Float32),
                  (cuRAND.rand_logn,Float32), (cuRAND.rand_poisson,Cuint),
                  (rand,Float64), (randn,Float64)),
        args in ((0,), (2,), (2, 2), (3,), (3, 3))
        A = f(args...)
        @test eltype(A) == T
    end

    # out-of-place, with type specified
    for (f,T) in ((cuRAND.rand,Float32), (cuRAND.randn,Float32), (cuRAND.rand_logn,Float32),
                  (cuRAND.rand,Float64), (cuRAND.randn,Float64), (cuRAND.rand_logn,Float64),
                  (cuRAND.rand_poisson,Cuint),
                  (rand,Float32), (randn,Float32),
                  (rand,Float64), (randn,Float64)),
        args in ((T, 0), (T, 2), (T, 2, 2), (T, (2, 2)), (T, 3), (T, 3, 3), (T, (3, 3)))
        A = f(args...)
        @test eltype(A) == T
    end

    # unsupported types that fall back to the GPUArrays RNG
    for (f,T) in ((cuRAND.rand,Int64), (cuRAND.randn,ComplexF64)),
        args in ((T, 0), (T, 2), (T, 2, 2), (T, (2, 2)), (T, 3), (T, 3, 3), (T, (3, 3)))
        A = f(args...)
        @test eltype(A) == T
    end
    for (f,T) in ((rand!,Int64), (randn!,ComplexF64)),
        d in (0, 2, (2,2), (2,2,2), 3, (3,3), (3,3,3))
        A = CuArray{T}(undef, d)
        f(A)
    end

    # the fallback supports the same keyword arguments
    for T in (Float16, ComplexF32)
        @test Array(cuRAND.randn(T, 3; mean=2, stddev=0)) == fill(T(2), 3)
        @test Array(cuRAND.randn(T, 2, 2; mean=2, stddev=0)) == fill(T(2), 2, 2)
        @test Array(randn!(CuArray{T}(undef, 3); mean=2, stddev=0)) == fill(T(2), 3)
    end

    @test_throws ErrorException rand_logn!(CuArray{Cuint}(undef, 10))
    @test_throws ErrorException rand_poisson!(CuArray{Float64}(undef, 10))

    # seeding
    cuRAND.seed!()
    cuRAND.seed!(1)
    ## cuRAND library
    cuRAND.seed!(1)
    A = cuRAND.rand(Float32, 1)
    cuRAND.seed!(1)
    B = cuRAND.rand(Float32, 1)
    @test all(A .== B)
    ## GPUArrays fallback
    cuRAND.seed!(1)
    A = cuRAND.rand(Int64, 1)
    cuRAND.seed!(1)
    B = cuRAND.rand(Int64, 1)
    @test all(A .== B)

    # scalar number generation
    CUDACore.@allowscalar let
        cuRAND.rand()
        cuRAND.rand(Float32)
        cuRAND.randn()
        cuRAND.randn(Float32)

        cuRAND.rand_logn()
        cuRAND.rand_logn(Float32)
        cuRAND.rand_poisson()
        cuRAND.rand_poisson(Cuint)

        rng = cuRAND.library_rng()
        @test rand(rng) isa Float32
        @test rand(rng, Float64) isa Float64
        @test randn(rng) isa Float32
        @test randn(rng, Float64; mean=1, stddev=0) == 1
        @test cuRAND.rand_logn(rng) isa Float32
        @test cuRAND.rand_poisson(rng) isa Cuint
        @test cuRAND.rand_poisson(rng, Cuint; lambda=5) isa Cuint
    end

    # CPU arrays and views, generated on the GPU and copied over
    rng = cuRAND.library_rng()
    for (f,T) in ((rand!,Float32), (rand!,Float64), (rand!,Cuint),
                  (randn!,Float32), (randn!,Float64),
                  (rand_logn!,Float32), (rand_logn!,Float64),
                  (rand_poisson!,Cuint))
        # a value to tell which elements were written (uniform UInt32 draws produce
        # typemax once in 2^32)
        x = T == Cuint ? typemax(Cuint) : T(NaN)

        for d in (0, 2, (2,2), 3, (3,3))
            A = fill(x, d)
            @test f(rng, A) === A
            @test all(!isequal(x), A)
        end

        B = fill(x, 4, 3)
        f(rng, view(B, 1:2:4, :))
        @test all(!isequal(x), B[1:2:4, :])
        @test all(isequal(x), B[2:2:4, :])

        C = CuArray(fill(x, 4, 3))
        f(rng, view(C, 1:2:4, :))
        @test all(!isequal(x), Array(C)[1:2:4, :])
        @test all(isequal(x), Array(C)[2:2:4, :])
    end
    @test randn!(rng, zeros(Float32, 3); mean=2, stddev=0) == fill(2f0, 3)
    @test rand_logn!(rng, zeros(Float64, 3); mean=0, stddev=0) == fill(1.0, 3)
    if VERSION >= v"1.11"
        # dense, but not an Array, which copyto! from a CuArray does not support
        M = fill!(Memory{Float32}(undef, 3), NaN32)
        @test all(!isnan, rand!(rng, M))
    end
    # element types that cuRAND does not generate keep using Random's fallback
    @test_throws MethodError rand!(rng, Vector{Union{Float32,Float64}}(undef, 2))
    @test all(>(1000), rand_poisson!(rng, zeros(Cuint, 3); lambda=10_000))
    C = CuArray(zeros(Float32, 4))
    randn!(rng, view(C, 1:2:4); mean=2, stddev=0)
    @test Array(C) == [2, 0, 2, 0]
end

@testset "seeding idempotency" begin
    t = @async begin
        Random.seed!(1)
        cuRAND.seed!(1)
        x = rand()

        Random.seed!(1)
        cuRAND.seed!(1)
        y = rand()

        x == y
    end
    @test fetch(t)
end

@testset "releasing cached generators" begin
    # destroying a generator may wait for the GPU, so that's deferred until reclaiming
    # memory, which also empties the cache it was returned to
    @noinline function borrow_and_return()
        ctx = context()
        rng = pop!(cuRAND.idle_library_rngs, ctx)
        push!(cuRAND.idle_library_rngs, ctx, rng)
        return WeakRef(rng)
    end
    weak = borrow_and_return()
    CUDACore.reclaim()
    GC.gc(true)
    @test weak.value === nothing
end
