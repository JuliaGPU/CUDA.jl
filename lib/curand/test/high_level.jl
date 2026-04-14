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

    # unsupported types that fall back to the native generator
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

    @test_throws ErrorException rand_logn!(CuArray{Cuint}(undef, 10))
    @test_throws ErrorException rand_poisson!(CuArray{Float64}(undef, 10))

    # seeding
    cuRAND.seed!()
    cuRAND.seed!(1)
    cuRAND.seed!(1)
    A = cuRAND.rand(Float32, 1)
    cuRAND.seed!(1)
    B = cuRAND.rand(Float32, 1)
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
    end
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
