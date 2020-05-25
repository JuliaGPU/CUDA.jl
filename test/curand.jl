using CUDA.CURAND

rng = CURAND.generator()
Random.seed!(rng)
Random.seed!(rng, nothing)
Random.seed!(rng, 1)
Random.seed!(rng, 1, 0)

# NOTE: tests should cover both pow2 and non-pow2 dims

# in-place
for (f,T) in ((rand!,Float32),
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

# unsupported types that fall back to GPUArrays
for (f,T) in ((CUDA.rand,Int64),),
    args in ((T, 2), (T, 2, 2), (T, (2, 2)), (T, 3), (T, 3, 3), (T, (3, 3)))
    A = f(args...)
    @test eltype(A) == T
end
for (f,T) in ((rand!,Int64),),
    d in (2, (2,2), (2,2,2), 3, (3,3), (3,3,3))
    A = CuArray{T}(undef, d)
    f(A)
end

@test_throws ErrorException randn!(CuArray{Cuint}(undef, 10))
@test_throws ErrorException rand_logn!(CuArray{Cuint}(undef, 10))
@test_throws ErrorException rand_poisson!(CuArray{Float64}(undef, 10))

# seeding of both generators
CURAND.seed!()
CURAND.seed!(1)
## CUDA CURAND
CURAND.seed!(1)
A = CUDA.rand(Float32, 1)
CURAND.seed!(1)
B = CUDA.rand(Float32, 1)
@test all(A .== B)
## GPUArrays fallback
CURAND.seed!(1)
A = CUDA.rand(Int64, 1)
CURAND.seed!(1)
B = CUDA.rand(Int64, 1)
@test all(A .== B)
