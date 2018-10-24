@testset "cuRAND" begin

using CuArrays.CURAND

CURAND.seed!()

# in-place
for (f,T) in ((rand!,Float32), (curand!,Float32),
              (randn!,Float32), (curandn!,Float32),
              (curand_logn!,Float32),
              (curand_poisson!,Cuint)),
    d in (2, (2,2), (2,2,2))
    A = CuArray{T}(d)
    f(A)
end

# out-of-place, with implicit type
for (f,T) in ((curand,Float32), (curandn,Float32), (curand_logn,Float32),
              (curand_poisson,Cuint)),
    args in ((2,), (2, 2))
    A = f(args...)
    @test eltype(A) == T
end

# out-of-place, with type specified
for (f,T) in ((curand,Float32), (curandn,Float32), (curand_logn,Float32),
              (curand,Float64), (curandn,Float64), (curand_logn,Float64),
              (curand_poisson,Cuint)),
    args in ((T, 2), (T, 2, 2), (T, (2, 2)))
    A = f(args...)
    @test eltype(A) == T
end

end