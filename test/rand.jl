@testset "CURAND" begin

using CuArrays.CURAND

CURAND.seed!()

# in-place
for (f,T) in ((rand!,Float32),
              (randn!,Float32),
              (rand_logn!,Float32),
              (rand_poisson!,Cuint)),
    d in (2, (2,2), (2,2,2))
    A = CuArray{T}(undef, d)
    f(A)
end

# out-of-place, with implicit type
for (f,T) in ((CuArrays.rand,Float32), (CuArrays.randn,Float32),
              (CuArrays.rand_logn,Float32), (CuArrays.rand_poisson,Cuint),
              (rand,Float64), (randn,Float64)),
    args in ((2,), (2, 2))
    A = f(args...)
    @test eltype(A) == T
end

# out-of-place, with type specified
for (f,T) in ((CuArrays.rand,Float32), (CuArrays.randn,Float32), (CuArrays.rand_logn,Float32),
              (CuArrays.rand,Float64), (CuArrays.randn,Float64), (CuArrays.rand_logn,Float64),
              (CuArrays.rand_poisson,Cuint),
              (rand,Float32), (randn,Float32),
              (rand,Float64), (randn,Float64)),
    args in ((T, 2), (T, 2, 2), (T, (2, 2)))
    A = f(args...)
    @test eltype(A) == T
end

# unsupported types that fall back to GPUArrays
for (f,T) in ((CuArrays.rand,Int64),),
    args in ((T, 2), (T, 2, 2), (T, (2, 2)))
    A = f(args...)
    @test eltype(A) == T
end
for (f,T) in ((rand!,Int64),),
    d in (2, (2,2), (2,2,2))
    A = CuArray{T}(undef, d)
    f(A)
end

@test_throws ErrorException randn!(CuArray{Cuint}(undef, 10)) 
@test_throws ErrorException rand_logn!(CuArray{Cuint}(undef, 10)) 
@test_throws ErrorException rand_poisson!(CuArray{Float64}(undef, 10)) 

end
