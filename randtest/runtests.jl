
using CURAND
using Base.Test

# smoke tests for high-level API
rng = create_generator()
n = 10
n = 10
m = 5.
mf = Float32(5.)
sd = .1
sdf = Float32(.1)
lambda = .5

curand(rng, Float32, n)
curand(Float32, n)
curand(rng, Float64, n)
curand(Float64, n)
curand(n)

curandn(rng, Float32, n, mf, sdf)
curandn(Float32, n, mf, sdf)
curandn(rng, Float64, n, m, sd)
curandn(Float64, n, m, sd)
curandn(n, m, sd)

curand_logn(rng, Float32, n, mf, sdf)
curand_logn(Float32, n, mf, sdf)
curand_logn(rng, Float64, n, m, sd)
curand_logn(Float64, n, m, sd)
curand_logn(n, m, sd)

curand_poisson(rng, n, lambda)
curand_poisson(n, lambda)

destroy_generator(rng)

println("ok.")
