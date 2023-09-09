using CUDA.CURAND

@test CURAND.version() isa VersionNumber

rng = CURAND.default_rng()
Random.seed!(rng)
Random.seed!(rng, nothing)
Random.seed!(rng, 1)
Random.seed!(rng, 1, 0)
