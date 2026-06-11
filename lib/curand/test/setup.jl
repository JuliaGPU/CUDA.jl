using Test
using Random

using CUDACore
using cuRAND
using cuRAND: rand_logn!, rand_logn, rand_poisson!, rand_poisson

rng = cuRAND.library_rng()
Random.seed!(rng)
Random.seed!(rng, nothing)
Random.seed!(rng, 1)
Random.seed!(rng, 1, 0)
