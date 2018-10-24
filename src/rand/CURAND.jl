module CURAND

using ..GPUArrays
using ..CuArrays: CuArray, libcurand

using Random

export curand,
       curandn,
       curand_logn, rand_logn!,
       curand_poisson, rand_poisson!

include("libcurand_defs.jl")
include("error.jl")
include("libcurand.jl")
include("highlevel.jl")

end # module
