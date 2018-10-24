module CURAND

using ..CuArrays: CuArray, libcurand

using Random

export curand, curand!,
       curandn, curandn!,
       curand_logn, curand_logn!,
       curand_poisson, curand_poisson!

include("libcurand_defs.jl")
include("error.jl")
include("libcurand.jl")
include("highlevel.jl")

end # module
