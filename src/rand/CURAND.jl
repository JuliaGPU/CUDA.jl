
module CURAND

export curand,
       curandn,
       curand_logn,
       curand_poisson

using ..CuArrays: CuArray, libcurand

include("defs.jl")
include("error.jl")
include("wrappers.jl")
include("highlevel.jl")

const _rng = Ref{RNG}()

function __init__()
    _rng[] = create_generator()
end

end # module
