module CURAND

import CUDAdrv: CUDAdrv, CuContext
import CUDAapi

using ..CuArrays
using ..CuArrays: libcurand, active_context

using GPUArrays

using Random

export curand,
       curandn,
       curand_logn, rand_logn!,
       curand_poisson, rand_poisson!

include("libcurand_types.jl")
include("error.jl")

const _generators = Dict{CuContext,RNG}()
const _generator = Ref{Union{Nothing,RNG}}(nothing)

function generator()
    if _generator[] == nothing
        @assert isassigned(active_context) # some other call should have initialized CUDA
        _generator[] = get!(_generators, active_context[]) do
            context = active_context[]
            generator = create_generator()
            # FIXME: crashes
            #atexit(()->CUDAdrv.isvalid(context) && destroy_generator(generator))
            generator
        end
    end

    return _generator[]::RNG
end

include("libcurand.jl")
include("highlevel.jl")

version() = VersionNumber(curandGetProperty(CUDAapi.MAJOR_VERSION),
                          curandGetProperty(CUDAapi.MINOR_VERSION),
                          curandGetProperty(CUDAapi.PATCH_LEVEL))

end
