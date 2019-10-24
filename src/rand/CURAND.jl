module CURAND

using ..CuArrays
using ..CuArrays: libcurand, active_context

using CUDAapi

using CUDAdrv
using CUDAdrv: CUstream

import CUDAnative

using CEnum

include("libcurand_common.jl")
include("error.jl")

version() = VersionNumber(curandGetProperty(CUDAapi.MAJOR_VERSION),
                          curandGetProperty(CUDAapi.MINOR_VERSION),
                          curandGetProperty(CUDAapi.PATCH_LEVEL))

include("libcurand.jl")
include("wrappers.jl")
include("random.jl")

const _generators = Dict{CuContext,RNG}()
const _generator = Ref{Union{Nothing,RNG}}(nothing)

function generator()
    if _generator[] == nothing
        CUDAnative.maybe_initialize("CURAND")
        _generator[] = get!(_generators, active_context[]) do
            context = active_context[]
            RNG()
        end
    end

    return _generator[]::RNG
end

end

const seed! = CURAND.seed!
const rand = CURAND.rand
const randn = CURAND.randn
const rand_logn = CURAND.rand_logn
const rand_poisson = CURAND.rand_poisson

@deprecate curand CuArrays.rand
@deprecate curandn CuArrays.randn
