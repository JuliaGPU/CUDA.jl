module CURAND

import CUDAapi

import CUDAdrv: CUDAdrv, CuContext, CuPtr

import CUDAnative

using ..CuArrays
using ..CuArrays: libcurand, active_context

using GPUArrays

using Random

export rand_logn!, rand_poisson!

include("libcurand_types.jl")
include("error.jl")

const _generators = Dict{CuContext,RNG}()
const _generator = Ref{Union{Nothing,RNG}}(nothing)

function generator()
    if _generator[] == nothing
        CUDAnative.maybe_initialize("CURAND")
        _generator[] = get!(_generators, active_context[]) do
            context = active_context[]
            generator = curandCreateGenerator()
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

const seed! = CURAND.seed!
const rand = CURAND.rand
const randn = CURAND.randn
const rand_logn = CURAND.rand_logn
const rand_poisson = CURAND.rand_poisson

@deprecate curand CuArrays.rand
@deprecate curandn CuArrays.randn
