module CURAND

using CUDAdrv

using Random

using ..GPUArrays
using ..CuArrays: CuArray, libcurand, active_context

export curand,
       curandn,
       curand_logn, rand_logn!,
       curand_poisson, rand_poisson!

include("libcurand_defs.jl")

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

include("error.jl")
include("libcurand.jl")
include("highlevel.jl")

end # module
