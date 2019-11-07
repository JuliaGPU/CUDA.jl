module CURAND

using ..CuArrays
using ..CuArrays: active_context

using CUDAapi

using CUDAdrv
using CUDAdrv: CUstream

import CUDAnative

using CEnum

const libcurand = if Sys.iswindows()
    # no ccall by soname, we need the filename
    # NOTE: we discover the full path here, while only the wordsize and toolkit versions
    #       would have been enough to construct "curand64_10.dll"
    toolkit = find_toolkit()
    path = find_cuda_library("curand", toolkit)
    if path === nothing
        error("Could not find libcurand")
    end
    basename(path)
else
    # ccall by soname; CuArrays.__init__ will have populated Libdl.DL_LOAD_PATH
    "libcurand"
end

# core library
include("libcurand_common.jl")
include("error.jl")
include("libcurand.jl")

# low-level wrappers
include("wrappers.jl")

# high-level integrations
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
