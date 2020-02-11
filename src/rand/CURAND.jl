module CURAND

using ..CuArrays
using ..CuArrays: libcurand

using CUDAapi

using CUDAdrv
using CUDAdrv: CUstream

using CUDAnative

using CEnum

# core library
include("libcurand_common.jl")
include("error.jl")
include("libcurand.jl")

# low-level wrappers
include("wrappers.jl")

# high-level integrations
include("random.jl")

const created_generators = IdDict{CuContext,RNG}()
const active_generators = Vector{Union{Nothing,RNG}}()

function generator()
    tid = Threads.threadid()
    if @inbounds active_generators[tid] === nothing
        ctx = context()
        active_generators[tid] = get!(created_generators, ctx) do
            RNG()
        end
    end
    @inbounds active_generators[tid]
end

function __init__()
    resize!(active_generators, Threads.nthreads())
    fill!(active_generators, nothing)

    CUDAnative.atcontextswitch() do tid, ctx
        # we don't eagerly initialize handles, but do so lazily when requested
        active_generators[tid] = nothing
    end
end

end

const seed! = CURAND.seed!
const rand = CURAND.rand
const randn = CURAND.randn
const rand_logn = CURAND.rand_logn
const rand_poisson = CURAND.rand_poisson

@deprecate curand CuArrays.rand
@deprecate curandn CuArrays.randn
