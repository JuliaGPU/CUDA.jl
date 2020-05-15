module CURAND

using ..CuArrays
using ..CuArrays: libcurand, @retry_reclaim

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

# thread cache for task-local library handles
const thread_generators = Vector{Union{Nothing,RNG}}()

function generator()
    tid = Threads.threadid()
    if @inbounds thread_generators[tid] === nothing
        ctx = context()
        thread_generators[tid] = get!(task_local_storage(), (:CURAND, ctx)) do
            rng = RNG()
            Random.seed!(rng)
            rng
        end
    end
    @inbounds thread_generators[tid]
end

function __init__()
    resize!(thread_generators, Threads.nthreads())
    fill!(thread_generators, nothing)

    CUDAnative.atcontextswitch() do tid, ctx
        thread_generators[tid] = nothing
    end

    CUDAnative.attaskswitch() do tid, task
        thread_generators[tid] = nothing
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
