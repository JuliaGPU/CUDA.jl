module CURAND

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, libraryPropertyType, DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
using ..CUDA: libcurand, @retry_reclaim

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
const CURAND_THREAD_RNGs = Vector{Union{Nothing,RNG}}()
const GPUARRAY_THREAD_RNGs = Vector{Union{Nothing,GPUArrays.RNG}}()

function default_rng()
    tid = Threads.threadid()
    if @inbounds CURAND_THREAD_RNGs[tid] === nothing
        ctx = context()
        CURAND_THREAD_RNGs[tid] = get!(task_local_storage(), (:CURAND, ctx)) do
            rng = RNG()
            Random.seed!(rng)
            rng
        end
    end
    @inbounds CURAND_THREAD_RNGs[tid]
end

function GPUArrays.default_rng(::Type{<:CuArray})
    tid = Threads.threadid()
    if @inbounds GPUARRAY_THREAD_RNGs[tid] === nothing
        ctx = context()
        GPUARRAY_THREAD_RNGs[tid] = get!(task_local_storage(), (:GPUArraysRNG, ctx)) do
            dev = device()
            N = attribute(dev, DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            state = CuArray{NTuple{4, UInt32}}(undef, N)
            rng = GPUArrays.RNG(state)
            Random.seed!(rng)
            rng
        end
    end
    @inbounds GPUARRAY_THREAD_RNGs[tid]
end

function __init__()
    resize!(CURAND_THREAD_RNGs, Threads.nthreads())
    fill!(CURAND_THREAD_RNGs, nothing)

    resize!(GPUARRAY_THREAD_RNGs, Threads.nthreads())
    fill!(GPUARRAY_THREAD_RNGs, nothing)

    CUDA.atdeviceswitch() do
        tid = Threads.threadid()
        CURAND_THREAD_RNGs[tid] = nothing
        GPUARRAY_THREAD_RNGs[tid] = nothing
    end

    CUDA.attaskswitch() do
        tid = Threads.threadid()
        CURAND_THREAD_RNGs[tid] = nothing
        GPUARRAY_THREAD_RNGs[tid] = nothing
    end
end

@deprecate seed!() CUDA.seed!()
@deprecate seed!(seed) CUDA.seed!(seed)

end
