module CURAND

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, libraryPropertyType, DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
using ..CUDA: libcurand, @retry_reclaim

using CEnum

using Memoize

using DataStructures


# core library
include("libcurand_common.jl")
include("error.jl")
include("libcurand.jl")

# low-level wrappers
include("wrappers.jl")

# high-level integrations
include("random.jl")

# cache for created, but unused handles
const idle_curand_rngs = HandleCache{CuContext,RNG}()
const idle_gpuarray_rngs = HandleCache{CuContext,GPUArrays.RNG}()

function default_rng()
    state = CUDA.active_state()
    rng = get!(task_local_storage(), (:CURAND, state.context)) do
        new_rng = pop!(idle_curand_rngs, state.context) do
            RNG()
        end

        finalizer(current_task()) do task
            push!(idle_curand_rngs, state.context, new_rng) do
                # no need to do anything, as the RNG is collected by its finalizer
            end
        end

        Random.seed!(new_rng)
        new_rng
    end::RNG

    return rng
end

function GPUArrays.default_rng(::Type{<:CuArray})
    ctx = context()
    get!(task_local_storage(), (:GPUArraysRNG, ctx)) do
        rng = pop!(idle_gpuarray_rngs, ctx) do
            dev = device()
            N = attribute(dev, DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            state = CuArray{NTuple{4, UInt32}}(undef, N)
            GPUArrays.RNG(state)
        end

        finalizer(current_task()) do task
            push!(idle_gpuarray_rngs, ctx, rng) do
                # no need to do anything, as the RNG is collected by its finalizer
            end
        end

        Random.seed!(rng)
        rng
    end::GPUArrays.RNG
end

@deprecate seed!() CUDA.seed!()
@deprecate seed!(seed) CUDA.seed!(seed)

end
