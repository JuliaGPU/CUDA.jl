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
const rng_cache_lock = ReentrantLock()
const active_curand_rngs = Set{RNG}()
const active_gpuarray_rngs = Set{GPUArrays.RNG}()
const idle_curand_rngs = DefaultDict{CuContext,Vector{RNG}}(()->RNG[])
const idle_gpuarray_rngs = DefaultDict{CuContext,Vector{GPUArrays.RNG}}(()->GPUArrays.RNG[])

function default_rng()
    ctx = context()
    get!(task_local_storage(), (:CURAND, ctx)) do
        rng = lock(rng_cache_lock) do
            rng = if isempty(idle_curand_rngs[ctx])
                RNG()
            else
                pop!(idle_curand_rngs[ctx])
            end

            # protect handles from the GC when the owning task is collected. we only
            # need to do this for CURAND, as handles typically don't have finalizers.
            push!(active_curand_rngs, rng)

            rng
        end

        finalizer(current_task()) do task
            lock(rng_cache_lock) do
                push!(idle_curand_rngs[ctx], rng)
                delete!(active_curand_rngs, rng)
            end
        end
        # TODO: curandDestroyGenerator to preserve memory, or at exit?

        curandSetStream(rng, stream())

        Random.seed!(rng)
        rng
    end::RNG
end

function GPUArrays.default_rng(::Type{<:CuArray})
    ctx = context()
    get!(task_local_storage(), (:GPUArraysRNG, ctx)) do
        rng = lock(rng_cache_lock) do
            rng = if isempty(idle_gpuarray_rngs[ctx])
                dev = device()
                N = attribute(dev, DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
                state = CuArray{NTuple{4, UInt32}}(undef, N)
                GPUArrays.RNG(state)
            else
                pop!(idle_gpuarray_rngs[ctx])
            end

            push!(active_gpuarray_rngs, rng)

            rng
        end

        finalizer(current_task()) do task
            lock(rng_cache_lock) do
                push!(idle_gpuarray_rngs[ctx], rng)
                delete!(active_gpuarray_rngs, rng)
            end
        end
        # TODO: destroy to preserve memory, or at exit?

        Random.seed!(rng)
        rng
    end::GPUArrays.RNG
end

@inline function set_stream(stream::CuStream)
    ctx = context()
    tls = task_local_storage()
    rng = get(tls, (:CURAND, ctx), nothing)
    if rng !== nothing
        curandSetStream(rng, stream)
    end
    return
end

@deprecate seed!() CUDA.seed!()
@deprecate seed!(seed) CUDA.seed!(seed)

end
