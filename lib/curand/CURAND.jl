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
    active_stream = stream()
    rng, chosen_stream = get!(task_local_storage(), (:CURAND, ctx)) do
        new_rng = @lock rng_cache_lock begin
            new_rng = if isempty(idle_curand_rngs[ctx])
                RNG()
            else
                pop!(idle_curand_rngs[ctx])
            end

            # protect handles from the GC when the owning task is collected. we only
            # need to do this for CURAND, as handles typically don't have finalizers.
            push!(active_curand_rngs, new_rng)

            new_rng
        end

        finalizer(current_task()) do task
            @spinlock rng_cache_lock begin
                push!(idle_curand_rngs[ctx], new_rng)
                delete!(active_curand_rngs, new_rng)
            end
        end
        # TODO: curandDestroyGenerator to preserve memory, or at exit?

        curandSetStream(new_rng, active_stream)

        Random.seed!(new_rng)
        new_rng, active_stream
    end::Tuple{RNG,CuStream}

    if chosen_stream != active_stream
        curandSetStream(rng, active_stream)
        task_local_storage((:CURAND, ctx), (rng, active_stream))
    end

    return rng
end

function GPUArrays.default_rng(::Type{<:CuArray})
    ctx = context()
    get!(task_local_storage(), (:GPUArraysRNG, ctx)) do
        rng = @lock rng_cache_lock begin
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
            @spinlock rng_cache_lock begin
                push!(idle_gpuarray_rngs[ctx], rng)
                delete!(active_gpuarray_rngs, rng)
            end
        end
        # TODO: destroy to preserve memory, or at exit?

        Random.seed!(rng)
        rng
    end::GPUArrays.RNG
end

@deprecate seed!() CUDA.seed!()
@deprecate seed!(seed) CUDA.seed!(seed)

end
