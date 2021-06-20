module CURAND

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, libraryPropertyType, DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
using ..CUDA: libcurand, @retry_reclaim

using CEnum

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
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{rng::RNG}
    states = get!(task_local_storage(), :CURAND) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        new_rng = pop!(idle_curand_rngs, cuda.context) do
            RNG()
        end

        finalizer(current_task()) do task
            push!(idle_curand_rngs, cuda.context, new_rng) do
                # no need to do anything, as the RNG is collected by its finalizer
            end
        end

        Random.seed!(new_rng)
        (; rng=new_rng)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    return state.rng
end

function GPUArrays.default_rng(::Type{<:CuArray})
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{rng::GPUArrays.RNG}
    states = get!(task_local_storage(), :GPUArraysRNG) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        new_rng = pop!(idle_gpuarray_rngs, cuda.context) do
            N = attribute(cuda.device, DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            buf = CuArray{NTuple{4, UInt32}}(undef, N)
            GPUArrays.RNG(buf)
        end

        finalizer(current_task()) do task
            push!(idle_gpuarray_rngs, cuda.context, new_rng) do
                # no need to do anything, as the RNG is collected by its finalizer
            end
        end

        Random.seed!(new_rng)
        (; rng=new_rng)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    return state.rng
end

@deprecate seed!() CUDA.seed!()
@deprecate seed!(seed) CUDA.seed!(seed)

end
