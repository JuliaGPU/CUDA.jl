module CURAND

using ..APIUtils

using ..CUDA_Runtime

using ..CUDA
using ..CUDA: CUstream, libraryPropertyType, DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
using ..CUDA: retry_reclaim, initialize_context

using CEnum: @cenum


# core library
include("libcurand.jl")

# low-level wrappers
include("error.jl")
include("wrappers.jl")

# high-level integrations
include("random.jl")


## handles

function handle_ctor(ctx)
    context!(ctx) do
        RNG()
    end
end
function handle_dtor(ctx, handle)
    context!(ctx; skip_destroyed=true) do
        # no need to do anything, as the RNG is collected by its finalizer
        # TODO: early free?
    end
end
const idle_curand_rngs = HandleCache{CuContext,RNG}(handle_ctor, handle_dtor)

function default_rng()
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{rng::RNG}
    states = get!(task_local_storage(), :CURAND) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        new_rng = pop!(idle_curand_rngs, cuda.context)
        finalizer(current_task()) do task
            push!(idle_curand_rngs, cuda.context, new_rng)
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
