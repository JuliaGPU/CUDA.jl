# GPUArrays RNG integration
#
# GPUArrays.default_rng is implemented here using
# GPUArrays' built-in RNG (no cuRAND dependency).

using Random

# GPUArrays.default_rng implementation
function _gpuarrays_rng_ctor(ctx)
    context!(ctx) do
        N = attribute(device(), DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
        buf = CuArray{NTuple{4, UInt32}}(undef, N)
        GPUArrays.RNG(buf)
    end
end
function _gpuarrays_rng_dtor(ctx, rng)
    context!(ctx; skip_destroyed=true) do
    end
end
const _idle_gpuarray_rngs = HandleCache{CuContext,GPUArrays.RNG}(_gpuarrays_rng_ctor, _gpuarrays_rng_dtor)

function GPUArrays.default_rng(::Type{<:CuArray})
    cuda = active_state()

    LibraryState = @NamedTuple{rng::GPUArrays.RNG}
    states = get!(task_local_storage(), :GPUArraysRNG) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    @noinline function new_state(cuda)
        new_rng = pop!(_idle_gpuarray_rngs, cuda.context)
        finalizer(current_task()) do task
            push!(_idle_gpuarray_rngs, cuda.context, new_rng)
        end
        Random.seed!(new_rng)
        (; rng=new_rng)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    return state.rng
end
