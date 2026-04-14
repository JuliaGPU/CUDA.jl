# GPUArrays RNG integration
#
# GPUArrays.default_rng is implemented here using
# GPUArrays' built-in Philox4x32 RNG (no cuRAND dependency, no GPU allocation).

using Random

function GPUArrays.default_rng(::Type{<:CuArray})
    cuda = active_state()

    LibraryState = @NamedTuple{rng::GPUArrays.RNG}
    states = get!(task_local_storage(), :GPUArraysRNG) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    @noinline function new_state(cuda)
        new_rng = GPUArrays.RNG()
        Random.seed!(new_rng)
        (; rng=new_rng)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    return state.rng
end
