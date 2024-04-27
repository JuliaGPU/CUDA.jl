module CUSPARSE

using ..APIUtils

using ..CUDA_Runtime

using ..CUDA
using ..CUDA: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType
using ..CUDA: unsafe_free!, retry_reclaim, initialize_context, i32, @allowscalar

using ..CUDA.GPUArrays

using CEnum: @cenum

using LinearAlgebra
using LinearAlgebra: HermOrSym

using Adapt: Adapt, adapt

using SparseArrays

const SparseChar = Char


# core library
include("libcusparse.jl")
include("libcusparse_deprecated.jl")

include("error.jl")
include("array.jl")
include("util.jl")
include("types.jl")
include("linalg.jl")


# low-level wrappers
include("helpers.jl")
include("management.jl")
include("level2.jl")
include("level3.jl")
include("extra.jl")
include("preconditioners.jl")
include("reorderings.jl")
include("conversions.jl")
include("generic.jl")

# high-level integrations
include("interfaces.jl")

# native functionality
include("device.jl")
include("broadcast.jl")
include("reduce.jl")

include("batched.jl")


## handles

function handle_ctor(ctx)
    context!(ctx) do
        cusparseCreate()
    end
end
function handle_dtor(ctx, handle)
    context!(ctx; skip_destroyed=true) do
        cusparseDestroy(handle)
    end
end
const idle_handles = HandleCache{CuContext,cusparseHandle_t}(handle_ctor, handle_dtor)

function handle()
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{handle::cusparseHandle_t, stream::CuStream}
    states = get!(task_local_storage(), :CUSPARSE) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context)
        finalizer(current_task()) do task
            push!(idle_handles, cuda.context, new_handle)
        end

        cusparseSetStream(new_handle, cuda.stream)

        (; handle=new_handle, cuda.stream)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    # update stream
    @noinline function update_stream(cuda, state)
        cusparseSetStream(state.handle, cuda.stream)
        (; state.handle, cuda.stream)
    end
    if state.stream != cuda.stream
        states[cuda.context] = state = update_stream(cuda, state)
    end

    return state.handle
end

end
