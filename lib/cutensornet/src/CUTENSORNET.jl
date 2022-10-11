module CUTENSORNET

using CUDA
using CUDA: CUstream, cudaDataType, @checked, HandleCache, with_workspace
using CUDA: libcutensornet, @retry_reclaim, initialize_context

using CEnum: @cenum

const cudaDataType_t = cudaDataType

# core library
include("libcutensornet_common.jl")
include("error.jl")
include("libcutensornet.jl")
include("libcutensornet_deprecated.jl")

include("types.jl")
include("tensornet.jl")

# cache for created, but unused handles
const idle_handles = HandleCache{CuContext,cutensornetHandle_t}()

function handle()
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{handle::cutensornetHandle_t}
    states = get!(task_local_storage(), :CUTENSORNET) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context) do
            handle = Ref{cutensornetHandle_t}()
            cutensornetCreate(handle)
            handle[]
        end

        finalizer(current_task()) do task
            push!(idle_handles, cuda.context, new_handle) do
                context!(cuda.context; skip_destroyed=true) do
                    cutensornetDestroy(new_handle)
                end
            end
        end

        (; handle=new_handle)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    return state.handle
end

end
