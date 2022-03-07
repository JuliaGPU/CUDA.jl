module CUSTATEVEC

using ..CUDA
using ..CUDA: CUstream, cudaDataType, @checked, HandleCache, with_workspace, libraryPropertyType
using ..CUDA: libcustatevec, unsafe_free!, @retry_reclaim, initialize_context

using CEnum: @cenum

const cudaDataType_t = cudaDataType

# core library
include("libcustatevec_common.jl")
include("error.jl")
include("libcustatevec.jl")
include("types.jl")
include("statevec.jl")

# cache for created, but unused handles
const idle_handles  = HandleCache{CuContext,custatevecHandle_t}()

function handle()
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{handle::custatevecHandle_t, stream::CuStream}
    states = get!(task_local_storage(), :CUQUANTUM) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context) do
            handle = Ref{custatevecHandle_t}()
            custatevecCreate(handle)
            handle[]
        end

        finalizer(current_task()) do task
            push!(idle_handles, cuda.context, new_handle) do
                context!(cuda.context; skip_destroyed=true) do
                    custatevecDestroy(new_handle)
                end
            end
        end

        custatevecSetStream(new_handle, cuda.stream)

        (; handle=new_handle, cuda.stream)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    # update stream
    @noinline function update_stream(cuda, state)
        custatevecSetStream(state.handle, cuda.stream)
        (; state.handle, cuda.stream)
    end
    if state.stream != cuda.stream
        states[cuda.context] = state = update_stream(cuda, state)
    end

    return state.handle
end

end
