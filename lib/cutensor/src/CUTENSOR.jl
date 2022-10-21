module CUTENSOR

using CUDA
using CUDA.APIUtils
using CUDA: CUstream, cudaDataType
using CUDA: @retry_reclaim, initialize_context

using CEnum: @cenum

using CUTENSOR_jll


export has_cutensor

function has_cutensor(show_reason::Bool=false)
    if !isdefined(CUTENSOR_jll, :libcutensor)
        show_reason && error("CUTENSOR library not found")
        return false
    end
    return true
end


const cudaDataType_t = cudaDataType

# core library
include("libcutensor_common.jl")
include("error.jl")
include("libcutensor.jl")

# low-level wrappers
include("tensor.jl")
include("wrappers.jl")

# high-level integrations
include("interfaces.jl")

# cache for created, but unused handles
const idle_handles = HandleCache{CuContext,Base.RefValue{cutensorHandle_t}}()

function handle()
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{handle::Base.RefValue{cutensorHandle_t}}
    states = get!(task_local_storage(), :CUTENSOR) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context) do
            handle = Ref{cutensorHandle_t}()
            cutensorInit(handle)
            handle
        end

        finalizer(current_task()) do task
            push!(idle_handles, cuda.context, new_handle) do
                # CUTENSOR doesn't need to actively destroy its handle
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
