module CUTENSOR

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, cudaDataType
using ..CUDA: libcutensor,  @retry_reclaim

using CEnum

using Memoize

using DataStructures


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
    ctx = context()
    get!(task_local_storage(), (:CUTENSOR, ctx)) do
        handle = pop!(idle_handles, ctx) do
            handle = Ref{cutensorHandle_t}()
            cutensorInit(handle)
            handle
        end

        finalizer(current_task()) do task
            push!(idle_handles, ctx, handle) do
                # CUTENSOR doesn't need to actively destroy its handle
            end
        end

        handle
    end::Base.RefValue{cutensorHandle_t}
end

end
