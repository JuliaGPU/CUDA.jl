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
const handle_cache_lock = ReentrantLock()
const idle_handles = DefaultDict{CuContext,Vector{Base.RefValue{cutensorHandle_t}}}(()->Base.RefValue{cutensorHandle_t}[])

function handle()
    ctx = context()
    active_stream = stream()
    get!(task_local_storage(), (:CUTENSOR, ctx)) do
        handle = @lock handle_cache_lock begin
            if isempty(idle_handles[ctx])
                handle = Ref{cutensorHandle_t}()
                cutensorInit(handle)
                handle
            else
                pop!(idle_handles[ctx])
            end
        end

        finalizer(current_task()) do task
            @spinlock handle_cache_lock begin
                push!(idle_handles[ctx], handle)
            end
        end
        # TODO: destroy to preserve memory, or at exit?

        handle
    end::Base.RefValue{cutensorHandle_t}
end

end
