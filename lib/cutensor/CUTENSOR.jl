module CUTENSOR

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, cudaDataType
using ..CUDA: libcutensor,  @retry_reclaim

using CEnum

using Memoize


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

# thread cache for task-local library handles
const thread_handles = Vector{Union{Nothing,Base.RefValue{cutensorHandle_t}}}()

function handle()
    tid = Threads.threadid()
    if @inbounds thread_handles[tid] === nothing
        ctx = context()
        thread_handles[tid] = get!(task_local_storage(), (:CUTENSOR, ctx)) do
            handle = Ref{cutensorHandle_t}()
            cutensorInit(handle)
            handle
        end
    end
    something(@inbounds thread_handles[tid])
end

function reset_stream()
    # CUTENSOR uses stream arguments per operation
end

function __init__()
    resize!(thread_handles, Threads.nthreads())
    fill!(thread_handles, nothing)

    CUDA.atdeviceswitch() do
        tid = Threads.threadid()
        thread_handles[tid] = nothing
    end

    CUDA.attaskswitch() do
        tid = Threads.threadid()
        thread_handles[tid] = nothing
    end
end

end
