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

# thread cache for task-local library handles
const thread_handles = Vector{Union{Nothing,Base.RefValue{cutensorHandle_t}}}()

# cache for created, but unused handles
const handle_cache_lock = ReentrantLock()
const idle_handles = DefaultDict{CuContext,Vector{Base.RefValue{cutensorHandle_t}}}(()->Base.RefValue{cutensorHandle_t}[])

function handle()
    CUDA.detect_state_changes()
    tid = Threads.threadid()
    if @inbounds thread_handles[tid] === nothing
        ctx = context()
        thread_handles[tid] = get!(task_local_storage(), (:CUTENSOR, ctx)) do
            handle = lock(handle_cache_lock) do
                if isempty(idle_handles[ctx])
                    handle = Ref{cutensorHandle_t}()
                    cutensorInit(handle)
                    handle
                else
                    pop!(idle_handles[ctx])
                end
            end

            finalizer(current_task()) do task
                lock(handle_cache_lock) do
                    push!(idle_handles[ctx], handle)
                end
            end
            # TODO: destroy to preserve memory, or at exit?

            handle
        end
    end
    something(@inbounds thread_handles[tid])
end

@inline function set_stream(stream::CuStream)
    # CUTENSOR uses stream arguments per operation
    return
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
