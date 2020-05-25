module CUSPARSE

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType
using ..CUDA: libcusparse, unsafe_free!, @retry_reclaim

using CEnum

const SparseChar = Char

# core library
include("libcusparse_common.jl")
include("error.jl")
include("libcusparse.jl")

# low-level wrappers
include("array.jl")
include("util.jl")
include("wrappers.jl")

# high-level integrations
include("interfaces.jl")

# thread cache for task-local library handles
const thread_handles = Vector{Union{Nothing,cusparseHandle_t}}()

function handle()
    tid = Threads.threadid()
    if @inbounds thread_handles[tid] === nothing
        ctx = context()
        thread_handles[tid] = get!(task_local_storage(), (:CUSPARSE, ctx)) do
            handle = cusparseCreate()
            finalizer(current_task()) do task
                CUDA.isvalid(ctx) || return
                context!(ctx) do
                    cusparseDestroy(handle)
                end
            end

            handle
        end
    end
    @inbounds thread_handles[tid]
end

function __init__()
    resize!(thread_handles, Threads.nthreads())
    fill!(thread_handles, nothing)

    CUDA.atcontextswitch() do tid, ctx
        thread_handles[tid] = nothing
    end

    CUDA.attaskswitch() do tid, task
        thread_handles[tid] = nothing
    end
end

end
