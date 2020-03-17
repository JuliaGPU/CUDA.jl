module CUSPARSE

using ..CuArrays
using ..CuArrays: libcusparse, unsafe_free!, @argout, @workspace, @retry_reclaim

using CUDAapi

using CUDAdrv
using CUDAdrv: CUstream

using CUDAnative

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
            atexit(()->CUDAdrv.isvalid(ctx) && cusparseDestroy(handle))
            handle
        end
    end
    @inbounds thread_handles[tid]
end

function __init__()
    resize!(thread_handles, Threads.nthreads())
    fill!(thread_handles, nothing)

    CUDAnative.atcontextswitch() do tid, ctx
        thread_handles[tid] = nothing
    end

    CUDAnative.attaskswitch() do tid, task
        thread_handles[tid] = nothing
    end
end

end
