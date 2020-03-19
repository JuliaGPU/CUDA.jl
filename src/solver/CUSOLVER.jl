module CUSOLVER

using ..CuArrays
using ..CuArrays: libcusolver, @allowscalar, unsafe_free!, @argout, @workspace, @retry_reclaim

using ..CUBLAS: cublasFillMode_t, cublasOperation_t, cublasSideMode_t, cublasDiagType_t
using ..CUSPARSE: cusparseMatDescr_t

using CUDAapi

using CUDAdrv
using CUDAdrv: CUstream

using CUDAnative

using CEnum

# core library
include("libcusolver_common.jl")
include("error.jl")
include("libcusolver.jl")

# low-level wrappers
include("util.jl")
include("wrappers.jl")

# high-level integrations
include("linalg.jl")

# thread cache for task-local library handles
const thread_dense_handles = Vector{Union{Nothing,cusolverDnHandle_t}}()
const thread_sparse_handles = Vector{Union{Nothing,cusolverSpHandle_t}}()

function dense_handle()
    tid = Threads.threadid()
    if @inbounds thread_dense_handles[tid] === nothing
        ctx = context()
        thread_dense_handles[tid] = get!(task_local_storage(), (:CUSOLVER, :dense, ctx)) do
            handle = cusolverDnCreate()
            atexit() do
                CUDAdrv.isvalid(ctx) || return
                context!(ctx) do
                    cusolverDnDestroy(handle)
                end
            end

            handle
        end
    end
    @inbounds thread_dense_handles[tid]
end

function sparse_handle()
    tid = Threads.threadid()
    if @inbounds thread_sparse_handles[tid] === nothing
        ctx = context()
        thread_sparse_handles[tid] = get!(task_local_storage(), (:CUSOLVER, :sparse, ctx)) do
            handle = cusolverSpCreate()
            atexit() do
                CUDAdrv.isvalid(ctx) || return
                context!(ctx) do
                    cusolverSpDestroy(handle)
                end
            end

            handle
        end
    end
    @inbounds thread_sparse_handles[tid]
end

function __init__()
    resize!(thread_dense_handles, Threads.nthreads())
    fill!(thread_dense_handles, nothing)

    resize!(thread_sparse_handles, Threads.nthreads())
    fill!(thread_sparse_handles, nothing)

    CUDAnative.atcontextswitch() do tid, ctx
        thread_dense_handles[tid] = nothing
        thread_sparse_handles[tid] = nothing
    end

    CUDAnative.attaskswitch() do tid, task
        thread_dense_handles[tid] = nothing
        thread_sparse_handles[tid] = nothing
    end
end

end
