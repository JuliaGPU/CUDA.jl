module CUSOLVER

using ..CuArrays
using ..CuArrays: libcusolver, @allowscalar, unsafe_free!, @argout, @workspace

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

const created_dense_handles = IdDict{CuContext,cusolverDnHandle_t}()
const created_sparse_handles = IdDict{CuContext,cusolverSpHandle_t}()
const active_dense_handles = Vector{Union{Nothing,cusolverDnHandle_t}}()
const active_sparse_handles = Vector{Union{Nothing,cusolverSpHandle_t}}()

function dense_handle()
    tid = Threads.threadid()
    if @inbounds active_dense_handles[tid] === nothing
        ctx = context()
        active_dense_handles[tid] = get!(created_dense_handles, ctx) do
            handle = cusolverDnCreate()
            atexit(()->CUDAdrv.isvalid(ctx) && cusolverDnDestroy(handle))
            handle
        end
    end
    @inbounds active_dense_handles[tid]
end

function sparse_handle()
    tid = Threads.threadid()
    if @inbounds active_sparse_handles[tid] === nothing
        ctx = context()
        active_sparse_handles[tid] = get!(created_sparse_handles, ctx) do
            handle = cusolverSpCreate()
            atexit(()->CUDAdrv.isvalid(ctx) && cusolverSpDestroy(handle))
            handle
        end
    end
    @inbounds active_sparse_handles[tid]
end

function __init__()
    resize!(active_dense_handles, Threads.nthreads())
    fill!(active_dense_handles, nothing)

    resize!(active_sparse_handles, Threads.nthreads())
    fill!(active_sparse_handles, nothing)

    CUDAnative.atcontextswitch() do tid, ctx
        # we don't eagerly initialize handles, but do so lazily when requested
        active_dense_handles[tid] = nothing
        active_sparse_handles[tid] = nothing
    end
end

end
