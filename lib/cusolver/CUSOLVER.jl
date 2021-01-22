module CUSOLVER

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType
using ..CUDA: libcusolver, @allowscalar, assertscalar, unsafe_free!, @retry_reclaim

using ..CUBLAS: cublasFillMode_t, cublasOperation_t, cublasSideMode_t, cublasDiagType_t
using ..CUSPARSE: cusparseMatDescr_t

using CEnum

using Memoize

using DataStructures


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

# cache for created, but unused handles
const old_dense_handles = DefaultDict{CuContext,Vector{cusolverDnHandle_t}}(()->cusolverDnHandle_t[])
const old_sparse_handles = DefaultDict{CuContext,Vector{cusolverSpHandle_t}}(()->cusolverSpHandle_t[])

function dense_handle()
    tid = Threads.threadid()
    if @inbounds thread_dense_handles[tid] === nothing
        ctx = context()
        thread_dense_handles[tid] = get!(task_local_storage(), (:CUSOLVER, :dense, ctx)) do
            handle = if isempty(old_dense_handles[ctx])
                cusolverDnCreate()
            else
                pop!(old_dense_handles[ctx])
            end

            finalizer(current_task()) do task
                push!(old_dense_handles[ctx], handle)
            end
            # TODO: cusolverDnDestroy to preserve memory, or at exit?

            handle
        end
        cusolverDnSetStream(thread_dense_handles[tid], stream())
    end
    something(@inbounds thread_dense_handles[tid])
end

function sparse_handle()
    tid = Threads.threadid()
    if @inbounds thread_sparse_handles[tid] === nothing
        ctx = context()
        thread_sparse_handles[tid] = get!(task_local_storage(), (:CUSOLVER, :sparse, ctx)) do
            handle = if isempty(old_sparse_handles[ctx])
                cusolverSpCreate()
            else
                pop!(old_sparse_handles[ctx])
            end

            finalizer(current_task()) do task
                push!(old_sparse_handles[ctx], handle)
            end
            # TODO: cusolverSpDestroy to preserve memory, or at exit?

            handle
        end
        cusolverSpSetStream(thread_sparse_handles[tid], stream())
    end
    something(@inbounds thread_sparse_handles[tid])
end

function reset_stream()
    # NOTE: we 'abuse' the thread cache here, as switching streams doesn't invalidate it,
    #       but we (re-)apply the current stream when populating that cache.
    tid = Threads.threadid()
    thread_dense_handles[tid] = nothing
    thread_sparse_handles[tid] = nothing
end

function __init__()
    resize!(thread_dense_handles, Threads.nthreads())
    fill!(thread_dense_handles, nothing)

    resize!(thread_sparse_handles, Threads.nthreads())
    fill!(thread_sparse_handles, nothing)

    CUDA.atdeviceswitch() do
        tid = Threads.threadid()
        thread_dense_handles[tid] = nothing
        thread_sparse_handles[tid] = nothing
    end

    CUDA.attaskswitch() do
        tid = Threads.threadid()
        thread_dense_handles[tid] = nothing
        thread_sparse_handles[tid] = nothing
    end
end

end
