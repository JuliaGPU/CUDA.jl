module CUSOLVER

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType
using ..CUDA: libcusolver, libcusolvermg, @allowscalar, assertscalar, unsafe_free!, @retry_reclaim

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
include("base.jl")
include("sparse.jl")
include("dense.jl")
include("multigpu.jl")

# high-level integrations
include("linalg.jl")

# cache for created, but unused handles
const handle_cache_lock = ReentrantLock()
const idle_dense_handles = DefaultDict{CuContext,Vector{cusolverDnHandle_t}}(()->cusolverDnHandle_t[])
const idle_sparse_handles = DefaultDict{CuContext,Vector{cusolverSpHandle_t}}(()->cusolverSpHandle_t[])

function dense_handle()::cusolverDnHandle_t
    ctx = context()
    get!(task_local_storage(), (:CUSOLVER, :dense, ctx)) do
        handle = lock(handle_cache_lock) do
            if isempty(idle_dense_handles[ctx])
                cusolverDnCreate()
            else
                pop!(idle_dense_handles[ctx])
            end
        end

        finalizer(current_task()) do task
            lock(handle_cache_lock) do
                push!(idle_dense_handles[ctx], handle)
            end
        end
        # TODO: cusolverDnDestroy to preserve memory, or at exit?

        cusolverDnSetStream(handle, stream())

        handle
    end
end

function sparse_handle()::cusolverSpHandle_t
    ctx = context()
    get!(task_local_storage(), (:CUSOLVER, :sparse, ctx)) do
        handle = if isempty(idle_sparse_handles[ctx])
            cusolverSpCreate()
        else
            pop!(idle_sparse_handles[ctx])
        end

        finalizer(current_task()) do task
            push!(idle_sparse_handles[ctx], handle)
        end
        # TODO: cusolverSpDestroy to preserve memory, or at exit?

        cusolverSpSetStream(handle, stream())

        handle
    end
end

function mg_handle()::cusolverMgHandle_t
    ctx = context()
    get!(task_local_storage(), (:CUSOLVER, :mg, ctx)) do
        # we can't reuse cusolverMg handles because they can only be assigned devices once
        handle = cusolverMgCreate()

        finalizer(current_task()) do task
            tls = task.storage
            if haskey(tls, (:CUSOLVER, :mg, ctx)) && CUDA.isvalid(ctx)
                # look-up the handle again, because it might have been destroyed already
                # (e.g. in `devices!`)
                handle = tls[(:CUSOLVER, :mg, ctx)]
                cusolverMgDestroy(handle)
            end
        end

        # select devices
        cusolverMgDeviceSelect(handle, ndevices(), devices())

        handle
    end
end

# module-local version of CUDA's devices/ndevices to support flexible device selection
# TODO: make this task-local
const __devices = Cint[0]
devices() = __devices
ndevices() = length(__devices)

# NOTE: this invalidates any existing cusolverMg handle
function devices!(devs::Vector{CuDevice})
    resize!(__devices, length(devs))
    __devices .= deviceid.(devs)

    # we can't select different devices _after_ having initialized the handle,
    # so just destroy it and wait for initialization to kick in again
    ctx = context()
    if haskey(task_local_storage(), (:CUSOLVER, :mg, ctx))
        handle = task_local_storage((:CUSOLVER, :mg, ctx))
        cusolverMgDestroy(handle)
        delete!(task_local_storage(), (:CUSOLVER, :mg, ctx))
    end

    return
end

@inline function set_stream(stream::CuStream)
    ctx = context()
    tls = task_local_storage()
    dense_handle = get(tls, (:CUSOLVER, :dense, ctx), nothing)
    if dense_handle !== nothing
        cusolverDnSetStream(dense_handle, stream)
    end
    sparse_handle = get(tls, (:CUSOLVER, :sparse, ctx), nothing)
    if sparse_handle !== nothing
        cusolverSpSetStream(sparse_handle, stream)
    end
    return
end

end
