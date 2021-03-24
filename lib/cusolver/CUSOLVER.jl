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

function dense_handle()
    ctx = context()
    active_stream = stream()
    handle, chosen_stream = get!(task_local_storage(), (:CUSOLVER, :dense, ctx)) do
        new_handle = @lock handle_cache_lock begin
            if isempty(idle_dense_handles[ctx])
                cusolverDnCreate()
            else
                pop!(idle_dense_handles[ctx])
            end
        end

        finalizer(current_task()) do task
            @spinlock handle_cache_lock begin
                push!(idle_dense_handles[ctx], new_handle)
            end
        end
        # TODO: cusolverDnDestroy to preserve memory, or at exit?

        cusolverDnSetStream(new_handle, active_stream)

        new_handle, active_stream
    end::Tuple{cusolverDnHandle_t,CuStream}

    if chosen_stream != active_stream
        cusolverDnSetStream(handle, active_stream)
        task_local_storage((:CUSOLVER, :dense, ctx), (handle, active_stream))
    end

    return handle
end

function sparse_handle()
    ctx = context()
    active_stream = stream()
    handle, chosen_stream = get!(task_local_storage(), (:CUSOLVER, :sparse, ctx)) do
        new_handle = @lock handle_cache_lock begin
            if isempty(idle_sparse_handles[ctx])
                cusolverSpCreate()
            else
                pop!(idle_sparse_handles[ctx])
            end
        end

        finalizer(current_task()) do task
            @spinlock handle_cache_lock begin
                push!(idle_sparse_handles[ctx], new_handle)
            end
        end
        # TODO: cusolverSpDestroy to preserve memory, or at exit?

        cusolverSpSetStream(new_handle, active_stream)

        new_handle, active_stream
    end::Tuple{cusolverSpHandle_t,CuStream}

    if chosen_stream != active_stream
        cusolverSpSetStream(handle, active_stream)
        task_local_storage((:CUSOLVER, :sparse, ctx), (handle, active_stream))
    end

    return handle
end

function mg_handle()
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
    end::cusolverMgHandle_t
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

end
