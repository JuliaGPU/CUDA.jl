module CUSOLVER

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType
using ..CUDA: libcusolver, libcusolvermg, @allowscalar, assertscalar, unsafe_free!, @retry_reclaim, @context!

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
const idle_dense_handles = HandleCache{CuContext,cusolverDnHandle_t}()
const idle_sparse_handles = HandleCache{CuContext,cusolverSpHandle_t}()

function dense_handle()
    state = CUDA.active_state()
    handle, stream = get!(task_local_storage(), (:CUSOLVER, :dense, state.context)) do
        new_handle = pop!(idle_dense_handles, state.context) do
            cusolverDnCreate()
        end

        finalizer(current_task()) do task
            push!(idle_dense_handles, state.context, new_handle) do
                @context! skip_destroyed=true state.context cusolverDnDestroy()
            end
        end

        cusolverDnSetStream(new_handle, state.stream)

        new_handle, state.stream
    end::Tuple{cusolverDnHandle_t,CuStream}

    if stream != state.stream
        cusolverDnSetStream(handle, state.stream)
        task_local_storage((:CUSOLVER, :dense, state.context), (handle, state.stream))
    end

    return handle
end

function sparse_handle()
    state = CUDA.active_state()
    handle, stream = get!(task_local_storage(), (:CUSOLVER, :sparse, state.context)) do
        new_handle = pop!(idle_sparse_handles, state.context) do
            cusolverSpCreate()
        end

        finalizer(current_task()) do task
            push!(idle_sparse_handles, state.context, new_handle) do
                @context! skip_destroyed=true state.context cusolverSpDestroy(new_handle)
            end
        end

        cusolverSpSetStream(new_handle, state.stream)

        new_handle, state.stream
    end::Tuple{cusolverSpHandle_t,CuStream}

    if stream != state.stream
        cusolverSpSetStream(handle, state.stream)
        task_local_storage((:CUSOLVER, :sparse, state.context), (handle, state.stream))
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
