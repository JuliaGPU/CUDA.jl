module CUSOLVER

using ..APIUtils

using ..CUDA_Runtime

using ..CUDA
using ..CUDA: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType
using ..CUDA: @allowscalar, assertscalar, unsafe_free!, retry_reclaim, initialize_context

using ..CUBLAS: cublasFillMode_t, cublasOperation_t, cublasSideMode_t, cublasDiagType_t
using ..CUSPARSE: cusparseMatDescr_t

using CEnum: @cenum


export has_cusolvermg

function has_cusolvermg(show_reason::Bool=false)
    if !@isdefined(libcusolverMg)
        show_reason && error("CUDA toolkit does not contain cuSolverMG")
        return false
    end
    return true
end


# core library
include("libcusolver.jl")
include("libcusolverMg.jl")
include("libcusolverRF.jl")

# low-level wrappers
include("error.jl")
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
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{handle::cusolverDnHandle_t, stream::CuStream}
    states = get!(task_local_storage(), :CUSOLVER_dense) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_dense_handles, cuda.context) do
            cusolverDnCreate()
        end

        finalizer(current_task()) do task
            push!(idle_dense_handles, cuda.context, new_handle) do
                context!(cuda.context; skip_destroyed=true) do
                    cusolverDnDestroy(new_handle)
                end
            end
        end

        cusolverDnSetStream(new_handle, cuda.stream)
        (; handle=new_handle, cuda.stream)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    # update stream
    @noinline function update_stream(cuda, state)
        cusolverDnSetStream(state.handle, cuda.stream)
        (; state.handle, cuda.stream)
    end
    if state.stream != cuda.stream
        states[cuda.context] = state = update_stream(cuda, state)
    end

    return state.handle
end

function sparse_handle()
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{handle::cusolverSpHandle_t, stream::CuStream}
    states = get!(task_local_storage(), :CUSOLVER_sparse) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get or create handle
    @noinline function new_state(cuda)
        new_handle = pop!(idle_sparse_handles, cuda.context) do
            cusolverSpCreate()
        end

        finalizer(current_task()) do task
            push!(idle_sparse_handles, cuda.context, new_handle) do
                context!(cuda.context; skip_destroyed=true) do
                    cusolverSpDestroy(new_handle)
                end
            end
        end

        cusolverSpSetStream(new_handle, cuda.stream)
        (; handle=new_handle, cuda.stream)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    # update stream
    @noinline function update_stream(cuda, state)
        cusolverSpSetStream(state.handle, cuda.stream)
        (; state.handle, cuda.stream)
    end
    if state.stream != cuda.stream
        states[cuda.context] = state = update_stream(cuda, state)
    end

    return state.handle
end

function mg_handle()
    cuda = CUDA.active_state()

    # every task maintains library state per set of devices
    LibraryState = @NamedTuple{handle::cusolverMgHandle_t}
    states = get!(task_local_storage(), :CUSOLVERmg) do
        Dict{UInt,LibraryState}()
    end::Dict{UInt,LibraryState}

    # derive a key from the active and selected devices
    key = hash(cuda.context)
    for dev in devices()
        # we hash the device context to support device resets
        key = hash(context(dev), key)
    end

    # get library state
    @noinline function new_state(cuda)
        # we can't reuse cusolverMg handles because they can only be assigned devices once
        new_handle = cusolverMgCreate()

        finalizer(current_task()) do task
            context!(cuda.context; skip_destroyed=true) do
                cusolverMgDestroy(new_handle)
            end
        end

        devs = convert.(Cint, devices())
        cusolverMgDeviceSelect(new_handle, length(devs), devs)

        (; handle=new_handle)
    end
    state = get!(states, key) do
        new_state(cuda)
    end

    return state.handle
end

function devices!(devs::Vector{CuDevice})
    task_local_storage(:CUSOLVERmg_devices, sort(devs; by=deviceid))
    return
end

devices() = get!(task_local_storage(), :CUSOLVERmg_devices) do
    # by default, select only the first device
    [first(CUDA.devices())]
    # TODO: select all devices
    #sort(collect(CUDA.devices()); by=deviceid)
end::Vector{CuDevice}

ndevices() = length(devices())

end
