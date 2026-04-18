module cuSOLVER

using CUDACore
using GPUToolbox

using CUDACore: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType, cudaEmulationStrategy_t, cudaEmulationMantissaControl_t, cudaEmulationSpecialValuesSupport_t
using CUDACore: @allowscalar, assertscalar, unsafe_free!, retry_reclaim, initialize_context

using cuBLAS
using cuBLAS: cublasFillMode_t, cublasOperation_t, cublasSideMode_t, cublasDiagType_t
using cuSPARSE
using cuSPARSE: cusparseMatDescr_t

using CEnum: @cenum

using LinearAlgebra
using LinearAlgebra: BlasFloat, Factorization

if CUDACore.local_toolkit
    using CUDA_Runtime_Discovery
else
    import CUDA_Runtime_jll
end


@public functional, has_cusolvermg

const _initialized = Ref{Bool}(false)
functional() = _initialized[]

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
include("helpers.jl")
include("error.jl")
include("base.jl")
include("sparse.jl")
include("sparse_factorizations.jl")
include("dense.jl")
include("dense_generic.jl")
include("multigpu.jl")

# high-level integrations
include("linalg.jl")


## dense handles

function dense_handle_ctor(ctx)
    context!(ctx) do
        cusolverDnCreate()
    end
end
function dense_handle_dtor(ctx, handle)
    context!(ctx; skip_destroyed=true) do
        cusolverDnDestroy(handle)
    end
end
const idle_dense_handles =
    HandleCache{CuContext,cusolverDnHandle_t}(dense_handle_ctor, dense_handle_dtor)

# fat handle, holds the raw cuSOLVER handle together with reusable workspace
# and info buffers. Mutable so the finalizer can be attached to the object
# itself: once no one references this struct (e.g. after the owning task dies
# or we clear it from task-local storage on reclaim), GC runs the finalizer
# which returns the buffers to the pool and the raw handle to the idle cache.
mutable struct cusolverDnHandle
    const handle::cusolverDnHandle_t
    const ctx::CuContext
    const workspace_gpu::CuVector{UInt8}
    const workspace_cpu::Vector{UInt8}
    const info::CuVector{Cint}
end
Base.unsafe_convert(::Type{Ptr{cusolverDnContext}}, handle::cusolverDnHandle) =
    handle.handle

function dense_handle_finalizer(dh::cusolverDnHandle)
    CUDACore.unsafe_free!(dh.workspace_gpu)
    CUDACore.unsafe_free!(dh.info)
    push!(idle_dense_handles, dh.ctx, dh.handle)
end

function dense_handle()
    cuda = CUDACore.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{handle::cusolverDnHandle, stream::CuStream}
    states = get!(task_local_storage(), :CUSOLVER_dense) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_dense_handles, cuda.context)

        workspace_gpu = CuVector{UInt8}(undef, 0)
        workspace_cpu = Vector{UInt8}(undef, 0)
        info = CuVector{Cint}(undef, 1)
        fat_handle = cusolverDnHandle(new_handle, cuda.context, workspace_gpu,
                                      workspace_cpu, info)
        finalizer(dense_handle_finalizer, fat_handle)

        cusolverDnSetStream(new_handle, cuda.stream)

        (; handle=fat_handle, cuda.stream)
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


## sparse handles

function sparse_handle_ctor(ctx)
    context!(ctx) do
        cusolverSpCreate()
    end
end
function sparse_handle_dtor(ctx, handle)
    context!(ctx; skip_destroyed=true) do
        cusolverSpDestroy(handle)
    end
end
const idle_sparse_handles =
    HandleCache{CuContext,cusolverSpHandle_t}(sparse_handle_ctor, sparse_handle_dtor)

# mutable wrapper so the raw sparse handle is released via an object-bound
# finalizer (see `cusolverDnHandle` for rationale).
mutable struct cusolverSpHandle
    const handle::cusolverSpHandle_t
    const ctx::CuContext
end
Base.unsafe_convert(::Type{cusolverSpHandle_t}, handle::cusolverSpHandle) =
    handle.handle

function sparse_handle_finalizer(sh::cusolverSpHandle)
    push!(idle_sparse_handles, sh.ctx, sh.handle)
end

function sparse_handle()
    cuda = CUDACore.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{handle::cusolverSpHandle, stream::CuStream}
    states = get!(task_local_storage(), :CUSOLVER_sparse) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get or create handle
    @noinline function new_state(cuda)
        new_handle = pop!(idle_sparse_handles, cuda.context)
        wrapped = cusolverSpHandle(new_handle, cuda.context)
        finalizer(sparse_handle_finalizer, wrapped)

        cusolverSpSetStream(new_handle, cuda.stream)

        (; handle=wrapped, cuda.stream)
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


## mg handles

function devices!(devs::Vector{CuDevice})
    task_local_storage(:CUSOLVERmg_devices, sort(devs; by=deviceid))
    return
end

devices() = get!(task_local_storage(), :CUSOLVERmg_devices) do
    # by default, select only the first device
    [first(CUDACore.devices())]
    # TODO: select all devices
    #sort(collect(CUDACore.devices()); by=deviceid)
end::Vector{CuDevice}

ndevices() = length(devices())

# mutable wrapper so the mg handle is destroyed when its owning state struct
# becomes unreachable, rather than being pinned for the lifetime of the task.
mutable struct cusolverMgHandle
    const handle::cusolverMgHandle_t
    const ctx::CuContext
end
Base.unsafe_convert(::Type{cusolverMgHandle_t}, handle::cusolverMgHandle) =
    handle.handle

function mg_handle_finalizer(mh::cusolverMgHandle)
    context!(mh.ctx; skip_destroyed=true) do
        cusolverMgDestroy(mh.handle)
    end
end

function mg_handle()
    cuda = CUDACore.active_state()

    # every task maintains library state per set of devices
    LibraryState = @NamedTuple{handle::cusolverMgHandle}
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
        wrapped = cusolverMgHandle(new_handle, cuda.context)
        finalizer(mg_handle_finalizer, wrapped)

        devs = convert.(Cint, devices())
        cusolverMgDeviceSelect(new_handle, length(devs), devs)

        (; handle=wrapped)
    end
    state = get!(states, key) do
        new_state(cuda)
    end

    return state.handle
end


function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0

    CUDACore.functional() || return

    # find the library
    global libcusolver, libcusolverMg
    if CUDACore.local_toolkit
        dirs = CUDA_Runtime_Discovery.find_toolkit()
        path = CUDA_Runtime_Discovery.get_library(dirs, "cusolver"; optional=true)
        if path === nothing
            precompiling || @error "cuSOLVER is not available on your system (looked in $(join(dirs, ", ")))"
            return
        end
        libcusolver = path
        path_mg = CUDA_Runtime_Discovery.get_library(dirs, "cusolverMg"; optional=true)
        if path_mg !== nothing
            libcusolverMg = path_mg
        end
    else
        libcusolver = CUDA_Runtime_jll.libcusolver
        if hasproperty(CUDA_Runtime_jll, :libcusolverMg)
            libcusolverMg = CUDA_Runtime_jll.libcusolverMg
        end
    end

    # Drop task-local cuSOLVER state on reclaim so the cached workspace
    # CuVectors (sometimes hundreds of MiB) become GC-able.
    push!(CUDACore.pre_reclaim_hooks, drop_library_state!)

    _initialized[] = true
end

function drop_library_state!()
    delete!(task_local_storage(), :CUSOLVER_dense)
    delete!(task_local_storage(), :CUSOLVER_sparse)
    delete!(task_local_storage(), :CUSOLVERmg)
    return
end

include("precompile.jl")

# deprecated binding for backwards compatibility
Base.@deprecate_binding CUSOLVER cuSOLVER false

end
