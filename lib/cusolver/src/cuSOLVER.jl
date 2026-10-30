module cuSOLVER

using CUDACore
using GPUToolbox

using CUDACore: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType, cudaEmulationStrategy_t, cudaEmulationMantissaControl_t, cudaEmulationSpecialValuesSupport_t
using CUDACore: @allowscalar, assertscalar, unsafe_free!, retry_reclaim, initialize_context, isdebug, resource_finalizer

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


@public functional, has_cusolvermg, enable_logging

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
    context!(ctx) do
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
mutable struct DnHandle
    const handle::cusolverDnHandle_t
    const ctx::CuContext
    const workspace_gpu::CuVector{UInt8}
    const workspace_cpu::Vector{UInt8}
    const info::CuVector{Cint}
end
Base.unsafe_convert(::Type{Ptr{cusolverDnContext}}, handle::DnHandle) =
    handle.handle

function dense_handle_finalizer(dh::DnHandle)
    CUDACore.unsafe_free!(dh.workspace_gpu)
    CUDACore.unsafe_free!(dh.info)
    push!(idle_dense_handles, dh.ctx, dh.handle)
end

const DenseLibraryState = @NamedTuple{handle::DnHandle, stream::CuStream}
const dense_state_cache = CUDACore.TaskLocalCache{CuContext, DenseLibraryState}(:CUSOLVER_dense)

function dense_handle()
    cuda = CUDACore.active_state()

    states = CUDACore.task_dict(dense_state_cache)

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_dense_handles, cuda.context)

        workspace_gpu = CuVector{UInt8}(undef, 0)
        workspace_cpu = Vector{UInt8}(undef, 0)
        info = CuVector{Cint}(undef, 1)
        fat_handle = DnHandle(new_handle, cuda.context, workspace_gpu,
                                      workspace_cpu, info)
        resource_finalizer(dense_handle_finalizer, fat_handle; blocking=false)

        cusolverDnSetStream(new_handle, cuda.stream)

        (; handle=fat_handle, cuda.stream)
    end
    state = get!(dense_state_cache, cuda.context) do
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
    context!(ctx) do
        cusolverSpDestroy(handle)
    end
end
const idle_sparse_handles =
    HandleCache{CuContext,cusolverSpHandle_t}(sparse_handle_ctor, sparse_handle_dtor)

# mutable wrapper so the raw sparse handle is released via an object-bound
# finalizer (see `DnHandle` for rationale).
mutable struct SpHandle
    const handle::cusolverSpHandle_t
    const ctx::CuContext
end
Base.unsafe_convert(::Type{cusolverSpHandle_t}, handle::SpHandle) =
    handle.handle

function sparse_handle_finalizer(sh::SpHandle)
    push!(idle_sparse_handles, sh.ctx, sh.handle)
end

const SparseLibraryState = @NamedTuple{handle::SpHandle, stream::CuStream}
const sparse_state_cache = CUDACore.TaskLocalCache{CuContext, SparseLibraryState}(:CUSOLVER_sparse)

function sparse_handle()
    cuda = CUDACore.active_state()

    states = CUDACore.task_dict(sparse_state_cache)

    # get or create handle
    @noinline function new_state(cuda)
        new_handle = pop!(idle_sparse_handles, cuda.context)
        wrapped = SpHandle(new_handle, cuda.context)
        resource_finalizer(sparse_handle_finalizer, wrapped; blocking=false)

        cusolverSpSetStream(new_handle, cuda.stream)

        (; handle=wrapped, cuda.stream)
    end
    state = get!(sparse_state_cache, cuda.context) do
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
mutable struct MgHandle
    const handle::cusolverMgHandle_t
    const ctx::CuContext
end
Base.unsafe_convert(::Type{cusolverMgHandle_t}, handle::MgHandle) =
    handle.handle

function mg_handle_finalizer(mh::MgHandle)
    context!(mh.ctx) do
        cusolverMgDestroy(mh.handle)
    end
end

const MgLibraryState = @NamedTuple{handle::MgHandle}
const mg_state_cache = CUDACore.TaskLocalCache{UInt, MgLibraryState}(:CUSOLVERmg)

function mg_handle()
    cuda = CUDACore.active_state()

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
        wrapped = MgHandle(new_handle, cuda.context)
        resource_finalizer(mg_handle_finalizer, wrapped)

        # cuSOLVERMg accesses the per-device buffers from the other devices in the grid
        # and from the context it is called in, so when using the stream-ordered
        # allocator, make each grid device's pool visible to all of them.
        for dev in filter(CUDACore.stream_ordered, devices())
            other_devs = filter(!isequal(dev), union(devices(), [cuda.device]))
            # only grant access to peer-capable devices: cuMemPoolSetAccess on a
            # fresh pool can succeed even when the devices are not peer capable,
            # deferring the failure to a later allocation.
            accessible_devs = filter(other -> CUDACore.can_access_peer(other, dev), other_devs)
            for other in setdiff(other_devs, accessible_devs)
                @warn "cuSOLVERMg: $other cannot access memory on $dev; operations may fail" maxlog=1
            end
            isempty(accessible_devs) && continue
            pool = CUDACore.pool_create(dev)
            access!(pool, accessible_devs, CUDACore.CU_MEM_ACCESS_FLAGS_PROT_READWRITE)
        end

        devs = convert.(Cint, devices())
        cusolverMgDeviceSelect(new_handle, length(devs), devs)

        (; handle=wrapped)
    end
    state = get!(mg_state_cache, key) do
        new_state(cuda)
    end

    return state.handle
end


## logging

# the logging API is documented and defined in the static library, but the shared library
# does not export it (checked from CUDA 11 through 13.4; NVIDIA bug #6738440)
function has_logging_api()
    lib = CUDACore.Libdl.dlopen(libcusolver)
    try
        CUDACore.Libdl.dlsym(lib, :cusolverDnLoggerSetCallback; throw_error=false) !== nothing
    finally
        CUDACore.Libdl.dlclose(lib)
    end
end

function log_message(level::Int32, function_name::Cstring, message::Cstring)
    CUDACore.library_log_callback(cuSOLVER, level, function_name, message)
    return
end

# without the callback API, the library can only be configured through its environment,
# which it reads once when first called. have it write to a file, which unlike a pipe cannot
# block the library when a single-threaded Julia isn't reading, and forward what it writes.
const log_file = Ref("")
const log_offset = Ref(0)
const log_forwarding = Ref(false)
const log_file_lock = ReentrantLock()
const log_timer = Ref{Timer}()

const log_line_regex = r"^\[[^\]]*\]\[[^\]]*\]\[\d+\]\[(\w+)\]\[(\w+)\]\s*(.*)$"

function parse_log_line(line::AbstractString)
    m = match(log_line_regex, line)
    m === nothing && return CUDACore.Debug, line
    category, name, message = m.captures
    level = if category == "Error"
        CUDACore.Error
    elseif category == "Hint"
        CUDACore.Info
    else
        CUDACore.Debug
    end
    return level, isempty(message) ? "$name(...)" : "$name: $message"
end

function forward_log_file()
    Base.@lock log_file_lock begin
        isfile(log_file[]) || return
        # the library has read its configuration; don't leak it into child processes
        if get(ENV, "CUSOLVERDN_LOG_FILE", nothing) == log_file[]
            delete!(ENV, "CUSOLVERDN_LOG_LEVEL")
            delete!(ENV, "CUSOLVERDN_LOG_FILE")
        end
        data = open(log_file[]) do io
            seek(io, log_offset[])
            read(io)
        end
        # a line that's still being written is picked up the next time
        last_newline = findlast(==(UInt8('\n')), data)
        last_newline === nothing && return
        log_offset[] += last_newline
        log_forwarding[] || return
        for line in eachline(IOBuffer(view(data, 1:last_newline)))
            level, message = parse_log_line(line)
            CUDACore.enqueue_log(cuSOLVER, level, message)
        end
    end
    return
end

# NOTE: must be called before the library is first used
function forward_log_file_start()
    CUDACore.init_logging()
    log_file[] = tempname()
    ENV["CUSOLVERDN_LOG_LEVEL"] = "5"
    ENV["CUSOLVERDN_LOG_FILE"] = log_file[]
    log_forwarding[] = true
    push!(CUDACore.log_sources, forward_log_file)
    log_timer[] = Timer(0.25; interval=0.25) do _
        try
            forward_log_file()
        catch err
            CUDACore.enqueue_log(CUDACore, CUDACore.Error,
                                 "Error reading the cuSOLVER log: " * sprint(showerror, err))
        end
    end
    return
end

"""
    cuSOLVER.enable_logging(enable::Bool=true)

Forward log messages from cuSOLVER to Julia's logging system. API and kernel traces are
reported at `Debug` level, performance hints at `Info` level, and problems at `Error`
level. Starting Julia with `JULIA_DEBUG=cuSOLVER` enables this automatically, and also shows
the `Debug`-level messages.

If the installed shared library does not export the logging API, messages are only available
when starting Julia with `JULIA_DEBUG=cuSOLVER`, in which case this function pauses or
resumes forwarding them. The library keeps generating messages while paused. Without
`JULIA_DEBUG`, enabling emits a warning; set `CUSOLVERDN_LOG_LEVEL` before starting Julia to
use cuSOLVER's own output instead.
"""
function enable_logging(enable::Bool=true)
    if !has_logging_api()
        if !isempty(log_file[])
            # drop or deliver what was generated before this call under the old setting
            forward_log_file()
            log_forwarding[] = enable
        elseif enable
            @warn "The cuSOLVER $(version()) shared library does not export the logging callback API; start Julia with JULIA_DEBUG=cuSOLVER, or set the CUSOLVERDN_LOG_LEVEL environment variable to have the library print its log to stdout instead" maxlog=1
        end
        return
    end
    if enable
        CUDACore.init_logging()
        callback = @cfunction(log_message, Nothing, (Int32, Cstring, Cstring))
        cusolverDnLoggerSetCallback(callback)
        # the library also writes to stdout, unless a log file is set
        cusolverDnLoggerOpenFile(CUDACore.devnull_path)
        cusolverDnLoggerSetLevel(5)
    else
        cusolverDnLoggerSetLevel(0)
    end
    return
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

    # forward the library's log messages when debugging, unless the user configured the
    # library's own output.
    if !precompiling && isdebug(cuSOLVER)
        if has_logging_api()
            enable_logging(true)
        elseif !haskey(ENV, "CUSOLVERDN_LOG_LEVEL")
            forward_log_file_start()
        end
    end

    CUDACore.register_reclaimable!(idle_dense_handles)
    CUDACore.register_reclaimable!(idle_sparse_handles)
    CUDACore.register_reclaimable!(dense_state_cache)
    CUDACore.register_reclaimable!(sparse_state_cache)
    CUDACore.register_reclaimable!(mg_state_cache)

    _initialized[] = true
end

include("precompile.jl")

# deprecated binding for backwards compatibility
Base.@deprecate_binding CUSOLVER cuSOLVER false

end
