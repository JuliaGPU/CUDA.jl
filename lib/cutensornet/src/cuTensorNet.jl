module cuTensorNet

using LinearAlgebra
using CUDACore
using CUDACore: CUstream, cudaDataType
using CUDACore: retry_reclaim, initialize_context, isdebug, cuDoubleComplex
using CUDACore: @checked, @gcsafe_ccall

using cuTENSOR
using cuTENSOR: CuTensor

using CEnum: @cenum

if CUDACore.local_toolkit
    using CUDA_Runtime_Discovery
else
    import cuQuantum_jll
end


@public functional

const _initialized = Ref{Bool}(false)
functional() = _initialized[]


const cudaDataType_t = cudaDataType

# core library
include("libcutensornet.jl")

# low-level wrappers
include("error.jl")
include("types.jl")
include("wrappers.jl")
include("tensornet.jl")


## handles

function handle_ctor(ctx)
    context!(ctx) do
        cutensornetCreate()
    end
end
function handle_dtor(ctx, handle)
    context!(ctx; skip_destroyed=true) do
        cutensornetDestroy(handle)
    end
end
const idle_handles = HandleCache{CuContext,cutensornetHandle_t}(handle_ctor, handle_dtor)

# mutable wrapper so the raw handle is released via an object-bound
# finalizer: when TLS state is cleared on reclaim (or the owning task is
# collected) and GC runs, the wrapper is collected and its finalizer
# returns the handle to the idle cache.
mutable struct cutensornetHandle
    const handle::cutensornetHandle_t
    const ctx::CuContext
end
Base.unsafe_convert(::Type{cutensornetHandle_t}, h::cutensornetHandle) = h.handle

function handle_finalizer(h::cutensornetHandle)
    push!(idle_handles, h.ctx, h.handle)
end

const LibraryState = @NamedTuple{handle::cutensornetHandle}
const state_cache = CUDACore.TaskLocalCache{CuContext, LibraryState}(:cuTensorNet)

function handle()
    cuda = CUDACore.active_state()

    states = CUDACore.task_dict(state_cache)

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context)
        wrapped = cutensornetHandle(new_handle, cuda.context)
        finalizer(handle_finalizer, wrapped)

        (; handle=wrapped)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    return state.handle
end


## logging

function log_message(log_level, function_name, message)
    function_name = unsafe_string(function_name)
    message = unsafe_string(message)
    output = if isempty(message)
        "$function_name(...)"
    else
        "$function_name: $message"
    end
    if log_level <= 1
        @error output
    else
        # the other log levels are different levels of tracing and hints
        @debug output
    end
    return
end

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0

    CUDACore.functional() || return

    # find the library
    global libcutensornet
    if CUDACore.local_toolkit
        dirs = CUDA_Runtime_Discovery.find_toolkit()
        path = CUDA_Runtime_Discovery.get_library(dirs, "cutensornet"; optional=true)
        if path === nothing
            precompiling || @error "cuQuantum is not available on your system (looked for cutensornet in $(join(dirs, ", ")))"
            return
        end
        libcutensornet = path
    else
        if !cuQuantum_jll.is_available()
            precompiling || @error "cuQuantum is not available for your platform ($(Base.BinaryPlatforms.triplet(cuQuantum_jll.host_platform)))"
            return
        end
        libcutensornet = cuQuantum_jll.libcutensornet
    end

    # register a log callback
    if isdebug(:init, cuTensorNet) || Base.JLOptions().debug_level >= 2
        callback = @cfunction(log_message, Nothing, (Int32, Cstring, Cstring))
        cutensornetLoggerSetCallback(callback)
        cutensornetLoggerOpenFile(Sys.iswindows() ? "NUL" : "/dev/null")
        cutensornetLoggerSetLevel(5)
    end

    CUDACore.register_reclaimable!(idle_handles)
    CUDACore.register_reclaimable!(state_cache)

    _initialized[] = true
end

include("precompile.jl")

end
