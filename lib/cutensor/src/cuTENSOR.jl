module cuTENSOR

using CUDACore
using CUDACore: CUstream, cudaDataType, @gcsafe_ccall, @checked, @enum_without_prefix
using CUDACore: retry_reclaim, initialize_context, isdebug

using CUDACore.GPUToolbox

using CEnum: @cenum

using Printf: @printf

if CUDACore.local_toolkit
    using CUDA_Runtime_Discovery
else
    import CUTENSOR_jll
end


@public functional

const _initialized = Ref{Bool}(false)
functional() = _initialized[]


const cudaDataType_t = cudaDataType

# core library
include("libcutensor.jl")

# low-level wrappers
include("error.jl")
include("utils.jl")
include("types.jl")
include("operations.jl")


# Block sparse wrappers
include("blocksparse/types.jl")
include("blocksparse/operations.jl")

# high-level integrations
include("interfaces.jl")
include("blocksparse/interfaces.jl")


## handles

function handle_ctor(ctx)
    context!(ctx) do
        cutensorCreate()
    end
end
function handle_dtor(ctx, handle)
    context!(ctx; skip_destroyed=true) do
        cutensorDestroy(handle)
    end
end
const idle_handles = HandleCache{CuContext,cutensorHandle_t}(handle_ctor, handle_dtor)

# mutable wrapper so the raw handle is released via an object-bound
# finalizer: when TLS state is cleared on reclaim (or the owning task is
# collected) and GC runs, the wrapper is collected and its finalizer
# returns the handle to the idle cache.
mutable struct cutensorHandle
    const handle::cutensorHandle_t
    const ctx::CuContext
end
Base.unsafe_convert(::Type{cutensorHandle_t}, h::cutensorHandle) = h.handle

function handle_finalizer(h::cutensorHandle)
    push!(idle_handles, h.ctx, h.handle)
end

const LibraryState = @NamedTuple{handle::cutensorHandle}
const state_cache = CUDACore.TaskLocalCache{CuContext, LibraryState}(:cuTENSOR)

function handle()
    cuda = CUDACore.active_state()

    states = CUDACore.task_dict(state_cache)

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context)
        wrapped = cutensorHandle(new_handle, cuda.context)
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
    global libcutensor
    if CUDACore.local_toolkit
        dirs = CUDA_Runtime_Discovery.find_toolkit()
        path = CUDA_Runtime_Discovery.get_library(dirs, "cutensor"; optional=true)
        if path === nothing
            precompiling || @error "cuTENSOR is not available on your system (looked in $(join(dirs, ", ")))"
            return
        end
        libcutensor = path
    else
        if !CUTENSOR_jll.is_available()
            precompiling || @error "cuTENSOR is not available for your platform ($(Base.BinaryPlatforms.triplet(CUTENSOR_jll.host_platform)))"
            return
        end
        libcutensor = CUTENSOR_jll.libcutensor
    end

    # register a log callback
    if !precompiling && (isdebug(:init, cuTENSOR) || Base.JLOptions().debug_level >= 2)
        callback = @cfunction(log_message, Nothing, (Int32, Cstring, Cstring))
        cutensorLoggerSetCallback(callback)
        cutensorLoggerOpenFile(Sys.iswindows() ? "NUL" : "/dev/null")
        cutensorLoggerSetLevel(5)
    end

    CUDACore.register_reclaimable!(idle_handles)
    CUDACore.register_reclaimable!(state_cache)

    _initialized[] = true
end

include("precompile.jl")

end
