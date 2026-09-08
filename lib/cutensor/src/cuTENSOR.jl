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


@public functional, enable_logging

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
    context!(ctx) do
        cutensorDestroy(handle)
    end
end
const idle_handles = HandleCache{CuContext,cutensorHandle_t}(handle_ctor, handle_dtor)

# mutable wrapper so the raw handle is released via an object-bound
# finalizer: when TLS state is cleared on reclaim (or the owning task is
# collected) and GC runs, the wrapper is collected and its finalizer
# returns the handle to the idle cache.
mutable struct Handle
    const handle::cutensorHandle_t
    const ctx::CuContext
end
Base.unsafe_convert(::Type{cutensorHandle_t}, h::Handle) = h.handle

function handle_finalizer(h::Handle)
    push!(idle_handles, h.ctx, h.handle)
end

const LibraryState = @NamedTuple{handle::Handle}
const state_cache = CUDACore.TaskLocalCache{CuContext, LibraryState}(:cuTENSOR)

function handle()
    cuda = CUDACore.active_state()

    states = CUDACore.task_dict(state_cache)

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context)
        wrapped = Handle(new_handle, cuda.context)
        finalizer(handle_finalizer, wrapped)

        (; handle=wrapped)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    return state.handle
end


## logging

function log_message(level::Int32, function_name::Cstring, message::Cstring)
    CUDACore.library_log_callback(cuTENSOR, level, function_name, message)
    return
end

"""
    cuTENSOR.enable_logging(enable::Bool=true)

Forward log messages from cuTENSOR to Julia's logging system. API and kernel traces are
reported at `Debug` level, performance hints at `Info` level, and problems at `Error`
level. Starting Julia with `JULIA_DEBUG=cuTENSOR` enables this automatically, and also shows
the `Debug`-level messages.
"""
function enable_logging(enable::Bool=true)
    if enable
        CUDACore.init_logging()
        callback = @cfunction(log_message, Nothing, (Int32, Cstring, Cstring))
        cutensorLoggerSetCallback(callback)
        # the library also writes to stdout, unless a log file is set
        cutensorLoggerOpenFile(CUDACore.devnull_path)
        cutensorLoggerSetLevel(5)
    else
        cutensorLoggerSetLevel(0)
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

    # forward the library's log messages when debugging
    if !precompiling && isdebug(cuTENSOR)
        enable_logging(true)
    end

    CUDACore.register_reclaimable!(idle_handles)
    CUDACore.register_reclaimable!(state_cache)

    _initialized[] = true
end

include("precompile.jl")

end
