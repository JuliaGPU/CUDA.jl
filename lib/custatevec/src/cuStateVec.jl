module cuStateVec

using CUDACore
using CUDACore: CUstream, cudaDataType, cudaEvent_t, libraryPropertyType
using CUDACore: unsafe_free!, retry_reclaim, initialize_context, isdebug
using CUDACore: @checked, @gcsafe_ccall

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
include("libcustatevec.jl")

# low-level wrappers
include("error.jl")
include("types.jl")
include("wrappers.jl")
include("statevec.jl")


## handles

function handle_ctor(ctx)
    context!(ctx) do
        custatevecCreate()
    end
end
function handle_dtor(ctx, handle)
    context!(ctx; skip_destroyed=true) do
        custatevecDestroy(handle)
    end
end
const idle_handles = HandleCache{CuContext,custatevecHandle_t}(handle_ctor, handle_dtor)

# fat handle: bundles the raw cuStateVec handle with a cache buffer and a
# context reference. Mutable so an object-bound finalizer can release both
# the buffer and the handle when the wrapper becomes unreachable (on
# reclaim or task GC).
mutable struct cuStateVecHandle
    const handle::custatevecHandle_t
    const ctx::CuContext
    const cache::CuVector{UInt8}
end
Base.unsafe_convert(::Type{Ptr{custatevecContext}}, handle::cuStateVecHandle) =
    handle.handle

function handle_finalizer(h::cuStateVecHandle)
    CUDACore.unsafe_free!(h.cache)
    push!(idle_handles, h.ctx, h.handle)
end

const LibraryState = @NamedTuple{handle::cuStateVecHandle, stream::CuStream}
const state_cache = CUDACore.TaskLocalCache{CuContext, LibraryState}(:CUQUANTUM)

function handle()
    cuda = CUDACore.active_state()

    states = CUDACore.task_dict(state_cache)

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context)

        cache = CuVector{UInt8}(undef, 0)
        fat_handle = cuStateVecHandle(new_handle, cuda.context, cache)
        finalizer(handle_finalizer, fat_handle)

        custatevecSetStream(new_handle, cuda.stream)

        (; handle=fat_handle, stream=cuda.stream)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    # update stream
    @noinline function update_stream(cuda, state)
        custatevecSetStream(state.handle, cuda.stream)
        (; state.handle, cuda.stream)
    end
    if state.stream != cuda.stream
        states[cuda.context] = state = update_stream(cuda, state)
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
    global libcustatevec
    if CUDACore.local_toolkit
        dirs = CUDA_Runtime_Discovery.find_toolkit()
        path = CUDA_Runtime_Discovery.get_library(dirs, "custatevec"; optional=true)
        if path === nothing
            precompiling || @error "cuQuantum is not available on your system (looked for custatevec in $(join(dirs, ", ")))"
            return
        end
        libcustatevec = path
    else
        if !cuQuantum_jll.is_available()
            precompiling || @error "cuQuantum is not available for your platform ($(Base.BinaryPlatforms.triplet(cuQuantum_jll.host_platform)))"
            return
        end
        libcustatevec = cuQuantum_jll.libcustatevec
    end

    # register a log callback
    if !precompiling && (isdebug(:init, cuStateVec) || Base.JLOptions().debug_level >= 2)
        callback = @cfunction(log_message, Nothing, (Int32, Cstring, Cstring))
        custatevecLoggerSetCallback(callback)
        custatevecLoggerOpenFile(Sys.iswindows() ? "NUL" : "/dev/null")
        custatevecLoggerSetLevel(5)
    end

    CUDACore.register_reclaimable!(idle_handles)
    CUDACore.register_reclaimable!(state_cache)

    _initialized[] = true
end

include("precompile.jl")

end
