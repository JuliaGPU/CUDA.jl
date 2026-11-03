module cuStateVec

using CUDACore
using CUDACore: CUstream, cudaDataType, cudaEvent_t, libraryPropertyType
using CUDACore: unsafe_free!, retry_reclaim, initialize_context, isdebug, resource_finalizer
using CUDACore: @checked, @gcsafe_ccall

using CEnum: @cenum

if CUDACore.local_toolkit
    using CUDA_Runtime_Discovery
else
    import cuQuantum_jll
end


@public functional, enable_logging

const _initialized = Ref{Bool}(false)
functional() = _initialized[]


const cudaDataType_t = cudaDataType

# core library
include("libcustatevec.jl")

include("error.jl")


## handles

function handle_ctor(ctx)
    context!(ctx) do
        custatevecCreate()
    end
end
function handle_dtor(ctx, handle)
    context!(ctx) do
        custatevecDestroy(handle)
    end
end
const idle_handles = HandleCache{CuContext,custatevecHandle_t}(handle_ctor, handle_dtor)

# fat handle: bundles the raw cuStateVec handle with a cache buffer and a
# context reference. Mutable so an object-bound finalizer can release both
# the buffer and the handle when the wrapper becomes unreachable (on
# reclaim or task GC).
mutable struct Handle
    const handle::custatevecHandle_t
    const ctx::CuContext
    const cache::CuVector{UInt8}
end
Base.unsafe_convert(::Type{Ptr{custatevecContext}}, handle::Handle) =
    handle.handle

function handle_finalizer(h::Handle)
    CUDACore.unsafe_free!(h.cache)
    push!(idle_handles, h.ctx, h.handle)
end

# BF16x9 emulation of FP32 is opt-in through the fast math mode, as with cuBLAS
function math_mode!(handle, mode, precision)
    custatevecSetMathMode(handle, if mode == CUDACore.FAST_MATH && precision === :BFloat16x9
        CUSTATEVEC_MATH_MODE_ALLOW_FP32_EMULATED_BF16X9
    elseif mode == CUDACore.PEDANTIC_MATH
        CUSTATEVEC_MATH_MODE_DISALLOW_FP32_EMULATED_BF16X9
    else
        CUSTATEVEC_MATH_MODE_DEFAULT
    end)
end

const LibraryState = @NamedTuple{handle::Handle, stream::CuStream, math_mode::CUDACore.MathMode, math_precision::Symbol}
const state_cache = CUDACore.TaskLocalCache{CuContext, LibraryState}(:CUQUANTUM)

function handle()
    cuda = CUDACore.active_state()

    states = CUDACore.task_dict(state_cache)

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context)

        cache = CuVector{UInt8}(undef, 0)
        fat_handle = Handle(new_handle, cuda.context, cache)
        resource_finalizer(handle_finalizer, fat_handle; blocking=false)

        custatevecSetStream(new_handle, cuda.stream)
        math_mode!(new_handle, cuda.math_mode, cuda.math_precision)

        (; handle=fat_handle, cuda.stream, cuda.math_mode, cuda.math_precision)
    end
    state = get!(state_cache, cuda.context) do
        new_state(cuda)
    end

    # update stream
    @noinline function update_stream(cuda, state)
        custatevecSetStream(state.handle, cuda.stream)
        (; state.handle, cuda.stream, state.math_mode, state.math_precision)
    end
    if state.stream != cuda.stream
        states[cuda.context] = state = update_stream(cuda, state)
    end

    # update math mode
    @noinline function update_math_mode(cuda, state)
        math_mode!(state.handle, cuda.math_mode, cuda.math_precision)
        (; state.handle, state.stream, cuda.math_mode, cuda.math_precision)
    end
    if state.math_mode != cuda.math_mode || state.math_precision != cuda.math_precision
        states[cuda.context] = state = update_math_mode(cuda, state)
    end

    return state.handle
end


# low-level wrappers
include("types.jl")
include("wrappers.jl")
include("statevec.jl")


## logging

function log_message(level::Int32, function_name::Cstring, message::Cstring)
    CUDACore.library_log_callback(cuStateVec, level, function_name, message)
    return
end

"""
    cuStateVec.enable_logging(enable::Bool=true)

Forward log messages from cuStateVec to Julia's logging system. API and kernel traces are
reported at `Debug` level, performance hints at `Info` level, and problems at `Error`
level. Starting Julia with `JULIA_DEBUG=cuStateVec` enables this automatically, and also shows
the `Debug`-level messages.
"""
function enable_logging(enable::Bool=true)
    if enable
        CUDACore.init_logging()
        callback = @cfunction(log_message, Nothing, (Int32, Cstring, Cstring))
        custatevecLoggerSetCallback(callback)
        # the library also writes to stdout, unless a log file is set
        custatevecLoggerOpenFile(CUDACore.devnull_path)
        custatevecLoggerSetLevel(5)
    else
        custatevecLoggerSetLevel(0)
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
            precompiling || @warn "cuQuantum is not available on your system (looked for custatevec in $(join(dirs, ", "))). cuStateVec.functional() will return false."
            return
        end
        libcustatevec = path
    else
        if !cuQuantum_jll.is_available()
            precompiling || @warn "cuQuantum is not available for your platform ($(Base.BinaryPlatforms.triplet(cuQuantum_jll.host_platform))). cuStateVec.functional() will return false."
            return
        end
        libcustatevec = cuQuantum_jll.libcustatevec
    end

    # forward the library's log messages when debugging
    if !precompiling && isdebug(cuStateVec)
        enable_logging(true)
    end

    CUDACore.register_reclaimable!(idle_handles)
    CUDACore.register_reclaimable!(state_cache)

    _initialized[] = true
end

include("precompile.jl")

end
