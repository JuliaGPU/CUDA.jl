"""
    cuDNN

High level interface to cuDNN functions. See
[README.md](https://github.com/JuliaGPU/CUDA.jl/blob/master/lib/cudnn/README.md) for a
design overview.
"""
module cuDNN

using CUDA
using CUDA.APIUtils
using CUDA: CUstream, libraryPropertyType
using CUDA: retry_reclaim, isdebug, initialize_context

using CEnum: @cenum

if CUDA.local_toolkit
    using CUDA_Runtime_Discovery
else
    import CUDNN_jll
end


export has_cudnn

const _initialized = Ref{Bool}(false)
has_cudnn() = _initialized[]

# core library
include("libcudnn.jl")

# low-level wrappers
include("error.jl")
include("util.jl")
include("base.jl")
include("descriptors.jl")
include("tensor.jl")
include("inplace.jl")
include("optensor.jl")
include("reduce.jl")
include("convolution.jl")
include("pooling.jl")
include("activation.jl")
include("softmax.jl")
include("dropout.jl")
include("rnn.jl")
include("multiheadattn.jl")
include("normalization.jl")


function math_mode(mode=CUDA.math_mode())
    if mode == CUDA.PEDANTIC_MATH
        # don't use tensor cores.
        # on A100, only use them for TF32
        CUDNN_DEFAULT_MATH
    elseif mode == CUDA.DEFAULT_MATH
        # allow tensor core usage
        CUDNN_TENSOR_OP_MATH
    elseif mode == CUDA.FAST_MATH
        # also downcast inputs
        CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION
    end
end


## handles

function handle_ctor(ctx)
    context!(ctx) do
        cudnnCreate()
    end
end
function handle_dtor(ctx, handle)
    context!(ctx; skip_destroyed=true) do
        cudnnDestroy(handle)
    end
end
const idle_handles = HandleCache{CuContext,cudnnHandle_t}(handle_ctor, handle_dtor)

function handle()
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{handle::cudnnHandle_t, stream::CuStream}
    states = get!(task_local_storage(), :cuDNN) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context)
        finalizer(current_task()) do task
            push!(idle_handles, cuda.context, new_handle)
        end

        cudnnSetStream(new_handle, cuda.stream)

        (; handle=new_handle, cuda.stream)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    # update stream
    @noinline function update_stream(cuda, state)
        cudnnSetStream(state.handle, cuda.stream)
        (; state.handle, cuda.stream)
    end
    if state.stream != cuda.stream
        states[cuda.context] = state = update_stream(cuda, state)
    end

    return state.handle
end


## logging

const log_messages = []
const log_lock = ReentrantLock()
const log_cond = Ref{Any}()    # root

function log_message(sev, udata, dbg_ptr, ptr)
    dbg = unsafe_load(dbg_ptr)

    # find the length of the message, as denoted by two null terminators
    len = 0
    while true && len < 10000
        if unsafe_load(ptr, len+1) == 0 && unsafe_load(ptr, len+2) == 0
            break
        end
        len += 1
    end
    str = unsafe_string(ptr, len)   # XXX: can this yield?

    # print asynchronously
    Base.@lock log_lock begin
        push!(log_messages, (; sev, dbg, str))
    end
    ccall(:uv_async_send, Cint, (Ptr{Cvoid},), udata)

    return
end

@gcunsafe_callback function _log_message(sev, dbg, str)
    lines = split(str, '\0')
    msg = join(lines, '\n')
    if sev == CUDNN_SEV_INFO
        @debug msg
    elseif sev == CUDNN_SEV_WARNING
        @warn msg
    elseif sev == CUDNN_SEV_ERROR
        @error msg
    elseif sev == CUDNN_SEV_FATAL
        error(msg)
    end
    return
end

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0

    CUDA.functional() || return

    # find the library
    global libcudnn
    if CUDA.local_toolkit
        dirs = CUDA_Runtime_Discovery.find_toolkit()
        path = CUDA_Runtime_Discovery.get_library(dirs, "cudnn"; optional=true)
        if path === nothing
            precompiling || @error "cuDNN is not available on your system (looked in $(join(dirs, ", ")))"
            return
        end
        libcudnn = path
    else
        if !CUDNN_jll.is_available()
            precompiling || @error "cuDNN is not available for your platform ($(Base.BinaryPlatforms.triplet(CUDNN_jll.host_platform)))"
            return
        end
        libcudnn = CUDNN_jll.libcudnn
    end

    # register a log callback
    if !precompiling && (isdebug(:init, cuDNN) || Base.JLOptions().debug_level >= 2)
        log_cond[] = Base.AsyncCondition() do async_cond
            message = Base.@lock log_lock popfirst!(log_messages)
            _log_message(message...)
        end

        callback = @cfunction(log_message, Nothing,
                              (cudnnSeverity_t, Ptr{Cvoid}, Ptr{cudnnDebug_t}, Ptr{UInt8}))
        cudnnSetCallback(typemax(UInt32), log_cond[], callback)
    end

    _initialized[] = true
end

end
