"""
    CUDA.CUDNN

High level interface to cuDNN functions. See
https://github.com/JuliaGPU/CUDA.jl/blob/master/lib/state/README.md
for a design overview.
"""
module CUDNN

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, libraryPropertyType
using ..CUDA: libcudnn, @retry_reclaim, isdebug, @context!

using CEnum

using DataStructures


# core library
include("libcudnn_common.jl")
include("error.jl")
include("libcudnn.jl")
include("libcudnn_deprecated.jl")

# low-level wrappers
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

# cache for created, but unused handles
const idle_handles = HandleCache{CuContext,cudnnHandle_t}()

function handle()
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{handle::cudnnHandle_t, stream::CuStream}
    states = get!(task_local_storage(), :CUDNN) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context) do
            cudnnCreate()
        end

        finalizer(current_task()) do task
            push!(idle_handles, cuda.context, new_handle) do
                @context! skip_destroyed=true cuda.context cudnnDestroy(new_handle)
            end
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
    @spinlock log_lock begin
        push!(log_messages, (; sev, dbg, str))
    end
    ccall(:uv_async_send, Cint, (Ptr{Cvoid},), udata)

    return
end

function _log_message(sev, dbg, str)
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

function __runtime_init__()
    if version() < v"8.0"
        @warn "This version of CUDA.jl only supports CUDNN 8.0 or higher"
    end

    # register a log callback
    if (isdebug(:init, CUDNN) || Base.JLOptions().debug_level >= 2) &&
       version() >= v"8.2"  # NVIDIA bug #3256123
        log_cond[] = Base.AsyncCondition() do async_cond
            message =  @lock log_lock popfirst!(log_messages)
            _log_message(message...)
        end

        callback = @cfunction(log_message, Nothing,
                              (cudnnSeverity_t, Ptr{Cvoid}, Ptr{cudnnDebug_t}, Ptr{UInt8}))
        cudnnSetCallback(typemax(UInt32), log_cond[], callback)
    end
end

end
