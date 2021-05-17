"""
    CUDA.CUDNN

High level interface to cuDNN functions. See
https://github.com/JuliaGPU/CUDA.jl/blob/master/lib/cudnn/README.md
for a design overview.
"""
module CUDNN

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, libraryPropertyType
using ..CUDA: libcudnn, @retry_reclaim, isdebug, @context!

using CEnum

using Memoize

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
    state = CUDA.active_state()
    handle, stream = get!(task_local_storage(), (:CUDNN, state.context)) do
        new_handle = pop!(idle_handles, state.context) do
            cudnnCreate()
        end

        finalizer(current_task()) do task
            push!(idle_handles, state.context, new_handle) do
                @context! skip_destroyed=true state.context cudnnDestroy(new_handle)
            end
        end

        cudnnSetStream(new_handle, state.stream)

        new_handle, state.stream
    end::Tuple{cudnnHandle_t,CuStream}

    if stream != state.stream
        cudnnSetStream(handle, state.stream)
        task_local_storage((:CUDNN, state.context), (handle, state.stream))
    end

    return handle
end

function log_message(sev, udata, dbg_ptr, ptr)
    # "Each line of this message is terminated by \0, and the end of the message is
    # terminated by \0\0"
    len = 0
    while true
        if unsafe_load(ptr, len+1) == '\0' && unsafe_load(ptr, len+2) == '\0'
            break
        end
        len += 1
    end
    str = unsafe_string(ptr, len)
    lines = split(str, '\0')
    msg = join(str, '\n')

    # TODO: inspect `sev` to generate an appropriate message (@debug, @info, etc)
    dbg = unsafe_load(dbg_ptr)
    println(msg)
    return
end

function __runtime_init__()
    if version() < v"8.0"
        @warn "This version of CUDA.jl only supports CUDNN 8.0 or higher"
    end

    # enable library logging when launched with JULIA_DEBUG=CUDNN
    # FIXME: this doesn't work, and the mask remains 0 (as observed with cudnnGetCallback)
    if isdebug(:init, CUDNN)
        callback = @cfunction(log_message, Nothing,
                              (cudnnSeverity_t, Ptr{Cvoid}, Ptr{cudnnDebug_t}, Ptr{UInt8}))
        cudnnSetCallback(typemax(UInt32), C_NULL, callback)
    end
end

end
