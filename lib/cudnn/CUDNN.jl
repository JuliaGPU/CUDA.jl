module CUDNN

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, libraryPropertyType
using ..CUDA: libcudnn, @retry_reclaim, isdebug

using CEnum

using Memoize

using DataStructures

import NNlib


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

# high-level integrations
include("nnlib.jl")
include("batchnorm.jl")


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

# thread cache for task-local library handles
const thread_handles = Vector{Union{Nothing,cudnnHandle_t}}()

# cache for created, but unused handles
const handle_cache_lock = ReentrantLock()
const idle_handles = DefaultDict{CuContext,Vector{cudnnHandle_t}}(()->cudnnHandle_t[])

function handle()
    CUDA.detect_state_changes()
    tid = Threads.threadid()
    if @inbounds thread_handles[tid] === nothing
        ctx = context()
        thread_handles[tid] = get!(task_local_storage(), (:CUDNN, ctx)) do
            handle = lock(handle_cache_lock) do
                if isempty(idle_handles[ctx])
                    cudnnCreate()
                else
                    pop!(idle_handles[ctx])
                end
            end

            finalizer(current_task()) do task
                lock(handle_cache_lock) do
                    push!(idle_handles[ctx], handle)
                end
            end
            # TODO: cudnnDestroy to preserve memory, or at exit?

            cudnnSetStream(handle, stream())

            handle
        end
    end
    something(@inbounds thread_handles[tid])
end

@inline function set_stream(stream::CuStream)
    ctx = context()
    tls = task_local_storage()
    handle = get(tls, (:CUDNN, ctx), nothing)
    if handle !== nothing
        cudnnSetStream(handle, stream)
    end
    return
end

function __init__()
    resize!(thread_handles, Threads.nthreads())
    fill!(thread_handles, nothing)

    CUDA.atdeviceswitch() do
        tid = Threads.threadid()
        thread_handles[tid] = nothing
    end

    CUDA.attaskswitch() do
        tid = Threads.threadid()
        thread_handles[tid] = nothing
    end
end

function log_message(sev, udata, dbg_ptr, cstr)
    # "Each line of this message is terminated by \0, and the end of the message is
    # terminated by \0\0"
    len = 0
    while true
        if unsafe_load(cstr, len+1) == '\0' && unsafe_load(cstr, len+2)
            break
        end
        len += 1
    end
    str = unsafe_string(cstr, len)
    lines = split(str, '\0')
    msg = join(str, '\n')

    # TODO: inspect `sev` to generate an appropriate message (@debug, @info, etc)
    dbg = unsafe_load(dbg_ptr)
    println(msg)
    return
end

function __runtime_init__()
    # enable library logging when launched with JULIA_DEBUG=CUDNN
    # FIXME: this doesn't work, and the mask remains 0 (as observed with cudnnGetCallback)
    if isdebug(:init, CUDNN)
        callback = @cfunction(log_message, Nothing,
                              (cudnnSeverity_t, Ptr{Cvoid}, Ptr{cudnnDebug_t}, Cstring))
        cudnnSetCallback(typemax(UInt32), C_NULL, callback)
    end
end

end
