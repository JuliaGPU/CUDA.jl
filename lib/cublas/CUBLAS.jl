module CUBLAS

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType
using ..CUDA: libcublas, unsafe_free!, @retry_reclaim, isdebug, @sync

using GPUArrays

using LinearAlgebra

using BFloat16s

using CEnum

using Memoize

using DataStructures


# core library
include("libcublas_common.jl")
include("error.jl")
include("libcublas.jl")
include("libcublas_deprecated.jl")

# low-level wrappers
include("util.jl")
include("wrappers.jl")

# high-level integrations
include("linalg.jl")

# thread cache for task-local library handles
const thread_handles = Vector{Union{Nothing,cublasHandle_t}}()
const thread_xt_handles = Vector{Union{Nothing,cublasXtHandle_t}}()

# cache for created, but unused handles
const old_handles = DefaultDict{CuContext,Vector{cublasHandle_t}}(()->cublasHandle_t[])
const old_xt_handles = DefaultDict{Vector{CuContext},Vector{cublasHandle_t}}(()->cublasHandle_t[])

function math_mode!(handle, mode)
    flags = 0

    # https://github.com/facebookresearch/faiss/issues/1385
    if version(handle) > v"11"
        flags = CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION
    end

    flags |= if mode == CUDA.PEDANTIC_MATH
        # prevent use of tensor cores
        if version(handle) < v"11"
            CUBLAS_DEFAULT_MATH
        else
            CUBLAS_PEDANTIC_MATH
        end
    elseif mode == CUDA.DEFAULT_MATH
        # use tensor cores, but don't reduce precision
        if version(handle) < v"11"
            CUBLAS_TENSOR_OP_MATH
        else
            CUBLAS_DEFAULT_MATH
        end
    elseif mode == CUDA.FAST_MATH
        # we'll additionally select a compute-mode with reduced precision whenever possible
        if version(handle) < v"11"
            CUBLAS_TENSOR_OP_MATH
        else
            CUBLAS_TF32_TENSOR_OP_MATH
        end
    end

    cublasSetMathMode(handle, flags)

    return
end

function handle()
    tid = Threads.threadid()
    if @inbounds thread_handles[tid] === nothing
        ctx = context()
        thread_handles[tid] = get!(task_local_storage(), (:CUBLAS, ctx)) do
            handle = if isempty(old_handles[ctx])
                cublasCreate()
                # FIXME: use cublasSetWorkspace? cublasSetStream reset it.
            else
                pop!(old_handles[ctx])
            end

            finalizer(current_task()) do task
                push!(old_handles[ctx], handle)
            end
            # TODO: cublasDestroy to preserve memory, or at exit?

            cublasSetStream_v2(handle, stream())

            math_mode!(handle, CUDA.math_mode())

            handle
        end
    end
    something(@inbounds thread_handles[tid])
end

function xt_handle()
    tid = Threads.threadid()
    if @inbounds thread_xt_handles[tid] === nothing
        ctxs = [context(dev) for dev in devices()]
        thread_xt_handles[tid] = get!(task_local_storage(), (:CUBLASxt, ctxs)) do
            handle = if isempty(old_xt_handles[ctxs])
                cublasXtCreate()
            else
                pop!(old_xt_handles[ctxs])
            end

            finalizer(current_task()) do task
                push!(old_xt_handles[ctxs], handle)
            end
            # TODO: cublasXtDestroy to preserve memory, or at exit?

            # select all devices
            devs = convert.(Cint, devices())
            cublasXtDeviceSelect(handle, length(devs), devs)

            handle
        end
    end
    something(@inbounds thread_xt_handles[tid])
end

@inline function set_stream(stream::CuStream)
    ctx = context()
    tls = task_local_storage()
    handle = get(tls, (:CUBLAS, ctx), nothing)
    if handle !== nothing
        cublasSetStream_v2(handle, stream)
    end
    return
end

function __init__()
    resize!(thread_handles, Threads.nthreads())
    fill!(thread_handles, nothing)

    resize!(thread_xt_handles, Threads.nthreads())
    fill!(thread_xt_handles, nothing)

    CUDA.atdevicereset() do dev
        fill!(thread_xt_handles, nothing)
    end

    CUDA.atdeviceswitch() do
        tid = Threads.threadid()
        thread_handles[tid] = nothing
    end

    CUDA.attaskswitch() do
        tid = Threads.threadid()
        thread_handles[tid] = nothing
        thread_xt_handles[tid] = nothing
    end
end

function log_message(cstr)
    # NOTE: we can't `@debug` these messages, because the logging function is called for
    #       every line... also not sure what the i!/I! prefixes mean (info?)
    # NOTE: turns out we even can't `print` these messages, as cublasXt callbacks might
    #       happen from a different thread! we could strdup + uv_async_send, but I couldn't
    #       find an easy way to attach data to that
    # TODO: use a pre-allocated lock-free global message buffer?
    len = ccall(:strlen, Csize_t, (Cstring,), cstr)
    ccall(:write, Cint, (Cint, Cstring, Csize_t), 0, cstr, len)
    return
end

function __runtime_init__()
    # enable library logging when launched with JULIA_DEBUG=CUBLAS
    if isdebug(:init, CUBLAS)
        callback = @cfunction(log_message, Nothing, (Cstring,))
        cublasSetLoggerCallback(callback)
    end
end

end
