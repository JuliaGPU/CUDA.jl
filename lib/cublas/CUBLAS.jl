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

# cache for created, but unused handles
const handle_cache_lock = ReentrantLock()
const idle_handles = DefaultDict{CuContext,Vector{cublasHandle_t}}(()->cublasHandle_t[])
const idle_xt_handles = DefaultDict{Any,Vector{cublasHandle_t}}(()->cublasHandle_t[])

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
    ctx = context()
    active_stream = stream()
    active_math_mode = CUDA.math_mode()
    handle, chosen_stream, chosen_math_mode = get!(task_local_storage(), (:CUBLAS, ctx)) do
        new_handle = @lock handle_cache_lock begin
            if isempty(idle_handles[ctx])
                cublasCreate()
                # FIXME: use cublasSetWorkspace? cublasSetStream reset it.
            else
                pop!(idle_handles[ctx])
            end
        end

        finalizer(current_task()) do task
            @spinlock handle_cache_lock begin
                push!(idle_handles[ctx], new_handle)
            end
        end
        # TODO: cublasDestroy to preserve memory, or at exit?

        cublasSetStream_v2(new_handle, active_stream)

        math_mode!(new_handle, active_math_mode)

        new_handle, active_stream, active_math_mode
    end::Tuple{cublasHandle_t,CuStream,CUDA.MathMode}

    if chosen_stream != active_stream
        cublasSetStream_v2(handle, active_stream)
        task_local_storage((:CUBLAS, ctx), (handle, active_stream, chosen_math_mode))
        chosen_stream = active_stream
    end

    if chosen_math_mode != active_math_mode
        math_mode!(handle, active_math_mode)
        task_local_storage((:CUBLAS, ctx), (handle, chosen_stream, active_math_mode))
        chosen_math_mode = active_math_mode
    end

    return handle
end

function xt_handle()
    ctxs = Tuple(context(dev) for dev in devices())
    get!(task_local_storage(), (:CUBLASxt, ctxs)) do
        handle = @lock handle_cache_lock begin
            if isempty(idle_xt_handles[ctxs])
                cublasXtCreate()
            else
                pop!(idle_xt_handles[ctxs])
            end
        end

        finalizer(current_task()) do task
            @spinlock handle_cache_lock begin
                push!(idle_xt_handles[ctxs], handle)
            end
        end
        # TODO: cublasXtDestroy to preserve memory, or at exit?

        # select all devices
        devs = convert.(Cint, devices())
        cublasXtDeviceSelect(handle, length(devs), devs)

        handle
    end::cublasHandle_t
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
