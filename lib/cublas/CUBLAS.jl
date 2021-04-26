module CUBLAS

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType
using ..CUDA: libcublas, unsafe_free!, @retry_reclaim, isdebug, @sync, @context!

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

# cache for created, but unused handles
const idle_handles = HandleCache{CuContext,cublasHandle_t}()
const idle_xt_handles = HandleCache{Any,cublasXtHandle_t}()

function handle()
    state = CUDA.active_state()
    handle, stream, math_mode = get!(task_local_storage(), (:CUBLAS, state.context)) do
        new_handle = pop!(idle_handles, state.context) do
            cublasCreate()
        end

        finalizer(current_task()) do task
            push!(idle_handles, state.context, new_handle) do
                @context! skip_destroyed=true state.context cublasDestroy_v2(handle)
            end
        end

        cublasSetStream_v2(new_handle, state.stream)

        math_mode!(new_handle, state.math_mode)

        new_handle, state.stream, state.math_mode
    end::Tuple{cublasHandle_t,CuStream,CUDA.MathMode}

    if stream != state.stream
        cublasSetStream_v2(handle, state.stream)
        task_local_storage((:CUBLAS, state.context), (handle, state.stream, math_mode))
        stream = state.stream
    end

    if math_mode != state.math_mode
        math_mode!(handle, state.math_mode)
        task_local_storage((:CUBLAS, state.context), (handle, stream, state.math_mode))
        math_mode = state.math_mode
    end

    return handle
end

function xt_handle()
    ctxs = Tuple(context(dev) for dev in devices())
    get!(task_local_storage(), (:CUBLASxt, ctxs)) do
        handle = pop!(idle_xt_handles, ctxs) do
            cublasXtCreate()
        end

        finalizer(current_task()) do task
            push!(idle_xt_handles, ctxs, handle) do
                # TODO: which context do we need to destroy this on?
                cublasXtDestroy(handle)
            end
        end

        # select all devices
        devs = convert.(Cint, devices())
        cublasXtDeviceSelect(handle, length(devs), devs)

        handle
    end::cublasXtHandle_t
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
