module CUBLAS

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType
using ..CUDA: libcublas, unsafe_free!, @retry_reclaim, isdebug, @sync, @context!

using GPUArrays

using LinearAlgebra

using BFloat16s

using CEnum

using Memoization

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
    if version() > v"11"
        flags = CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION
    end

    flags |= if mode == CUDA.PEDANTIC_MATH
        # prevent use of tensor cores
        if version() < v"11"
            CUBLAS_DEFAULT_MATH
        else
            CUBLAS_PEDANTIC_MATH
        end
    elseif mode == CUDA.DEFAULT_MATH
        # use tensor cores, but don't reduce precision
        if version() < v"11"
            CUBLAS_TENSOR_OP_MATH
        else
            CUBLAS_DEFAULT_MATH
        end
    elseif mode == CUDA.FAST_MATH
        # we'll additionally select a compute-mode with reduced precision whenever possible
        if version() < v"11"
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


## logging

const log_entries = []
const log_lock = ReentrantLock()
const log_cond = Ref{Any}()    # root

function log_message(ptr)
    str = unsafe_string(ptr)

    # print asynchronously
    @spinlock log_lock begin
        push!(log_entries, strip(str))
    end
    ccall(:uv_async_send, Cint, (Ptr{Cvoid},), log_cond[])

    return
end

function _log_message(blob)
    # the message format isn't documented, but it looks like a message starts with a capital
    # and the severity (e.g. `I!`), and subsequent lines start with a lowercase mark (`!i`)
    for message in split(blob, r"\n(?=[A-Z]!)")
        code = message[1]
        lines = split(message[3:end], r"\n[a-z]!")
        submessage = join(lines, '\n')
        if code == 'I'
            @debug submessage
        elseif code == 'W'
            @warn submessage
        elseif code == 'E'
            @error submessage
        elseif code == 'F'
            error(submessage)
        else
            @info "Unknown log message, please file an issue.\n$message"
        end
    end
    return
end

function __runtime_init__()
    # register a log callback
    log_cond[] = Base.AsyncCondition() do async_cond
        blob =  @lock log_lock begin
            blob = join(log_entries, '\n')
            empty!(log_entries)
            blob
        end
        _log_message(blob)
        return
    end
    callback = @cfunction(log_message, Nothing, (Cstring,))
    cublasSetLoggerCallback(callback)
end

end
