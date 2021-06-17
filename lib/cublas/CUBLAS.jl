module CUBLAS

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType
using ..CUDA: libcublas, unsafe_free!, @retry_reclaim, isdebug, @sync, @context!

using GPUArrays

using LinearAlgebra

using BFloat16s

using CEnum

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

const MAX_LOG_BUFLEN = UInt(1024*1024)
const log_buffer = Vector{UInt8}(undef, MAX_LOG_BUFLEN)
const log_cursor = Threads.Atomic{UInt}(0)
const log_cond = Ref{Base.AsyncCondition}()    # root

function log_message(ptr)
    # NOTE: this function may be called from unmanaged threads (by cublasXt),
    #       so we can't even allocate, let alone perform I/O.
    len = @ccall strlen(ptr::Cstring)::Csize_t
    old_cursor = log_cursor[]
    new_cursor = old_cursor + len+1
    if new_cursor >= MAX_LOG_BUFLEN
        # overrun
        return
    end

    @ccall memmove((pointer(log_buffer)+old_cursor)::Ptr{Nothing},
                   pointer(ptr)::Ptr{Nothing}, (len+1)::Csize_t)::Nothing
    log_cursor[] = new_cursor   # the consumer handles CAS'ing this value
    @ccall uv_async_send(log_cond[]::Ptr{Nothing})::Cint

    return
end

function _log_message(blob)
    # the message format isn't documented, but it looks like a message starts with a capital
    # and the severity (e.g. `I!`), and subsequent lines start with a lowercase mark (`!i`)
    #
    # lines are separated by a \0 if they came in separately, but there may also be multiple
    # actual lines separated by \n in each message.
    for message in split(blob, r"[\0\n]+(?=[A-Z]!)")
        code = message[1]
        lines = split(message[3:end], r"[\0\n]+[a-z]!")
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
        blob = ""
        while true
            message_length = log_cursor[]
            blob = unsafe_string(pointer(log_buffer), message_length)
            if Threads.atomic_cas!(log_cursor, message_length, UInt(0)) == message_length
                break
            end
        end
        _log_message(blob)
        return
    end
    if (isdebug(:init, CUBLAS) || Base.JLOptions().debug_level >= 2) &&
       !Sys.iswindows() # NVIDIA bug #3321130
        callback = @cfunction(log_message, Nothing, (Cstring,))
        cublasSetLoggerCallback(callback)
    end
end

end
