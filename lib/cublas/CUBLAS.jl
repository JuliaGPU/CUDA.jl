module CUBLAS

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType
using ..CUDA: libcublas, unsafe_free!, @retry_reclaim, isdebug, @sync

using GPUArrays

using LinearAlgebra

using BFloat16s

using CEnum

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

function math_mode!(handle, mode)
    flags = 0

    # https://github.com/facebookresearch/faiss/issues/1385
    if version(handle) > v"11"
        flags = CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION
    end

    flags |= if mode == CUDA.PEDANTIC_MATH
        # prevent use of tensor cores
        if VERSION < v"11"
            CUBLAS_DEFAULT_MATH
        else
            CUBLAS_PEDANTIC_MATH
        end
    elseif mode == CUDA.DEFAULT_MATH
        # use tensor cores, but don't reduce precision
        if VERSION < v"11"
            CUBLAS_TENSOR_OP_MATH
        else
            CUBLAS_DEFAULT_MATH
        end
    elseif mode == CUDA.FAST_MATH
        # we'll additionally select a compute-mode with reduced precision whenever possible
        if VERSION < v"11"
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
            handle = cublasCreate()
            finalizer(current_task()) do task
                CUDA.isvalid(ctx) || return
                context!(ctx) do
                    cublasDestroy_v2(handle)
                end
            end

            cublasSetStream_v2(handle, CuStreamPerThread())

            if version(handle) >= v"11.2"
                workspace = CuArray{UInt8}(undef, 4*1024*1024)  # TODO: 256-byte aligned
                cublasSetWorkspace_v2(handle, workspace, sizeof(workspace))
                task_local_storage((:CUBLAS, :workspace), workspace)
            end

            math_mode!(handle, CUDA.math_mode())

            handle
        end
    end
    something(@inbounds thread_handles[tid])
end

function xt_handle()
    tid = Threads.threadid()
    if @inbounds thread_xt_handles[tid] === nothing
        ctx = context()
        thread_xt_handles[tid] = get!(task_local_storage(), (:CUBLASxt, ctx)) do
            handle = cublasXtCreate()
            finalizer(current_task()) do task
                CUDA.isvalid(ctx) || return
                context!(ctx) do
                    cublasXtDestroy(handle)
                end
            end

            # select the devices
            # TODO: this is weird, since we typically use a single device per thread/context
            devs = convert.(Cint, CUDA.devices())
            cublasXtDeviceSelect(handle, length(devs), devs)

            handle
        end
    end
    something(@inbounds thread_xt_handles[tid])
end

function __init__()
    resize!(thread_handles, Threads.nthreads())
    fill!(thread_handles, nothing)

    resize!(thread_xt_handles, Threads.nthreads())
    fill!(thread_xt_handles, nothing)

    CUDA.atdeviceswitch() do
        tid = Threads.threadid()
        thread_handles[tid] = nothing
        thread_xt_handles[tid] = nothing
    end

    CUDA.attaskswitch() do
        tid = Threads.threadid()
        thread_handles[tid] = nothing
        thread_xt_handles[tid] = nothing
    end
end

function log_message(cstr)
    str = unsafe_string(cstr)
    print(str)
    # NOTE: we can't `@debug` these messages, because the logging function is called for
    #       every line... also not sure what the i!/I! prefixes mean (info?)
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
