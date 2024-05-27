module CUBLAS

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType, i32
using ..CUDA: unsafe_free!, retry_reclaim, isdebug, @sync, initialize_context

using ..CUDA: CUDA_Runtime
using ..CUDA_Runtime

using GPUArrays

using LinearAlgebra

using BFloat16s: BFloat16

import LLVM
using LLVM.Interop: assume

using CEnum: @cenum


const cudaDataType_t = cudaDataType

# core library
include("libcublas.jl")
include("libcublasLt.jl")
include("libcublas_deprecated.jl")

# low-level wrappers
include("error.jl")
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

    cublasSetMathMode(handle, cublasMath_t(flags))

    return
end


## handles

function handle_ctor(ctx)
    context!(ctx) do
        cublasCreate()
    end
end
function handle_dtor(ctx, handle)
    context!(ctx; skip_destroyed=true) do
        cublasDestroy_v2(handle)
    end
end
const idle_handles = HandleCache{CuContext,cublasHandle_t}(handle_ctor, handle_dtor)

function handle()
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{handle::cublasHandle_t, stream::CuStream, math_mode::CUDA.MathMode}
    states = get!(task_local_storage(), :CUBLAS) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context)
        finalizer(current_task()) do task
            push!(idle_handles, cuda.context, new_handle)
        end

        cublasSetStream_v2(new_handle, cuda.stream)
        math_mode!(new_handle, cuda.math_mode)

        (; handle=new_handle, cuda.stream, cuda.math_mode)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    # update stream
    @noinline function update_stream(cuda, state)
        cublasSetStream_v2(state.handle, cuda.stream)
        (; state.handle, stream=cuda.stream, state.math_mode)
    end
    if state.stream != cuda.stream
        states[cuda.context] = state = update_stream(cuda, state)
    end

    # update math mode
    @noinline function update_math_mode(cuda, state)
        math_mode!(state.handle, cuda.math_mode)
        (; state.handle, state.stream, math_mode=cuda.math_mode)
    end
    if state.math_mode != cuda.math_mode
        states[cuda.context] = state = update_math_mode(cuda, state)
    end

    return state.handle
end


## xt handles

function xt_handle_ctor(ctxs)
    cublasXtCreate()
end
function xt_handle_dtor(ctxs, handle)
    for ctx in ctxs
        CUDA.isvalid(ctx) || return
    end
    cublasXtDestroy(handle)
end
const idle_xt_handles =
    HandleCache{Vector{CuContext},cublasXtHandle_t}(xt_handle_ctor, xt_handle_dtor)

function devices!(devs::Vector{CuDevice})
    task_local_storage(:CUBLASxt_devices, sort(devs; by=deviceid))
    return
end

devices() = get!(task_local_storage(), :CUBLASxt_devices) do
    # by default, select all devices
    sort(collect(CUDA.devices()); by=deviceid)
end::Vector{CuDevice}

ndevices() = length(devices())

function xt_handle()
    cuda = CUDA.active_state()

    # every task maintains library state per set of devices
    LibraryState = @NamedTuple{handle::cublasXtHandle_t}
    states = get!(task_local_storage(), :CUBLASxt) do
        Dict{UInt,LibraryState}()
    end::Dict{UInt,LibraryState}

    # for performance, don't use a tuple of contexts to index the TLS
    key = zero(UInt)
    for dev in devices()
        key = hash(context(dev), key)
    end

    # get library state
    @noinline function new_state(cuda)
        # these are the actual contexts
        ctxs = [context(dev) for dev in devices()]

        new_handle = pop!(idle_xt_handles, ctxs)
        finalizer(current_task()) do task
            push!(idle_xt_handles, ctxs, new_handle)
        end

        # if we're using the stream-ordered allocator,
        # make sure allocations are visible on all devices
        async_devs = filter(memory_pools_supported, devices())
        for dev in async_devs
            other_devs = filter(!isequal(dev), async_devs)
            pool = CUDA.pool_create(dev)
            access!(pool, other_devs, CUDA.CU_MEM_ACCESS_FLAGS_PROT_READWRITE)
        end

        devs = convert.(Cint, devices())
        cublasXtDeviceSelect(new_handle, length(devs), devs)

        (; handle=new_handle)
    end
    state = get!(states, key) do
        new_state(cuda)
    end

    return state.handle
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

    # avoid code that depends on the runtime (even the unsafe_convert from ccall does?!)
    assume(isassigned(log_cond))
    @ccall uv_async_send(log_cond[].handle::Ptr{Nothing})::Cint

    return
end

@gcunsafe_callback function _log_message(blob)
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

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0

    if !CUDA_Runtime.is_available()
        #precompiling || @error "cuBLAS is not available"
        return
    end

    # register a log callback
    if !Sys.iswindows() && # NVIDIA bug #3321130 &&
       !precompiling && (isdebug(:init, CUBLAS) || Base.JLOptions().debug_level >= 2)
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

        callback = @cfunction(log_message, Nothing, (Cstring,))
        cublasSetLoggerCallback(callback)
    end
end

end
