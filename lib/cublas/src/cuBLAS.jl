module cuBLAS

using CUDACore
using GPUToolbox

using CUDACore: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType, cudaEmulationSpecialValuesSupport, cudaEmulationMantissaControl
using CUDACore: unsafe_free!, retry_reclaim, isdebug, @sync, initialize_context

using GPUArrays

using LinearAlgebra

using BFloat16s: BFloat16

import LLVM
using LLVM.Interop: assume

using CEnum: @cenum

using Adapt: adapt

if CUDACore.local_toolkit
    using CUDA_Runtime_Discovery
else
    import CUDA_Runtime_jll
end


@public functional, enable_logging

const _initialized = Ref{Bool}(false)
functional() = _initialized[]

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

function math_mode!(handle, mode, precision=CUDACore.math_precision())
    flags = 0

    # https://github.com/facebookresearch/faiss/issues/1385
    if version() > v"11"
        flags = CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION
    end

    flags |= if mode == CUDACore.PEDANTIC_MATH
        # Prevent use of tensor cores. On cuBLAS 10 they are opt-in,
        # so the default math mode already accomplishes this.
        version() < v"11" ? CUBLAS_DEFAULT_MATH : CUBLAS_PEDANTIC_MATH
    elseif mode == CUDACore.DEFAULT_MATH
        # On cuBLAS 10, tensor-op math may convert FP32 inputs to FP16. Keep the
        # default mode's guarantee of using at least the requested precision.
        CUBLAS_DEFAULT_MATH
    elseif mode == CUDACore.FAST_MATH
        # downcast to a reduced-precision compute mode whenever possible; the
        # emulation modes additionally engage matching emulated kernels for
        # plain (non-gemmEx) GEMMs. See also `gemmExComputeType`.
        if precision === :BFloat16x9 && version() >= v"12.9"
            CUBLAS_FP32_EMULATED_BF16X9_MATH
        elseif precision === :FixedPoint && version() >= v"13.1"
            CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH
        elseif version() < v"11"
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
    context!(ctx) do
        cublasDestroy_v2(handle)
    end
end
const idle_handles = HandleCache{CuContext,cublasHandle_t}(handle_ctor, handle_dtor)

# mutable wrapper so the raw handle is released via an object-bound finalizer:
# when TLS state is cleared (e.g. on reclaim) and GC runs, the wrapper is
# collected and its finalizer returns the handle to the idle cache instead
# of the handle being pinned for the entire lifetime of the owning task.
mutable struct Handle
    const handle::cublasHandle_t
    const ctx::CuContext
end
Base.unsafe_convert(::Type{cublasHandle_t}, handle::Handle) = handle.handle

function handle_finalizer(h::Handle)
    push!(idle_handles, h.ctx, h.handle)
end

const LibraryState = @NamedTuple{handle::Handle, stream::CuStream, math_mode::CUDACore.MathMode, math_precision::Symbol}
const state_cache = CUDACore.TaskLocalCache{CuContext, LibraryState}(:CUBLAS)

function handle()
    cuda = CUDACore.active_state()

    states = CUDACore.task_dict(state_cache)

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context)
        wrapped = Handle(new_handle, cuda.context)
        finalizer(handle_finalizer, wrapped)

        cublasSetStream_v2(new_handle, cuda.stream)
        cublasSetPointerMode_v2(new_handle, CUBLAS_POINTER_MODE_DEVICE)
        math_mode!(new_handle, cuda.math_mode, cuda.math_precision)

        (; handle=wrapped, cuda.stream, cuda.math_mode, cuda.math_precision)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    # update stream
    @noinline function update_stream(cuda, state)
        cublasSetStream_v2(state.handle, cuda.stream)
        (; state.handle, stream=cuda.stream, state.math_mode, state.math_precision)
    end
    if state.stream != cuda.stream
        states[cuda.context] = state = update_stream(cuda, state)
    end

    # update math mode (the precision feeds into the emulation math modes)
    @noinline function update_math_mode(cuda, state)
        math_mode!(state.handle, cuda.math_mode, cuda.math_precision)
        (; state.handle, state.stream, math_mode=cuda.math_mode, math_precision=cuda.math_precision)
    end
    if state.math_mode != cuda.math_mode || state.math_precision != cuda.math_precision
        states[cuda.context] = state = update_math_mode(cuda, state)
    end

    return state.handle
end


## xt handles

function xt_handle_ctor(ctxs)
    cublasXtCreate()
end
function xt_handle_dtor(ctxs, handle)
    cublasXtDestroy(handle)
end
const idle_xt_handles =
    HandleCache{Vector{CuContext},cublasXtHandle_t}(xt_handle_ctor, xt_handle_dtor)

# mutable wrapper for the xt handle, see `Handle` for rationale.
mutable struct XtHandle
    const handle::cublasXtHandle_t
    const ctxs::Vector{CuContext}
end
Base.unsafe_convert(::Type{cublasXtHandle_t}, h::XtHandle) = h.handle

function xt_handle_finalizer(h::XtHandle)
    push!(idle_xt_handles, h.ctxs, h.handle)
end

function devices!(devs::Vector{CuDevice})
    task_local_storage(:CUBLASxt_devices, sort(devs; by=deviceid))
    return
end

devices() = get!(task_local_storage(), :CUBLASxt_devices) do
    # by default, select all devices
    sort(collect(CUDACore.devices()); by=deviceid)
end::Vector{CuDevice}

ndevices() = length(devices())

const XtLibraryState = @NamedTuple{handle::XtHandle}
const xt_state_cache = CUDACore.TaskLocalCache{UInt, XtLibraryState}(:CUBLASxt)

function xt_handle()
    cuda = CUDACore.active_state()

    states = CUDACore.task_dict(xt_state_cache)

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
        wrapped = XtHandle(new_handle, ctxs)
        finalizer(xt_handle_finalizer, wrapped)

        # if we're using the stream-ordered allocator,
        # make sure allocations are visible on all devices
        async_devs = filter(memory_pools_supported, devices())
        for dev in async_devs
            other_devs = filter(!isequal(dev), async_devs)
            # only grant access to peer-capable devices: cuMemPoolSetAccess on a
            # fresh pool can succeed even when the devices are not peer capable,
            # deferring the failure to a later allocation.
            accessible_devs = filter(other -> CUDACore.can_access_peer(other, dev), other_devs)
            for other in setdiff(other_devs, accessible_devs)
                @warn "cublasXt: $other cannot access memory on $dev; operations on device arrays may fail" maxlog=1
            end
            isempty(accessible_devs) && continue
            pool = CUDACore.pool_create(dev)
            access!(pool, accessible_devs, CUDACore.CU_MEM_ACCESS_FLAGS_PROT_READWRITE)
        end

        devs = convert.(Cint, devices())
        cublasXtDeviceSelect(new_handle, length(devs), devs)

        (; handle=wrapped)
    end
    state = get!(states, key) do
        new_state(cuda)
    end

    return state.handle
end


## logging

# cuBLAS invokes the callback once per line of a message, from the calling thread, which
# for cuBLASXt can be a worker thread. Lines are assembled per thread into complete messages.
const log_lock = Threads.SpinLock()
const log_buffers = Dict{Int,IOBuffer}()
const log_atexit = Ref(false)

function log_message(ptr::Cstring)
    CUDACore.guarded_callback() do
        line = unsafe_string(ptr)
        Base.@lock log_lock begin
            buf = get!(IOBuffer, log_buffers, Threads.threadid())
            # a message starts with a line marked by an uppercase severity code (e.g. `I!`),
            # and ends with a chunk of several lines (the time, process and compiler details)
            if buf.size > 0 && ncodeunits(line) >= 2 && 'A' <= Char(codeunit(line, 1)) <= 'Z' &&
               codeunit(line, 2) == UInt8('!')
                flush_log_buffer(buf)
            end
            println(buf, line)
            if occursin('\n', chop(line))
                flush_log_buffer(buf)
            end
        end
    end
    return
end

# NOTE: must be called with the log lock held
function flush_log_buffer(buf::IOBuffer)
    message = String(take!(buf))
    isempty(message) && return

    # the first line is marked with the severity (e.g. `I!`), subsequent ones with a
    # lowercase code (`i!`), optionally followed by a space
    code = message[1]
    lines = map(eachline(IOBuffer(message))) do line
        strip(ncodeunits(line) >= 2 && codeunit(line, 2) == UInt8('!') ? line[3:end] : line)
    end
    filter!(!isempty, lines)
    message = join(lines, '\n')

    level = if code == 'I'
        CUDACore.Debug
    elseif code == 'W'
        CUDACore.Warn
    elseif code == 'E'
        CUDACore.Error
    elseif code == 'F'
        CUDACore.Error
    else
        CUDACore.Info
    end
    CUDACore.enqueue_log(cuBLAS, level, message)
    return
end

# report incomplete messages, e.g. at exit
function flush_log_buffers()
    Base.@lock log_lock begin
        for buf in values(log_buffers)
            flush_log_buffer(buf)
        end
    end
    return
end

# cuBLASLt uses the logging design shared by other libraries
function lt_log_message(level::Int32, function_name::Cstring, message::Cstring)
    CUDACore.library_log_callback(cuBLAS, level, function_name, message)
    return
end

"""
    cuBLAS.enable_logging(enable::Bool=true)

Forward log messages from cuBLAS and cuBLASLt to Julia's logging system. API traces are
reported at `Debug` level, performance hints at `Info` level, and problems at `Warn` or
`Error` level. Starting Julia with `JULIA_DEBUG=cuBLAS` enables this automatically, and also
shows the `Debug`-level messages.
"""
function enable_logging(enable::Bool=true)
    if enable
        CUDACore.init_logging()
        # the cuBLAS logging callback crashes on Windows (NVIDIA bug #3321130)
        if !Sys.iswindows()
            callback = @cfunction(log_message, Nothing, (Cstring,))
            cublasSetLoggerCallback(callback)
            if !log_atexit[]
                atexit(flush_log_buffers)
                log_atexit[] = true
            end
        end
        callback = @cfunction(lt_log_message, Nothing, (Int32, Cstring, Cstring))
        cublasLtLoggerSetCallback(callback)
        cublasLtLoggerOpenFile(CUDACore.devnull_path)
        cublasLtLoggerSetLevel(5)
    else
        Sys.iswindows() || cublasLoggerConfigure(0, 0, 0, C_NULL)
        cublasLtLoggerSetLevel(0)
    end
    return
end

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0

    CUDACore.functional() || return

    # find the library
    global libcublas, libcublasLt
    if CUDACore.local_toolkit
        dirs = CUDA_Runtime_Discovery.find_toolkit()
        path = CUDA_Runtime_Discovery.get_library(dirs, "cublas"; optional=true)
        if path === nothing
            precompiling || @error "cuBLAS is not available on your system (looked in $(join(dirs, ", ")))"
            return
        end
        libcublas = path
        path_lt = CUDA_Runtime_Discovery.get_library(dirs, "cublasLt"; optional=true)
        if path_lt !== nothing
            libcublasLt = path_lt
        end
    else
        libcublas = CUDA_Runtime_jll.libcublas
        libcublasLt = CUDA_Runtime_jll.libcublasLt
    end

    # forward the library's log messages when debugging
    if !precompiling && isdebug(cuBLAS)
        enable_logging(true)
    end

    # wire up reclaim (precompile-captured constructors can't push into
    # CUDACore's registry themselves)
    CUDACore.register_reclaimable!(idle_handles)
    CUDACore.register_reclaimable!(idle_xt_handles)
    CUDACore.register_reclaimable!(state_cache)
    CUDACore.register_reclaimable!(xt_state_cache)

    _initialized[] = true
end

include("precompile.jl")

# deprecated binding for backwards compatibility
Base.@deprecate_binding CUBLAS cuBLAS false

end
