# global state management
#
# The functionality in this file serves to create a coherent environment to perform
# computations in, with support for Julia constructs like tasks (and executing those on
# multiple threads), using a GPU of your choice (with the ability to reset that device, or
# use different devices on different threads).
#
# this is complicated by the fact that CUDA uses thread-bound contexts, while we want to be
# able to switch between tasks executing on the same thread. the current approach involves
# prefixing every CUDA API call with code that ensures the CUDA thread-bound state is
# identical to Julia's task-local one.

export context, context!, device, device!, device_reset!, deviceid, stream, stream!


## initialization

# the default device unitialized tasks will use, set when switching devices.
# this behavior differs from the CUDA Runtime, where device 0 is always used.
# this setting won't be used when switching tasks on a pre-initialized thread.
const default_device = Ref{Union{Nothing,CuDevice}}(nothing)

@inline function initialize_cuda_context()
    julia_ctx = get(task_local_storage(), :CuContext, nothing)::Union{CuContext,Nothing}
    # NOTE: the context may be invalid if another task reset it (which we detect here
    #       since we can't touch other tasks' local state from `device_reset!`)
    if julia_ctx === nothing || !isvalid(julia_ctx)
        dev = something(default_device[], CuDevice(0))
        device!(dev)
        # NOTE: we can't be compatible with externally initialize contexts here (i.e., reuse
        #       the CuCurrentContext we don't know about) because of how device_reset! works:
        #       contexts that got reset remain bound, and are not discernible from regular ones.
    else
        #cuda_ctx = CuCurrentContext()
        # NOTE: CuCurrentContext() is too slow to use here (taking a lock, accessing a dict)
        #       so we use the raw handle. is that safe though, when we reset the device?
        cuda_ctx = Ref{CUcontext}()
        cuCtxGetCurrent(cuda_ctx)
        if cuda_ctx[] != julia_ctx.handle
            context!(julia_ctx)
        end
    end

    return
end


## context-based API

"""
    context()::CuContext

Get or create a CUDA context for the current thread (as opposed to
`CuCurrentContext` which may return `nothing` if there is no context bound to the
current thread).
"""
@inline function context()
    initialize_cuda_context()
    task_local_storage(:CuContext)::CuContext
end

"""
    context!(ctx::CuContext)

Bind the current host thread to the context `ctx`. Returns the previously-bound context.

Note that the contexts used with this call should be previously acquired by calling
[`context`](@ref), and not arbitrary contexts created by calling the `CuContext` constructor.
"""
function context!(ctx::CuContext)
    old_ctx = get(task_local_storage(), :CuContext, nothing)::Union{CuContext,Nothing}
    if old_ctx != ctx
        task_local_storage(:CuContext, ctx)
        activate(ctx)
    end
    return old_ctx
end

macro context!(ex...)
    body = ex[end]
    ctx = ex[end-1]
    kwargs = ex[1:end-2]

    skip_destroyed = false
    for kwarg in kwargs
        Meta.isexpr(kwarg, :(=)) || throw(ArgumentError("non-keyword argument like option '$kwarg'"))
        key, val = kwarg.args
        isa(key, Symbol) || throw(ArgumentError("non-symbolic keyword '$key'"))

        if key == :skip_destroyed
            skip_destroyed = val
        else
            throw(ArgumentError("unrecognized keyword argument '$kwarg'"))
        end
    end

    quote
        ctx = $(esc(ctx))
        if isvalid(ctx)
            old_ctx = context!(ctx)
            try
                $(esc(body))
            finally
                if old_ctx !== nothing && old_ctx != ctx && isvalid(old_ctx)
                    context!(old_ctx)
                end
            end
        elseif !$(esc(skip_destroyed))
            error("Cannot switch to an invalidated context.")
        end
    end
end

"""
    context!(f, ctx; [skip_destroyed=false])

Sets the active context for the duration of `f`.
"""
@inline function context!(f::Function, ctx::CuContext; skip_destroyed::Bool=false)
    @context! skip_destroyed=skip_destroyed ctx f()
end


## device-based API

"""
    device()::CuDevice

Get the CUDA device for the current thread, similar to how [`context()`](@ref) works
compared to [`CuCurrentContext()`](@ref).
"""
@inline function device()
    initialize_cuda_context()
    CuCurrentDevice()::CuDevice
end

const __device_contexts = Union{Nothing,CuContext}[]
device_context(i) = @after_init(@inbounds __device_contexts[i])
device_context!(i, ctx) = @after_init(@inbounds __device_contexts[i] = ctx)

function context(dev::CuDevice)
    devidx = deviceid(dev)+1

    # querying the primary context for a device is expensive (~100ns), so cache it
    ctx = device_context(devidx)
    if ctx !== nothing
        return ctx
    end

    if capability(dev) < v"5"
        @warn("""Your $(name(dev)) GPU does not meet the minimal required compute capability ($(capability(dev)) < 5.0).
                 Some functionality might not work. For a fully-supported set-up, please use an older version of CUDA.jl""",
              maxlog=1, _id=devidx)
    end

    # configure the primary context
    pctx = CuPrimaryContext(dev)
    ctx = CuContext(pctx)
    device_context!(devidx, ctx)
    return ctx
end

"""
    device!(dev::Integer)
    device!(dev::CuDevice)

Sets `dev` as the current active device for the calling host thread. Devices can be
specified by integer id, or as a `CuDevice` (slightly faster).
"""
function device!(dev::CuDevice, flags=nothing)
    # configure the primary context flags
    if flags !== nothing
        devidx = deviceid(dev)+1
        if device_context(devidx) !== nothing
            error("Cannot set flags for an active device. Do so before calling any CUDA function, or reset the device first.")
        end
        cuDevicePrimaryCtxSetFlags(dev, flags)
    end

    # make this device the new default
    default_device[] = dev

    # switch contexts
    ctx = context(dev)
    context!(ctx)
end

macro device!(dev, body)
    quote
        ctx = context($(esc(dev)))
        @context! ctx $(esc(body))
    end
end

"""
    device!(f, dev)

Sets the active device for the duration of `f`.

Note that this call is intended for temporarily switching devices, and does not change the
default device used to initialize new threads or tasks.
"""
@inline function device!(f::Function, dev::CuDevice)
    @device! dev f()
end

"""
    device_reset!(dev::CuDevice=device())

Reset the CUDA state associated with a device. This call with release the underlying
context, at which point any objects allocated in that context will be invalidated.

!!! warning

    This function is broken on CUDA 11.2 when using the CUDA memory pool (the default).
    If you need to reset the device, use another memory pool by setting the
    `JULIA_CUDA_MEMORY_POOL` environment variable to, e.g., `binned` before importing
    this package.
"""
function device_reset!(dev::CuDevice=device())
    if any_stream_ordered()
        @error """Due to a bug in CUDA, resetting the device is not possible on CUDA 11.2 when using the stream-ordered memory allocator.

                  If you are calling this function to free memory, that may not be required anymore
                  as the stream-ordered memory allocator releases memory much more eagerly.
                  If you do need this functionality, switch to another memory pool by setting
                  the `JULIA_CUDA_MEMORY_POOL` environment variable to, e.g., `binned`."""
        return
    end

    # unconditionally reset the primary context (don't just release it),
    # as there might be users outside of CUDA.jl
    pctx = CuPrimaryContext(dev)
    unsafe_reset!(pctx)

    # wipe the device-specific state
    devidx = deviceid(dev)+1
    device_context!(devidx, nothing)

    return
end


## integer device-based API

device!(dev::Integer, flags=nothing) = device!(CuDevice(dev), flags)
device!(f::Function, dev::Integer) = device!(f, CuDevice(dev))

"""
    deviceid()::Int
    deviceid(dev::CuDevice)::Int

Get the ID number of the current device of execution. This is a 0-indexed number,
corresponding to the device ID as known to CUDA.
"""
deviceid(dev::CuDevice=device()) = Int(convert(CUdevice, dev))


## helpers

# helper struct to maintain state per device and invalidate it when the device is reset
struct PerDevice{T,F} <: AbstractDict{T,F}
    inner::Dict{CuDevice,Tuple{CuContext,T}}
    ctor::F

    PerDevice{T,F}(ctor::F) where {T,F<:Function} = new(Dict{CuDevice,Tuple{CuContext,T}}(), ctor)
end

PerDevice{T}(ctor::F) where {T,F} = PerDevice{T,F}(ctor)

function Base.getindex(x::PerDevice, dev::CuDevice)
    entry = get(x.inner, dev, nothing)
    if entry === nothing || !isvalid(entry[1])
        ctx = context(dev)
        val = x.ctor(dev)
        x.inner[dev] = (ctx, val)
        val
    else
        entry[2]
    end
end
Base.setindex!(x::PerDevice, val, dev::CuDevice) = (x.inner[dev] = (context(dev), val); )

Base.length(x::PerDevice) = length(x.inner)
Base.size(x::PerDevice) = size(x.inner)
Base.keys(x::PerDevice) = keys(x.inner)

function Base.show(io::IO, mime::MIME"text/plain", x::PerDevice{T}) where {T}
    print(io, "PerDevice{$T} with $(length(x)) entries")
end


## math mode

@enum MathMode begin
    # use prescribed precision and standardized arithmetic for all calculations.
    # this may serialize operations, and reduce performance.
    PEDANTIC_MATH

    # use at least the required precision, and allow reordering operations for performance.
    DEFAULT_MATH

    # additionally allow downcasting operations for better use of hardware resources.
    # whenever possible the `precision` flag passed to `math_mode!` will be used
    # to constrain those downcasts.
    FAST_MATH
end

# math mode and precision are sticky (once set on a task, inherit to newly created tasks)
const default_math_mode = Ref{Union{Nothing,MathMode}}(nothing)
const default_math_precision = Ref{Union{Nothing,Symbol}}(nothing)

function math_mode!(mode::MathMode; precision=nothing)
    # make sure we initialize first, or recursion might overwrite the math mode
    ctx = context()

    tls = task_local_storage()
    tls[(:CUDA, :math_mode)] = mode
    default_math_mode[] = mode
    if precision !== nothing
        tls[(:CUDA, :math_precision)] = precision
        default_math_precision[] = precision
    end

    # reapply the CUBLAS math mode if it had been set already
    cublas_handle = get(tls, (:CUBLAS, ctx), nothing)
    if cublas_handle !== nothing
        CUBLAS.math_mode!(cublas_handle, mode)
    end

    return
end

math_mode() =
    get!(task_local_storage(), (:CUDA, :math_mode)) do
        something(default_math_mode[],
                  Base.JLOptions().fast_math==1 ? FAST_MATH : DEFAULT_MATH)
    end
math_precision() =
    get!(task_local_storage(), (:CUDA, :math_precision)) do
        something(default_math_precision[], :TensorFloat32)
    end


## streams

"""
    stream()

Get the CUDA stream that should be used as the default one for the currently executing task.
"""
@inline function stream()
    ctx = context()
    get!(task_local_storage(), (:CuStream, ctx)) do
        stream = CuStream()

        # register the name of this task
        t = current_task()
        tptr = pointer_from_objref(current_task())
        tptrstr = string(convert(UInt, tptr), base=16, pad=Sys.WORD_SIZE>>2)
        NVTX.nvtxNameCuStreamA(stream, "Task(0x$tptrstr)")

        stream
    end::CuStream
end

function set_library_streams(s)
    CUBLAS.set_stream(s)
    CUSPARSE.set_stream(s)
    CUSOLVER.set_stream(s)
    CURAND.set_stream(s)
    CUFFT.set_stream(s)

    CUDNN.set_stream(s)
    CUTENSOR.set_stream(s)
end

function stream!(s::CuStream)
    ctx = context()
    task_local_storage((:CuStream, ctx), s)

    set_library_streams(s)
end

function stream!(f::Function, s::CuStream)
    ctx = context()
    old_s = get(task_local_storage(), (:CuStream, ctx), nothing)::Union{CuStream,Nothing}
    try
        return task_local_storage((:CuStream, ctx), s) do
            set_library_streams(s)
            f()
        end
    finally
        if old_s !== nothing
            set_library_streams(old_s)
        end
    end
end
