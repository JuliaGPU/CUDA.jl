# global state management
#
# The functionality in this file serves to create a coherent environment to perform
# computations in, with support for Julia constructs like tasks (and executing those on
# multiple threads), using a GPU of your choice (with the ability to reset that device, or
# use different devices on different threads).

export context, context!, device, device!, device_reset!, deviceid, stream, stream!, pool


## hooks

"""
    CUDA.attaskswitch(f::Function)

Register a function to be called after switching to or initializing a task on a thread.

Use this hook to invalidate thread-local state that depends on the current task.
"""
attaskswitch(f::Function) = (pushfirst!(task_hooks, f); nothing)
const task_hooks = []
_attaskswitch() = foreach(f->Base.invokelatest(f), task_hooks)

"""
    CUDA.atdeviceswitch(f::Function)

Register a function to be called after switching to or initializing a device on a thread.

Use this hook to invalidate thread-local state that depends on the current device. If that
state is also context dependent, be sure to query the context in your callback.
"""
atdeviceswitch(f::Function) = (pushfirst!(device_switch_hooks, f); nothing)
const device_switch_hooks = []
_atdeviceswitch() = foreach(f->Base.invokelatest(f), device_switch_hooks)

"""
    CUDA.atdevicereset(f::Function)

Register a function to be called after resetting devices. The function is passed one
argument: the device which has been reset.

Use this hook to invalidate global state that depends on the current device.
"""
atdevicereset(f::Function) = (pushfirst!(device_reset_hooks, f); nothing)
const device_reset_hooks = []
_atdevicereset(dev) = foreach(f->Base.invokelatest(f, dev), device_reset_hooks)


## initialization

# Many CUDA APIs, like the CUDA driver API used by CUDA.jl, use implicit thread-local state
# to determine, e.g., which device to use. With Julia however, code is grouped in tasks.
# Execution can switch between them, and tasks can be executing on (and in the future
# migrate between) different threads. To synchronize these two worlds, we try to detect task
# and thread switches, making sure CUDA's state is mirrored appropriately.

# If you need to maintain your own task-local state, subscribe to device and task switch
# events using [`CUDA.atdeviceswitch`](@ref) and [`CUDA.attaskswitch`](@ref) for proper
# invalidation. If your state is device-specific, but global (i.e. not task-bound), it
# suffices to index your state with the current [`deviceid()`](@ref) and invalidate that
# state when the device is reset by subscribing to [`CUDA.atdevicereset()`](@ref).

@inline function detect_state_changes()
    detect_task_switches()

    # make sure to initialize the CUDA context, which is responsible for triggering a
    # device switch event after the device has been reset.
    initialize_cuda_context()

    return
end

@inline function detect_task_switches()
    tid = Threads.threadid()

    # detect when a different task is now executing on a thread
    task = @inbounds thread_tasks[tid]
    if task === nothing || task.value === nothing || task.value::Task !== current_task()
        switched_tasks(tid, current_task())
    end

    return
end

const thread_tasks = Union{Nothing,WeakRef}[]
@noinline function switched_tasks(tid::Int, task::Task)
    thread_tasks[tid] = WeakRef(task)
    _attaskswitch()

    # switch contexts if task switched to was already bound to one
    ctx = get(task_local_storage(), :CuContext, nothing)
    if ctx !== nothing && isvalid(ctx)
        # NOTE: the context may be invalid if another task reset it (which we detect here
        #       since we can't touch other tasks' local state from `device_reset!`)
        context!(ctx)
    else
        thread_state[tid] = nothing  # trigger `initialize_thread` down the line
        # NOTE: actually deactivating the CUDA context would be more correct,
        #       but that confuses CUDA and leads to invalid contexts later on.
    end
end

# prepare for a CUDA API call, which requires a CUDA device to be selected.
# we can't do this eagerly because it consumes quite some memory.
@inline function initialize_cuda_context()
    tid = Threads.threadid()

    # initialize CUDA state when first executing on a thread
    state = @inbounds thread_state[tid]
    if state === nothing
        initialize_current_thread()
    end

    return
end

# the default device unitialized tasks will use, set when switching devices.
# this behavior differs from the CUDA Runtime, where device 0 is always used.
# this setting won't be used when switching tasks on a pre-initialized thread.
const default_device = Ref{Union{Nothing,CuDevice}}(nothing)

# CUDA uses thread-bound state, but calling CuCurrent* all the time is expensive,
# so we maintain our own thread-local copy keeping track of the current CUDA state.
const CuCurrentState = NamedTuple{(:ctx, :dev), Tuple{CuContext,CuDevice}}
const thread_state = Union{Nothing,CuCurrentState}[]
@noinline function initialize_current_thread()
    dev = something(default_device[], CuDevice(0))
    device!(dev)

    # NOTE: we can't be compatible with externally initialize contexts here (i.e., reuse
    #       the CuCurrentContext we don't know about) because of how device_reset! works:
    #       contexts that got reset remain bound, and are not discernible from regular ones.
end


## context-based API

"""
    context()::CuContext

Get or create a CUDA context for the current thread (as opposed to
`CuCurrentContext` which may return `nothing` if there is no context bound to the
current thread).
"""
@inline function context()
    detect_state_changes()
    tid = Threads.threadid()

    state = @inbounds thread_state[tid]::CuCurrentState

    if Base.JLOptions().debug_level >= 2
        @assert state.ctx == CuCurrentContext()
    end
    state.ctx
end

"""
    context!(ctx::CuContext)

Bind the current host thread to the context `ctx`.

Note that the contexts used with this call should be previously acquired by calling
[`context`](@ref), and not arbitrary contexts created by calling the `CuContext` constructor.
"""
function context!(ctx::CuContext)
    detect_task_switches()
    tid = Threads.threadid()

    # update the thread-local state
    state = @inbounds thread_state[tid]
    if state === nothing || state.ctx != ctx
        activate(ctx)
        dev = CuCurrentDevice()::CuDevice
        thread_state[tid] = (;ctx=ctx, dev=dev)
        _atdeviceswitch()
    end

    # update the task-local state
    task_local_storage(:CuContext, ctx)

    return
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
            detect_task_switches()
            tid = Threads.threadid()

            # NOTE: we don't use `context()` here, since that initializes
            old_state = @inbounds thread_state[tid]
            if old_state === nothing || old_state.ctx != ctx
                context!(ctx)   # XXX: context! performs a lot of the same checks...
            end
            try
                $(esc(body))
            finally
                if old_state !== nothing && ctx != old_state.ctx
                    context!(old_state.ctx)
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
    detect_state_changes()
    tid = Threads.threadid()

    state = @inbounds thread_state[tid]::CuCurrentState

    if Base.JLOptions().debug_level >= 2
        @assert state.dev == CuCurrentDevice()
    end
    state.dev
end

const __device_contexts = Union{Nothing,CuContext}[]
device_context(i) = @after_init(@inbounds __device_contexts[i])
device_context!(i, ctx) = @after_init(@inbounds __device_contexts[i] = ctx)

function context(dev::CuDevice)
    devidx = deviceid(dev)+1

    # querying the primary context for a device is expensive, so cache it
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

If your library or code needs to perform an action when the active device changes,
add a hook using [`CUDA.atdeviceswitch`](@ref).
"""
function device!(dev::CuDevice, flags=nothing)
    detect_task_switches()  # we got a CuDevice, so CUDA is initialized already
    tid = Threads.threadid()

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

    # bail out if switching to the current device
    state = @inbounds thread_state[tid]
    if state !== nothing && state.dev == dev
        return
    end

    # actually switch contexts
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

If your library or code needs to perform an action when the active context changes,
add a hook using [`CUDA.atdevicereset`](@ref). Resetting the device will also cause
the [`CUDA.atdeviceswitch`](@ref) hook to fire when `initialize_cuda_context` is called,
so it is generally not needed to subscribe to the reset hook specifically.

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

    # wipe the thread-local state for all threads using this device
    for (tid, state) in enumerate(thread_state)
        if state !== nothing && state.dev == dev
            thread_state[tid] = nothing
        end
    end

    # wipe the device-specific state
    devidx = deviceid(dev)+1
    device_context!(devidx, nothing)

    _atdevicereset(dev)

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

# helper struct to maintain state per device
# - make it possible to index directly with CuDevice (without converting to integer index)
# - initialize function to fill state based on a constructor function
# - automatically wiping state on device reset
struct PerDevice{T,F} <: AbstractVector{T}
    inner::Vector{T}
    ctor::F

    PerDevice{T,F}(ctor::F) where {T,F<:Function} = new(Vector{T}(), ctor)
end

PerDevice{T}(ctor::F) where {T,F} = PerDevice{T,F}(ctor)

function initialize!(x::PerDevice, n::Integer)
    @assert isempty(x.inner)

    resize!(x.inner, n)
    for i in 0:n-1
        x[i] = x.ctor(CuDevice(i))
    end

    atdevicereset() do dev
        x[dev] = x.ctor(dev)
    end

    return
end

# 0-based indexing for using CUDA device identifiers
Base.getindex(x::PerDevice, devidx::Integer) = x.inner[devidx+1]
Base.setindex!(x::PerDevice, val, devidx::Integer) = (x.inner[devidx+1] = val; )

# indexing using CuDevice objecs
Base.getindex(x::PerDevice, dev::CuDevice) = x[deviceid(dev)]
Base.setindex!(x::PerDevice, val, dev::CuDevice) = (x[deviceid(dev)] = val; )

Base.length(x::PerDevice) = length(x.inner)
Base.size(x::PerDevice) = size(x.inner)

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

# thread cache for task-local streams
const thread_streams = Vector{Union{Nothing,CuStream}}()

"""
    stream()

Get the CUDA stream that should be used as the default one for the currently executing task.
"""
@inline function stream()
    detect_state_changes()
    tid = Threads.threadid()

    if @inbounds thread_streams[tid] === nothing
        ctx = context()
        thread_streams[tid] = get!(task_local_storage(), (:CuStream, ctx)) do
            stream = CuStream(; flags=STREAM_NON_BLOCKING)

            t = current_task()
            tptr = pointer_from_objref(current_task())
            tptrstr = string(convert(UInt, tptr), base=16, pad=Sys.WORD_SIZE>>2)
            NVTX.nvtxNameCuStreamA(stream, "Task(0x$tptrstr)")

            stream
        end
    end
    something(@inbounds thread_streams[tid])
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
    # task switch detected by context()
    tid = Threads.threadid()

    ctx = context()
    task_local_storage((:CuStream, ctx), s)

    # update the thread cache
    @inbounds thread_streams[tid] = s

    set_library_streams(s)
end

function stream!(f::Function, s::CuStream)
    # task switch detected by stream()
    tid = Threads.threadid()

    # NOTE: we can't read `thread_streams` directly here, or could end up with `nothing`,
    #       and we need a valid stream to fall back to and reset the library handles with.
    old_s = stream()
    try
        return task_local_storage(:CuStream, s) do
            thread_streams[tid] = s
            set_library_streams(s)
            f()
        end
    finally
        thread_streams[tid] = old_s
        set_library_streams(old_s)
    end
end
