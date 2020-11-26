# global state management
#
# The functionality in this file serves to create a coherent environment to perform
# computations in, with support for Julia constructs like tasks (and executing those on
# multiple threads), using a GPU of your choice (with the ability to reset that device, or
# use different devices on different threads).

export context, context!, device, device!, device_reset!, deviceid


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

"""
    CUDA.prepare_cuda_call()

Prepare state for calling CUDA API functions.

Many CUDA APIs, like the CUDA driver API used by CUDA.jl, use implicit thread-local state
to determine, e.g., which device to use. With Julia however, code is grouped in tasks.
Execution can switch between them, and tasks can be executing on (and in the future migrate
between) different threads. To synchronize these two worlds, call this function before any
CUDA API call to update thread-local state based on the current task and its context.

If you need to maintain your own task-local state, subscribe to device and task switch
events using [`CUDA.atdeviceswitch`](@ref) and [`CUDA.attaskswitch`](@ref) for
proper invalidation. If your state is device-specific, but global (i.e. not task-bound), it
suffices to index your state with the current [`deviceid()`](@ref) and invalidate that state
when the device is reset by subscribing to [`CUDA.atdevicereset()`](@ref).
"""
@inline function prepare_cuda_call()
    tid = Threads.threadid()

    # detect when a different task is now executing on a thread
    task = @inbounds thread_tasks[tid]
    if task === nothing || task.value === nothing || task.value::Task !== current_task()
        switched_tasks(tid, current_task())
    end

    # initialize CUDA state when first executing on a thread
    state = @inbounds thread_state[tid]
    if state === nothing
        initialize_thread(tid)
    end

    # FIXME: this is expensive. Maybe kernels should return a `wait`able object, a la KA.jl,
    #        which then performs the necessary checks. Or only check when launching kernels.
    check_exceptions()

    return
end

# Julia executes with tasks, so we need to keep track of the active task for each thread
# in order to detect task switches and update the thread-local state accordingly.
# doing so using task_local_storage is too expensive.
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

# the default device unitialized tasks will use, set when switching devices.
# this behavior differs from the CUDA Runtime, where device 0 is always used.
# this setting won't be used when switching tasks on a pre-initialized thread.
const default_device = Ref{Union{Nothing,CuDevice}}(nothing)

# CUDA uses thread-bound state, but calling CuCurrent* all the time is expensive,
# so we maintain our own thread-local copy keeping track of the current CUDA state.
const CuCurrentState = NamedTuple{(:ctx, :dev), Tuple{CuContext,CuDevice}}
const thread_state = Union{Nothing,CuCurrentState}[]
@noinline function initialize_thread(tid::Int)
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
    tid = Threads.threadid()

    prepare_cuda_call()
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
    # update the thread-local state
    tid = Threads.threadid()
    state = @inbounds thread_state[tid]
    if state === nothing || state.ctx != ctx
        activate(ctx)
        dev = CuCurrentDevice()
        thread_state[tid] = (;ctx=ctx, dev=dev)
        _atdeviceswitch()
    end

    # update the task-local state
    task_local_storage(:CuContext, ctx)

    return
end

"""
    context!(f, ctx)

Sets the active context for the duration of `f`.
"""
function context!(f::Function, ctx::CuContext)
    @assert isvalid(ctx)
    old_ctx = CuCurrentContext()
    if ctx != old_ctx
        context!(ctx)
    end
    try
        f()
    finally
        if ctx !== old_ctx && old_ctx !== nothing
            context!(old_ctx)
        end
    end
end

# macro version for maximal performance (avoiding closures)
macro context!(ctx_expr, expr)
    quote
        ctx = $(esc(ctx_expr))
        @assert isvalid(ctx)
        old_ctx = CuCurrentContext()
        if ctx != old_ctx
            context!(ctx)
        end
        try
            $(esc(expr))
        finally
            if ctx !== old_ctx && old_ctx !== nothing
                context!(old_ctx)
            end
        end
    end
end


## device-based API

"""
    device()::CuDevice

Get the CUDA device for the current thread, similar to how [`context()`](@ref) works
compared to [`CuCurrentContext()`](@ref).
"""
@inline function device()
    tid = Threads.threadid()

    prepare_cuda_call()
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
    tid = Threads.threadid()
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
    tid = Threads.threadid()
    devidx = deviceid(dev)+1

    # configure the primary context flags
    if flags !== nothing
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

"""
    device!(f, dev)

Sets the active device for the duration of `f`.

Note that this call is intended for temporarily switching devices, and does not change the
default device used to initialize new threads or tasks.
"""
function device!(f::Function, dev::CuDevice)
    ctx = context(dev)
    context!(f, ctx)
end

macro device!(dev_expr, expr)
    quote
        dev = $(esc(dev_expr))
        ctx = context(dev)
        @context! ctx $(esc(expr))
    end
end

"""
    device_reset!(dev::CuDevice=device())

Reset the CUDA state associated with a device. This call with release the underlying
context, at which point any objects allocated in that context will be invalidated.

If your library or code needs to perform an action when the active context changes,
add a hook using [`CUDA.atdevicereset`](@ref). Resetting the device will also cause
subsequent API calls to fire the [`CUDA.atdeviceswitch`](@ref) hook.
"""
function device_reset!(dev::CuDevice=device())
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
