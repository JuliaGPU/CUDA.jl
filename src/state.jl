# global state management

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

If you need to maintain your own thread-local state, subscribe to context and task switch
events using [`CUDA.atdeviceswitch`](@ref) and [`CUDA.attaskswitch`](@ref) for
proper invalidation.
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

# the default device unitialized tasks will use, set when switching devices.
# this behavior differs from the CUDA Runtime, where device 0 is always used.
# this setting won't be used when switching tasks on a pre-initialized thread.
const default_device = Ref{Union{Nothing,CuDevice}}(nothing)

# CUDA uses thread-bound state, but calling CuCurrent* all the time is expensive,
# so we maintain our own thread-local copy keeping track of the current CUDA state.
CuCurrentState = NamedTuple{(:ctx, :dev), Tuple{CuContext,CuDevice}}
const thread_state = Union{Nothing,CuCurrentState}[]
@noinline function initialize_thread(tid::Int)
    ctx = CuCurrentContext()
    if ctx === nothing
        dev = something(default_device[], CuDevice(0))
        device!(dev)
    else
        # compatibility with externally-initialized contexts
        dev = CuCurrentDevice()
        thread_state[tid] = (;ctx,dev)
    end
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
    if ctx !== nothing
        context!(ctx)
    end
    # NOTE: deactivating the context in the case ctx===nothing would be more correct,
    #       but that confuses CUDA and leads to invalid contexts later on.
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
        thread_state[tid] = (;ctx, dev)
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
    old_ctx = CuCurrentContext()
    try
        context!(ctx)
        f()
    finally
        if old_ctx != nothing
            context!(old_ctx)
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

"""
    device!(dev::Integer)
    device!(dev::CuDevice)

Sets `dev` as the current active device for the calling host thread. Devices can be
specified by integer id, or as a `CuDevice` (slightly faster).

Although this call is fairly cheap (50-100ns), it is only intended for interactive use, or
for initial set-up of the environment. If you need to switch devices on a regular basis,
work with contexts instead and call [`context!`](@ref) directly (5-10ns).

If your library or code needs to perform an action when the active context changes,
add a hook using [`CUDA.atdeviceswitch`](@ref).
"""
function device!(dev::CuDevice, flags=nothing)
    tid = Threads.threadid()

    # configure the primary context
    pctx = CuPrimaryContext(dev)
    if flags !== nothing
        @assert !isactive(pctx) "Cannot set flags for an active device. Do so before calling any CUDA function, or reset the device first."
        cuDevicePrimaryCtxSetFlags(dev, flags)
    end

    # bail out if switching to the current device
    state = @inbounds thread_state[tid]
    if state !== nothing && state.dev == dev
        return
    end

    # have new threads use this device as well
    default_device[] = dev

    # activate a context
    ctx = CuContext(pctx)
    context!(ctx)
end
device!(dev::Integer, flags=nothing) = device!(CuDevice(dev), flags)

"""
    device!(f, dev)

Sets the active device for the duration of `f`.
"""
function device!(f::Function, dev::CuDevice)
    old_ctx = CuCurrentContext()
    try
        device!(dev)
        f()
    finally
        if old_ctx != nothing
            context!(old_ctx)
        end
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

    _atdevicereset(dev)

    return
end


## integer device-based API

deviceid() = Int(convert(CUdevice, device()))

device!(f::Function, dev::Integer) = device!(f, CuDevice(dev))
