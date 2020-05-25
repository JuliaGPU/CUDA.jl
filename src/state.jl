# global state management

export context, context!, device!, device_reset!


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
events using [`CUDA.atcontextswitch`](@ref) and [`CUDA.attaskswitch`](@ref) for
proper invalidation.
"""
@inline function prepare_cuda_call()
    tid = Threads.threadid()
    task = current_task()

    # detect when a different task is now executing on a thread
    if @inbounds thread_tasks[tid] != task
        switched_tasks(tid, task)
    end

    # initialize a CUDA context when first executing on a thread
    if @inbounds thread_contexts[tid] === nothing
        initialize_thread(tid)
    end

    check_exceptions()

    return
end

# the default device unitialized tasks will use, set when switching devices.
# this behavior differs from the CUDA Runtime, where device 0 is always used.
# this setting won't be used when switching tasks on a pre-initialized thread.
const default_device = Ref{Union{Nothing,CuDevice}}(nothing)

# CUDA uses thread-bound contexts, but calling CuCurrentContext all the time is expensive,
# so we maintain our own thread-local state keeping track of the current context.
const thread_contexts = Union{Nothing,CuContext}[]
@noinline function initialize_thread(tid::Int)
    dev = something(default_device[], CuDevice(0))
    device!(dev)

    # NOTE: we can't be compatible with externally initialize contexts here (i.e., reuse
    #       the CuCurrentContext we don't know about) because of how device_reset! works:
    #       contexts that got reset remain bound, and are not discernible from regular ones.
end

# Julia executes with tasks, so we need to keep track of the active task for each thread
# in order to detect task switches and update the thread-local state accordingly.
# doing so using task_local_storage is too expensive.
const thread_tasks = Union{Nothing,WeakRef}[]
@noinline function switched_tasks(tid::Int, task::Task)
    thread_tasks[tid] = WeakRef(task)
    _attaskswitch(tid, task)

    # switch contexts if task switched to was already bound to one
    ctx = get(task_local_storage(), :CuContext, nothing)
    if ctx !== nothing
        context!(ctx)
    end
    # NOTE: deactivating the context in the case ctx===nothing would be more correct,
    #       but that confuses CUDA and leads to invalid contexts later on.
end

"""
    CUDA.attaskswitch(f::Function)

Register a function to be called after switching tasks on a thread. The function is passed
two arguments: the thread ID, and the task switched to.

Use this hook to invalidate thread-local state that depends on the current task.
"""
attaskswitch(f::Function) = (pushfirst!(task_hooks, f); nothing)
const task_hooks = []
_attaskswitch(tid, task) = foreach(f->Base.invokelatest(f, tid, task), task_hooks)


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
    ctx = @inbounds thread_contexts[tid]::CuContext

    if Base.JLOptions().debug_level >= 2
        @assert ctx == CuCurrentContext()
    end

    ctx
end

"""
    context!(ctx::CuContext)

Bind the current host thread to the context `ctx`.

Note that the contexts used with this call should be previously acquired by calling
[`context`](@ref), and not arbirary contexts created by calling the `CuContext` constructor.
"""
function context!(ctx::CuContext)
    # update the thread-local state
    tid = Threads.threadid()
    thread_ctx = @inbounds thread_contexts[tid]
    if thread_ctx != ctx
        thread_contexts[tid] = ctx
        activate(ctx)
        _atcontextswitch(tid, ctx)
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

"""
    CUDA.atcontextswitch(f::Function)

Register a function to be called after switching contexts on a thread. The function is
passed two arguments: the thread ID, and the context switched to.

If the new context is `nothing`, this indicates that the context is being unbound from this
thread (typically during device reset).

Use this hook to invalidate thread-local state that depends on the current device or context.
"""
atcontextswitch(f::Function) = (pushfirst!(context_hooks, f); nothing)
const context_hooks = []
_atcontextswitch(tid, ctx) = foreach(f->Base.invokelatest(f, tid, ctx), context_hooks)


## device-based API

"""
    device!(dev::Integer)
    device!(dev::CuDevice)

Sets `dev` as the current active device for the calling host thread. Devices can be
specified by integer id, or as a `CuDevice` (slightly faster).

Although this call is fairly cheap (50-100ns), it is only intended for interactive use, or
for initial set-up of the environment. If you need to switch devices on a regular basis,
work with contexts instead and call [`context!`](@ref) directly (5-10ns).

If your library or code needs to perform an action when the active context changes,
add a hook using [`CUDA.atcontextswitch`](@ref).
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
    if @inbounds thread_contexts[tid] !== nothing && dev == device()
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
device!(f::Function, dev::Integer) = device!(f, CuDevice(dev))

"""
    device_reset!(dev::CuDevice=device())

Reset the CUDA state associated with a device. This call with release the underlying
context, at which point any objects allocated in that context will be invalidated.
"""
function device_reset!(dev::CuDevice=device())
    pctx = CuPrimaryContext(dev)
    ctx = CuContext(pctx)

    # unconditionally reset the primary context (don't just release it),
    # as there might be users outside of CUDA.jl
    unsafe_reset!(pctx)

    # wipe the context handles for all threads using this device
    for (tid, thread_ctx) in enumerate(thread_contexts)
        if thread_ctx == ctx
            thread_contexts[tid] = nothing
            _atcontextswitch(tid, nothing)
        end
    end

    return
end
