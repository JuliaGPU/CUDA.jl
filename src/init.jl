# CUDA Initialization and Context Management

export context, context!, device!, device_reset!


## initialization

"""
    CUDAnative.prepare_cuda_call()

Prepare state for calling CUDA API functions.

Many CUDA APIs, like the CUDA driver API used by CUDAnative, use global thread-local state
to determine, e.g., which device to use. With Julia however, code is grouped in tasks.
Execution can switch between them, and tasks can be executing on (and in the future migrate
between) different threads. To synchronize these two worlds, call this function before any
CUDA API call to update thread-local state based on the current task.

If you need to maintain your own thread-local state, subscribe to context and task switch
events using [`CUDAnative.atcontextswitch`](@ref) and [`CUDAnative.attaskswitch`](@ref) for
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

# the default device new threads will use, set when switching devices.
# this behavior differs from the CUDA Runtime, where device 0 is always used.
const default_device = Ref{Union{Nothing,CuDevice}}(nothing)

# CUDA uses thread-bound contexts, but calling CuCurrentContext all the time is expensive,
# so we maintain our own thread-local state keeping track of the current context.
const thread_contexts = Union{Nothing,CuContext}[]
@noinline function initialize_thread(tid::Int)
    ctx = CuCurrentContext()
    dev = if ctx === nothing
        something(default_device[], CuDevice(0))
    else
        # compatibility with externally-initialized contexts
        device()
    end

    device!(dev)
end

# Julia executes with tasks, so we need to keep track of the active task for each thread
# in order to detect task switches and update the thread-local state accordingly.
# doing so using task_local_storage is too expensive.
const thread_tasks = Union{Nothing,Task}[]
@noinline function switched_tasks(tid::Int, new_task::Task)
    old_task = thread_tasks[tid]
    thread_tasks[tid] = new_task

    # switch contexts
    old_ctx = thread_contexts[tid]
    new_ctx = get(task_contexts, new_task, nothing)
    if new_ctx != old_ctx
        context!(something(new_ctx, old_ctx))
    end

    _attaskswitch(tid, old_task, new_task)
end

# for resetting devices, we need to be able to iterate tasks and find their contexts.
# Julia supports neither iterating tasks nor inspecing other tasks' local storage.
const task_contexts = Dict{Task,CuContext}()

"""
    CUDAnative.attaskswitch(f::Function)

Register a function to be called after switching tasks on a thread. The function is
passed three arguments: the thread number, the old task, and the new one.

Use this hook to invalidate thread-local state.
"""
attaskswitch(f::Function) = (pushfirst!(task_hooks, f); nothing)
const task_hooks = []
_attaskswitch(tid, old, new) = foreach(listener->listener(tid, old, new), task_hooks)


## context-based API

"""
    context()::CuContext

Get or create a CUDA context for the current thread (as opposed to
`CUDAdrv.CuCurrentContext` which may return `nothing` if there is no context bound to the
current thread).
"""
@inline function context()
    tid = Threads.threadid()

    prepare_cuda_call()
    ctx = @inbounds thread_contexts[tid]::CuContext

    if Base.JLOptions().debug_level >= 2
        @assert ctx == CuCurrentContext()
        @assert ctx == task_contexts[current_task()]
    end

    ctx
end

"""
    context!(ctx::CuContext)

Bind the current host thread to the context `ctx`.

Note that the contexts used with this call should be previously acquired by calling
[`context`](@ref), and not arbirary contexts created by calling the `CuContext` constructor.

If your library or code needs to perform an action when the active context changes,
add a hook using [`CUDAnative.atcontextswitch`](@ref).
"""
function context!(new::CuContext)
    # update the thread-local state
    tid = Threads.threadid()
    old = @inbounds thread_contexts[tid]
    if old != new
        @inbounds thread_contexts[tid] = new
        activate(new)
    end

    # update the task-local state
    task = current_task()
    old = get(task_contexts, task, nothing)
    if old != new
        task_contexts[task] = new
        _atcontextswitch(task, old, new)
    end

    return
end

"""
    CUDAnative.atcontextswitch(f::Function)

Register a function to be called after switching contexts in a task. The function is
passed three arguments: a task, the old context, and the new one.

If the new context is `nothing`, this indicates that the context is being unbound from its
task (typically during device reset). An old context being `nothing` happens during the
first API call.

Use this hook to invalidate thread-local state.
"""
atcontextswitch(f::Function) = (pushfirst!(context_hooks, f); nothing)
const context_hooks = []
_atcontextswitch(task, old, new) = foreach(listener->listener(task, old, new), context_hooks)


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
add a hook using [`CUDAnative.atcontextswitch`](@ref).
"""
function device!(dev::CuDevice, flags=nothing)
    tid = Threads.threadid()

    # configure the primary context
    pctx = CuPrimaryContext(dev)
    if flags !== nothing
        @assert !isactive(pctx) "Cannot set flags for an active device. Do so before calling any CUDA function, or reset the device first."
        CUDAdrv.cuDevicePrimaryCtxSetFlags(dev, flags)
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
    # as there might be users outside of CUDAnative.jl
    unsafe_reset!(pctx)

    # unbind the context from threads using it
    # NOTE: we don't actually deactive the contexts, since that confuses CUDA and requires
    #       executing code in another thread, but just updates our thread-local state
    #       causing a reset upon the next API call preparation.
    replace!(thread_contexts, ctx => nothing)

    # wipe the context handles for all tasks using this device
    for (task, task_ctx) in task_contexts
        if task_ctx == ctx
            old = task_contexts[task]
            delete!(task_contexts, task)
            _atcontextswitch(task, old, nothing)
        end
    end

    return
end
