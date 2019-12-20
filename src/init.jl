# Initialization

export device!, device_reset!, CuGetContext

const thread_contexts = Union{Nothing,CuContext}[]

# FIXME: support for flags (see `cudaSetDeviceFlags`)

"""
    CuGetContext()::CuContext

Get or create a CUDA context for the current thread (as opposed to
`CUDAdrv.CuCurrentContext` which may return `nothing` if there is no context bound to the
current thread).
"""
function CuGetContext()::CuContext
    maybe_initialize("CuGetContext")
    ctx = CuCurrentContext()
    @assert @inbounds thread_contexts[Threads.threadid()] === ctx
    # TODO: once we trust our initialization logic, we can just return from thread_contexts
    ctx
end

# TODO: for proper threading, we need a setter-counterpart of CuGetContext. For example,
#       when creating an array we use `CuGetContext` to initialize and get a device context,
#       when doing operations on it we will need to make sure that context is also active.
#
#       The API will need to be very fast (so, e.g., we won't be able to go from devices to
#       or from contexts because both cuGetDevice and cuPrimaryContextRetain are slow),
#       and should also not look to much like the low-level CUDAdrv context API
#       (`CuGetContext` is probably too generic a name in that regard).
#
#       We should also think about reducing the necessary code changes, i.e., having every
#       CuArray method call `CuSetContext(a.ctx)` is much too invasive. Worst case, we just
#       document that users should do this, e.g., with a user-friendly `activate(CuArray)`.

"""
    CUDAnative.maybe_initialize(apicall::Symbol)

Initialize a GPU device if none is bound to the current thread yet. Call this function
before any functionality that requires a functioning GPU context.

This is designed to be a very fast call (couple of ns).
"""
function maybe_initialize(apicall)
    tid = Threads.threadid()
    if @inbounds thread_contexts[tid] !== nothing
        check_exceptions() # FIXME: This doesn't really belong here
        return
    end

    initialize(apicall)
end

@noinline function initialize(apicall)
    @debug "Initializing CUDA on thread $(Threads.threadid()) after call to $apicall"
    device!(CuDevice(0))
end

"""
    CUDAnative.atcontextswitch(f::Function)

Register a function to be called after switching contexts on a thread. The function is
passed three arguments: the thread ID, new context, and corresponding device. If the context
is nothing, this indicates that the thread is unbounds with the device representing the
previously bound device.
"""
atcontextswitch(f::Function) = (pushfirst!(context_hooks, f); nothing)
const context_hooks = []
_atcontextswitch(tid, ctx, dev) = foreach(listener->listener(tid, ctx, dev), context_hooks)

"""
    device!(dev)

Sets `dev` as the current active device for the calling host thread. Devices can be
specified by integer id, or as a `CuDevice`. This is intended to be a low-cost operation,
only performing significant work when calling it for the first time for each device on each
thread.

If your library or code needs to perform an action when the active device context changes,
add a hook using [`CUDAnative.atcontextswitch`](@ref).
"""
function device!(dev::CuDevice)
    tid = Threads.threadid()

    # NOTE: this call is fairly "expensive" (50-100ns); see TODO on top

    # bail out if switching to the current device
    if @inbounds thread_contexts[tid] !== nothing && dev == device() # FIXME: expensive
        return
    end

    # get the primary context
    pctx = CuPrimaryContext(dev)
    ctx = CuContext(pctx)   # FIXME: expensive

    # update the thread-local state
    @inbounds thread_contexts[tid] = ctx
    activate(ctx)

    _atcontextswitch(tid, ctx, dev)
end
device!(dev::Integer) = device!(CuDevice(dev))

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

    # wipe the context handles for all threads using this device
    for (tid, thread_ctx) in enumerate(thread_contexts)
        if thread_ctx == ctx
            thread_contexts[tid] = nothing
            _atcontextswitch(tid, nothing, dev)
            # TODO: actually unbind the CUDA threads with `activate(CuContext(CU_NULL))`?
        end
    end

    return
end

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
            activate(old_ctx)
        end
    end
end
device!(f::Function, dev::Integer) = device!(f, CuDevice(dev))
