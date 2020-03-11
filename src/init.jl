# CUDA Initialization and Context Management

export context, context!, device!, device_reset!


## initialization

# the default device new threads will use, set when switching devices.
# this behavior differs from the CUDA Runtime, where device 0 is always used.
const default_device = Ref{Union{Nothing,CuDevice}}(nothing)

const thread_contexts = Union{Nothing,CuContext}[]

# FIXME: support for flags (see `cudaSetDeviceFlags`)

"""
    CUDAnative.initialize_context()

Initialize a CUDA context for the current thread.

This function is designed to be very fast, and should be called befor almost every CUDA
interaction.
"""
@inline function initialize_context()
    tid = Threads.threadid()
    if @inbounds thread_contexts[tid] === nothing
        _initialize_context()
    end

    check_exceptions() # FIXME: This doesn't really belong here

    return
end

@noinline function _initialize_context()
    @debug "Initializing CUDA on thread $(Threads.threadid())"
    ctx = CuCurrentContext()
    dev = if ctx === nothing
        something(default_device[], CuDevice(0))
    else
        device()
    end
    device!(dev)
end

"""
    CUDAnative.atcontextswitch(f::Function)

Register a function to be called after switching contexts on a thread. The function is
passed two arguments: the thread ID, and a context object. If the context is `nothing`, this
indicates that the thread is unbounds from its context (typically during device reset).
"""
atcontextswitch(f::Function) = (pushfirst!(context_hooks, f); nothing)
const context_hooks = []
_atcontextswitch(tid, ctx) = foreach(listener->listener(tid, ctx), context_hooks)


## context-based API

"""
    context()::CuContext

Get or create a CUDA context for the current thread (as opposed to
`CUDAdrv.CuCurrentContext` which may return `nothing` if there is no context bound to the
current thread).
"""
@inline function context()
    tid = Threads.threadid()

    initialize_context()
    ctx = @inbounds thread_contexts[tid]::CuContext

    if Base.JLOptions().debug_level >= 2
        @assert ctx === CuCurrentContext()
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
function context!(ctx::CuContext)
    tid = Threads.threadid()

    # bail out if switching to the current context
    if @inbounds thread_contexts[tid] === ctx
        return
    end

    # update the thread-local state
    @inbounds thread_contexts[tid] = ctx
    activate(ctx)

    _atcontextswitch(tid, ctx)

    return
end


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
function device!(dev::CuDevice)
    tid = Threads.threadid()

    # bail out if switching to the current device
    if @inbounds thread_contexts[tid] !== nothing && dev == device()
        return
    end

    # have new threads use this device as well
    default_device[] = dev

    # get the primary context
    pctx = CuPrimaryContext(dev)
    ctx = CuContext(pctx)

    context!(ctx)
end
device!(dev::Integer) = device!(CuDevice(dev))

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

    # wipe the context handles for all threads using this device
    for (tid, thread_ctx) in enumerate(thread_contexts)
        if thread_ctx == ctx
            thread_contexts[tid] = nothing
            _atcontextswitch(tid, nothing)
        end
    end

    return
end
