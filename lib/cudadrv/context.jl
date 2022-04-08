# Context management

export
    CuPrimaryContext, CuContext, current_context, has_context, activate,
    unsafe_reset!, isactive, flags, setflags!,
    device, device_synchronize


## construction and destruction

@enum_without_prefix CUctx_flags CU_

"""
    CuPrimaryContext(dev::CuDevice)

Create a primary CUDA context for a given device.

Each primary context is unique per device and is shared with CUDA runtime API. It is meant
for interoperability with (applications using) the runtime API.
"""
struct CuPrimaryContext
    dev::CuDevice
end

"""
    CuContext(dev::CuDevice, flags=CTX_SCHED_AUTO)
    CuContext(f::Function, ...)

Create a CUDA context for device. A context on the GPU is analogous to a process on the CPU,
with its own distinct address space and allocated resources. When a context is destroyed,
the system cleans up the resources allocated to it.

When you are done using the context, call [`CUDA.unsafe_destroy!`](@ref) to mark it for
deletion, or use do-block syntax with this constructor.

"""
mutable struct CuContext
    handle::CUcontext
    valid::Bool

    function new_unique(handle)
        # XXX: this makes it dangerous to call this function from finalizers.
        #      can we do without the lock?
        Base.@lock context_lock get!(valid_contexts, handle) do
            new(handle, true)
        end
    end

    function CuContext(dev::CuDevice, flags=0)
        handle_ref = Ref{CUcontext}()
        cuCtxCreate_v2(handle_ref, flags, dev)
        new_unique(handle_ref[])
    end

    function CuContext(pctx::CuPrimaryContext)
        handle_ref = Ref{CUcontext}()
        cuDevicePrimaryCtxRetain(handle_ref, pctx.dev)
        return new_unique(handle_ref[])
    end

    global function current_context()
        handle_ref = Ref{CUcontext}()
        cuCtxGetCurrent(handle_ref)
        handle_ref[] == C_NULL && throw(UndefRefError())
        new_unique(handle_ref[])
    end

    # for outer constructors
    global _CuContext(handle::CUcontext) = new_unique(handle)
end

"""
    CuContext(pctx::CuPrimaryContext)

Retain the primary context on the GPU, returning a context compatible with the driver API.
The primary context will be released when the returned driver context is finalized.

As these contexts are refcounted by CUDA, you should not call [`CUDA.unsafe_destroy!`](@ref)
on them but use [`CUDA.unsafe_release!`](@ref) instead (available with do-block syntax as
well).
"""
CuContext(pctx::CuPrimaryContext)

"""
    current_context()

Returns the current context.

!!! warning

    This is a low-level API, returning the current context as known to the CUDA driver.
    For most users, it is recommended to use the [`context`](@ref) method instead.
"""
current_context()

"""
    has_context()

Returns whether there is an active context.
"""
function has_context()
    handle_ref = Ref{CUcontext}()
    cuCtxGetCurrent(handle_ref)
    handle_ref[] != C_NULL
end

# the `valid` bit serves two purposes: make sure we don't double-free a context (in case we
# early-freed it ourselves before the GC kicked in), and to make sure we don't free derived
# resources after the owning context has been destroyed (which can happen due to
# out-of-order finalizer execution)
const valid_contexts = Dict{CUcontext,CuContext}()
const context_lock = ReentrantLock()
isvalid(ctx::CuContext) = ctx.valid
# NOTE: we can't just look up by the handle, because contexts derived from a primary one
#       have the same handle even though they might have been destroyed in the meantime.
function invalidate!(ctx::CuContext)
    Base.@lock context_lock delete!(valid_contexts, ctx.handle)
    ctx.valid = false
    return
end

"""
    unsafe_destroy!(ctx::CuContext)

Immediately destroy a context, freeing up all resources associated with it. This does not
respect any users of the context, and might make other objects unusable.
"""
function unsafe_destroy!(ctx::CuContext)
    if isvalid(ctx)
        cuCtxDestroy_v2(ctx)
        invalidate!(ctx)
    end
end

Base.unsafe_convert(::Type{CUcontext}, ctx::CuContext) = ctx.handle

# NOTE: we don't implement `isequal` or `hash` in order to fall back to `===` and `objectid`
#       as contexts are unique, and with primary device contexts identical handles might be
#       returned after resetting the context (device) and all associated resources.

function Base.show(io::IO, ctx::CuContext)
    if ctx.handle != C_NULL
        fields = [@sprintf("%p", ctx.handle), @sprintf("instance %x", objectid(ctx))]
        if !isvalid(ctx)
            push!(fields, "invalidated")
        end

        print(io, "CuContext(", join(fields, ", "), ")")
    else
        print(io, "CuContext(NULL)")
    end
end


## core context API

"""
    push!(CuContext, ctx::CuContext)

Pushes a context on the current CPU thread.
"""
Base.push!(::Type{CuContext}, ctx::CuContext) = cuCtxPushCurrent_v2(ctx)

"""
    pop!(CuContext)

Pops the current CUDA context from the current CPU thread.
"""
function Base.pop!(::Type{CuContext})
    handle_ref = Ref{CUcontext}()
    cuCtxPopCurrent_v2(handle_ref)
    # we don't return the context here, because it may be unused
    # (and constructing the unique object is expensive)
end

"""
    activate(ctx::CuContext)

Binds the specified CUDA context to the calling CPU thread.
"""
activate(ctx::CuContext) = cuCtxSetCurrent(ctx)

function CuContext(f::Function, dev::CuDevice, args...)
    ctx = CuContext(dev, args...)    # implicitly pushes
    try
        f(ctx)
    finally
        pop!(CuContext)
        unsafe_destroy!(ctx)
    end
end


## primary context management

"""
    CUDA.unsafe_release!(ctx::CuContext)

Lower the refcount of a context, possibly freeing up all resources associated with it. This
does not respect any users of the context, and might make other objects unusable.
"""
function unsafe_release!(ctx::CuContext)
    if isvalid(ctx)
        dev = device(ctx)
        pctx = CuPrimaryContext(dev)
        if version() >= v"11"
            cuDevicePrimaryCtxRelease_v2(dev)
        else
            cuDevicePrimaryCtxRelease(dev)
        end
        isactive(pctx) || invalidate!(ctx)
    end
    return
end

function CuContext(f::Function, pctx::CuPrimaryContext)
    ctx = CuContext(pctx)
    try
        f(ctx)
    finally
        unsafe_release!(ctx)
    end
end

"""
    unsafe_reset!(pctx::CuPrimaryContext)

Explicitly destroys and cleans up all resources associated with a device's primary context
in the current process. Note that this forcibly invalidates all contexts derived from this
primary context, and as a result outstanding resources might become invalid.
"""
function unsafe_reset!(pctx::CuPrimaryContext)
    ctx = CuContext(pctx)
    invalidate!(ctx)
    if version() >= v"11"
        cuDevicePrimaryCtxReset_v2(pctx.dev)
    else
        cuDevicePrimaryCtxReset(pctx.dev)
    end
    return
end

function state(pctx::CuPrimaryContext)
    flags = Ref{Cuint}()
    active = Ref{Cint}()
    cuDevicePrimaryCtxGetState(pctx.dev, flags, active)
    return (flags[], active[] == one(Cint))
end

"""
    isactive(pctx::CuPrimaryContext)

Query whether a primary context is active.
"""
isactive(pctx::CuPrimaryContext) = state(pctx)[2]

"""
    flags(pctx::CuPrimaryContext)

Query the flags of a primary context.
"""
flags(pctx::CuPrimaryContext) = state(pctx)[1]

"""
    setflags!(pctx::CuPrimaryContext)

Set the flags of a primary context.
"""
function setflags!(pctx::CuPrimaryContext, flags)
    if version() >= v"11"
        cuDevicePrimaryCtxSetFlags_v2(pctx.dev, flags)
    else
        cuDevicePrimaryCtxSetFlags(pctx.dev, flags)
    end
end


## context properties

"""
    device(::CuContext)

Returns the device for a context.
"""
function device(ctx::CuContext)
    push!(CuContext, ctx)
    dev = current_device()
    pop!(CuContext)
    return dev
end

"""
    synchronize(ctx::Context)

Block for the all operations on `ctx` to complete. This is a heavyweight operation,
typically you only need to call [`synchronize`](@ref) which only synchronizes the stream
associated with the current task.
"""
function synchronize(ctx::CuContext)
    push!(CuContext, ctx)
    try
        nonblocking_synchronize()
    finally
        pop!(CuContext)
    end
end

# same, but without the context switch
"""
    device_synchronize()

Block for the all operations on `ctx` to complete. This is a heavyweight operation,
typically you only need to call [`synchronize`](@ref) which only synchronizes the stream
associated with the current task.

On the device, `device_synchronize` acts as a synchronization point for child grids in the
context of dynamic parallelism.
"""
device_synchronize() = nonblocking_synchronize()
# XXX: can we put the device docstring in dynamic_parallelism.jl?

@inline function nonblocking_synchronize()
    # perform as much of the sync as possible without blocking in CUDA.
    # XXX: remove this using a yield callback, or by synchronizing on a dedicated thread?
    nonblocking_synchronize(legacy_stream())

    # even though the GPU should be idle now, CUDA hooks work to the actual API call.
    # see NVIDIA bug #3383169 for more details.
    cuCtxSynchronize()

    check_exceptions()
end


## cache config

export cache_config, cache_config!

@enum_without_prefix CUfunc_cache CU_

function cache_config()
    config = Ref{CUfunc_cache}()
    cuCtxGetCacheConfig(config)
    return config[]
end

function cache_config!(config::CUfunc_cache)
    cuCtxSetCacheConfig(config)
end


## shared memory config

export shmem_config, shmem_config!

@enum_without_prefix CUsharedconfig CU_

function shmem_config()
    config = Ref{CUsharedconfig}()
    cuCtxGetSharedMemConfig(config)
    return config[]
end

function shmem_config!(config::CUsharedconfig)
    cuCtxSetSharedMemConfig(config)
end


## limits

export limit, limit!

@enum_without_prefix CUlimit CU_

function limit(lim::CUlimit)
    val = Ref{Csize_t}()
    cuCtxGetLimit(val, lim)
    return Int(val[])
end

limit!(lim::CUlimit, val) = cuCtxSetLimit(lim, val)


## p2p

export enable_peer_access, disable_peer_access

enable_peer_access(peer::CuContext, flags=0) =
    cuCtxEnablePeerAccess(peer, flags)

disable_peer_access(peer::CuContext) = cuCtxDisablePeerAccess(peer)
