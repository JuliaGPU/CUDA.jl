# Context management

export
    CuPrimaryContext, CuContext, current_context, has_context, activate,
    unsafe_reset!, isactive, flags, setflags!,
    device, device_synchronize


## construction and destruction

@enum_without_prefix CUctx_flags CU_

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
    device::CuDevice
    valid::Bool

    function CuContext(dev::CuDevice, flags=0)
        handle_ref = Ref{CUcontext}()
        cuCtxCreate_v2(handle_ref, flags, dev)
        UniqueCuContext(handle_ref[])
    end

    global function current_context()
        handle_ref = Ref{CUcontext}()
        cuCtxGetCurrent(handle_ref)
        handle_ref[] == C_NULL && throw(UndefRefError())
        UniqueCuContext(handle_ref[])
    end

    global function UnsafeCuContext(handle::CUcontext)
        # because of context uniqueing, this function is called rarely,
        # so it's OK to look up the device by temporarily activating the context
        cuCtxPushCurrent_v2(handle)
        dev = try
            current_device()
        finally
            cuCtxPopCurrent_v2(Ref{CUcontext}())
        end

        new(handle, dev, true)
    end

    unsafe
end

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

# we need to know when a context has been destroyed, to make sure we don't destroy resources
# after the owning context has been destroyed already. this is complicated by the fact that
# contexts obtained from a primary context have the same handle before and after primary
# context destruction, so we cannot use a simple mapping from context handle to a validity
# bit. instead, we unique the context objects and put a validity bit in there.
isvalid(ctx::CuContext) = ctx.valid
function invalidate!(ctx::CuContext)
    ctx.valid = false
    return
end
# to make this work, every function returning a context (e.g. `cuCtxGetCurrent`, attribute
# functions, etc) need to return the same context objects. because looking up a context is a
# very common operation (often executed from finalizers), we need to ensure this look-up is
# fast and does not switch tasks. we do this by scanning a simple linear vector.
const MAX_CONTEXTS = 1024
const context_objects = Vector{CuContext}(undef, MAX_CONTEXTS)
const context_lock = Base.ThreadSynchronizer()
function UniqueCuContext(handle::CUcontext)
    @assert handle != C_NULL
    @lock context_lock begin
        # look if there's an existing object for this handle
        i = 1
        @inbounds while i <= MAX_CONTEXTS && isassigned(context_objects, i)
            if context_objects[i].handle == handle
                if isvalid(context_objects[i])
                    return context_objects[i]
                else
                    # this object was invalidated, so we can reuse its slot
                    break
                end
            end
            i += 1
        end
        if i == MAX_CONTEXTS
            error("Exceeded maximum amount of CUDA contexts. This is unexpected; please file an issue.")
        end

        # we've got a slot we can write to
        new_object = UnsafeCuContext(handle)
        @inbounds context_objects[i] = new_object
        return new_object
    end
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
        fields = [@sprintf("%p", ctx.handle),
                  "$(ctx.device)",
                  @sprintf("instance %x", objectid(ctx))]
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
    CuPrimaryContext(dev::CuDevice)

Create a primary CUDA context for a given device.

Each primary context is unique per device and is shared with CUDA runtime API. It is meant
for interoperability with (applications using) the runtime API.
"""
struct CuPrimaryContext
    dev::CuDevice
end

# we need to keep track of contexts derived from primary contexts,
# so that we can invalidate them when the primary context is reset.
const derived_contexts = Dict{CuPrimaryContext,CuContext}()
const derived_lock = ReentrantLock()

"""
    CuContext(pctx::CuPrimaryContext)

Derive a context from a primary context.

Calling this function increases the reference count of the primary context. The returned
context *should not* be free with the `unsafe_destroy!` function that's used with ordinary
contexts. Instead, the refcount of the primary context should be decreased by calling
`unsafe_release!`, or set to zero by calling `unsafe_reset!`. The easiest way to do this is
by using the `do`-block syntax.
"""
function CuContext(pctx::CuPrimaryContext)
    handle_ref = Ref{CUcontext}()
    cuDevicePrimaryCtxRetain(handle_ref, pctx.dev)
    ctx = UniqueCuContext(handle_ref[])
    Base.@lock derived_lock derived_contexts[pctx] = ctx
    return ctx
end

function CuContext(f::Function, pctx::CuPrimaryContext)
    ctx = CuContext(pctx)
    try
        f(ctx)
    finally
        unsafe_release!(pctx)
    end
end

"""
    CUDA.unsafe_release!(pctx::CuPrimaryContext)

Lower the refcount of a context, possibly freeing up all resources associated with it. This
does not respect any users of the context, and might make other objects unusable.
"""
function unsafe_release!(pctx::CuPrimaryContext)
    if driver_version() >= v"11"
        cuDevicePrimaryCtxRelease_v2(dev)
    else
        cuDevicePrimaryCtxRelease(dev)
    end

    # if this releases the last reference, invalidate all derived contexts
    if !isactive(pctx)
        ctx = @lock derived_lock get(derived_contexts, pctx, nothing)
        if ctx !== nothing
            invalidate!(ctx)
        end
    end
end

"""
    unsafe_reset!(pctx::CuPrimaryContext)

Explicitly destroys and cleans up all resources associated with a device's primary context
in the current process. Note that this forcibly invalidates all contexts derived from this
primary context, and as a result outstanding resources might become invalid.
"""
function unsafe_reset!(pctx::CuPrimaryContext)
    if driver_version() >= v"11"
        cuDevicePrimaryCtxReset_v2(pctx.dev)
    else
        cuDevicePrimaryCtxReset(pctx.dev)
    end

    # invalidate all derived contexts
    ctx = @lock derived_lock get(derived_contexts, pctx, nothing)
    if ctx !== nothing
        invalidate!(ctx)
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
    if driver_version() >= v"11"
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
    # cuCtxGetDevice assumes the context is current, so instead we store the device
    # in the context object
    return ctx.device
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
        device_synchronize()
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
device_synchronize()
# XXX: can we put the device docstring in dynamic_parallelism.jl?


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
