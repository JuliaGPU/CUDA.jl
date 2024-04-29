# Context management

export
    CuPrimaryContext, CuContext, current_context, has_context, activate,
    unsafe_reset!, isactive, flags, setflags!, unique_id, api_version,
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
struct CuContext
    handle::CUcontext
    id::UInt64

    function CuContext(handle::CUcontext)
        handle == C_NULL && throw(UndefRefError())

        id = if driver_version() >= v"12"
            id_ref = Ref{Culonglong}()
            res = unchecked_cuCtxGetId(handle, id_ref)
            res == ERROR_CONTEXT_IS_DESTROYED && throw(UndefRefError())
            res != SUCCESS && throw_api_error(res)
            id_ref[]
        else
            typemax(UInt64)
        end

        new(handle, id)
    end
end

function CuContext(dev::CuDevice, flags=0)
    handle_ref = Ref{CUcontext}()
    cuCtxCreate_v2(handle_ref, flags, dev)
    CuContext(handle_ref[])
end

"""
    current_context()

Returns the current context. Throws an undefined reference error if the current thread
has no context bound to it, or if the bound context has been destroyed.

!!! warning

    This is a low-level API, returning the current context as known to the CUDA driver.
    For most users, it is recommended to use the [`context`](@ref) method instead.
"""
function current_context()
    handle_ref = Ref{CUcontext}()
    cuCtxGetCurrent(handle_ref)
    handle_ref[] == C_NULL && throw(UndefRefError())
    CuContext(handle_ref[])
end

function isvalid(ctx::CuContext)
    # we first try an API call to see if the context handle is usable
    if driver_version() >= v"12"
        id_ref = Ref{Culonglong}()
        res = unchecked_cuCtxGetId(ctx, id_ref)
        res == ERROR_CONTEXT_IS_DESTROYED && return false
        res != SUCCESS && throw_api_error(res)

        # detect handle reuse, which happens when destroying and re-creating a context, by
        # looking at the context's unique ID (which does change on re-creation)
        return ctx.id == id_ref[]
    else
        version_ref = Ref{Cuint}()
        res = unchecked_cuCtxGetApiVersion(ctx, version_ref)
        res == ERROR_INVALID_CONTEXT && return false

        # we can't detect handle reuse, so we just assume the context is valid
        return true
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
    end
end

Base.unsafe_convert(::Type{CUcontext}, ctx::CuContext) = ctx.handle

Base.show(io::IO, ctx::CuContext) = @printf io "CuContext(%p)" ctx.handle

function Base.show(io::IO, ::MIME"text/plain", ctx::CuContext)
    fields = [@sprintf("%p", ctx.handle)]
    if driver_version() >= v"12"
        push!(fields, "id=$(ctx.id)")
    end
    if !isvalid(ctx)
        push!(fields, "destroyed")
    end

    print(io, "CuContext(", join(fields, ", "), ")")
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

function unique_id(ctx::CuContext)
    id_ref = Ref{Culonglong}()
    cuCtxGetId(ctx, id_ref)
    return id_ref[]
end

function api_version(ctx::CuContext)
    version = Ref{Cuint}()
    cuCtxGetApiVersion(ctx, version)
    return version[]
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
    CuContext(handle_ref[])
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
        cuDevicePrimaryCtxRelease_v2(pctx.dev)
    else
        cuDevicePrimaryCtxRelease(pctx.dev)
    end

    return
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

# matrix of set-up peer accesses:
# - -1: unsupported
# -  0: not set-up yet
# -  1: supported
const peer_access = Ref{Matrix{Int}}()
function maybe_enable_peer_access(src::CuDevice, dst::CuDevice)
    global peer_access

    src_idx = deviceid(src)+1
    dst_idx = deviceid(dst)+1

    if !isassigned(peer_access)
        peer_access[] = Base.zeros(Int8, ndevices(), ndevices())
    end

    # we need to take care only to enable P2P access when it is supported,
    # as well as not to call this function multiple times, to avoid errors.
    if peer_access[][src_idx, dst_idx] == 0
        if can_access_peer(src, dst)
            device!(src) do
                try
                    enable_peer_access(context(dst))
                    peer_access[][src_idx, dst_idx] = 1
                catch err
                    @warn "Enabling peer-to-peer access between $src and $dst failed; please file an issue." exception=(err,catch_backtrace())
                    peer_access[][src_idx, dst_idx] = -1
                end
            end
        else
            peer_access[][src_idx, dst_idx] = -1
        end
    end

    return peer_access[][src_idx, dst_idx]
end
