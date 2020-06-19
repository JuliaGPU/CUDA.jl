# Primary context management
#
# This is meant for interoperability with the CUDA runtime API

export
    CuPrimaryContext, unsafe_reset!, isactive, flags, setflags!

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

Retain the primary context on the GPU, returning a context compatible with the driver API.
The primary context will be released when the returned driver context is finalized.

As these contexts are refcounted by CUDA, you should not call [`CUDA.unsafe_destroy!`](@ref) on
them but use [`CUDA.unsafe_release!`](@ref) instead (available with do-block syntax as well).
"""
function CuContext(pctx::CuPrimaryContext)
    handle = Ref{CUcontext}()
    cuDevicePrimaryCtxRetain(handle, pctx.dev)
    return CuContext(handle[])
end

"""
    CUDA.unsafe_release!(ctx::CuContext)

Lower the refcount of a context, possibly freeing up all resources associated with it. This
does not respect any users of the context, and might make other objects unusable.
"""
function unsafe_release!(ctx::CuContext)
    if isvalid(ctx)
        dev = device(ctx)
        pctx = CuPrimaryContext(dev)
        cuDevicePrimaryCtxRelease(dev)
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
    cuDevicePrimaryCtxReset(pctx.dev)
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
setflags!(pctx::CuPrimaryContext, flags) = cuDevicePrimaryCtxSetFlags(pctx.dev, flags)
