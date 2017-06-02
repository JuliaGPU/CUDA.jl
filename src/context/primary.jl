# Primary context management
#
# This is meant for interoperability with the CUDA runtime API

export
    CuPrimaryContext, isactive, flags, setflags!

"""
    CuPrimaryContext(dev::Int)

Create a primary CUDA context for a given device.

Each primary context is unique per device and is shared with CUDA runtime API. It is meant
for interoperability with (applications using) the runtime API.
"""
type CuPrimaryContext
    dev::Int
end

"""
    CuContext(pctx::CuPrimaryContext)
    CuContext(f::Function, pctx::CuPrimaryContext)

Retain the primary context on the GPU, returning a context compatible with the driver API.
The primary context will be released when the returned driver context is finalized. For that
reason, it is advised to use this function with `do` block syntax.
"""
function CuContext(pctx::CuPrimaryContext)
    handle = Ref{CuContext_t}()
    @apicall(:cuDevicePrimaryCtxRetain, (Ptr{CuContext_t}, CuDevice_t,), handle, pctx.dev)
    ctx = CuContext(handle[], false)    # CuContext shouldn't destroy this ctx
    finalizer(ctx, (ctx)->begin
        isvalid(ctx) && @apicall(:cuDevicePrimaryCtxRelease, (CuDevice_t,), pctx.dev)
    end)
    return ctx
end

function state(pctx::CuPrimaryContext)
    flags = Ref{Cuint}()
    active = Ref{Cint}()
    @apicall(:cuDevicePrimaryCtxGetState, (CuDevice_t, Ptr{Cuint}, Ptr{Cint}),
             pctx.dev, flags, active)
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
flags(pctx::CuPrimaryContext) = CUctx_flags(state(pctx)[1])

"""
    setflags!(pctx::CuPrimaryContext)

Set the flags of a primary context.
"""
setflags!(pctx::CuPrimaryContext, flags::CUctx_flags) =
    @apicall(:cuDevicePrimaryCtxSetFlags, (CuDevice_t, Cuint), pctx.dev, flags)
