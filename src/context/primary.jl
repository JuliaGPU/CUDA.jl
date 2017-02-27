# Primary context management
#
# This is meant for interoperability with the CUDA runtime API

export
    CuPrimaryContext, destroy, reset, state, isactive, flags, setflags!

type CuPrimaryContext
    dev::Int
end

function CuContext(pctx::CuPrimaryContext)
    handle = Ref{CuContext_t}()
    @apicall(:cuDevicePrimaryCtxRetain, (Ptr{CuContext_t}, CuDevice_t,), handle, pctx.dev)
    ctx = CuContext(handle[], false)    # CuContext shouldn't destroy this ctx
    finalizer(ctx, (ctx)->begin
        @apicall(:cuDevicePrimaryCtxRelease, (CuDevice_t,), pctx.dev)
    end)
    return ctx
end

# TODO: query the underlying, active context and check with `can_finalize`
#       if there's any outstanding references

reset(pctx::CuPrimaryContext) =
    @apicall(:cuDevicePrimaryCtxReset, (CuDevice_t,), pctx.dev)

function state(pctx)
    flags = Ref{Cuint}()
    active = Ref{Cint}()
    @apicall(:cuDevicePrimaryCtxGetState, (CuDevice_t, Ptr{Cuint}, Ptr{Cint}),
             pctx.dev, flags, active)
    return (flags[], active[] == one(Cint))
end

isactive(pctx::CuPrimaryContext) = state(pctx)[2]

flags(pctx::CuPrimaryContext) = CUctx_flags(state(pctx)[1])

setflags!(pctx::CuPrimaryContext, flags::CUctx_flags) =
    @apicall(:cuDevicePrimaryCtxSetFlags, (CuDevice_t, Cuint), pctx.dev, flags)
