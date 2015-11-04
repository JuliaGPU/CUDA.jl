# Context management

export
    CuContext, destroy, push, pop


const CTX_SCHED_AUTO            = 0x00
const CTX_SCHED_SPIN            = 0x01
const CTX_SCHED_YIELD           = 0x02
const CTX_SCHED_BLOCKING_SYNC   = 0x04
const CTX_MAP_HOST              = 0x08
const CTX_LMEM_RESIZE_TO_MAX    = 0x10

immutable CuContext
    handle::Ptr{Void}

    function CuContext(dev::CuDevice, flags::Integer)
            pctx_ref = Ref{Ptr{Void}}()
            @cucall(:cuCtxCreate, (Ptr{Ptr{Void}}, Cuint, Cint),
                                  pctx_ref, flags, dev.handle)
            new(pctx_ref[])
    end

    CuContext(dev::CuDevice) = CuContext(dev, 0)

    function CuContext()
            pctx_ref = Ref{Ptr{Void}}()
            @cucall(:cuCtxGetCurrent, (Ptr{Ptr{Void}},), pctx_ref)
            new(pctx_ref[])
    end
end

function destroy(ctx::CuContext)
    @cucall(:cuCtxDestroy, (Ptr{Void},), ctx.handle)
end

function push(ctx::CuContext)
    @cucall(:cuCtxPushCurrent, (Ptr{Void},), ctx.handle)
end

function pop(ctx::CuContext)
    pctx_ref = Ref{Ptr{Void}}()
    @cucall(:cuCtxPopCurrent, (Ptr{Ptr{Void}},), pctx_ref)
    return CuContext(pctx_ref[])
end

function device(ctx::CuContext)
    @assert CuContext() == ctx
    # TODO: cuCtxGetDevice returns the device ordinal, but as a CUDevice*?
    #       This can't be right...
    device_ref = Ref{Cint}()
    @cucall(:cuCtxGetDevice, (Ptr{Cint},), device_ref)
    return CuDevice(device_ref[])
end

synchronize(ctx::CuContext) = @cucall(:cuCtxSynchronize, (Ptr{Void},), ctx.handle)
