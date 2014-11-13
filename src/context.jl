# CUDA CuContext

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
            pctx_box = ptrbox(Ptr{Void})
            @cucall(:cuCtxCreate, (Ptr{Ptr{Void}}, Cuint, Cint),
                                  pctx_box, flags, dev.handle)
            new(ptrunbox(pctx_box))
    end

    CuContext(dev::CuDevice) = CuContext(dev, 0)

    function CuContext()
            pctx_box = ptrbox(Ptr{Void})
            @cucall(:cuCtxGetCurrent, (Ptr{Ptr{Void}},), pctx_box)
            new(ptrunbox(pctx_box))
    end
end

function destroy(ctx::CuContext)
    @cucall(:cuCtxDestroy, (Ptr{Void},), ctx.handle)
end

function push(ctx::CuContext)
    @cucall(:cuCtxPushCurrent, (Ptr{Void},), ctx.handle)
end

function pop(ctx::CuContext)
    pctx_box = ptrbox(Ptr{Void})
    @cucall(:cuCtxPopCurrent, (Ptr{Ptr{Void}},), a)
    return CuContext(ptrunbox(pctx_box))
end

function device(ctx::CuContext)
    # TODO: cuCtxGetDevice returns the device ordinal, but as a CUDevice*?
    #       This can't be right...
    device_box = ptrbox(Cint)
    @cucall(:cuCtxGetDevice, (Ptr{Cint},), device_box)
    return CuDevice(ptrunbox(device_box))
end
