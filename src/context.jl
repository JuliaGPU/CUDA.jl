# Context management

export
    CuContext, destroy, push, pop, synchronize

@enum(CUctx_flags, CTX_SCHED_AUTO           = 0x00,
                   CTX_SCHED_SPIN           = 0x01,
                   CTX_SCHED_YIELD          = 0x02,
                   CTX_SCHED_BLOCKING_SYNC  = 0x04,
                   CTX_MAP_HOST             = 0x08,
                   CTX_LMEM_RESIZE_TO_MAX   = 0x10)

immutable CuContext
    handle::Ptr{Void}

    """
    Create a CUDA context for device. A context on the GPU is analogous to a process on the
    CPU, with its own distinct address space and allocated resources. When a context is
    destroyed, the system cleans up the resources allocated to it.
    """
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

"Destroy the CUDA context, releasing resources allocated to it."
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
