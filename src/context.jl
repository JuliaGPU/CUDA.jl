# Context management

export
    CuContext, destroy, current_context,
    push, pop, synchronize, device

@enum(CUctx_flags, CTX_SCHED_AUTO           = 0x00,
                   CTX_SCHED_SPIN           = 0x01,
                   CTX_SCHED_YIELD          = 0x02,
                   CTX_SCHED_BLOCKING_SYNC  = 0x04,
                   CTX_MAP_HOST             = 0x08,
                   CTX_LMEM_RESIZE_TO_MAX   = 0x10)

"""
Create a CUDA context for device. A context on the GPU is analogous to a process on the
CPU, with its own distinct address space and allocated resources. When a context is
destroyed, the system cleans up the resources allocated to it.
"""
immutable CuContext
    handle::Ptr{Void}
end

function CuContext(dev::CuDevice, flags::Integer)
    pctx_ref = Ref{Ptr{Void}}()
    @apicall(:cuCtxCreate, (Ptr{Ptr{Void}}, Cuint, Cint),
                          pctx_ref, flags, dev.handle)
    CuContext(pctx_ref[])
end

CuContext(dev::CuDevice) = CuContext(dev, 0)

"Destroy the CUDA context, releasing resources allocated to it."
function destroy(ctx::CuContext)
    @apicall(:cuCtxDestroy, (Ptr{Void},), ctx.handle)
end

function current_context()
    pctx_ref = Ref{Ptr{Void}}()
    @apicall(:cuCtxGetCurrent, (Ptr{Ptr{Void}},), pctx_ref)
    CuContext(pctx_ref[])
end

function push(ctx::CuContext)
    @apicall(:cuCtxPushCurrent, (Ptr{Void},), ctx.handle)
end

function pop()
    pctx_ref = Ref{Ptr{Void}}()
    @apicall(:cuCtxPopCurrent, (Ptr{Ptr{Void}},), pctx_ref)
    return nothing
end

function device(ctx::CuContext)
    if current_context() != ctx
        # TODO: we could push and pop here
        throw(ArgumentError("context should be active"))
    end

    # TODO: cuCtxGetDevice returns the device ordinal, but as a CUDevice*?
    #       This can't be right...
    device_ref = Ref{Cint}()
    @apicall(:cuCtxGetDevice, (Ptr{Cint},), device_ref)
    return CuDevice(device_ref[])
end

synchronize(ctx::CuContext) = @apicall(:cuCtxSynchronize, (Ptr{Void},), ctx.handle)
