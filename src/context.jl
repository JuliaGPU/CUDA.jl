# Context management

import Base: unsafe_convert, @deprecate_binding, ==

export
    CuContext, destroy, current_context,
    push, pop, synchronize, device

@enum(CUctx_flags, SCHED_AUTO           = 0x00,
                   SCHED_SPIN           = 0x01,
                   SCHED_YIELD          = 0x02,
                   SCHED_BLOCKING_SYNC  = 0x04,
                   MAP_HOST             = 0x08,
                   LMEM_RESIZE_TO_MAX   = 0x10)
@deprecate_binding BLOCKING_SYNC SCHED_BLOCKING_SYNC

typealias CuContext_t Ptr{Void}

"""
Create a CUDA context for device. A context on the GPU is analogous to a process on the
CPU, with its own distinct address space and allocated resources. When a context is
destroyed, the system cleans up the resources allocated to it.
"""
immutable CuContext
    handle::CuContext_t
end

unsafe_convert(::Type{CuContext_t}, ctx::CuContext) = ctx.handle

==(a::CuContext, b::CuContext) = a.handle == b.handle

function CuContext(dev::CuDevice, flags::CUctx_flags)
    handle_ref = Ref{CuContext_t}()
    @apicall(:cuCtxCreate, (Ptr{CuContext_t}, Cuint, Cint),
                           handle_ref, flags, dev)
    CuContext(handle_ref[])
end

CuContext(dev::CuDevice) = CuContext(dev, SCHED_AUTO)

"Destroy the CUDA context, releasing resources allocated to it."
function destroy(ctx::CuContext)
    @apicall(:cuCtxDestroy, (CuContext_t,), ctx)
end

function current_context()
    handle_ref = Ref{CuContext_t}()
    @apicall(:cuCtxGetCurrent, (Ptr{CuContext_t},), handle_ref)
    CuContext(handle_ref[])
end

function push(ctx::CuContext)
    @apicall(:cuCtxPushCurrent, (CuContext_t,), ctx)
end

function pop()
    handle_ref = Ref{CuContext_t}()
    @apicall(:cuCtxPopCurrent, (Ptr{CuContext_t},), handle_ref)
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

synchronize(ctx::CuContext) = @apicall(:cuCtxSynchronize, (CuContext_t,), ctx)
