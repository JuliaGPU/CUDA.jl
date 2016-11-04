# Context management

import Base: unsafe_convert, @deprecate_binding, ==

export
    CuContext, CuCurrentContext,
    push, pop, synchronize, device

@enum(CUctx_flags, SCHED_AUTO           = 0x00,
                   SCHED_SPIN           = 0x01,
                   SCHED_YIELD          = 0x02,
                   SCHED_BLOCKING_SYNC  = 0x04,
                   MAP_HOST             = 0x08,
                   LMEM_RESIZE_TO_MAX   = 0x10)
@deprecate_binding BLOCKING_SYNC SCHED_BLOCKING_SYNC

typealias CuContext_t Ptr{Void}


## construction and destruction

"""
Create a CUDA context for device. A context on the GPU is analogous to a process on the
CPU, with its own distinct address space and allocated resources. When a context is
destroyed, the system cleans up the resources allocated to it.
"""
type CuContext
    handle::CuContext_t

    function CuContext(handle::CuContext_t)
        obj = new(handle)
        if handle!=C_NULL
            instances = get(context_instances, handle, 0)
            context_instances[handle] = instances+1
            finalizer(obj, finalize)
        end
        obj
    end
end
const context_instances = Dict{CuContext_t,Int}()

unsafe_convert(::Type{CuContext_t}, ctx::CuContext) = ctx.handle

function finalize(ctx::CuContext)
    instances = context_instances[ctx.handle]
    context_instances[ctx.handle] = instances-1

    if instances == 1
        delete!(context_instances, ctx.handle)
        @apicall(:cuCtxDestroy, (CuContext_t,), ctx)
    end
end

# finalizers are run out-of-order (see JuliaLang/julia#3067), so we need to keep it alive as
# long as there's any consumer to prevent the context getting finalized ahead of consumers
const context_consumers = Set{Pair{Ptr{Void},CuContext}}()
function track(ctx::CuContext, consumer::ANY)
    push!(context_consumers, Pair(Base.pointer_from_objref(consumer),ctx))
end
function untrack(ctx::CuContext, consumer::ANY)
    delete!(context_consumers, Pair(Base.pointer_from_objref(consumer),ctx))
end

Base.deepcopy_internal(::CuContext, ::ObjectIdDict) =
    throw(ArgumentError("CuContext cannot be copied"))

==(a::CuContext, b::CuContext) = a.handle == b.handle

function CuContext(dev::CuDevice, flags::CUctx_flags)
    handle_ref = Ref{CuContext_t}()
    @apicall(:cuCtxCreate, (Ptr{CuContext_t}, Cuint, Cint),
                           handle_ref, flags, dev)
    CuContext(handle_ref[])
end

CuContext(dev::CuDevice) = CuContext(dev, SCHED_AUTO)

"Return the current context."
function CuCurrentContext()
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


## context properties

function device(ctx::CuContext)
    if CuCurrentContext() != ctx
        # TODO: we could push and pop here
        throw(ArgumentError("context should be active"))
    end

    # TODO: cuCtxGetDevice returns the device ordinal, but as a CUDevice*?
    #       This can't be right...
    device_ref = Ref{Cint}()
    @apicall(:cuCtxGetDevice, (Ptr{Cint},), device_ref)
    return CuDevice(device_ref[])
end

synchronize(ctx::CuContext=CuCurrentContext()) =
    @apicall(:cuCtxSynchronize, (CuContext_t,), ctx)
