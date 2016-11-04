# Context management

export
    CuContext, CuCurrentContext,
    synchronize, device

@enum(CUctx_flags, SCHED_AUTO           = 0x00,
                   SCHED_SPIN           = 0x01,
                   SCHED_YIELD          = 0x02,
                   SCHED_BLOCKING_SYNC  = 0x04,
                   MAP_HOST             = 0x08,
                   LMEM_RESIZE_TO_MAX   = 0x10)
Base.@deprecate_binding BLOCKING_SYNC SCHED_BLOCKING_SYNC

typealias CuContext_t Ptr{Void}


## construction and destruction

"""
Create a CUDA context for device. A context on the GPU is analogous to a process on the
CPU, with its own distinct address space and allocated resources. When a context is
destroyed, the system cleans up the resources allocated to it.

Contexts are reference counted based on the underlying handle. If all outstanding references
to a context have been dropped, and the reference count has consequently dropped to 0, the
context will be destroyed.
"""
type CuContext
    handle::CuContext_t

    function CuContext(handle::CuContext_t)
        obj = new(handle)
        if handle != C_NULL
            instances = get(context_instances, handle, 0)
            context_instances[handle] = instances+1
            finalizer(obj, finalize)
        end
        obj
    end
end
const context_instances = Dict{CuContext_t,Int}()

Base.unsafe_convert(::Type{CuContext_t}, ctx::CuContext) = ctx.handle
Base.:(==)(a::CuContext, b::CuContext) = a.handle == b.handle

function finalize(ctx::CuContext)
    instances = context_instances[ctx.handle]
    context_instances[ctx.handle] = instances-1

    if instances == 1
        delete!(context_instances, ctx.handle)
        @apicall(:cuCtxDestroy, (CuContext_t,), ctx)
    end
end

Base.deepcopy_internal(::CuContext, ::ObjectIdDict) =
    throw(ArgumentError("CuContext cannot be copied"))

# finalizers are run out-of-order (see JuliaLang/julia#3067), so we need to keep it alive as
# long as there's any consumer to prevent the context getting finalized ahead of consumers
const context_consumers = Set{Pair{Ptr{Void},CuContext}}()
function track(ctx::CuContext, consumer::ANY)
    push!(context_consumers, Pair(Base.pointer_from_objref(consumer),ctx))
end
function untrack(ctx::CuContext, consumer::ANY)
    delete!(context_consumers, Pair(Base.pointer_from_objref(consumer),ctx))
end

function CuContext(dev::CuDevice, flags::CUctx_flags=SCHED_AUTO)
    handle_ref = Ref{CuContext_t}()
    @apicall(:cuCtxCreate, (Ptr{CuContext_t}, Cuint, Cint),
                           handle_ref, flags, dev)
    CuContext(handle_ref[])
end

"Return the current context."
function CuCurrentContext()
    handle_ref = Ref{CuContext_t}()
    @apicall(:cuCtxGetCurrent, (Ptr{CuContext_t},), handle_ref)
    CuContext(handle_ref[])
end

activate(ctx::CuContext) = @apicall(:cuCtxSetCurrent, (CuContext_t,), ctx)

"Create a context, and activate it temporarily."
function CuContext(f::Function, args...)
    ctx = CuContext(args...)    # implicitly pushes
    try
        f(ctx)
    finally
        handle_ref = Ref{CuContext_t}()
        @apicall(:cuCtxPopCurrent, (Ptr{CuContext_t},), handle_ref[])
    end
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
