# Context management

export
    CuContext, destroy, CuCurrentContext, activate,
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

Contexts are unique instances which need to be `destroy`ed after use. For automatic
management, prefer the `do` block syntax, which implicitly calls `destroy`.
"""
type CuContext
    handle::CuContext_t

    function CuContext(handle::CuContext_t)
        handle == C_NULL && return new(C_NULL)

        # we need unique context instances for garbage collection reasons
        #
        # refcounting the context handle doesn't work, because having multiple instances can
        # cause the first instance (if without any use) getting gc'd before a new instance
        # (eg. through getting the current context, _with_ actual uses) is created
        #
        # instead, we force unique instances, and keep a reference alive in a global dict.
        # this prevents contexts from getting collected, requiring the user to destroy it.
        return get!(context_instances, handle) do
            new(handle)
        end
    end
end
const context_instances = Dict{CuContext_t,CuContext}()

"""
Mark a context for destruction.

This does not immediately destroy the context, as there might still be dependent resources
which have not been collected yet.  It will get freed as soon as all outstanding instances
have gone out-of-scope.
"""
function destroy(ctx::CuContext)
    @static if DEBUG
        children = length(gc_children(ctx))
        if children > 0
            debug("Request to destroy context $ctx while there are still $children remaining consumers")
        end
    end
    delete!(context_instances, ctx.handle)
    ctx = CuContext(C_NULL)
    return
end

Base.unsafe_convert(::Type{CuContext_t}, ctx::CuContext) = ctx.handle
Base.:(==)(a::CuContext, b::CuContext) = a.handle == b.handle

function finalize(ctx::CuContext)
    # during teardown, finalizer order does not respect _any_ order
    # (ie. it doesn't respect active instances carefully set-up in `gc_keepalive`)
    # TODO: can we check this only happens during teardown?
    children = length(gc_children(ctx))
    if children == 0
        @apicall(:cuCtxDestroy, (CuContext_t,), ctx)
    else
        trace("Not destroying context $ctx because of out-of-order finalizer run")
    end
end

Base.deepcopy_internal(::CuContext, ::ObjectIdDict) =
    error("CuContext cannot be copied")

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
    # NOTE: this could be implemented with context pushing and popping,
    #       but that functionality / our implementation of it hasn't been reliable
    old_ctx = CuCurrentContext()
    ctx = CuContext(args...)    # implicitly activates
    try
        f(ctx)
    finally
        destroy(ctx)
        activate(old_ctx)
    end
end


## context properties

function device(ctx::CuContext)
    if CuCurrentContext() != ctx
        # TODO: should we push and pop here?
        error("context should be active")
    end

    # TODO: cuCtxGetDevice returns the device ordinal, but as a CUDevice*?
    #       This can't be right...
    device_ref = Ref{Cint}()
    @apicall(:cuCtxGetDevice, (Ptr{Cint},), device_ref)
    return CuDevice(device_ref[])
end

synchronize(ctx::CuContext=CuCurrentContext()) =
    @apicall(:cuCtxSynchronize, (CuContext_t,), ctx)
