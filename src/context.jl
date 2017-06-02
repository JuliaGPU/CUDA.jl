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

const CuContext_t = Ptr{Void}


## construction and destruction

"""
    CuContext(dev::CuDevice, flags::CUctx_flags=SCHED_AUTO)
    CuContext(f::Function, ...)

Create a CUDA context for device. A context on the GPU is analogous to a process on the CPU,
with its own distinct address space and allocated resources. When a context is destroyed,
the system cleans up the resources allocated to it.

Contexts are unique instances which need to be `destroy`ed after use. For automatic
management, prefer the `do` block syntax, which implicitly calls `destroy`.

"""
type CuContext
    handle::CuContext_t
    owned::Bool
    valid::Bool

    """
    The `owned` argument indicates whether the caller owns this context. If so, the context
    should get destroyed when it goes out of scope. If not, it is up to the caller to do so.
    """
    function CuContext(handle::CuContext_t, owned=true)
        handle == C_NULL && return new(C_NULL, false, true)

        # we need unique context instances for garbage collection reasons
        #
        # refcounting the context handle doesn't work, because having multiple instances can
        # cause the first instance (if without any use) getting gc'd before a new instance
        # (eg. through getting the current context, _with_ actual uses) is created
        #
        # instead, we force unique instances, and keep a reference alive in a global dict.
        # this prevents contexts from getting collected, requiring the user to destroy it.
        ctx = get!(context_instances, handle) do
            obj = new(handle, owned, true)
            finalizer(obj, unsafe_destroy!)
            return obj
        end

        if owned && !ctx.owned
            # trying to get a non-owned handle on an already-owned context is common,
            # eg. construct a context, and call a function doing CuCurrentContext()
            #
            # the inverse isn't true: constructing an owning-object on using a handle
            # which was the result of a non-owning API call really shouldn't happen
            warn("Ownership conflict on context $ctx")
        end

        ctx
    end
end
const context_instances = Dict{CuContext_t,CuContext}()

isvalid(ctx::CuContext) = ctx.valid
function invalidate!(ctx::CuContext)
    @trace("Invalidating CuContext at $(Base.pointer_from_objref(ctx))")
    ctx.valid = false
    nothing
end

function unsafe_destroy!(ctx::CuContext)
    @trace("Finalizing CuContext at $(Base.pointer_from_objref(ctx))")
    if !ctx.owned
        @trace("Not destroying context $ctx because we don't own it")
    elseif isvalid(ctx)
        @apicall(:cuCtxDestroy, (CuContext_t,), ctx)
        invalidate!(ctx)
    else
        # this is due to finalizers not respecting _any_ order during process teardown
        # (ie. it doesn't respect active instances carefully set-up in `gc.jl`)
        # TODO: can we check this only happens during teardown?
        @trace("Not destroying context $ctx because of out-of-order finalizer run")
    end
end

Base.unsafe_convert(::Type{CuContext_t}, ctx::CuContext) = ctx.handle

Base.:(==)(a::CuContext, b::CuContext) = a.handle == b.handle
Base.hash(ctx::CuContext, h::UInt) = hash(ctx.handle, h)

"""
    destroy(ctx::CuContext)

Mark a context for destruction.

This does not immediately destroy the context, as there might still be dependent resources
which have not been collected yet. The context will get freed as soon as all outstanding
instances have been finalized.
"""
function destroy(ctx::CuContext)
    delete!(context_instances, ctx.handle)
    return
end

Base.deepcopy_internal(::CuContext, ::ObjectIdDict) =
    error("CuContext cannot be copied")

function CuContext(dev::CuDevice, flags::CUctx_flags=SCHED_AUTO)
    handle_ref = Ref{CuContext_t}()
    @apicall(:cuCtxCreate, (Ptr{CuContext_t}, Cuint, Cint),
                           handle_ref, flags, dev)
    CuContext(handle_ref[])
end

"""
    CuCurrentContext()

Return the current context, or a NULL context if there is no active context (see
[`isnull`](@ref)).
"""
function CuCurrentContext()
    handle_ref = Ref{CuContext_t}()
    @apicall(:cuCtxGetCurrent, (Ptr{CuContext_t},), handle_ref)
    CuContext(handle_ref[], false)
end

"""
    isnull(ctx::CuContext)

Indicates whether the current context is an invalid NULL context.
"""
Base.isnull(ctx::CuContext) = (ctx.handle == C_NULL)

"""
    activate(ctx::CuContext)

Binds the specified CUDA context to the calling CPU thread.
"""
activate(ctx::CuContext) = @apicall(:cuCtxSetCurrent, (CuContext_t,), ctx)

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

"""
    device(ctx::Cucontext)

Returns the device for the current context. The `ctx` parameter is to make sure that the
current context is really active, and hence the returned device is valid.
"""
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

"""
    synchronize(ctx::CuContext=CuCurrentContext())

Block for a context's tasks to complete.

The `ctx` parameter defaults to the current active context.
"""
synchronize(ctx::CuContext=CuCurrentContext()) =
    @apicall(:cuCtxSynchronize, (CuContext_t,), ctx)
