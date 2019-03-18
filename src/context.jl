# Context management

export
    CuContext, destroy!, CuCurrentContext, activate,
    synchronize, device

@enum(CUctx_flags, SCHED_AUTO           = 0x00,
                   SCHED_SPIN           = 0x01,
                   SCHED_YIELD          = 0x02,
                   SCHED_BLOCKING_SYNC  = 0x04,
                   MAP_HOST             = 0x08,
                   LMEM_RESIZE_TO_MAX   = 0x10)
Base.@deprecate_binding BLOCKING_SYNC SCHED_BLOCKING_SYNC

const CuContext_t = Ptr{Cvoid}


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
mutable struct CuContext
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
            finalizer(unsafe_destroy!, obj)
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
    ctx.valid = false
    nothing
end

function unsafe_destroy!(ctx::CuContext)
    # finalizers do not respect _any_ oder during process teardown 
    # (ie. it doesn't respect active instances carefully set-up in `gc.jl`)
    # TODO: can we check this only happens during teardown?
    if ctx.owned && isvalid(ctx)
        @apicall(:cuCtxDestroy, (CuContext_t,), ctx)
        invalidate!(ctx)
    end
end

Base.unsafe_convert(::Type{CuContext_t}, ctx::CuContext) = ctx.handle

Base.:(==)(a::CuContext, b::CuContext) = a.handle == b.handle
Base.hash(ctx::CuContext, h::UInt) = hash(ctx.handle, h)

"""
    destroy!(ctx::CuContext)

Mark a context for destruction.

This does not immediately destroy the context, as there might still be dependent resources
which have not been collected yet. The context will get freed as soon as all outstanding
instances have been finalized.
"""
function destroy!(ctx::CuContext)
    delete!(context_instances, ctx.handle)
    return
end

Base.deepcopy_internal(::CuContext, ::IdDict) =
    error("CuContext cannot be copied")

function CuContext(dev::CuDevice, flags::CUctx_flags=SCHED_AUTO)
    handle_ref = Ref{CuContext_t}()
    @apicall(:cuCtxCreate, (Ptr{CuContext_t}, Cuint, Cint),
                           handle_ref, flags, dev)
    CuContext(handle_ref[])
end

"""
    CuCurrentContext()

Return the current context, or `nothing` if there is no active context.
"""
function CuCurrentContext()
    handle_ref = Ref{CuContext_t}()
    @apicall(:cuCtxGetCurrent, (Ptr{CuContext_t},), handle_ref)
    if handle_ref[] == C_NULL
        return nothing
    else
        return CuContext(handle_ref[], false)
    end
end

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
        destroy!(ctx)
        activate(old_ctx)
    end
end


## context properties

"""
    device()
    device(ctx::Cucontext)

Returns the device for the current context. The optional `ctx` parameter is to make sure
that the current context is really active, and hence the returned device is valid.
"""
function device(ctx::CuContext)
    if CuCurrentContext() != ctx
        # TODO: should we push and pop here?
        error("context should be active")
    end

    return device()
end
function device()
    device_ref = Ref{CuDevice_t}()
    @apicall(:cuCtxGetDevice, (Ptr{Cint},), device_ref)
    return CuDevice(Bool, device_ref[])
end

"""
    synchronize(ctx::CuContext=CuCurrentContext())

Block for a context's tasks to complete.

The `ctx` parameter defaults to the current active context.
"""
synchronize(ctx::CuContext=CuCurrentContext()) =
    @apicall(:cuCtxSynchronize, (CuContext_t,), ctx)


## cache config

export cache_config, cache_config!

@enum(CUfunc_cache, FUNC_CACHE_PREFER_NONE   = 0x00,
                    FUNC_CACHE_PREFER_SHARED = 0x01,
                    FUNC_CACHE_PREFER_L1     = 0x02,
                    FUNC_CACHE_PREFER_EQUAL  = 0x03)

function cache_config()
    config = Ref{CUfunc_cache}()
    @apicall(:cuCtxGetCacheConfig, (Ptr{CUfunc_cache},), config)
    return config[]
end

function cache_config!(config::CUfunc_cache)
    @apicall(:cuCtxSetCacheConfig, (CUfunc_cache,), config)
end


## shared memory config

export shmem_config, shmem_config!

@enum(CUsharedconfig, SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE    = 0x00,
                      SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE  = 0x01,
                      SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 0x02)

function shmem_config()
    config = Ref{CUsharedconfig}()
    @apicall(:cuCtxGetSharedMemConfig, (Ptr{CUsharedconfig},), config)
    return config[]
end

function shmem_config!(config::CUsharedconfig)
    @apicall(:cuCtxSetSharedMemConfig, (CUsharedconfig,), config)
end
