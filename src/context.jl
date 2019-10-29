# Context management

export
    CuContext, destroy!, CuCurrentContext, activate,
    synchronize, device


## construction and destruction

"""
    CuContext(dev::CuDevice, flags::CUctx_flags=CTX_SCHED_AUTO)
    CuContext(f::Function, ...)

Create a CUDA context for device. A context on the GPU is analogous to a process on the CPU,
with its own distinct address space and allocated resources. When a context is destroyed,
the system cleans up the resources allocated to it.

Contexts are unique instances which need to be `destroy`ed after use. For automatic
management, prefer the `do` block syntax, which implicitly calls `destroy`.
"""
mutable struct CuContext
    handle::CUcontext
    owned::Bool
    valid::Bool

    """
    The `owned` argument indicates whether the caller owns this context. If so, the context
    should get destroyed when it goes out of scope. If not, it is up to the caller to do so.
    """
    function CuContext(handle::CUcontext, owned=true)
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
const context_instances = Dict{CUcontext,CuContext}()

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
        cuCtxDestroy(ctx)
        invalidate!(ctx)
    end
end

Base.unsafe_convert(::Type{CUcontext}, ctx::CuContext) = ctx.handle

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

@enum_without_prefix CUctx_flags CU_

function CuContext(dev::CuDevice, flags::CUctx_flags=CTX_SCHED_AUTO)
    handle_ref = Ref{CUcontext}()
    cuCtxCreate(handle_ref, flags, dev)
    CuContext(handle_ref[])
end

"""
    CuCurrentContext()

Return the current context, or `nothing` if there is no active context.
"""
function CuCurrentContext()
    handle_ref = Ref{CUcontext}()
    cuCtxGetCurrent(handle_ref)
    if handle_ref[] == C_NULL
        return nothing
    else
        return CuContext(handle_ref[], false)
    end
end

"""
    push!(CuContext, ctx::CuContext)

Pushes a context on the current CPU thread.
"""
Base.push!(::Type{CuContext}, ctx::CuContext) =
    cuCtxPushCurrent(ctx)

"""
    pop!(CuContext)

Pops the current CUDA context from the current CPU thread, and returns that context.
"""
function Base.pop!(::Type{CuContext})
    handle_ref = Ref{CUcontext}()
    cuCtxPopCurrent(handle_ref)
    CuContext(handle_ref[], false)
end

"""
    activate(ctx::CuContext)

Binds the specified CUDA context to the calling CPU thread.
"""
activate(ctx::CuContext) = cuCtxSetCurrent(ctx)

function CuContext(f::Function, args...)
    ctx = CuContext(args...)    # implicitly pushes
    try
        f(ctx)
    finally
        @assert pop!(CuContext) == ctx
        destroy!(ctx)
    end
end


## context properties

"""
    device()
    device(ctx::Cucontext)

Returns the device for a context.
"""
function device(ctx::CuContext)
    push!(CuContext, ctx)
    device_ref = Ref{CUdevice}()
    cuCtxGetDevice(device_ref)
    pop!(CuContext)
    return CuDevice(Bool, device_ref[])
end
function device()
    device_ref = Ref{CUdevice}()
    cuCtxGetDevice(device_ref)
    return CuDevice(Bool, device_ref[])
end

"""
    synchronize()

Block for the current context's tasks to complete.
"""
synchronize() = cuCtxSynchronize()


## cache config

export cache_config, cache_config!

@enum_without_prefix CUfunc_cache CU_

function cache_config()
    config = Ref{CUfunc_cache}()
    cuCtxGetCacheConfig(config)
    return config[]
end

function cache_config!(config::CUfunc_cache)
    cuCtxSetCacheConfig(config)
end


## shared memory config

export shmem_config, shmem_config!

@enum_without_prefix CUsharedconfig CU_

function shmem_config()
    config = Ref{CUsharedconfig}()
    cuCtxGetSharedMemConfig(config)
    return config[]
end

function shmem_config!(config::CUsharedconfig)
    cuCtxSetSharedMemConfig(config)
end


## limits

export limit, limit!

@enum_without_prefix CUlimit CU_

function limit(lim::CUlimit)
    val = Ref{Csize_t}()
    cuCtxGetLimit(val, lim)
    return Int(val[])
end

limit!(lim::CUlimit, val) = cuCtxSetLimit(lim, val)
