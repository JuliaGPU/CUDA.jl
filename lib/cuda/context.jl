# Context management

export
    CuContext, CuCurrentContext, activate,
    synchronize, device


## construction and destruction

"""
    CuContext(dev::CuDevice, flags=CTX_SCHED_AUTO)
    CuContext(f::Function, ...)

Create a CUDA context for device. A context on the GPU is analogous to a process on the CPU,
with its own distinct address space and allocated resources. When a context is destroyed,
the system cleans up the resources allocated to it.

When you are done using the context, call [`unsafe_destroy!`](@ref) to mark it for deletion,
or use do-block syntax with this constructor.

"""
mutable struct CuContext
    handle::CUcontext

    function CuContext(handle::CUcontext)
        handle == C_NULL && return new(C_NULL)
        return Base.@lock context_lock get!(valid_contexts, handle) do
            new(handle)
        end
    end
end

# NOTE: we don't implement `isequal` or `hash` in order to fall back to `===` and `objectid`
#       as contexts are unique, and with primary device contexts identical handles might be
#       returned after resetting the context (device) and all associated resources.

# the `valid` bit serves two purposes: make sure we don't double-free a context (in case we
# early-freed it ourselves before the GC kicked in), and to make sure we don't free derived
# resources after the owning context has been destroyed (which can happen due to
# out-of-order finalizer execution)
const valid_contexts = Dict{CUcontext,CuContext}()
const context_lock = ReentrantLock()
isvalid(ctx::CuContext) = any(x->x==ctx, values(valid_contexts))
# NOTE: we can't just look up by the handle, because contexts derived from a primary one
#       have the same handle even though they might have been destroyed in the meantime.
function invalidate!(ctx::CuContext)
    delete!(valid_contexts, ctx.handle)
    return
end

"""
    unsafe_destroy!(ctx::CuContext)

Immediately destroy a context, freeing up all resources associated with it. This does not
respect any users of the context, and might make other objects unusable.
"""
function unsafe_destroy!(ctx::CuContext)
    if isvalid(ctx)
        cuCtxDestroy(ctx)
        invalidate!(ctx)
    end
end

Base.unsafe_convert(::Type{CUcontext}, ctx::CuContext) = ctx.handle

@enum_without_prefix CUctx_flags CU_

function CuContext(dev::CuDevice, flags=0)
    handle_ref = Ref{CUcontext}()
    cuCtxCreate(handle_ref, flags, dev)
    return CuContext(handle_ref[])
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
        return CuContext(handle_ref[])
    end
end

"""
    push!(CuContext, ctx::CuContext)

Pushes a context on the current CPU thread.
"""
Base.push!(::Type{CuContext}, ctx::CuContext) = cuCtxPushCurrent(ctx)

"""
    pop!(CuContext)

Pops the current CUDA context from the current CPU thread, and returns that context.
"""
function Base.pop!(::Type{CuContext})
    handle_ref = Ref{CUcontext}()
    cuCtxPopCurrent(handle_ref)
    CuContext(handle_ref[])
end

"""
    activate(ctx::CuContext)

Binds the specified CUDA context to the calling CPU thread.
"""
activate(ctx::CuContext) = cuCtxSetCurrent(ctx)

function CuContext(f::Function, dev::CuDevice, args...)
    ctx = CuContext(dev, args...)    # implicitly pushes
    try
        f(ctx)
    finally
        @assert pop!(CuContext) == ctx
        unsafe_destroy!(ctx)
    end
end


## context properties

"""
    device()
    device(ctx::CuContext)

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
