# Primary context management
#
# This is meant for interoperability with the CUDA runtime API

export
    CuPrimaryContext, unsafe_reset!, isactive, flags, setflags!

"""
    CuPrimaryContext(dev::CuDevice)

Create a primary CUDA context for a given device.

Each primary context is unique per device and is shared with CUDA runtime API. It is meant
for interoperability with (applications using) the runtime API.
"""
immutable CuPrimaryContext
    dev::CuDevice

    function CuPrimaryContext(dev::CuDevice)
        pctx = new(dev)
        get!(pctx_instances, pctx, Set{WeakRef}())
        return pctx
    end
end

# keep a list of the contexts derived from a primary context.
# these need to be invalidated when we reset the primary context forcibly
# (as opposed to waiting for all derived contexts going out of scope).
const pctx_instances = Dict{CuPrimaryContext,Set{WeakRef}}()

"""
    CuContext(pctx::CuPrimaryContext)
    CuContext(f::Function, pctx::CuPrimaryContext)

Retain the primary context on the GPU, returning a context compatible with the driver API.
The primary context will be released when the returned driver context is finalized. For that
reason, it is advised to use this function with `do` block syntax.
"""
function CuContext(pctx::CuPrimaryContext)
    handle = Ref{CuContext_t}()
    @apicall(:cuDevicePrimaryCtxRetain, (Ptr{CuContext_t}, CuDevice_t,), handle, pctx.dev)
    ctx = CuContext(handle[], false)    # CuContext shouldn't destroy this ctx
    finalizer(ctx, (ctx)->begin
        info("yay")
        @assert isvalid(ctx)    # not owned by CuContext, so shouldn't have been invalidated
        @apicall(:cuDevicePrimaryCtxRelease, (CuDevice_t,), pctx.dev)
        delete!(pctx_instances[pctx], WeakRef(ctx))
        invalidate!(ctx)
    end)
    push!(pctx_instances[pctx], WeakRef(ctx))
    return ctx
end

function state(pctx::CuPrimaryContext)
    flags = Ref{Cuint}()
    active = Ref{Cint}()
    @apicall(:cuDevicePrimaryCtxGetState, (CuDevice_t, Ptr{Cuint}, Ptr{Cint}),
             pctx.dev, flags, active)
    return (flags[], active[] == one(Cint))
end

"""
    reset(pctx::CuPrimaryContext)

Explicitly destroys and cleans up all resources associated with a device's primary context
in the current process. Note that this forcibly invalidates all contexts derived from this
primary context, and as a result outstanding resources might become invalid.

It is normally unnecessary to call this function, as resource are automatically freed when
contexts go out of scope. In the case of primary contexts, they are collected when all
contexts derived from that primary context have gone out of scope.
"""
function unsafe_reset!(pctx::CuPrimaryContext)
    if haskey(pctx_instances, pctx)
        for ref in pctx_instances[pctx]
            ctx = ref.value
            info("forcibly finalizing $ctx")
            finalize(ctx)
        end
    end
    @assert !isactive(pctx)

    # NOTE: we don't support/perform the actual call to cuDevicePrimaryCtxReset, because of
    #       what's probably a bug in CUDA. Calling cuDevicePrimaryCtxReset makes CUDA ignore
    #       all future calls to cuDevicePrimaryCtxRelease, even if those would be necessary
    #       to make the refcount of corresponding cuDevicePrimaryCtxRetain calls drop to 0.
    #       As a result, calling cuDevicePrimaryCtxReset keeps
    #
    #       However, we don't _need_ cuDevicePrimaryCtxReset because we already forced
    #       finalization (and hence cuDevicePrimaryCtxRelease) on all derived contexts
    #       through the GC, and asserted that the primary context is inactive now.
    #@apicall(:cuDevicePrimaryCtxReset, (CuDevice_t,), pctx.dev)

    return
end

"""
    isactive(pctx::CuPrimaryContext)

Query whether a primary context is active.
"""
isactive(pctx::CuPrimaryContext) = state(pctx)[2]

"""
    flags(pctx::CuPrimaryContext)

Query the flags of a primary context.
"""
flags(pctx::CuPrimaryContext) = CUctx_flags(state(pctx)[1])

"""
    setflags!(pctx::CuPrimaryContext)

Set the flags of a primary context.
"""
setflags!(pctx::CuPrimaryContext, flags::CUctx_flags) =
    @apicall(:cuDevicePrimaryCtxSetFlags, (CuDevice_t, Cuint), pctx.dev, flags)
