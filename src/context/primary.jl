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
struct CuPrimaryContext
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
    ctx = CuContext(handle[], false)    # CuContext shouldn't manage this ctx
    @compat finalizer((ctx)->begin
        @trace("Finalizing derived CuContext object at $(Base.pointer_from_objref(ctx)))")
        @assert isvalid(ctx)    # not owned by CuContext, so shouldn't have been invalidated
        @apicall(:cuDevicePrimaryCtxRelease, (CuDevice_t,), pctx.dev)
        invalidate!(ctx)
        delete!(pctx_instances[pctx], WeakRef(ctx))
    end, ctx)
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
    unsafe_reset!(pctx::CuPrimaryContext, [checked::Bool=true])

Explicitly destroys and cleans up all resources associated with a device's primary context
in the current process. Note that this forcibly invalidates all contexts derived from this
primary context, and as a result outstanding resources might become invalid.

It is normally unnecessary to call this function, as resource are automatically freed when
contexts go out of scope. In the case of primary contexts, they are collected when all
contexts derived from that primary context have gone out of scope.

The `checked` argument determines whether to verify that the primary context has become
inactive after resetting the derived driver contexts. This may not be possible, eg. if the
CUDA runtime API itself has retained an additional context instance.
"""
function unsafe_reset!(pctx::CuPrimaryContext, checked::Bool=true)
    for ref in pctx_instances[pctx]
        ctx = ref.value
        destroy!(ctx)
        finalize(ctx)
    end
    @assert isempty(pctx_instances[pctx])

    # NOTE: we don't support/perform the actual call to cuDevicePrimaryCtxReset, because of
    #       what's probably a bug in CUDA: calling cuDevicePrimaryCtxReset makes CUDA ignore
    #       all future calls to cuDevicePrimaryCtxRelease, even if those would be necessary
    #       to make the refcount of derived contexts instantiated through
    #       cuDevicePrimaryCtxRetain drop to 0. As a result, calling cuDevicePrimaryCtxReset
    #       makes that future activated primary contexts remain active indefinitely.
    #
    #       However, we don't _need_ cuDevicePrimaryCtxReset because we already forced
    #       finalization (and hence cuDevicePrimaryCtxRelease) on all derived contexts
    #       through the GC, and asserted that there's no derived contexts left.
    #@apicall(:cuDevicePrimaryCtxReset, (CuDevice_t,), pctx.dev)

    if checked
        # NOTE: having finalized all derived contexts doesn't mean the primary context is
        #       inactive now, because external consumers (eg. CUDArt) might hold another
        #       instance. That doesn't mean this logic isn't required, for correctness &
        #       ease of debugging wrt. data structures managed by CUDAdrv tied to contexts
        #       allocated via CUDAdrv. However, it requires us to still perform eg. a
        #       cudaDeviceReset call in CUDArt.
        @assert !isactive(pctx)
    end

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
