"""
    cudnnDropoutForward(x; dropout=0.5)
    cudnnDropoutForward(x, d::cudnnDropoutDescriptor)
    cudnnDropoutForward!(y, x; dropout=0.5)
    cudnnDropoutForward!(y, x, d::cudnnDropoutDescriptor)

Return a new array similar to `x` where approximately `dropout` fraction of the values are
replaced by a 0, and the rest are scaled by `1/(1-dropout)`.  Optionally `y` holds the
result and `d` specifies the operation. `y` should be similar to `x` if specified.

The user can set the global seed `cudnnDropoutSeed[]` to a positive number to always drop
the same values deterministically for debugging. Note that this slows down the operation by
about 40x.

Each cuDNN handle has its own random number generator state, which is shared by the dropout
descriptors of that handle.
"""
cudnnDropoutForward, cudnnDropoutForward!, cudnnDropoutSeed


# Public methods
cudnnDropoutForward(x; o...)     = cudnnDropoutForwardWithDefaults(x; o...)
cudnnDropoutForward!(y, x; o...) = cudnnDropoutForwardWithDefaults(x; y, o...)
cudnnDropoutForward(x, d::cudnnDropoutDescriptor; o...)     = cudnnDropoutForwardWithDefaults(x; dropoutDesc=d, o...)
cudnnDropoutForward!(y, x, d::cudnnDropoutDescriptor; o...) = cudnnDropoutForwardWithDefaults(x; y, dropoutDesc=d, o...)


# Private method
function cudnnDropoutForwardWithDefaults(
    x;
    y = similar(x),
    dropout::Real = 0.5,
    dropoutDesc::Union{Nothing,cudnnDropoutDescriptor} = nothing,
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x),
    yDesc::cudnnTensorDescriptor = xDesc,
    reserveSpace::CuArray = cudnnDropoutReserveSpace(xDesc)
)
    h = handle()
    dropoutDesc = dropoutDesc === nothing ? dropout_descriptor(h, Cfloat(dropout)) :
                                            dropout_descriptor(h, dropoutDesc)
    if cudnnDropoutSeed[] >= 0 && dropoutDesc.state !== nothing
        state = dropoutDesc.state
        cudnnSetDropoutDescriptor(dropoutDesc, h, dropoutDesc.dropout, state, sizeof(state),
                                  cudnnDropoutSeed[])
    end
    cudnnDropoutForwardAD(x; handle=h, xDesc, y, yDesc, dropoutDesc, reserveSpace)
end

function cudnnDropoutReserveSpace(td::cudnnTensorDescriptor)
    # reserveSpace is ~1/8 of tensor size and passes info between forw and back
    rss = Csize_t[0]; cudnnDropoutGetReserveSpaceSize(td, rss)
    return cudnnTempSpace(rss[1])
end


# AD method
function cudnnDropoutForwardAD(x; handle, xDesc, y, yDesc, dropoutDesc, reserveSpace)
    state = dropoutDesc.state
    GC.@preserve state begin
        # the descriptor hides the state from the operation, so take its pointer to make it
        # ready for use on the current stream (like the operation's own arguments)
        state === nothing || pointer(state)
        cudnnDropoutForward(handle, dropoutDesc, xDesc, x, yDesc, y, reserveSpace,
                            sizeof(reserveSpace))
    end
    return y
end


# Random number generator state: cuDNN docs for cudnnDropoutForward() say "This function
# should not be running concurrently with another cudnnDropoutForward() function using the
# same states", so each handle (which is used by a single task at a time) has its own state,
# shared by the handle's dropout descriptors. Initializing the state is expensive, so this
# is done once, after which other descriptors are pointed at it.

# RNN descriptors refer to a dropout descriptor, so they are cached per handle too.
mutable struct DropoutCache
    state::Union{Nothing,CuVector{UInt8}}
    const descriptors::Dict{Cfloat,cudnnDropoutDescriptor}
    const rnn_descriptors::Dict{Tuple,cudnnRNNDescriptor}
end
DropoutCache() = DropoutCache(nothing, Dict{Cfloat,cudnnDropoutDescriptor}(),
                              Dict{Tuple,cudnnRNNDescriptor}())

function dropout_descriptor(h, dropout::Cfloat)
    cache = h.dropout
    d = get(cache.descriptors, dropout, nothing)
    d === nothing || return d

    state = cache.state
    if state === nothing && is_capturing()
        throw(CaptureError("""cannot initialize the cuDNN dropout state while capturing a graph.
                              Perform a dropout operation before capturing."""))
    end

    initialized = state !== nothing
    if !initialized
        # the state lives as long as the handle, so it should not be owned by an active
        # allocation cache
        state = GPUArrays.@uncached CuVector{UInt8}(undef, cudnnDropoutGetStatesSize(h))
    end

    ptr = Ref{cudnnDropoutDescriptor_t}()
    cudnnCreateDropoutDescriptor(ptr)
    d = cudnnDropoutDescriptor(ptr[], dropout, state)
    resource_finalizer(x->cudnnDestroyDropoutDescriptor(x.ptr), d; blocking=false)
    if initialized
        cudnnRestoreDropoutDescriptor(d, h, dropout, state, sizeof(state), 0)
    else
        cudnnSetDropoutDescriptor(d, h, dropout, state, sizeof(state), time_ns())
        cache.state = state
    end
    cache.descriptors[dropout] = d
    return d
end

# a descriptor that belongs to another handle (e.g. created by another task, or on another
# device) is replaced by the current handle's descriptor for the same probability
function dropout_descriptor(h, d::cudnnDropoutDescriptor)
    (d.state === nothing || d.state === h.dropout.state) && return d
    return dropout_descriptor(h, d.dropout)
end


# Global dropout seed: To debug gradients set cudnnDropoutSeed[] >= 0 which makes all
# dropout operations deterministic but about 40x more expensive.

const cudnnDropoutSeed = Ref{Int}(-1)


function cudnnGetDropoutDescriptor(d::cudnnDropoutDescriptor)
    dropout, states, seed = Ref{Cfloat}(0), Ref{CuPtr{Nothing}}(0), Ref{Culonglong}(0)
    cudnnGetDropoutDescriptor(d::cudnnDropoutDescriptor, handle(), dropout, states, seed)
    return (dropout[], states[], seed[])
end

function cudnnDropoutGetStatesSize(h=handle())
    ssize = Ref{Csize_t}(0)
    cudnnDropoutGetStatesSize(h, ssize)
    @assert ssize[] > 0
    return ssize[]
end
