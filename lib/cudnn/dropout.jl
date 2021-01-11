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

The global constant `cudnnDropoutState::Dict` holds the random number generator state for
each CUDNN handle.
"""
cudnnDropoutForward, cudnnDropoutForward!, cudnnDropoutSeed, cudnnDropoutState


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
    dropoutDesc::cudnnDropoutDescriptor = cudnnDropoutDescriptor(Cfloat(dropout)),
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x),
    yDesc::cudnnTensorDescriptor = xDesc,
    reserveSpace::CuArray = cudnnDropoutReserveSpace(xDesc)
)
    if cudnnDropoutSeed[] >= 0
        # This is a very expensive call (40x dropout), so only use for debugging
        @warn "CUDA.CUDNN.cudnnDropoutSeed[] >= 0: dropout operations will be deterministic but 40x more expensive" maxlog=1
        dropout, states, seed = cudnnGetDropoutDescriptor(dropoutDesc)
        hstate = cudnnDropoutState[handle()]
        @assert states == pointer(hstate)
        @retry_reclaim(isequal(CUDNN_STATUS_ALLOC_FAILED),
                       cudnnSetDropoutDescriptor(dropoutDesc, handle(), dropout, hstate, sizeof(hstate), cudnnDropoutSeed[]))
    end
    cudnnDropoutForwardAD(x; xDesc, y, yDesc, dropoutDesc, reserveSpace)
end

function cudnnDropoutReserveSpace(td::cudnnTensorDescriptor)
    # reserveSpace is ~1/8 of tensor size and passes info between forw and back
    rss = Csize_t[0]; cudnnDropoutGetReserveSpaceSize(td, rss)
    return cudnnTempSpace(rss[1])
end


# AD method
function cudnnDropoutForwardAD(x; xDesc, y, yDesc, dropoutDesc, reserveSpace)
    cudnnDropoutForward(handle(), dropoutDesc, xDesc, x, yDesc, y, reserveSpace, sizeof(reserveSpace))
    return y
end


# Global RNG state: This should NOT be reallocated for each descriptor! However
# cudnnDropoutForward() doc says: "This function should not be running concurrently with
# another cudnnDropoutForward() function using the same states."  So I am going to assume
# using a single buffer per handle is ok.

const cudnnDropoutState = Dict{Ptr,CuArray}() # handle -> state

# Global dropout seed: To debug gradients set cudnnDropoutSeed[] >= 0 which makes all
# dropout operations deterministic but about 40x more expensive.

const cudnnDropoutSeed = Ref{Int}(-1)


# Helper for cudnnDropoutDescriptor constructor from float:
# Calls to cudnnDropoutDescriptor with identical Cfloats will return the same object thanks
# to caching. If the user wants to set the seed to replicate an experiment, that is taken
# care of during the forward call.

function cudnnSetDropoutDescriptorFromFloat(ptr::cudnnDropoutDescriptor_t, dropout::Real)
    hstate = get!(cudnnDropoutState, handle()) do
        cudnnTempSpace(cudnnDropoutGetStatesSize())
    end
    seed = floor(Culonglong,time())
    @retry_reclaim(isequal(CUDNN_STATUS_ALLOC_FAILED),
                   cudnnSetDropoutDescriptor(ptr, handle(), Cfloat(dropout), hstate, sizeof(hstate), seed))
end


function cudnnGetDropoutDescriptor(d::cudnnDropoutDescriptor)
    dropout, states, seed = Ref{Cfloat}(0), Ref{CuPtr{Nothing}}(0), Ref{Culonglong}(0)
    cudnnGetDropoutDescriptor(d::cudnnDropoutDescriptor, handle(), dropout, states, seed)
    return (dropout[], states[], seed[])
end

function cudnnDropoutGetStatesSize()
    ssize = Ref{Csize_t}(0)
    cudnnDropoutGetStatesSize(handle(), ssize)
    @assert ssize[] > 0
    return ssize[]
end
