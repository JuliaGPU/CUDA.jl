@doc raw"""
    cudnnMultiHeadAttnForward(weights, queries, keys, values; o...)
    cudnnMultiHeadAttnForward!(out, weights, queries, keys, values; o...)
    cudnnMultiHeadAttnForward(weights, queries, keys, values, d::cudnnAttnDescriptor; o...)
    cudnnMultiHeadAttnForward!(out, weights, queries, keys, values, d::cudnnAttnDescriptor; o...)

Return the multi-head attention result with `weights`, `queries`, `keys`, and `values`,
overwriting `out` if provided, according to keyword arguments or the attention descriptor
`d`.  The multi-head attention model can be described by the following equations:

```math
\begin{aligned}
&h_i = (W_{V,i} V) \operatorname{softmax}(\operatorname{smScaler}(K^T W^T_{K,i}) (W_{Q,i} q))
&\operatorname(MultiHeadAttn)(q,K,V,W_Q,W_K,W_V,W_O) = \sum_{i=1}^{\operatorname{nHeads}-1} W_{O,i} h_i
\end{aligned}
```

The input arguments are:
* `out`: Optional output tensor.
* `weights`: A weight buffer that contains ``W_Q, W_K, W_V, W_O``.
* `queries`: A query tensor ``Q`` which may contain a batch of queries (the above equations were for a single query vector ``q`` for simplicity).
* `keys`: The keys tensor ``K``.
* `values`: The values tensor ``V``.

Keyword arguments describing the tensors:
* `axes::Vector{cudnnSeqDataAxis_t} = [CUDNN_SEQDATA_VECT_DIM, CUDNN_SEQDATA_BATCH_DIM, CUDNN_SEQDATA_TIME_DIM, CUDNN_SEQDATA_BEAM_DIM]`: an array of length 4 that specifies the role of (Julia) dimensions. VECT has to be the first dimension, all 6 permutations of the remaining three are supported.
* `seqLengthsQO::Vector{<:Integer}`: sequence lengths in the queries and out containers. By default sequences are assumed to be full length of the TIME dimension.
* `seqLengthsKV::Vector{<:Integer}`: sequence lengths in the keys and values containers. By default sequences are assumed to be full length of the TIME dimension.

Keyword arguments describing the attention operation when `d` is not given:
* `attnMode::Unsigned = CUDNN_ATTN_QUERYMAP_ALL_TO_ONE | CUDNN_ATTN_DISABLE_PROJ_BIASES`: bitwise flags indicating various attention options. See cudnn docs for details.
* `nHeads::Integer = 1`: number of attention heads.
* `smScaler::Real = 1`: softmax smoothing (1.0 >= smScaler >= 0.0) or sharpening (smScaler > 1.0) coefficient. Negative values are not accepted.
* `mathType::cudnnMathType_t = math_mode()`: NVIDIA Tensor Core settings.
* `qProjSize, kProjSize, vProjSize, oProjSize`: vector lengths after projections, set to 0 by default which disables projections.
* `qoMaxSeqLength::Integer`: largest sequence length expected in queries and out, set to their TIME dim by default.
* `kvMaxSeqLength::Integer`: largest sequence length expected in keys and values, set to their TIME dim by default.
* `maxBatchSize::Integer`: largest batch size expected in any container, set to the BATCH dim of queries by default.
* `maxBeamSize::Integer`: largest beam size expected in any container, set to the BEAM dim of queries by default.

Other keyword arguments:
* `residuals = nothing`: optional tensor with the same size as queries that can be used to implement residual connections (see figure in cudnn docs). When residual connections are enabled, the vector length in `queries` should match the vector length in `out`, so that a vector addition is feasible. 
* `currIdx::Integer = -1`: Time-step (0-based) in queries to process. When the currIdx argument is negative, all Q time-steps are processed. When currIdx is zero or positive, the forward response is computed for the selected time-step only. The latter input can be used in inference mode only, to process one time-step while updating the next attention window and Q, R, K, V inputs in-between calls.
* `loWinIdx, hiWinIdx::Array{Cint}`: Two host integer arrays specifying the start and end (0-based) indices of the attention window for each Q time-step. The start index in K, V sets is inclusive, and the end index is exclusive. By default set at 0 and `kvMaxSeqLength` respectively.
"""
cudnnMultiHeadAttnForward, cudnnMultiHeadAttnForward!


# The axes argument in the functions below specifies the role of the axes using Julia order: VECT,BATCH,TIME,BEAM by default. Missing trailing dims assumed 1.
const cudnnSeqDataDefaultAxes = [CUDNN_SEQDATA_VECT_DIM, CUDNN_SEQDATA_BATCH_DIM, CUDNN_SEQDATA_TIME_DIM, CUDNN_SEQDATA_BEAM_DIM]


# Public methods
cudnnMultiHeadAttnForward(w,q,k,v; o...) = cudnnMultiHeadAttnForward(w,q,k,v, cudnnAttnDescriptor(q,k,v;o...); o...)
cudnnMultiHeadAttnForward(w,q,k,v, d::cudnnAttnDescriptor; o...) = cudnnMultiHeadAttnForward!(cudnnAttnOutput(q,k,v,d), w,q,k,v,d; o...)
cudnnMultiHeadAttnForward!(out, w,q,k,v; o...) = cudnnMultiHeadAttnForward!(out, w,q,k,v, cudnnAttnDescriptor(q,k,v;o...); o...)

function cudnnMultiHeadAttnForward!(
    out, weights, queries, keys, values, attnDesc::cudnnAttnDescriptor;

    # Input tensor descriptors
    axes::Vector{cudnnSeqDataAxis_t} = cudnnSeqDataDefaultAxes,
    seqLengthsQO::Vector{<:Integer} = fill(Cint(sdim(queries,axes,CUDNN_SEQDATA_TIME_DIM)), sdim(queries,axes,CUDNN_SEQDATA_BATCH_DIM)*sdim(queries,axes,CUDNN_SEQDATA_BEAM_DIM)),
    seqLengthsKV::Vector{<:Integer} = fill(Cint(sdim(keys,axes,CUDNN_SEQDATA_TIME_DIM)), sdim(keys,axes,CUDNN_SEQDATA_BATCH_DIM)*sdim(keys,axes,CUDNN_SEQDATA_BEAM_DIM)),
    devSeqLengthsQO::CuVector{Cint} = convert(CuVector{Cint}, seqLengthsQO),
    devSeqLengthsKV::CuVector{Cint} = convert(CuVector{Cint}, seqLengthsKV),
    qDesc::cudnnSeqDataDescriptor = cudnnSeqDataDescriptor(queries; axes, seqLengthArray=seqLengthsQO),
    kDesc::cudnnSeqDataDescriptor = cudnnSeqDataDescriptor(keys;    axes, seqLengthArray=seqLengthsKV),
    vDesc::cudnnSeqDataDescriptor = cudnnSeqDataDescriptor(values;  axes, seqLengthArray=seqLengthsKV),
    oDesc::cudnnSeqDataDescriptor = cudnnSeqDataDescriptor(out;     axes, seqLengthArray=seqLengthsQO),

    # forw parameters
    residuals = nothing,
    currIdx::Integer = -1,
    loWinIdx::Union{Array{Cint},Nothing} = nothing,
    hiWinIdx::Union{Array{Cint},Nothing} = nothing,
    workspace::Union{CuArray,Nothing}    = nothing, 
    reserveSpace::Union{CuArray,Nothing} = nothing,

    # Buffers for gradients
    dweights::Ref = Ref{Any}(),
    dqueries::Ref = Ref{Any}(),
    dkeys::Ref    = Ref{Any}(),
    dvalues::Ref  = Ref{Any}(),
    o...
)
    d = cudnnGetAttnDescriptor(attnDesc)
    dt = juliaDataType(d.dataType)
    @assert dt == eltype(out) == eltype(queries) == eltype(keys) == eltype(values)
    qSize = (d.qProjSize > 0 ? d.qProjSize : size(queries,1))
    kSize = (d.kProjSize > 0 ? d.kProjSize : size(keys,1))
    @assert kSize == qSize  "key size $kSize does not match query size $qSize."
    vSize = (d.vProjSize > 0 ? d.vProjSize : size(values,1))
    @assert size(keys)[2:end] == size(values)[2:end]  "key tensor $(size(keys)) does not match value tensor $(size(values))"
    oSize = (d.oProjSize > 0 ? d.oProjSize : d.nHeads * vSize)
    oDims = (oSize, size(queries)[2:end]...)
    @assert size(out) == oDims  "output size should be $(oDims)"
    @assert residuals === nothing || size(residuals) == oDims  "residual size should be $(oDims)"
    loWinIdx === nothing ? loWinIdx = fill(Cint(0), d.qoMaxSeqLength) : @assert length(loWinIdx) == d.qoMaxSeqLength
    hiWinIdx === nothing ? hiWinIdx = fill(typemax(Cint), d.qoMaxSeqLength) : @assert length(hiWinIdx) == d.qoMaxSeqLength

    @assert axes[1] == CUDNN_SEQDATA_VECT_DIM  "The most inner dimension of the containers should be the vector dimension"
    @assert d.smScaler >= 0  "smScaler should be non-negative"
    @assert d.qoMaxSeqLength >= sdim(queries, axes, CUDNN_SEQDATA_TIME_DIM)
    @assert d.kvMaxSeqLength >= sdim(keys,    axes, CUDNN_SEQDATA_TIME_DIM)
    @assert d.maxBatchSize   >= sdim(queries, axes, CUDNN_SEQDATA_BATCH_DIM)
    @assert d.maxBeamSize    >= sdim(queries, axes, CUDNN_SEQDATA_BEAM_DIM)
    @assert sdim(keys, axes, CUDNN_SEQDATA_BATCH_DIM) == sdim(queries, axes, CUDNN_SEQDATA_BATCH_DIM)  "keys/values and queries have different batch sizes"
    if d.attnMode & CUDNN_ATTN_QUERYMAP_ONE_TO_ONE > 0
        @assert sdim(keys, axes, CUDNN_SEQDATA_BEAM_DIM) == sdim(queries, axes, CUDNN_SEQDATA_BEAM_DIM)  "keys/values and queries have different beam sizes when attnMode is CUDNN_ATTN_QUERYMAP_ONE_TO_ONE"
    else
        @assert sdim(keys, axes, CUDNN_SEQDATA_BEAM_DIM) == 1  "keys/values should have beam=1 when attnMode is CUDNN_ATTN_QUERYMAP_ALL_TO_ONE"
    end

    # Backward called separately on each variable. We will calculate all gradients on first call. Use `dready` to avoid subsequent calls.
    dready = Ref{Bool}(false)   # this will be turned to `true` by the first backward call.

    cudnnMultiHeadAttnForwardAD(
        weights, queries, keys, values, residuals;
        dready, dweights, dqueries, dkeys, dvalues, # dresiduals is equal to dout
        attnDesc, currIdx, loWinIdx, hiWinIdx,
        devSeqLengthsQO, devSeqLengthsKV,
        qDesc, kDesc, vDesc, oDesc,
        out, workspace, reserveSpace)
end


# AD method
function cudnnMultiHeadAttnForwardAD(
    weights, queries, keys, values, residuals;
    dready, dweights, dqueries, dkeys, dvalues,
    attnDesc, currIdx, loWinIdx, hiWinIdx,
    devSeqLengthsQO, devSeqLengthsKV,
    qDesc, kDesc, vDesc, oDesc,
    out, workspace, reserveSpace
)
    # Cannot use @workspace here because it is shared between forw and back calls
    (weightSize, workspaceSize, reserveSpaceSize) = cudnnMultiHeadAttnBuffers(attnDesc)
    if workspaceSize > 0 && workspace === nothing; workspace = cudnnTempSpace(workspaceSize); end
    if reserveSpaceSize > 0 && reserveSpace === nothing; reserveSpace = cudnnTempSpace(reserveSpaceSize); end
    @assert sizeof(weights) >= weightSize  "weights should be at least $weightSize bytes."
    @assert sizeof(workspace) >= workspaceSize  "worksSpace should be at least $workspaceSize bytes"
    @assert sizeof(reserveSpace) >= reserveSpaceSize  "reserveSpace should be at least $reserveSpaceSize bytes"

    cudnnMultiHeadAttnForward(
        handle(), attnDesc, currIdx,
        loWinIdx, hiWinIdx,
        devSeqLengthsQO, devSeqLengthsKV,
        qDesc, queries, something(residuals, CU_NULL),
        kDesc, keys,
        vDesc, values,
        oDesc, out,
        sizeof(weights), something(weights, CU_NULL),
        sizeof(workspace), something(workspace, CU_NULL),
        sizeof(reserveSpace), something(reserveSpace, CU_NULL)
    )
    return out
end


# Helper methods


function cudnnAttnDescriptor(
    queries, keys, values;
    axes = cudnnSeqDataDefaultAxes,
    attnMode::Unsigned = CUDNN_ATTN_QUERYMAP_ALL_TO_ONE | CUDNN_ATTN_DISABLE_PROJ_BIASES |> Cuint,
    nHeads::Integer = Cint(1),
    smScaler::Real = Cdouble(1),
    # dataType::DataType = eltype(queries),
    # computePrec::DataType = eltype(queries),  ## No other option according to 8.0.2
    mathType::cudnnMathType_t = math_mode(),
    # attnDropout::Real = 0, ## The dropout option is currently not supported by the multi-head attention API
    # postDropout::Real = 0, ## The dropout option is currently not supported by the multi-head attention API
    qProjSize::Integer = 0, # Use zero to disable the corresponding projection
    kProjSize::Integer = 0,
    vProjSize::Integer = 0,
    oProjSize::Integer = 0,
    qoMaxSeqLength::Integer = sdim(queries,axes,CUDNN_SEQDATA_TIME_DIM),
    kvMaxSeqLength::Integer = sdim(keys,axes,CUDNN_SEQDATA_TIME_DIM),
    maxBatchSize::Integer = sdim(queries,axes,CUDNN_SEQDATA_BATCH_DIM),
    maxBeamSize::Integer = sdim(queries,axes,CUDNN_SEQDATA_BEAM_DIM),
    o...
)
    cudnnAttnDescriptor(
        Cuint(attnMode),
        Cint(nHeads),
        Cdouble(smScaler),
        cudnnDataType(eltype(queries)),    # dataType
        cudnnDataType(eltype(queries)),    # computePrec
        mathType,
        C_NULL,  # attnDropout
        C_NULL,  # postDropout
        Cint(sdim(queries,axes,CUDNN_SEQDATA_VECT_DIM)), # qSize
        Cint(sdim(keys,   axes,CUDNN_SEQDATA_VECT_DIM)), # kSize
        Cint(sdim(values, axes,CUDNN_SEQDATA_VECT_DIM)), # vSize
        Cint(qProjSize),
        Cint(kProjSize),
        Cint(vProjSize),
        Cint(oProjSize),
        Cint(qoMaxSeqLength),
        Cint(kvMaxSeqLength),
        Cint(maxBatchSize),
        Cint(maxBeamSize)
    )
end

function cudnnGetAttnDescriptor(d::cudnnAttnDescriptor)
    (attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize) = (Ref{Cuint}(), Ref{Cint}(), Ref{Cdouble}(), Ref{cudnnDataType_t}(), Ref{cudnnDataType_t}(), Ref{cudnnMathType_t}(), Ref{cudnnDropoutDescriptor_t}(), Ref{cudnnDropoutDescriptor_t}(), Ref{Cint}(), Ref{Cint}(), Ref{Cint}(), Ref{Cint}(), Ref{Cint}(), Ref{Cint}(), Ref{Cint}(), Ref{Cint}(), Ref{Cint}(), Ref{Cint}(), Ref{Cint}())
    cudnnGetAttnDescriptor(d, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize)
    (attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize) = (x->x[]).((attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize))
    return (; attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize)
end


function cudnnAttnOutput(queries, keys, values, attnDesc::cudnnAttnDescriptor)
    d = cudnnGetAttnDescriptor(attnDesc)
    vSize = (d.vProjSize > 0 ? d.vProjSize : size(values,1))
    oSize = (d.oProjSize > 0 ? d.oProjSize : d.nHeads * vSize)
    oDims = (oSize, size(queries)[2:end]...)
    out = similar(values, oDims)
    out .= 0 # currIdx >= 0 only fills part of this, zero the rest for consistency
    return out
end


function cudnnMultiHeadAttnBuffers(attnDesc::cudnnAttnDescriptor; training=false)
    weightSize, workspaceSize = Ref{Csize_t}(0), Ref{Csize_t}(0)
    # Assigning NULL to the reserveSpaceSizeInBytes argument indicates that the user does not plan to invoke multi-head attention gradient functions
    reserveSpaceSize = training ? Ref{Csize_t}(0) : C_NULL
    cudnnGetMultiHeadAttnBuffers(handle(), attnDesc, weightSize, workspaceSize, reserveSpaceSize)
    return (weightSize[], workspaceSize[], reserveSpaceSize === C_NULL ? 0 : reserveSpaceSize[])
end


# If we have less than four dims, assume they are padded by 1s on the right for Julia, on the left for cudnn.
# We keep `axes` in Julia order, e.g. axes[1] refers to the function of the first Julia dimension and is always VECT.
"""
    sdim(x,axes,dim)
    sdim(x,axes)

The first form returns the size of `x` in the dimension specified with
`dim::cudnnSeqDataAxis_t` (e.g. CUDNN_SEQDATA_TIME_DIM), i.e. return `size(x,i)` such that
`axes[i]==dim`.

The second form returns an array of length 4 `dims::Vector{Cint}` such that `dims[1+dim] ==
sdim(x,axes,dim)` where `dim::cudnnSeqDataAxis_t` specifies the role of the dimension
(e.g. dims[CUDNN_SEQDATA_TIME_DIM]==5).

The `axes::Vector{cudnnSeqDataAxis_t}` argument is an array of length 4 that
specifies the role of Julia dimensions, e.g. `axes[3]=CUDNN_SEQDATA_TIME_DIM`.
"""
function sdim(x,axes,dim)
    for i in 1:length(axes)
        if axes[i] === dim      # axes[i] = CUDNN_SEQDATA_XXX_DIM
            return size(x,i)
        end
    end
    error("Cannot find $dim in axes")
end

function sdim(x,axes)
    dims = Array{Cint}(undef, 4)
    for dim in (CUDNN_SEQDATA_VECT_DIM, CUDNN_SEQDATA_BATCH_DIM, CUDNN_SEQDATA_TIME_DIM, CUDNN_SEQDATA_BEAM_DIM)
        dims[1+dim] = sdim(x,axes,dim)
    end
    return dims                 # dims[1+CUDNN_SEQDATA_XXX_DIM] = how many XXX
end


# Alternative cudnnSeqDataDescriptor constructor for array
function cudnnSeqDataDescriptor(
    array; 
    axes::Vector{cudnnSeqDataAxis_t} = cudnnSeqDataDefaultAxes,
    dimA::Vector{Cint} = sdim(array,axes),
    seqLengthArray::Vector{<:Integer} = fill(Cint(sdim(array,axes,CUDNN_SEQDATA_TIME_DIM)), sdim(array,axes,CUDNN_SEQDATA_BATCH_DIM)*sdim(array,axes,CUDNN_SEQDATA_BEAM_DIM)), # cudnn-doc: The seqLengthArray[] must specify all sequence lengths in the container so the total size of this array should be dimA[CUDNN_SEQDATA_BATCH_DIM] * dimA[CUDNN_SEQDATA_BEAM_DIM].
    paddingFill::Ptr{Cvoid} = C_NULL, # cudnn-doc: Currently, the only supported value for paddingFill is NULL which means this option should be ignored.
)
    nbDims::Cint = CUDNN_SEQDATA_DIM_COUNT          # Currently, the value of this argument should be four. The actual size of the dimA[] and axes[] arrays should be declared using the CUDNN_SEQDATA_DIM_COUNT macro.
    @assert length(axes) == length(dimA) == CUDNN_SEQDATA_DIM_COUNT # cudnn-doc: The number of active dimensions in the dimA[] and axes[] arrays is defined by the nbDims argument. 
    seqLengthArraySize = Csize_t(sdim(array,axes,CUDNN_SEQDATA_BATCH_DIM) * sdim(array,axes,CUDNN_SEQDATA_BEAM_DIM))
    @assert length(seqLengthArray) == seqLengthArraySize
    cudnnSeqDataDescriptor(cudnnDataType(eltype(array)), nbDims, dimA, reverse(axes), # cudnn uses reverse order for dims
                           seqLengthArraySize, convert(Vector{Cint}, seqLengthArray), 
                           paddingFill)
end

cudnnSeqDataDescriptor(::Nothing; o...) = nothing
