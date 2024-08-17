using cuDNN:
    cudnnMultiHeadAttnForward,
    cudnnMultiHeadAttnForward!,
    cudnnMultiHeadAttnBackwardData,
    cudnnMultiHeadAttnBackwardWeights,
    cudnnGetMultiHeadAttnBuffers,
    cudnnGetMultiHeadAttnWeights,
    cudnnAttnDescriptor,
        cudnnAttnDescriptor_t,
        cudnnCreateAttnDescriptor,
        cudnnDestroyAttnDescriptor,
        cudnnSetAttnDescriptor,
        cudnnGetAttnDescriptor,
        cudnnDataType_t,
        cudnnDropoutDescriptor_t,
    #cudnnAttnQueryMap_t,
        CUDNN_ATTN_QUERYMAP_ALL_TO_ONE, # 0         /* multiple Q-s map to a single (K,V) set when beam size > 1, beam sizes for (K,V) = 1 */
        CUDNN_ATTN_QUERYMAP_ONE_TO_ONE, # (1U << 0) /* multiple Q-s map to multiple (K,V) sets when beam size > 1, beam sizes for (K,V) = beam size for (Q) */
        CUDNN_ATTN_DISABLE_PROJ_BIASES, # 0         /* no biases in attention input and output projections */
        CUDNN_ATTN_ENABLE_PROJ_BIASES,  # (1U << 1) /* use biases in attention input and output projections */
    cudnnMultiHeadAttnWeightKind_t,
        CUDNN_MH_ATTN_Q_WEIGHTS, # 0, /* input projection weights for 'queries' */
        CUDNN_MH_ATTN_K_WEIGHTS, # 1, /* input projection weights for 'keys' */
        CUDNN_MH_ATTN_V_WEIGHTS, # 2, /* input projection weights for 'values' */
        CUDNN_MH_ATTN_O_WEIGHTS, # 3, /* output projection weights */
        CUDNN_MH_ATTN_Q_BIASES,  # 4, /* input projection bias tensor for 'queries' */
        CUDNN_MH_ATTN_K_BIASES,  # 5, /* input projection bias for 'keys' */
        CUDNN_MH_ATTN_V_BIASES,  # 6, /* input projection bias for 'values' */
        CUDNN_MH_ATTN_O_BIASES,  # 7, /* output projection biases */
    cudnnMathType_t,
        CUDNN_DEFAULT_MATH,                    # 0,
        CUDNN_TENSOR_OP_MATH,                  # 1,
        CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION, # 2,
        CUDNN_FMA_MATH,                        # 3,
    cudnnWgradMode_t,
        CUDNN_WGRAD_MODE_ADD,  # 0,
        CUDNN_WGRAD_MODE_SET,  # 1,
    cudnnSeqDataDescriptor,
        cudnnSeqDataDescriptor_t,
        cudnnCreateSeqDataDescriptor,
        cudnnDestroySeqDataDescriptor,
        cudnnSetSeqDataDescriptor,
        cudnnGetSeqDataDescriptor,
    cudnnSeqDataAxis_t,
        CUDNN_SEQDATA_TIME_DIM,  # 0, /* index in time */
        CUDNN_SEQDATA_BATCH_DIM, # 1, /* index in batch */
        CUDNN_SEQDATA_BEAM_DIM,  # 2, /* index in beam */
        CUDNN_SEQDATA_VECT_DIM,  # 3  /* index in vector */
        CUDNN_SEQDATA_DIM_COUNT, # 4
    cudnnDataType,
    cudnnSeqDataDefaultAxes,
    math_mode,
    sdim

function mhatest(;
    # Input tensor descriptors
    axes::Vector{cudnnSeqDataAxis_t} = cudnnSeqDataDefaultAxes,
    seqLengthsQO::Vector{<:Integer} = fill(Cint(sdim(queries,axes,CUDNN_SEQDATA_TIME_DIM)), sdim(queries,axes,CUDNN_SEQDATA_BATCH_DIM)*sdim(queries,axes,CUDNN_SEQDATA_BEAM_DIM)),
    seqLengthsKV::Vector{<:Integer} = fill(Cint(sdim(keys,axes,CUDNN_SEQDATA_TIME_DIM)), sdim(keys,axes,CUDNN_SEQDATA_BATCH_DIM)*sdim(keys,axes,CUDNN_SEQDATA_BEAM_DIM)),
    #devSeqLengthsQO::CuVector{Cint} = convert(CuVector{Cint}, seqLengthsQO),
    #devSeqLengthsKV::CuVector{Cint} = convert(CuVector{Cint}, seqLengthsKV),
    #qDesc::cudnnSeqDataDescriptor = cudnnSeqDataDescriptor(queries; axes, seqLengthArray=seqLengthsQO),
    #kDesc::cudnnSeqDataDescriptor = cudnnSeqDataDescriptor(keys;    axes, seqLengthArray=seqLengthsKV),
    #vDesc::cudnnSeqDataDescriptor = cudnnSeqDataDescriptor(values;  axes, seqLengthArray=seqLengthsKV),

    # attnDesc parameters
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

    # forw parameters
    residuals = nothing,
    currIdx::Integer = -1,
    loWinIdx::Array{Cint} = fill(Cint(0), qoMaxSeqLength),
    hiWinIdx::Array{Cint} = fill(Cint(kvMaxSeqLength), qoMaxSeqLength),
    #workspace::Union{CuArray,Nothing}    = nothing,
    #reserveSpace::Union{CuArray,Nothing} = nothing,
)
    attnDesc::cudnnAttnDescriptor = cudnnAttnDescriptor(
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
    y = cudnnMultiHeadAttnForward(weights, queries, keys, values; axes, seqLengthsQO,
                                    seqLengthsKV, attnMode, nHeads, smScaler, mathType,
                                    qProjSize, kProjSize, vProjSize, oProjSize,
                                    qoMaxSeqLength, kvMaxSeqLength, maxBatchSize,
                                    maxBeamSize, residuals, currIdx, loWinIdx, hiWinIdx)
    @test Array(y) ≈ cudnnMultiHeadAttnForward!(zero(y), weights, queries, keys, values; axes,
                                                seqLengthsQO, seqLengthsKV, attnMode, nHeads,
                                                smScaler, mathType, qProjSize, kProjSize,
                                                vProjSize, oProjSize, qoMaxSeqLength,
                                                kvMaxSeqLength, maxBatchSize, maxBeamSize,
                                                residuals, currIdx, loWinIdx, hiWinIdx) |> Array
    @test Array(y) ≈ cudnnMultiHeadAttnForward(weights, queries, keys, values, attnDesc;
                                                axes, seqLengthsQO, seqLengthsKV, residuals,
                                                currIdx, loWinIdx, hiWinIdx) |> Array
    @test Array(y) ≈ cudnnMultiHeadAttnForward!(zero(y), weights, queries, keys, values, attnDesc;
                                                axes, seqLengthsQO, seqLengthsKV, residuals,
                                                currIdx, loWinIdx, hiWinIdx) |> Array
end

Q,K,V,B,T,F = 6,6,5,4,3,Float32

weights, queries, keys, values = (CUDA.randn(x...) for x in ((F,100),(F,Q,B,T),(F,K,B,T),(F,V,B,T)))
mhatest()
mhatest(attnMode = CUDNN_ATTN_QUERYMAP_ALL_TO_ONE | CUDNN_ATTN_ENABLE_PROJ_BIASES |> Cuint, vProjSize=7)
mhatest(seqLengthsQO = Cint[1,2,3,1])
mhatest(seqLengthsKV = Cint[1,2,3,1])
mhatest(nHeads = 2)
mhatest(smScaler = 2)
mhatest(mathType = CUDNN_DEFAULT_MATH)
mhatest(mathType = CUDNN_TENSOR_OP_MATH)
mhatest(mathType = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)
mhatest(mathType = CUDNN_FMA_MATH)
mhatest(kProjSize = 7, qProjSize = 7) # k and q have to match
mhatest(vProjSize = 7)
mhatest(oProjSize = 7)
mhatest(qoMaxSeqLength = 7)
mhatest(kvMaxSeqLength = 7)
mhatest(maxBatchSize = 7)
mhatest(maxBeamSize = 7)
mhatest(loWinIdx = fill(Cint(1),T))
mhatest(hiWinIdx = fill(Cint(1),T))
mhatest(currIdx = 0)

# Test residuals: residuals and output (and thus values unless oProjSize>0) must match queries in vector size
values, residuals = (CUDA.randn(x...) for x in ((F,Q,B,T),(F,Q,B,T)))
mhatest(residuals = residuals)

# Test nonstandard axes order
weights, queries, keys, values = (CUDA.randn(x...) for x in ((F,100),(F,Q,T,B),(F,K,T,B),(F,V,T,B)))
mhatest(axes = [CUDNN_SEQDATA_VECT_DIM, CUDNN_SEQDATA_TIME_DIM, CUDNN_SEQDATA_BATCH_DIM, CUDNN_SEQDATA_BEAM_DIM])

# Test beam handling
weights, queries, keys, values = (CUDA.randn(x...) for x in ((F,100),(F,Q,B,T,2),(F,K,B,T,1),(F,V,B,T,1)))
mhatest()

# CUDNN_ATTN_QUERYMAP_ONE_TO_ONE does not seem to be supported
#weights, queries, keys, values = (CUDA.randn(x...) for x in ((F,100),(F,Q,B,T,M),(F,K,B,T,M),(F,V,B,T,M)))
#mhatest(attnMode = CUDNN_ATTN_QUERYMAP_ONE_TO_ONE | CUDNN_ATTN_DISABLE_PROJ_BIASES |> Cuint)
