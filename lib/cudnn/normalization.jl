"""
    cudnnNormalizationForward(x, xmean, xvar, bias, scale; o...)
    cudnnNormalizationForward!(y, x, xmean, xvar, bias, scale; o...)

Return batch normalization applied to `x`:

    y .= ((x .- mean(x; dims)) ./ sqrt.(epsilon .+ var(x; dims))) .* scale .+ bias  # training
    y .= ((x .- xmean) ./ sqrt.(epsilon .+ xvar)) .* scale .+ bias                  # inference


Bias and scale are trainable parameters, xmean and xvar are modified to collect statistics
during training and treated as constants during inference. Note that during inference the
values given by xmean and xvar arguments are used in the formula whereas during training the
actual mean and variance of the minibatch are used in the formula: the xmean/xvar arguments
are only used to collect statistics. In the original paper bias is referred to as beta and
scale as gamma (Batch Normalization: Accelerating Deep Network Training by Reducing Internal
Covariate Shift, S. Ioffe, C. Szegedy, 2015).

Keyword arguments:
* `epsilon = 1e-5`: epsilon value used in the normalization formula
* `exponentialAverageFactor = 0.1`: factor used in running mean/variance calculation: `runningMean = runningMean*(1-factor) + newMean*factor`
* `training = false`: boolean indicating training vs inference mode
* `mode::cudnnNormMode_t = CUDNN_NORM_PER_CHANNEL`: Per-channel layer is based on the paper. In this mode `scale` etc. have dimensions (1,1,C,1). The other alternative is `CUDNN_NORM_PER_ACTIVATION` where `scale` etc. have dimensions `(W,H,C,1)`.
* `algo::cudnnNormAlgo_t = CUDNN_NORM_ALGO_STANDARD`: The other alternative, `CUDNN_NORM_ALGO_PERSIST`, triggers the new semi-persistent NHWC kernel when certain conditions are met (see cudnn docs).
* `normOps::cudnnNormOps_t = CUDNN_NORM_OPS_NORM`: Currently the other alternatives, `CUDNN_NORM_OPS_NORM_ACTIVATION` and `CUDNN_NORM_OPS_NORM_ADD_ACTIVATION` are not supported.
* `z = nothing`: for residual addition to the result of the normalization operation, prior to the activation (will be supported when CUDNN_NORM_OPS_NORM_ADD_ACTIVATION is supported)
* `groupCnt = 1`: Place holder for future work, should be set to 1 now
* `alpha = 1; beta = 0`: scaling parameters: return `alpha * new_y + beta * old_y`

"""
cudnnNormalizationForward, cudnnNormalizationForward!


# Public methods
cudnnNormalizationForward(x, xmean, xvar, bias, scale; o...) = cudnnNormalizationForwardWithDefaults(x, xmean, xvar, bias, scale; o...)
cudnnNormalizationForward!(y, x, xmean, xvar, bias, scale; o...) = cudnnNormalizationForwardWithDefaults(x, xmean, xvar, bias, scale; y, o...)


# Private method
function cudnnNormalizationForwardWithDefaults(
    x, mean, variance, bias, scale;

    # Inference parameters:
    y = similar(x),
    z = nothing, # for residual addition to the result of the normalization operation, prior to the activation
    mode::cudnnNormMode_t = CUDNN_NORM_PER_CHANNEL, # Per-channel layer is based on the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, S. Ioffe, C. Szegedy, 2015.
    normOps::cudnnNormOps_t = CUDNN_NORM_OPS_NORM,  # Currently CUDNN_NORM_OPS_NORM_ACTIVATION and CUDNN_NORM_OPS_NORM_ADD_ACTIVATION are only supported in the NHWC layout (training,backward), not supported (inference)
    algo::cudnnNormAlgo_t = CUDNN_NORM_ALGO_STANDARD, # trigger the new semi-persistent NHWC kernel when CUDNN_NORM_ALGO_PERSIST
    alpha::Real = 1,
    beta::Real = 0,
    epsilon::Real = Cdouble(1e-5), # Has to be >= 0. Should be the same in forward and backward functions.
    groupCnt::Integer = Cint(1),   # Place hold for future work, should be set to 1 now

    # Training-only parameters:
    training = false,
    exponentialAverageFactor::Real = Cdouble(0.1),
    savedMean = nothing, # Optionally save intermediate results from the forward pass here - can be reused to speed up backward pass. NULL if unused.
    savedInvVariance = nothing,

    # Activation parameters:
    activationMode::cudnnActivationMode_t = CUDNN_ACTIVATION_IDENTITY,
    activationReluNanOpt::cudnnNanPropagation_t = CUDNN_NOT_PROPAGATE_NAN,
    activationCoef::Real = 1,
    activationDesc::Union{Nothing,cudnnActivationDescriptor} = (normOps == CUDNN_NORM_OPS_NORM ? nothing : cudnnActivationDescriptor(activationMode, activationReluNanOpt, Cdouble(activationCoef))),

    # Tensor descriptors:
    format::cudnnTensorFormat_t = CUDNN_TENSOR_NCHW,
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x; format),
    yDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(y; format),
    zDesc::Union{Nothing,cudnnTensorDescriptor} = (z === nothing ? nothing : cudnnTensorDescriptor(z; format)),
    normScaleBiasDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(scale; format),
    normMeanVarDesc::Union{Nothing,cudnnTensorDescriptor} = (mean === nothing ? nothing : cudnnTensorDescriptor(mean; format)),

    # Temporary space used in training:
    workspace = nothing,
    reserveSpace = nothing,
    dx = Ref{Any}(),
    dscale = Ref{Any}(),
    dbias = Ref{Any}(),
    dz = Ref{Any}(),
)
    @assert epsilon >= 0 && exponentialAverageFactor >= 0  "epsilon and exponentialAverageFactor should be non-negative."
    @assert groupCnt == 1  "Currently only groupCnt=1 is supported."
    @assert normOps === CUDNN_NORM_OPS_NORM "Currently only normOps=CUDNN_NORM_OPS_NORM is supported."
    alpha, beta = (a->scalingParameter(eltype(x),a)).((alpha, beta))
    # Backward called separately on each variable. We will calculate all gradients on first call. Use `dready` to avoid subsequent calls.
    dready = Ref{Bool}(false)   # this will be turned to `true` by the first backward call.
    cudnnNormalizationForwardAD(x, scale, bias, z; training, mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz, dready)
end


# AD method:
function cudnnNormalizationForwardAD(x, scale, bias, z; training, mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz, dready)
    issimilar(x,y) = (typeof(x) === typeof(y) && (x === nothing || size(x) === size(y)))
    if training
        mean === nothing ? savedMean = nothing : savedMean === nothing ? savedMean = similar(mean) : @assert issimilar(mean, savedMean)
        variance === nothing ? savedInvVariance = nothing : savedInvVariance === nothing ? savedInvVariance = similar(variance) : @assert issimilar(variance, savedInvVariance)
        workspaceSize, reserveSpaceSize = cudnnNormalizationTempSpaceSizes(mode, normOps, algo, xDesc, zDesc, yDesc, normScaleBiasDesc, activationDesc, normMeanVarDesc, groupCnt)
        if reserveSpaceSize > 0 && reserveSpace === nothing; reserveSpace = cudnnTempSpace(reserveSpaceSize); end
        @assert sizeof(reserveSpace) >= reserveSpaceSize  "reserveSpace should be at least $(reserveSpaceSize) bytes"
        if workspaceSize > 0 && workspace === nothing; workspace = cudnnTempSpace(workspaceSize); end
        @assert sizeof(workspace) >= workspaceSize  "workspace should be at least $(workspaceSize) bytes"
        cudnnNormalizationForwardTraining(handle(), mode, normOps, algo, alpha, beta, xDesc, x, normScaleBiasDesc, scale, bias, exponentialAverageFactor, something(normMeanVarDesc,C_NULL), something(mean,CU_NULL), something(variance,CU_NULL), epsilon, something(savedMean,CU_NULL), something(savedInvVariance,CU_NULL), something(activationDesc,C_NULL), something(zDesc,C_NULL), something(z,CU_NULL), yDesc, y, something(workspace,CU_NULL), sizeof(workspace), something(reserveSpace,CU_NULL), sizeof(reserveSpace), groupCnt)
    else
        @assert mean !== nothing && variance !== nothing && normMeanVarDesc !== nothing "normalization mean and variance are required in inference mode."
        cudnnNormalizationForwardInference(handle(), mode, normOps, algo, alpha, beta, xDesc, x, normScaleBiasDesc, scale, bias, normMeanVarDesc, mean, variance, something(zDesc,C_NULL), something(z,CU_NULL), something(activationDesc,C_NULL), yDesc, y, epsilon, groupCnt)
    end
    return y
end


# Helper functions
function cudnnNormalizationTempSpaceSizes(mode, normOps, algo, xDesc, zDesc, yDesc, normScaleBiasDesc, activationDesc, normMeanVarDesc, groupCnt)
    workspaceSize, reserveSpaceSize = Ref{Csize_t}(0), Ref{Csize_t}(0)
    cudnnGetNormalizationForwardTrainingWorkspaceSize(handle(), mode, normOps, algo, xDesc, something(zDesc,C_NULL), yDesc, normScaleBiasDesc, something(activationDesc,C_NULL), something(normMeanVarDesc,C_NULL), workspaceSize, groupCnt)
    cudnnGetNormalizationTrainingReserveSpaceSize(handle(), mode, normOps, algo, something(activationDesc,C_NULL), xDesc, reserveSpaceSize, groupCnt)
    workspaceSize[], reserveSpaceSize[]
end
