using Test, Random, Statistics, CUDA

using CUDA.CUDNN:
    cudnnNormalizationForward,
    cudnnNormalizationForward!,
    cudnnNormalizationForwardInference,
    cudnnNormalizationForwardTraining,
    cudnnNormalizationBackward,
    cudnnActivationDescriptor,
    cudnnNormMode_t,
        CUDNN_NORM_PER_ACTIVATION, # 0, bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per CHW...-slice, normalized over N slice)
        CUDNN_NORM_PER_CHANNEL,    # 1, bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors)
    cudnnNormOps_t,
        CUDNN_NORM_OPS_NORM,                # 0, /* do normalization only */
        CUDNN_NORM_OPS_NORM_ACTIVATION,     # 1, /* do Norm, then activation */
        CUDNN_NORM_OPS_NORM_ADD_ACTIVATION, # 2, /* do Norm, then elemWiseAdd, then activation */
    cudnnNormAlgo_t,
        CUDNN_NORM_ALGO_STANDARD, # 0
        CUDNN_NORM_ALGO_PERSIST,  # 1
    cudnnActivationMode_t,
        CUDNN_ACTIVATION_SIGMOID,      # 0
        CUDNN_ACTIVATION_RELU,         # 1
        CUDNN_ACTIVATION_TANH,         # 2
        CUDNN_ACTIVATION_CLIPPED_RELU, # 3
        CUDNN_ACTIVATION_ELU,          # 4
        CUDNN_ACTIVATION_IDENTITY,     # 5
    cudnnNanPropagation_t,
        CUDNN_NOT_PROPAGATE_NAN, # 0
        CUDNN_PROPAGATE_NAN,     # 1
    cudnnTensorFormat_t,
        CUDNN_TENSOR_NCHW,        # 0, /* row major (wStride = 1, hStride = w) */
        CUDNN_TENSOR_NHWC,        # 1, /* feature maps interleaved ( cStride = 1 )*/
        CUDNN_TENSOR_NCHW_VECT_C, # 2, /* each image point is vector of element of C, vector length in data type */
    handle


@testset "cudnn/normalization" begin

    function normtest(
        x;

        training = false,
        
        # Inference parameters:
        z = nothing, # for residual addition to the result of the normalization operation, prior to the activation
        mode::cudnnNormMode_t = CUDNN_NORM_PER_CHANNEL, # Per-channel layer is based on the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, S. Ioffe, C. Szegedy, 2015.
        normOps::cudnnNormOps_t = CUDNN_NORM_OPS_NORM,  # Currently CUDNN_NORM_OPS_NORM_ACTIVATION and CUDNN_NORM_OPS_NORM_ADD_ACTIVATION are only supported in the NHWC layout (training,backward), not supported (inference)
        algo::cudnnNormAlgo_t = CUDNN_NORM_ALGO_STANDARD, # trigger the new semi-persistent NHWC kernel when CUDNN_NORM_ALGO_PERSIST
        alpha::Real = 1,
        beta::Real = 0,
        epsilon::Real = 1e-5, # Has to be >= 0. Should be the same in forward and backward functions.
        groupCnt::Integer = 1, # Place hold for future work, should be set to 1 now

        # Main argument defaults:
        format::cudnnTensorFormat_t = CUDNN_TENSOR_NCHW, # or NHWC
        _sdims = (mode == CUDNN_NORM_PER_CHANNEL    && format == CUDNN_TENSOR_NCHW ? (1,1,size(x,3),1) :
                  mode == CUDNN_NORM_PER_CHANNEL    && format == CUDNN_TENSOR_NHWC ? (size(x,1),1,1,1) :
                  mode == CUDNN_NORM_PER_ACTIVATION && format == CUDNN_TENSOR_NCHW ? (size(x)[1:3]...,1) :
                  mode == CUDNN_NORM_PER_ACTIVATION && format == CUDNN_TENSOR_NHWC ? (size(x)[1:3]...,1) :
                  error("Unknown mode $mode and format $format")),
        scale = fill!(similar(x, _sdims), 1),
        bias =  fill!(similar(x, _sdims), 0),
        xmean =  fill!(similar(x, _sdims), 0),
        xvar = fill!(similar(x, _sdims), 1),

        # Training-only parameters:
        exponentialAverageFactor::Real = 0.1,
        savedMean = nothing, # Optionally save intermediate results from the forward pass here - can be reused to speed up backward pass. NULL if unused.
        savedInvVariance = nothing,

        # Activation parameters:
        activationMode::cudnnActivationMode_t = CUDNN_ACTIVATION_IDENTITY,
        activationReluNanOpt::cudnnNanPropagation_t = CUDNN_NOT_PROPAGATE_NAN,
        activationCoef::Real = 1,
        activationDesc::Union{Nothing,cudnnActivationDescriptor} = (normOps == CUDNN_NORM_OPS_NORM ? nothing : cudnnActivationDescriptor(activationMode, activationReluNanOpt, Cdouble(activationCoef))),
    )
        if training
            dims = findall(size(xmean) .== 1)
            m = mean(x; dims)
            v = var(x; dims, mean=m, corrected=false)
            y = bias .+ scale .* (x .- m) ./ sqrt.(epsilon .+ v)
        else
            y = bias .+ scale .* (x .- xmean) ./ sqrt.(epsilon .+ xvar)
        end
        y0 = randn!(similar(x))
        y1 = alpha * y
        y2 = y1 + beta * y0
        (y1 ≈ cudnnNormalizationForward(x, xmean, xvar, bias, scale; training, z, mode, normOps, algo, alpha, epsilon, groupCnt, format, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc) &&
         y2 ≈ cudnnNormalizationForward!(copy(y0), x, xmean, xvar, bias, scale; training, z, mode, normOps, algo, alpha, beta, epsilon, groupCnt, format, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc))
    end

    x, z, s = (CUDA.randn(x...) for x in ((5,4,3,2),(5,4,3,2),(1,1,3,1)))
    @test normtest(x)
    @test normtest(x; training = true)
    @test normtest(x; mode = CUDNN_NORM_PER_ACTIVATION)
    @test normtest(x; algo = CUDNN_NORM_ALGO_PERSIST)
    @test normtest(x; algo = CUDNN_NORM_ALGO_PERSIST, format = CUDNN_TENSOR_NHWC)
    @test normtest(x; alpha = 2)
    @test normtest(x; beta = 2)
    @test normtest(x; epsilon = 0)
    @test normtest(x; format = CUDNN_TENSOR_NHWC)
    @test normtest(x; scale = fill!(s, 2))
    @test normtest(x; bias  = fill!(s, 2))
    @test normtest(x; xmean  = fill!(s, 2))
    @test normtest(x; xvar = fill!(s, 2))
    @test normtest(x; exponentialAverageFactor = 0.01)
    @test normtest(x; savedMean = similar(s))
    @test normtest(x; savedInvVariance = similar(s))
    # cudnn-8.0.5: Currently, CUDNN_NORM_OPS_NORM_ACTIVATION and CUDNN_NORM_OPS_NORM_ADD_ACTIVATION are not supported in inference.
    #@test normtest(x; normOps = CUDNN_NORM_OPS_NORM_ACTIVATION, activationMode = CUDNN_ACTIVATION_RELU, format = CUDNN_TENSOR_NHWC)
    #@test normtest(x; normOps = CUDNN_NORM_OPS_NORM_ADD_ACTIVATION, activationMode = CUDNN_ACTIVATION_RELU, z, format = CUDNN_TENSOR_NHWC)
    #@test normtest(x; groupCnt = 2) # cudnn-8.0.5: Currently only groupCnt=1 is supported
end
