@run_if has_cudnn() begin

using CUDA.CUDNN
import NNlib

using Random, Statistics

@testset "CUDNN" begin

############################################################################################

@testcase "essentials" begin
    @test CUDNN.version() isa VersionNumber
end

############################################################################################

@testcase "normalization" begin
    function normtest(
        x;

        training = false,

        # Inference parameters:
        z = nothing, # for residual addition to the result of the normalization operation, prior to the activation
        mode::CUDNN.cudnnNormMode_t = CUDNN.CUDNN_NORM_PER_CHANNEL, # Per-channel layer is based on the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, S. Ioffe, C. Szegedy, 2015.
        normOps::CUDNN.cudnnNormOps_t = CUDNN.CUDNN_NORM_OPS_NORM,  # Currently CUDNN.CUDNN_NORM_OPS_NORM_ACTIVATION and CUDNN.CUDNN_NORM_OPS_NORM_ADD_ACTIVATION are only supported in the NHWC layout (training,backward), not supported (inference)
        algo::CUDNN.cudnnNormAlgo_t = CUDNN.CUDNN_NORM_ALGO_STANDARD, # trigger the new semi-persistent NHWC kernel when CUDNN.CUDNN_NORM_ALGO_PERSIST
        alpha::Real = 1,
        beta::Real = 0,
        epsilon::Real = 1e-5, # Has to be >= 0. Should be the same in forward and backward functions.
        groupCnt::Integer = 1, # Place hold for future work, should be set to 1 now

        # Main argument defaults:
        format::CUDNN.cudnnTensorFormat_t = CUDNN.CUDNN_TENSOR_NCHW, # or NHWC
        _sdims = (mode == CUDNN.CUDNN_NORM_PER_CHANNEL    && format == CUDNN.CUDNN_TENSOR_NCHW ? (1,1,size(x,3),1) :
                  mode == CUDNN.CUDNN_NORM_PER_CHANNEL    && format == CUDNN.CUDNN_TENSOR_NHWC ? (size(x,1),1,1,1) :
                  mode == CUDNN.CUDNN_NORM_PER_ACTIVATION && format == CUDNN.CUDNN_TENSOR_NCHW ? (size(x)[1:3]...,1) :
                  mode == CUDNN.CUDNN_NORM_PER_ACTIVATION && format == CUDNN.CUDNN_TENSOR_NHWC ? (size(x)[1:3]...,1) :
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
        activationMode::CUDNN.cudnnActivationMode_t = CUDNN.CUDNN_ACTIVATION_IDENTITY,
        activationReluNanOpt::CUDNN.cudnnNanPropagation_t = CUDNN.CUDNN_NOT_PROPAGATE_NAN,
        activationCoef::Real = 1,
        activationDesc::Union{Nothing,CUDNN.cudnnActivationDescriptor} = (normOps == CUDNN.CUDNN_NORM_OPS_NORM ? nothing : CUDNN.cudnnActivationDescriptor(activationMode, activationReluNanOpt, Cdouble(activationCoef))),
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
        (y1 ≈ CUDNN.cudnnNormalizationForward(x, xmean, xvar, bias, scale; training, z, mode, normOps, algo, alpha, epsilon, groupCnt, format, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc) &&
         y2 ≈ CUDNN.cudnnNormalizationForward!(copy(y0), x, xmean, xvar, bias, scale; training, z, mode, normOps, algo, alpha, beta, epsilon, groupCnt, format, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc))
    end

    x, z, s = (CUDA.randn(x...) for x in ((5,4,3,2),(5,4,3,2),(1,1,3,1)))
    @test normtest(x)
    @test normtest(x; training = true)
    @test normtest(x; mode = CUDNN.CUDNN_NORM_PER_ACTIVATION)
    @test normtest(x; algo = CUDNN.CUDNN_NORM_ALGO_PERSIST)
    @test normtest(x; algo = CUDNN.CUDNN_NORM_ALGO_PERSIST, format = CUDNN.CUDNN_TENSOR_NHWC)
    @test normtest(x; alpha = 2)
    @test normtest(x; beta = 2)
    @test normtest(x; epsilon = 0)
    @test normtest(x; format = CUDNN.CUDNN_TENSOR_NHWC)
    @test normtest(x; scale = fill!(s, 2))
    @test normtest(x; bias  = fill!(s, 2))
    @test normtest(x; xmean  = fill!(s, 2))
    @test normtest(x; xvar = fill!(s, 2))
    @test normtest(x; exponentialAverageFactor = 0.01)
    @test normtest(x; savedMean = similar(s))
    @test normtest(x; savedInvVariance = similar(s))
    # CUDNN.cudnn-8.0.5: Currently, CUDNN.CUDNN_NORM_OPS_NORM_ACTIVATION and CUDNN.CUDNN_NORM_OPS_NORM_ADD_ACTIVATION are not supported in inference.
    #@test normtest(x; normOps = CUDNN.CUDNN_NORM_OPS_NORM_ACTIVATION, activationMode = CUDNN.CUDNN_ACTIVATION_RELU, format = CUDNN.CUDNN_TENSOR_NHWC)
    #@test normtest(x; normOps = CUDNN.CUDNN_NORM_OPS_NORM_ADD_ACTIVATION, activationMode = CUDNN.CUDNN_ACTIVATION_RELU, z, format = CUDNN.CUDNN_TENSOR_NHWC)
    #@test normtest(x; groupCnt = 2) # CUDNN.cudnn-8.0.5: Currently only groupCnt=1 is supported
end

############################################################################################

@testcase "optensor" begin
    @test CUDNN.cudnnOpTensorDescriptor(C_NULL) isa CUDNN.cudnnOpTensorDescriptor
    @test Base.unsafe_convert(Ptr, CUDNN.cudnnOpTensorDescriptor(C_NULL)) isa Ptr
    @test CUDNN.cudnnOpTensorDescriptor(CUDNN.CUDNN_OP_TENSOR_ADD,CUDNN.cudnnDataType(Float32),CUDNN.CUDNN_NOT_PROPAGATE_NAN) isa CUDNN.cudnnOpTensorDescriptor

    (ax1,ax2,ay) = rand.((10,10,10))
    (cx1,cx2,cy) = CuArray.((ax1,ax2,ay))

    function optensortest(
        ;op=CUDNN.CUDNN_OP_TENSOR_ADD,
        nanOpt=CUDNN.CUDNN_NOT_PROPAGATE_NAN,
        compType=(eltype(ax1) <: Float64 ? Float64 : Float32),
        alpha1=1,
        alpha2=1,
        beta=0,
    )
        f1 = (op === CUDNN.CUDNN_OP_TENSOR_ADD ? alpha1*ax1 .+ alpha2*ax2 :
              op === CUDNN.CUDNN_OP_TENSOR_MUL ? (alpha1*ax1) .* (alpha2*ax2) :
              op === CUDNN.CUDNN_OP_TENSOR_MIN ? min.(alpha1*ax1, alpha2*ax2) :
              op === CUDNN.CUDNN_OP_TENSOR_MAX ? max.(alpha1*ax1, alpha2*ax2) :
              op === CUDNN.CUDNN_OP_TENSOR_SQRT ? sqrt.(alpha1*ax1) :
              op === CUDNN.CUDNN_OP_TENSOR_NOT ? 1 .- ax1 :
              error("Unknown optensor"))
        f2 = f1 .+ beta * ay
        d = CUDNN.cudnnOpTensorDescriptor(op,CUDNN.cudnnDataType(compType),nanOpt)
        ((f1 ≈ CUDNN.cudnnOpTensor(cx1, cx2; op, compType, nanOpt, alpha1, alpha2) |> Array) &&
         (f1 ≈ CUDNN.cudnnOpTensor(cx1, cx2, d; alpha1, alpha2) |> Array) &&
         (f2 ≈ CUDNN.cudnnOpTensor!(copy(cy), cx1, cx2; op, compType, nanOpt, alpha1, alpha2, beta) |> Array) &&
         (f2 ≈ CUDNN.cudnnOpTensor!(copy(cy), cx1, cx2, d; alpha1, alpha2, beta) |> Array))
    end

    @test optensortest(op = CUDNN.CUDNN_OP_TENSOR_ADD)
    @test optensortest(op = CUDNN.CUDNN_OP_TENSOR_MUL)
    @test optensortest(op = CUDNN.CUDNN_OP_TENSOR_MIN)
    @test optensortest(op = CUDNN.CUDNN_OP_TENSOR_MAX)
    @test optensortest(op = CUDNN.CUDNN_OP_TENSOR_SQRT)
    @test optensortest(op = CUDNN.CUDNN_OP_TENSOR_NOT)
    @test optensortest(nanOpt = CUDNN.CUDNN_PROPAGATE_NAN)
    @test optensortest(alpha1 = 2)
    @test optensortest(alpha2 = 2)
    @test optensortest(beta = 2)
end

############################################################################################

@testcase "pooling" begin
    function pooltest(;
                      mode = CUDNN.CUDNN_POOLING_MAX,
                      nanOpt = CUDNN.CUDNN_NOT_PROPAGATE_NAN,
                      window = 2,
                      padding = 0,
                      stride = window,
                      format = CUDNN.CUDNN_TENSOR_NCHW,
                      dataType = Float32,
                      alpha = 1,
                      beta = 0)
        ax = randn(dataType,12,6,4,2)
        N = ndims(ax)
        window = expand(Val(N-2), window)
        stride = expand(Val(N-2), stride)
        padding = expand(Val(N-2), padding)
        pdims = NNlib.PoolDims(ax, window; padding = padding, stride = stride)
        #=
        if mode == CUDNN.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
            @warn "Pool mode=$mode not yet implemented in NNlib, using INCLUDE instead. See https://github.com/FluxML/NNlib.jl/issues/218" maxlog=1
        end
        if mode == CUDNN.CUDNN_POOLING_MAX_DETERMINISTIC
            @warn "Pool mode=$mode not yet implemented in NNlib, using MAX instead." maxlog=1
        end
        if nanOpt == CUDNN.CUDNN_NOT_PROPAGATE_NAN
            @warn "Pool nanOpt=$nanOpt not yet implemented in NNlib, using PROPAGATE instead. See https://github.com/FluxML/NNlib.jl/issues/218" maxlog=1
        end
        =#
        ay1 = (mode == CUDNN.CUDNN_POOLING_MAX ? NNlib.maxpool(ax, pdims) :
               mode == CUDNN.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING ? NNlib.meanpool(ax, pdims) :
               mode == CUDNN.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING ? NNlib.meanpool(ax, pdims) :
               mode == CUDNN.CUDNN_POOLING_MAX_DETERMINISTIC ? NNlib.maxpool(ax, pdims) :
               error("mode=$mode is not supported."))
        ay1 = alpha * ay1
        ay  = randn!(similar(ay1))
        ay2 = ay1 .+ beta * ay
        d = CUDNN.cudnnPoolingDescriptor(mode, nanOpt, Cint(max(2,ndims(ax)-2)), CUDNN.pooldims(window,size(ax)), CUDNN.pooldims(padding,size(ax)), CUDNN.pooldims(stride,size(ax)))
        nhwc(a) = permutedims(a,(3,1,2,4))
        if format === CUDNN.CUDNN_TENSOR_NCHW
            cx, cy = CuArray.((ax, ay))
        else
            cx, cy = CuArray.(nhwc.((ax,ay)))
            ay1, ay2 = nhwc.((ay1, ay2))
        end
        ((ay1 ≈ CUDNN.cudnnPoolingForward(cx; mode, nanOpt, window, padding, stride, format, alpha) |> Array) &&
         (ay1 ≈ CUDNN.cudnnPoolingForward(cx, d; format, alpha) |> Array) &&
         (ay2 ≈ CUDNN.cudnnPoolingForward!(copy(cy), cx; mode, nanOpt, window, padding, stride, format, alpha, beta) |> Array) &&
         (ay2 ≈ CUDNN.cudnnPoolingForward!(copy(cy), cx, d; format, alpha, beta) |> Array))
    end

    expand(::Val{N}, i::NTuple{N}) where {N} = i
    expand(::Val{N}, i::Integer) where {N} = ntuple(_ -> i, N)


    @test pooltest()
    @test pooltest(mode = CUDNN.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
    @test pooltest(mode = CUDNN.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
    @test pooltest(mode = CUDNN.CUDNN_POOLING_MAX_DETERMINISTIC)
    @test pooltest(nanOpt = CUDNN.CUDNN_PROPAGATE_NAN)
    @test pooltest(window = 3)
    @test pooltest(padding = 1)
    @test pooltest(stride = 1)
    @test pooltest(format = CUDNN.CUDNN_TENSOR_NHWC)
    @test pooltest(dataType = Float16)
    @test pooltest(alpha = 2)
    @test pooltest(beta = 2)
end

############################################################################################

@testcase "reduce" begin
    @test CUDNN.cudnnReduceTensorDescriptor(C_NULL) isa CUDNN.cudnnReduceTensorDescriptor
    @test Base.unsafe_convert(Ptr, CUDNN.cudnnReduceTensorDescriptor(C_NULL)) isa Ptr
    @test CUDNN.cudnnReduceTensorDescriptor(CUDNN.CUDNN_REDUCE_TENSOR_ADD,CUDNN.cudnnDataType(Float32),CUDNN.CUDNN_NOT_PROPAGATE_NAN,CUDNN.CUDNN_REDUCE_TENSOR_NO_INDICES,CUDNN.CUDNN_32BIT_INDICES) isa CUDNN.cudnnReduceTensorDescriptor

    (ax,ay) = randn(Float32,10,10), randn(Float32,10,1)
    (cx,cy) = CuArray.((ax,ay))

    function reducetensortest(
        ; op::CUDNN.cudnnReduceTensorOp_t = CUDNN.CUDNN_REDUCE_TENSOR_ADD,
        compType::DataType = (eltype(ax) <: Float64 ? Float64 : Float32),
        nanOpt::CUDNN.cudnnNanPropagation_t = CUDNN.CUDNN_NOT_PROPAGATE_NAN,
        indices::Union{Vector{<:Unsigned},Nothing} = nothing,
        d::CUDNN.cudnnReduceTensorDescriptor = CUDNN.cudnnReduceTensorDescriptor(op, CUDNN.cudnnDataType(compType), nanOpt, CUDNN.cudnnReduceTensorIndices(op, indices), CUDNN.cudnnIndicesType(indices)),
        alpha::Real = 1,
        beta::Real = 0,
    )
        f0 = (op === CUDNN.CUDNN_REDUCE_TENSOR_ADD          ? sum(ax, dims=2) :
              op === CUDNN.CUDNN_REDUCE_TENSOR_MUL          ? prod(ax, dims=2) :
              op === CUDNN.CUDNN_REDUCE_TENSOR_MIN          ? minimum(ax, dims=2) :
              op === CUDNN.CUDNN_REDUCE_TENSOR_MAX          ? maximum(ax, dims=2) :
              op === CUDNN.CUDNN_REDUCE_TENSOR_AMAX         ? maximum(abs, ax, dims=2) :
              op === CUDNN.CUDNN_REDUCE_TENSOR_AVG          ? mean(ax, dims=2) :
              op === CUDNN.CUDNN_REDUCE_TENSOR_NORM1        ? sum(abs, ax, dims=2) :
              op === CUDNN.CUDNN_REDUCE_TENSOR_NORM2        ? sqrt.(sum(abs2, ax, dims=2)) :
              op === CUDNN.CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS ? (ax1=copy(ax);ax1[ax.==0].=1;prod(ax1,dims=2)) :
              error("Unknown reducetensor"))
        f1 = alpha * f0
        f2 = f1 + beta * ay
        dims = size(ay)
        ((f1 ≈ CUDNN.cudnnReduceTensor(cx; dims, op, compType, nanOpt, indices, alpha) |> Array) &&
         (f1 ≈ CUDNN.cudnnReduceTensor(cx, d; dims, indices, alpha) |> Array) &&
         (f2 ≈ CUDNN.cudnnReduceTensor!(copy(cy), cx; op, compType, nanOpt, indices, alpha, beta) |> Array) &&
         (f2 ≈ CUDNN.cudnnReduceTensor!(copy(cy), cx, d; indices, alpha, beta) |> Array))
    end

    @test reducetensortest()
    @test reducetensortest(op = CUDNN.CUDNN_REDUCE_TENSOR_MUL)
    @test reducetensortest(op = CUDNN.CUDNN_REDUCE_TENSOR_MIN)
    @test reducetensortest(op = CUDNN.CUDNN_REDUCE_TENSOR_MAX)
    @test reducetensortest(op = CUDNN.CUDNN_REDUCE_TENSOR_AMAX)
    @test reducetensortest(op = CUDNN.CUDNN_REDUCE_TENSOR_AVG)
    @test reducetensortest(op = CUDNN.CUDNN_REDUCE_TENSOR_NORM1)
    @test reducetensortest(op = CUDNN.CUDNN_REDUCE_TENSOR_NORM2)
    @test reducetensortest(op = CUDNN.CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS)
    @test reducetensortest(nanOpt = CUDNN.CUDNN_PROPAGATE_NAN)
    @test reducetensortest(alpha = 2)
    @test reducetensortest(beta = 2)
end

############################################################################################

@testcase "rnn" begin
    X,H,B,T = 8,8,4,2
    w = CUDA.randn(10000)
    x = CUDA.randn(X,B,T)
    hx1 = CUDA.randn(H,B,1)
    cx1 = CUDA.randn(H,B,1)

    function rnntest(
        ;hx = nothing,
        cx = nothing,
        hy = nothing,
        cy = nothing,
        layout::CUDNN.cudnnRNNDataLayout_t = CUDNN.CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
        seqLengthArray::Union{Nothing,Vector{Cint}} = nothing,
        fwdMode::CUDNN.cudnnForwardMode_t = CUDNN.CUDNN_FWD_MODE_INFERENCE,
        # descriptor keywords
        hiddenSize::Integer = H,
        algo::CUDNN.cudnnRNNAlgo_t = CUDNN.CUDNN_RNN_ALGO_STANDARD,
        cellMode::CUDNN.cudnnRNNMode_t = CUDNN.CUDNN_LSTM,
        biasMode::CUDNN.cudnnRNNBiasMode_t = CUDNN.CUDNN_RNN_DOUBLE_BIAS,
        dirMode::CUDNN.cudnnDirectionMode_t = CUDNN.CUDNN_UNIDIRECTIONAL,
        inputMode::CUDNN.cudnnRNNInputMode_t = CUDNN.CUDNN_LINEAR_INPUT,
        mathPrec::DataType = eltype(x),
        mathType::CUDNN.cudnnMathType_t = CUDNN.math_mode(),
        inputSize::Integer = size(x,1),
        projSize::Integer = hiddenSize,
        numLayers::Integer = 1,
        dropout::Real = 0,
        auxFlags::Integer = CUDNN.CUDNN_RNN_PADDED_IO_ENABLED,
    )
        d = CUDNN.cudnnRNNDescriptor(algo, cellMode, biasMode, dirMode, inputMode, CUDNN.cudnnDataType(eltype(x)), CUDNN.cudnnDataType(mathPrec), mathType, Int32(inputSize), Int32(hiddenSize), Int32(projSize), Int32(numLayers), CUDNN.cudnnDropoutDescriptor(Cfloat(dropout)), UInt32(auxFlags))
        y = CUDNN.cudnnRNNForward(w, x; hx, cx, hy, cy, layout, seqLengthArray, fwdMode, hiddenSize, algo, cellMode, biasMode, dirMode, inputMode, mathPrec, mathType, inputSize, projSize, numLayers, dropout, auxFlags)
        _y = copy(y)
        _hy = (hy === nothing ? hy : copy(hy[]))
        _cy = (cy === nothing ? cy : copy(cy[]))
        (_y ≈ CUDNN.cudnnRNNForward!(y, w, x; hx, cx, hy, cy, layout, seqLengthArray, fwdMode, hiddenSize, algo, cellMode, biasMode, dirMode, inputMode, mathPrec, mathType, inputSize, projSize, numLayers, dropout, auxFlags) &&
         (_hy === hy === nothing || _hy ≈ hy[]) &&
         (_cy === cy === nothing || _cy ≈ cy[]) &&
         _y ≈ CUDNN.cudnnRNNForward(w, x, d; hx, cx, hy, cy, layout, seqLengthArray, fwdMode) &&
         (_hy === hy === nothing || _hy ≈ hy[]) &&
         (_cy === cy === nothing || _cy ≈ cy[]) &&
         _y ≈ CUDNN.cudnnRNNForward!(y, w, x, d; hx, cx, hy, cy, layout, seqLengthArray, fwdMode) &&
         (_hy === hy === nothing || _hy ≈ hy[]) &&
         (_cy === cy === nothing || _cy ≈ cy[]))
    end

    @test rnntest()
    @test rnntest(hx=hx1)
    @test rnntest(cx=cx1)
    @test rnntest(hy=Ref{Any}())
    @test rnntest(cy=Ref{Any}())
    @test rnntest(layout=CUDNN.CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED)
    @test rnntest(layout=CUDNN.CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED)
    @test rnntest(seqLengthArray=Cint[1,2,1,2])
    @test rnntest(fwdMode=CUDNN.CUDNN_FWD_MODE_TRAINING)
    @test rnntest(hiddenSize=16)
    @test rnntest(algo=CUDNN.CUDNN_RNN_ALGO_PERSIST_STATIC)
    #@test rnntest(algo=CUDNN.CUDNN_RNN_ALGO_PERSIST_DYNAMIC) # causes segfault
    @test rnntest(cellMode=CUDNN.CUDNN_RNN_RELU)
    @test rnntest(cellMode=CUDNN.CUDNN_RNN_TANH)
    @test rnntest(cellMode=CUDNN.CUDNN_GRU)
    @test rnntest(biasMode=CUDNN.CUDNN_RNN_NO_BIAS)
    @test rnntest(biasMode=CUDNN.CUDNN_RNN_SINGLE_INP_BIAS)
    @test rnntest(biasMode=CUDNN.CUDNN_RNN_SINGLE_REC_BIAS)
    @test rnntest(dirMode=CUDNN.CUDNN_BIDIRECTIONAL)
    @test rnntest(inputMode=CUDNN.CUDNN_SKIP_INPUT)
    @test rnntest(mathPrec=Float32) # only possible option for F32 input
    @test rnntest(mathType=CUDNN.CUDNN_DEFAULT_MATH)
    @test rnntest(mathType=CUDNN.CUDNN_TENSOR_OP_MATH)
    @test rnntest(mathType=CUDNN.CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)
    @test rnntest(projSize=4)
    @test rnntest(numLayers=2)
    @test rnntest(dropout=0.5)
    @test rnntest(auxFlags=CUDNN.CUDNN_RNN_PADDED_IO_DISABLED)
    @test rnntest(auxFlags=CUDNN.CUDNN_RNN_PADDED_IO_ENABLED)
end

############################################################################################

@testcase "softmax" begin
    ax,ay = randn(Float32,10,10),randn(Float32,10,10)
    cx,cy = CuArray.((ax,ay))

    function softmaxtest(
        ; alpha=1,
        beta=0,
        mode=CUDNN.CUDNN_SOFTMAX_MODE_INSTANCE,
        algo=CUDNN.CUDNN_SOFTMAX_FAST
    )
        d = mode === CUDNN.CUDNN_SOFTMAX_MODE_INSTANCE ? 1 : 2
        x = ax .- maximum(ax, dims=d)
        y = x .- log.(sum(exp.(x), dims=d))
        if algo !== CUDNN.CUDNN_SOFTMAX_LOG; y = exp.(y); end
        add1(x)=reshape(x, (size(x)..., 1))
        if mode === CUDNN.CUDNN_SOFTMAX_MODE_CHANNEL
            y,cx1,cy1 = add1.((y,cx,cy))
        else
            cx1,cy1 = cx,cy
        end
        y0 = alpha * y
        y1 = y0 .+ beta * ay
        ((y0 ≈ CUDNN.cudnnSoftmaxForward(cx1; algo, mode, alpha) |> Array) &&
         (y1 ≈ CUDNN.cudnnSoftmaxForward!(copy(cy1), cx1; algo, mode, alpha, beta) |> Array))
    end

    @test softmaxtest()
    @test softmaxtest(alpha=2)
    @test softmaxtest(beta=2)
    @test softmaxtest(mode=CUDNN.CUDNN_SOFTMAX_MODE_INSTANCE)
    @test softmaxtest(mode=CUDNN.CUDNN_SOFTMAX_MODE_CHANNEL)
    @test softmaxtest(algo=CUDNN.CUDNN_SOFTMAX_FAST)
    @test softmaxtest(algo=CUDNN.CUDNN_SOFTMAX_ACCURATE)
    @test softmaxtest(algo=CUDNN.CUDNN_SOFTMAX_LOG)
end

############################################################################################

@testcase "tensor" begin
    x = CUDA.rand(1,1,1,2)

    TD = CUDNN.cudnnTensorDescriptor
    FD = CUDNN.cudnnFilterDescriptor
    DT = CUDNN.cudnnDataType

    @test TD(x) isa TD
    @test TD(CUDNN.CUDNN_TENSOR_NCHW, DT(eltype(x)), Cint(ndims(x)), Cint[reverse(size(x))...]) isa TD
    td = TD(x)
    @test TD(td.ptr) isa TD
    @test Base.unsafe_convert(Ptr, TD(td.ptr)) isa Ptr

    @test FD(x) isa FD
    @test FD(DT(eltype(x)),CUDNN.CUDNN_TENSOR_NCHW,Cint(ndims(x)),Cint[reverse(size(x))...]) isa FD
    fd = FD(x)
    @test FD(fd.ptr) isa FD
    @test Base.unsafe_convert(Ptr, FD(fd.ptr)) isa Ptr

    @test DT(Float32) isa CUDNN.cudnnDataType_t

    @test (CUDA.@retry_reclaim(x->(x!==CUDNN.CUDNN_STATUS_SUCCESS),CUDNN.cudnnCreateTensorDescriptor(Ref{Ptr{Cvoid}}(C_NULL)))) isa Nothing
end

############################################################################################

@testcase "activation" begin
    @test CUDNN.cudnnActivationDescriptor(C_NULL) isa CUDNN.cudnnActivationDescriptor
    @test Base.unsafe_convert(Ptr, CUDNN.cudnnActivationDescriptor(C_NULL)) isa Ptr
    @test CUDNN.cudnnActivationDescriptor(CUDNN.CUDNN_ACTIVATION_RELU,CUDNN.CUDNN_NOT_PROPAGATE_NAN,0) isa CUDNN.cudnnActivationDescriptor

    (ax,ay) = randn.((10,10))
    (cx,cy) = CuArray.((ax,ay))

    function activationtest(
        ;mode=CUDNN.CUDNN_ACTIVATION_SIGMOID,
        nanOpt=CUDNN.CUDNN_NOT_PROPAGATE_NAN,
        coef=1,
        alpha=1,
        beta=0,
    )
        fx = (mode === CUDNN.CUDNN_ACTIVATION_SIGMOID ? 1 ./ (1 .+ exp.(-ax)) :
              mode === CUDNN.CUDNN_ACTIVATION_RELU ? max.(0,ax) :
              mode === CUDNN.CUDNN_ACTIVATION_TANH ? tanh.(ax) :
              mode === CUDNN.CUDNN_ACTIVATION_CLIPPED_RELU ? clamp.(ax,0,coef) :
              mode === CUDNN.CUDNN_ACTIVATION_ELU ? (x->(x >= 0 ? x : coef*(exp(x)-1))).(ax) :
              error("Unknown activation"))
        d = CUDNN.cudnnActivationDescriptor(mode,nanOpt,Cfloat(coef))
        y0 = alpha * fx
        y1 = y0 .+ beta * ay
        ((y0 ≈ CUDNN.cudnnActivationForward(cx; mode, nanOpt, coef, alpha) |> Array) &&
         (y0 ≈ CUDNN.cudnnActivationForward(cx, d; alpha) |> Array) &&
         (y1 ≈ CUDNN.cudnnActivationForward!(copy(cy), cx; mode, nanOpt, coef, alpha, beta) |> Array) &&
         (y1 ≈ CUDNN.cudnnActivationForward!(copy(cy), cx, d; alpha, beta) |> Array))
    end

    @test activationtest(mode=CUDNN.CUDNN_ACTIVATION_SIGMOID)
    @test activationtest(mode=CUDNN.CUDNN_ACTIVATION_RELU)
    @test activationtest(mode=CUDNN.CUDNN_ACTIVATION_TANH)
    @test activationtest(mode=CUDNN.CUDNN_ACTIVATION_CLIPPED_RELU)
    @test activationtest(mode=CUDNN.CUDNN_ACTIVATION_ELU)
    @test activationtest(nanOpt=CUDNN.CUDNN_PROPAGATE_NAN)
    @test activationtest(coef=2,mode=CUDNN.CUDNN_ACTIVATION_CLIPPED_RELU)
    @test activationtest(coef=2,mode=CUDNN.CUDNN_ACTIVATION_ELU)
    @test activationtest(alpha=2)
    @test activationtest(beta=2)
end

############################################################################################

@testcase "convolution" begin
    T = Float32
    ax,aw,ab = randn(T,8,8,4,4),randn(T,3,3,4,4),randn(T,1,1,4,1)
    cx,cw,cb = CuArray.((ax,aw,ab))

    function convtest(;
                      blendz=false,
                      bias=nothing,
                      activation = CUDNN.CUDNN_ACTIVATION_IDENTITY,
                      mode = CUDNN.CUDNN_CONVOLUTION,
                      padding = 0,
                      stride = 1,
                      dilation = 1,
                      group = 1,
                      dataType = eltype(cx),
                      mathType = CUDNN.math_mode(),
                      reorderType = CUDNN.CUDNN_DEFAULT_REORDER,
                      alpha = 1,
                      beta = 0)
        if group == 1
            cdims = NNlib.DenseConvDims(ax, aw; stride, padding, dilation, flipkernel = (mode === CUDNN.CUDNN_CROSS_CORRELATION))
            ay = NNlib.conv(ax, aw, cdims)
            cw0 = cw
        else
            # Implement grouped convolution
            xchan = size(aw,3)÷group
            ychan = size(aw,4)÷group
            xdims = (size(ax,1),size(ax,2),xchan,size(ax,4))
            wdims = (size(aw,1),size(aw,2),xchan,ychan)
            cdims = NNlib.DenseConvDims(xdims, wdims; stride, padding, dilation, flipkernel = (mode === CUDNN.CUDNN_CROSS_CORRELATION))
            ay = nothing
            for g in 1:group
                xrange = 1+(g-1)*xchan:g*xchan
                yrange = 1+(g-1)*ychan:g*ychan
                ay0 = NNlib.conv(ax[:,:,xrange,:], aw[:,:,1:xchan,yrange], cdims)
                ay = (ay === nothing ? ay0 : cat(ay, ay0; dims=3))
            end
            cw0 = CuArray(aw[:,:,1:xchan,:])
        end

        if alpha != 1; ay = alpha * ay; end
        if bias != nothing; ay = ay .+ Array(bias); end

        act = (activation === CUDNN.CUDNN_ACTIVATION_RELU ? NNlib.relu :
               activation === CUDNN.CUDNN_ACTIVATION_IDENTITY ? identity :
               error("Unsupported activation $activation"))
        ay1 = act.(ay)

        az0 = randn(T,size(ay)...)
        ay0 = randn(T,size(ay)...)
        cy0, cy1 = CuArray.((ay0,ay0))
        if blendz
            cz0 = cz1 = CuArray(az0)
            ay2 = act.(ay .+ beta * az0)
        else
            cz0, cz1 = cy0, cy1
            ay2 = act.(ay .+ beta * ay0)
        end
        d = CUDNN.cudnnConvolutionDescriptor(CUDNN.convdims(padding,size(ax)), CUDNN.convdims(stride,size(ax)),CUDNN. convdims(dilation,size(ax)), mode, CUDNN.cudnnDataType(dataType), mathType, reorderType, Cint(group))
        ((ay1 ≈ CUDNN.cudnnConvolutionForward(cw0, cx; bias, activation, mode, padding, stride, dilation, group, mathType, reorderType, alpha) |> Array) &&
         (ay1 ≈ CUDNN.cudnnConvolutionForward(cw0, cx, d; bias, activation, alpha) |> Array) &&
         (ay2 ≈ CUDNN.cudnnConvolutionForward!(cy0, cw0, cx; z=cz0, bias, activation, mode, padding, stride, dilation, group, mathType, reorderType, alpha, beta) |> Array) &&
         (ay2 ≈ CUDNN.cudnnConvolutionForward!(cy1, cw0, cx, d; z=cz1, bias, activation, alpha, beta) |> Array))
    end

    # These call CUDNN.cudnnConvolutionForward
    @test convtest()
    @test convtest(padding=1)
    @test convtest(stride=2)
    @test convtest(dilation=2)
    @test convtest(group=2) # See https://blog.yani.ai/filter-group-tutorial/
    @test convtest(mathType=CUDNN.CUDNN_DEFAULT_MATH)
    @test convtest(mathType=CUDNN.CUDNN_TENSOR_OP_MATH)
    @test convtest(mathType=CUDNN.CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)
    @test convtest(reorderType=CUDNN.CUDNN_NO_REORDER)
    @test convtest(alpha=2)
    @test convtest(beta=2)

    # These call CUDNN.cudnnConvolutionBiasActivationForward
    @test convtest(bias=cb)
    @test convtest(blendz=true)
    @test convtest(activation=CUDNN.CUDNN_ACTIVATION_RELU)
    @test convtest(bias=cb,blendz=true)
    @test convtest(bias=cb,activation=CUDNN.CUDNN_ACTIVATION_RELU)
    @test convtest(bias=cb,padding=1)
    @test convtest(bias=cb,stride=2)
    @test convtest(bias=cb,dilation=2)
    @test convtest(bias=cb,group=2)
    @test convtest(bias=cb,mathType=CUDNN.CUDNN_DEFAULT_MATH)
    @test convtest(bias=cb,mathType=CUDNN.CUDNN_TENSOR_OP_MATH)
    @test convtest(bias=cb,mathType=CUDNN.CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)
    @test convtest(bias=cb,reorderType=CUDNN.CUDNN_NO_REORDER)
    @test convtest(bias=cb,alpha=2)
    @test convtest(bias=cb,beta=2)
    @test convtest(bias=cb,beta=2,blendz=true)

    # Test tensor format
    cx2,cw2,cb2 = (x->permutedims(x,(3,1,2,4))).((cx,cw,cb))
    whcn = CUDNN.cudnnConvolutionForward(cw,cx)
    cwhn = CUDNN.cudnnConvolutionForward(cw2,cx2,format=CUDNN.CUDNN_TENSOR_NHWC)
    @test cwhn ≈ permutedims(whcn,(3,1,2,4))
    whcn = CUDNN.cudnnConvolutionForward(cw,cx;bias=cb)
    cwhn = CUDNN.cudnnConvolutionForward(cw2,cx2;bias=cb2,format=CUDNN.CUDNN_TENSOR_NHWC)
    @test cwhn ≈ permutedims(whcn,(3,1,2,4))
end


############################################################################################

@testcase "dropout" begin
    @test CUDNN.cudnnDropoutDescriptor(C_NULL) isa CUDNN.cudnnDropoutDescriptor
    @test Base.unsafe_convert(Ptr, CUDNN.cudnnDropoutDescriptor(C_NULL)) isa Ptr
    @test CUDNN.cudnnDropoutDescriptor(0.5) isa CUDNN.cudnnDropoutDescriptor

    N,P = 1000, 0.7
    x = CUDA.rand(N)
    d = CUDNN.cudnnDropoutDescriptor(P)
    CUDNN.cudnnDropoutSeed[] = 1
    y = CUDNN.cudnnDropoutForward(x; dropout = P)
    @test isapprox(mean(Array(y).==0), P; atol = 3/sqrt(N))
    @test y == CUDNN.cudnnDropoutForward(x, d)
    @test y == CUDNN.cudnnDropoutForward!(similar(x), x; dropout = P)
    @test y == CUDNN.cudnnDropoutForward!(similar(x), x, d)
    CUDNN.cudnnDropoutSeed[] = -1
end

############################################################################################

@testcase "inplace" begin
    x = CUDA.rand(10)
    CUDNN.cudnnSetTensor!(x, 7)
    @test all(isequal(7), Array(x))
    ax = rand(10)
    cx = CuArray(ax)
    @test 7*ax ≈ CUDNN.cudnnScaleTensor(cx, 7) |> Array
    @test 7*ax ≈ CUDNN.cudnnScaleTensor!(similar(cx), cx, 7) |> Array
    ax,ab = rand(5,4,3,2),rand(1,1,3,1)
    cx,cb = CuArray.((ax,ab))
    @test ax .+ ab ≈ CUDNN.cudnnAddTensor(cx, cb) |> Array
    @test ax .+ 7*ab ≈ CUDNN.cudnnAddTensor(cx, cb, alpha=7) |> Array
    @test 7*ax .+ ab ≈ CUDNN.cudnnAddTensor(cx, cb, beta=7) |> Array
    @test ax .+ ab ≈ CUDNN.cudnnAddTensor!(similar(cx), cx, cb) |> Array
    @test ax .+ 7*ab ≈ CUDNN.cudnnAddTensor!(similar(cx), cx, cb, alpha=7) |> Array
    @test 7*ax .+ ab ≈ CUDNN.cudnnAddTensor!(similar(cx), cx, cb, beta=7) |> Array
    @test ax .+ ab ≈ CUDNN.cudnnAddTensor!(cx, cx, cb) |> Array
    @test ax .+ ab ≈ cx |> Array
    ax,ab = rand(3,5,4,2),rand(3,1,1,1)
    cx,cb = CuArray.((ax,ab))
    @test ax .+ ab ≈ CUDNN.cudnnAddTensor(cx, cb, format=CUDNN.CUDNN_TENSOR_NHWC) |> Array
end

############################################################################################

@testcase "multiheadattn" begin
    function mhatest(
        # Input tensor descriptors
        ;axes::Vector{CUDNN.cudnnSeqDataAxis_t} = CUDNN.cudnnSeqDataDefaultAxes,
        seqLengthsQO::Vector{<:Integer} = fill(Cint(CUDNN.sdim(queries,axes,CUDNN.CUDNN_SEQDATA_TIME_DIM)), CUDNN.sdim(queries,axes,CUDNN.CUDNN_SEQDATA_BATCH_DIM)*CUDNN.sdim(queries,axes,CUDNN.CUDNN_SEQDATA_BEAM_DIM)),
        seqLengthsKV::Vector{<:Integer} = fill(Cint(CUDNN.sdim(keys,axes,CUDNN.CUDNN_SEQDATA_TIME_DIM)), CUDNN.sdim(keys,axes,CUDNN.CUDNN_SEQDATA_BATCH_DIM)*CUDNN.sdim(keys,axes,CUDNN.CUDNN_SEQDATA_BEAM_DIM)),
        #devSeqLengthsQO::CuVector{Cint} = convert(CuVector{Cint}, seqLengthsQO),
        #devSeqLengthsKV::CuVector{Cint} = convert(CuVector{Cint}, seqLengthsKV),
        #qDesc::CUDNN.cudnnSeqDataDescriptor = CUDNN.cudnnSeqDataDescriptor(queries; axes, seqLengthArray=seqLengthsQO),
        #kDesc::CUDNN.cudnnSeqDataDescriptor = CUDNN.cudnnSeqDataDescriptor(keys;    axes, seqLengthArray=seqLengthsKV),
        #vDesc::CUDNN.cudnnSeqDataDescriptor = CUDNN.cudnnSeqDataDescriptor(values;  axes, seqLengthArray=seqLengthsKV),

        # attnDesc parameters
        attnMode::Unsigned = CUDNN.CUDNN_ATTN_QUERYMAP_ALL_TO_ONE | CUDNN.CUDNN_ATTN_DISABLE_PROJ_BIASES |> Cuint,
        nHeads::Integer = Cint(1),
        smScaler::Real = Cdouble(1),
        # dataType::DataType = eltype(queries),
        # computePrec::DataType = eltype(queries),  ## No other option according to 8.0.2
        mathType::CUDNN.cudnnMathType_t = CUDNN.math_mode(),
        # attnDropout::Real = 0, ## The dropout option is currently not supported by the multi-head attention API
        # postDropout::Real = 0, ## The dropout option is currently not supported by the multi-head attention API
        qProjSize::Integer = 0, # Use zero to disable the corresponding projection
        kProjSize::Integer = 0,
        vProjSize::Integer = 0,
        oProjSize::Integer = 0,
        qoMaxSeqLength::Integer = CUDNN.sdim(queries,axes,CUDNN.CUDNN_SEQDATA_TIME_DIM),
        kvMaxSeqLength::Integer = CUDNN.sdim(keys,axes,CUDNN.CUDNN_SEQDATA_TIME_DIM),
        maxBatchSize::Integer = CUDNN.sdim(queries,axes,CUDNN.CUDNN_SEQDATA_BATCH_DIM),
        maxBeamSize::Integer = CUDNN.sdim(queries,axes,CUDNN.CUDNN_SEQDATA_BEAM_DIM),

        # forw parameters
        residuals = nothing,
        currIdx::Integer = -1,
        loWinIdx::Array{Cint} = fill(Cint(0), qoMaxSeqLength),
        hiWinIdx::Array{Cint} = fill(Cint(kvMaxSeqLength), qoMaxSeqLength),
        #workspace::Union{CuArray,Nothing}    = nothing,
        #reserveSpace::Union{CuArray,Nothing} = nothing,
    )
        attnDesc::CUDNN.cudnnAttnDescriptor = CUDNN.cudnnAttnDescriptor(
            Cuint(attnMode),
            Cint(nHeads),
            Cdouble(smScaler),
            CUDNN.cudnnDataType(eltype(queries)),    # dataType
            CUDNN.cudnnDataType(eltype(queries)),    # computePrec
            mathType,
            C_NULL,  # attnDropout
            C_NULL,  # postDropout
            Cint(CUDNN.sdim(queries,axes,CUDNN.CUDNN_SEQDATA_VECT_DIM)), # qSize
            Cint(CUDNN.sdim(keys,   axes,CUDNN.CUDNN_SEQDATA_VECT_DIM)), # kSize
            Cint(CUDNN.sdim(values, axes,CUDNN.CUDNN_SEQDATA_VECT_DIM)), # vSize
            Cint(qProjSize),
            Cint(kProjSize),
            Cint(vProjSize),
            Cint(oProjSize),
            Cint(qoMaxSeqLength),
            Cint(kvMaxSeqLength),
            Cint(maxBatchSize),
            Cint(maxBeamSize)
        )
        y = CUDNN.cudnnMultiHeadAttnForward(weights, queries, keys, values; axes, seqLengthsQO, seqLengthsKV, attnMode, nHeads, smScaler, mathType, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize, residuals, currIdx, loWinIdx, hiWinIdx)
        (y ≈ CUDNN.cudnnMultiHeadAttnForward!(zero(y), weights, queries, keys, values; axes, seqLengthsQO, seqLengthsKV, attnMode, nHeads, smScaler, mathType, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize, residuals, currIdx, loWinIdx, hiWinIdx) &&
         y ≈ CUDNN.cudnnMultiHeadAttnForward(weights, queries, keys, values, attnDesc; axes, seqLengthsQO, seqLengthsKV, residuals, currIdx, loWinIdx, hiWinIdx) &&
         y ≈ CUDNN.cudnnMultiHeadAttnForward!(zero(y), weights, queries, keys, values, attnDesc; axes, seqLengthsQO, seqLengthsKV, residuals, currIdx, loWinIdx, hiWinIdx))
    end

    Q,K,V,B,T,F = 6,6,5,4,3,Float32

    weights, queries, keys, values = (CUDA.randn(x...) for x in ((F,100),(F,Q,B,T),(F,K,B,T),(F,V,B,T)))
    @test mhatest()
    @test mhatest(attnMode = CUDNN.CUDNN_ATTN_QUERYMAP_ALL_TO_ONE | CUDNN.CUDNN_ATTN_ENABLE_PROJ_BIASES |> Cuint, vProjSize=7)
    @test mhatest(seqLengthsQO = Cint[1,2,3,1])
    @test mhatest(seqLengthsKV = Cint[1,2,3,1])
    @test mhatest(nHeads = 2)
    @test mhatest(smScaler = 2)
    @test mhatest(mathType = CUDNN.CUDNN_DEFAULT_MATH)
    @test mhatest(mathType = CUDNN.CUDNN_TENSOR_OP_MATH)
    @test mhatest(mathType = CUDNN.CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)
    @test mhatest(mathType = CUDNN.CUDNN_FMA_MATH)
    @test mhatest(kProjSize = 7, qProjSize = 7) # k and q have to match
    @test mhatest(vProjSize = 7)
    @test mhatest(oProjSize = 7)
    @test mhatest(qoMaxSeqLength = 7)
    @test mhatest(kvMaxSeqLength = 7)
    @test mhatest(maxBatchSize = 7)
    @test mhatest(maxBeamSize = 7)
    @test mhatest(loWinIdx = fill(Cint(1),T))
    @test mhatest(hiWinIdx = fill(Cint(1),T))
    @test mhatest(currIdx = 0)

    # Test residuals: residuals and output (and thus values unless oProjSize>0) must match queries in vector size
    values, residuals = (CUDA.randn(x...) for x in ((F,Q,B,T),(F,Q,B,T)))
    @test mhatest(residuals = residuals)

    # Test nonstandard axes order
    weights, queries, keys, values = (CUDA.randn(x...) for x in ((F,100),(F,Q,T,B),(F,K,T,B),(F,V,T,B)))
    @test mhatest(axes = [CUDNN.CUDNN_SEQDATA_VECT_DIM, CUDNN.CUDNN_SEQDATA_TIME_DIM, CUDNN.CUDNN_SEQDATA_BATCH_DIM, CUDNN.CUDNN_SEQDATA_BEAM_DIM])

    # Test beam handling
    weights, queries, keys, values = (CUDA.randn(x...) for x in ((F,100),(F,Q,B,T,2),(F,K,B,T,1),(F,V,B,T,1)))
    @test mhatest()
    # CUDNN.CUDNN_ATTN_QUERYMAP_ONE_TO_ONE does not seem to be supported
    # weights, queries, keys, values = (CUDA.randn(x...) for x in ((F,100),(F,Q,B,T,M),(F,K,B,T,M),(F,V,B,T,M)))
    # @test mhatest(attnMode = CUDNN.CUDNN_ATTN_QUERYMAP_ONE_TO_ONE | CUDNN.CUDNN_ATTN_DISABLE_PROJ_BIASES |> Cuint) ## Not supported
end

############################################################################################

@testset "NNlib" begin

using NNlib
using NNlib: ∇conv_data, ∇conv_filter,
    maxpool, meanpool, ∇maxpool, ∇meanpool,
    softmax, ∇softmax, logsoftmax, ∇logsoftmax

@testcase "essentials" begin
    a, b, c = rand(Float64, 10, 10, 3, 1), rand(Float64, 2, 2, 3, 4), rand(Float64, 9, 9, 4, 1)
    da, db, dc = CuArray(a), CuArray(b), CuArray(c)
    cdims = DenseConvDims(a, b)
    @test NNlib.conv(a, b, cdims) ≈ collect(NNlib.conv(da, db, cdims))
    @test ∇conv_data(c, b, cdims) ≈ collect(∇conv_data(dc, db, cdims))
    @test ∇conv_filter(a, c, cdims) ≈ collect(∇conv_filter(da, dc, cdims))

    # Test for agreement between CPU NNlib and CuDNN versions, across a variety of kwargs
    for num_spatial_dims in (1, 2, 3)
        # Initialize data we'll run our tests over
        C_in = 3
        C_out = 4
        batch_size = 1
        x = rand(Float64, fill(8, num_spatial_dims)..., C_in, batch_size)
        w = rand(Float64, fill(2, num_spatial_dims)..., C_in, C_out)
        b = rand(Float64, fill(1, num_spatial_dims)..., C_in, C_out)
        options = (Dict(), Dict(:dilation => 2), Dict(:flipkernel => true), Dict(:stride => 2), Dict(:padding => 1))

        # @denizyuret: algo option deprecated for nnlib, handling in CUDNN.cudnn
        # algos = (1, 0, 1, 1,)
        # for (opts, algo) in zip(options, algos)

        for opts in options
            cdims = DenseConvDims(x, w; opts...)
            y = NNlib.conv(x, w, cdims)

            # Test that basic convolution is equivalent across GPU/CPU
            @test testf((x, w) -> NNlib.conv(x, w, cdims), x, w)
            @test testf((y, w) -> NNlib.∇conv_data(y, w, cdims), y, w)
            @test testf((x, y) -> NNlib.∇conv_filter(x, y, cdims), x, y)

            # Scaling factors
            @test testf((x, w) -> NNlib.conv(x, w, cdims; alpha=2.0), x, w)
            @test testf((y, w) -> NNlib.∇conv_data(y, w, cdims; alpha=2.0), y, w)
            @test testf((x, y) -> NNlib.∇conv_filter(x, y, cdims; alpha=2.0), x, y)

            @test testf((y, x, w) -> NNlib.conv!(copy(y), x, w, cdims; beta=2.0), y, x, w)
            # @test testf((x, y, w) -> NNlib.∇conv_data!(copy(x), y, w, cdims; beta=2.0), x, y, w)
            @test testf((w, x, y) -> NNlib.∇conv_filter!(copy(w), x, y, cdims; beta=2.0), w, x, y)

            # Test the compatibility shims
            cy,cx,cw = CuArray{Float32}.((y,x,w))
            opts2 = Dict((k==:padding ? :pad : k)=>v for (k,v) in opts)
            @test NNlib.conv!(similar(cy),cx,cw; opts2...) ≈ NNlib.conv!(similar(cy),cx,cw,cdims)
            @test NNlib.∇conv_filter!(similar(cw),cy,cx; opts2...) ≈ NNlib.∇conv_filter!(similar(cw),cx,cy,cdims)
        end

        # Test that pooling is equivalent across GPU/CPU
        pdims = PoolDims(x, 2)
        y = maxpool(x, pdims)
        dy = ones(size(y))
        @test testf(x -> maxpool(x, pdims), x)
        @test testf((dy, y, x) -> ∇maxpool(dy, y, x, pdims), dy, y, x)
        @test testf(x -> maxpool(x, pdims), x)
        @test testf((dy, y, x) -> ∇maxpool(dy, y, x, pdims), dy, y, x)

        # Test the compatibility shims for pooling
        cx,cy,cdy = CuArray{Float32}.((x,y,dy))
        win,pad=2,1
        @test maxpool!(similar(cy), cx, win; pad=pad, stride=win) ≈ maxpool!(similar(cy), cx, PoolDims(cx, win; padding=pad, stride=win))
        @test meanpool!(similar(cy), cx, win; pad=pad, stride=win) ≈ meanpool!(similar(cy), cx, PoolDims(cx, win; padding=pad, stride=win))

        # CPU implementation of ∇conv_bias!
        db = zeros(Float64, 1, 1, 3, 1)
        dy = randn(Float64, 8, 8, 3, 1)
        function ∇conv_bias!(db, dy)
            db .= sum(dy, dims=(1:(ndims(dy)-2)))
            return db
        end
        @test testf(∇conv_bias!, db, dy)
    end

    for dims in [(5,5), (5,)]
        x = randn(Float64,dims)
        y = softmax(x)
        dy = randn(Float64,dims)
        @test testf(softmax, x)
        @test testf(∇softmax, dy, x) # add y when NNlib implements it
        y = logsoftmax(x)
        @test testf(logsoftmax, x)
        @test testf(∇logsoftmax, dy, x) # add y when NNlib implements it
    end
end

@testcase "ops" begin
    @test testf(CUDNN.cudnnAddTensor, CUDA.rand(Float32, 10, 10, 3, 1), CUDA.rand(Float32, 10, 10, 3, 1))
    @test testf(CUDNN.cudnnActivationForward!, CUDA.rand(Float32, 10, 10, 3, 1), CUDA.rand(Float32, 10, 10, 3, 1))
    # @denizyuret: no high level api for backward functions, see CUDA/lib/cudnn/README.md
    # @test testf(CUDNN.cudnnActivationBackward, CUDA.rand(Float32, 10, 10, 3, 1), CUDA.rand(Float32, 10, 10, 3, 1), CUDA.rand(Float32, 10, 10, 3, 1), CUDA.rand(Float32, 10, 10, 3, 1))

    # activations defined in src/nnlib.jl
    ACTIVATION_FUNCTIONS = [σ, logσ, hardσ, hardtanh, relu, leakyrelu, relu6, rrelu,
                            elu, gelu, celu, swish, lisht, selu, trelu, softplus,
                            softsign, logcosh, mish, tanhshrink, softshrink];
    for dims in ((5,5), (5,))
        for f in filter(x -> x != rrelu, ACTIVATION_FUNCTIONS)
            @test testf(x -> f.(x), rand(Float64, dims))
        end
    end

    # softplus does not give `Inf` for large arguments
    x = CuArray([1000.])
    @test all(softplus.(x) .== x)

    # optimized activation overwrote inputs
    let
        x = CUDA.ones(1)
        @test Array(x) == [1f0]
        tanh.(x)
        @test Array(x) == [1f0]
        y = tanh.(x)
        @test Array(x) == [1f0]
        @test Array(y) == [tanh(1f0)]
        x .= tanh.(y)
        @test Array(y) == [tanh(1f0)]
        @test Array(x) == [tanh(tanh(1f0))]
    end
end

@testcase "batchnorm" begin
    v = CUDA.rand(Float32, 2)
    m = CUDA.rand(Float32, 2, 5)
    for training in (false, true)
        CUDNN.batchnorm(v, v, m, v, v, 1.0; training=training)
    end
end

end

############################################################################################

end

end
