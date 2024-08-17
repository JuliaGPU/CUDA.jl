using cuDNN:
    cudnnRNNForward,
    cudnnRNNForward!,
    cudnnRNNBackwardData_v8,
    cudnnRNNBackwardWeights_v8,
    cudnnRNNDescriptor,
    cudnnRNNDescriptor_t,
    cudnnSetRNNDescriptor_v8,
    cudnnGetRNNWeightSpaceSize,
    cudnnGetRNNTempSpaceSizes,
    cudnnRNNAlgo_t,
        CUDNN_RNN_ALGO_STANDARD,        # 0, robust performance across a wide range of network parameters
        CUDNN_RNN_ALGO_PERSIST_STATIC,  # 1, fast when the first dimension of the input tensor is small (meaning, a small minibatch), cc>=6.0
        CUDNN_RNN_ALGO_PERSIST_DYNAMIC, # 2, similar to static, optimize using the specific parameters of the network and active GPU, cc>=6.0
        CUDNN_RNN_ALGO_COUNT,           # 3
    cudnnRNNMode_t,
        CUDNN_RNN_RELU, # 0, /* basic RNN cell type with ReLu activation */
        CUDNN_RNN_TANH, # 1, /* basic RNN cell type with tanh activation */
        CUDNN_LSTM,     # 2, /* LSTM with optional recurrent projection and clipping */
        CUDNN_GRU,      # 3, /* Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1); */
    cudnnRNNBiasMode_t,
        CUDNN_RNN_NO_BIAS,         # 0, /* rnn cell formulas do not use biases */
        CUDNN_RNN_SINGLE_INP_BIAS, # 1, /* rnn cell formulas use one input bias in input GEMM */
        CUDNN_RNN_DOUBLE_BIAS,     # 2, /* default, rnn cell formulas use two bias vectors */
        CUDNN_RNN_SINGLE_REC_BIAS, # 3  /* rnn cell formulas use one recurrent bias in recurrent GEMM */
    cudnnDirectionMode_t,
        CUDNN_UNIDIRECTIONAL, # 0, /* single direction network */
        CUDNN_BIDIRECTIONAL,  # 1, /* output concatination at each layer */
    cudnnRNNInputMode_t,
        CUDNN_LINEAR_INPUT, # 0, /* adjustable weight matrix in first layer input GEMM */
        CUDNN_SKIP_INPUT,   # 1, /* fixed identity matrix in the first layer input GEMM */
    cudnnMathType_t,
        CUDNN_DEFAULT_MATH,                    # 0,
        CUDNN_TENSOR_OP_MATH,                  # 1,
        CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION, # 2,
        CUDNN_FMA_MATH,                        # 3,
    #/* For auxFlags in cudnnSetRNNDescriptor_v8() */
        CUDNN_RNN_PADDED_IO_DISABLED, # 0
        CUDNN_RNN_PADDED_IO_ENABLED,  # (1U << 0)
    cudnnForwardMode_t,
        CUDNN_FWD_MODE_INFERENCE, # 0
        CUDNN_FWD_MODE_TRAINING,  # 1
    cudnnRNNDataDescriptor_t,
    cudnnSetRNNDataDescriptor,
    cudnnRNNDataLayout_t,
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,   # 0, /* padded, outer stride from one time-step to the next */
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,     # 1, /* sequence length sorted and packed as in basic RNN api */
        CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED, # 2, /* padded, outer stride from one batch to the next */
    cudnnWgradMode_t,
        CUDNN_WGRAD_MODE_ADD, # 0, /* add partial gradients to wgrad output buffers */
        CUDNN_WGRAD_MODE_SET, # 1, /* write partial gradients to wgrad output buffers */
    cudnnTensorDescriptor,
    cudnnDropoutDescriptor,
    cudnnDataType,
    math_mode

    X,H,B,T = 8,8,4,2
    w = CUDA.randn(10000)
    x = CUDA.randn(X,B,T)
    hx1 = CUDA.randn(H,B,1)
    cx1 = CUDA.randn(H,B,1)

function rnntest(;
    hx = nothing,
    cx = nothing,
    hy = nothing,
    cy = nothing,
    layout::cudnnRNNDataLayout_t = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
    seqLengthArray::Union{Nothing,Vector{Cint}} = nothing,
    fwdMode::cudnnForwardMode_t = CUDNN_FWD_MODE_INFERENCE,
    # descriptor keywords
    hiddenSize::Integer = H,
    algo::cudnnRNNAlgo_t = CUDNN_RNN_ALGO_STANDARD,
    cellMode::cudnnRNNMode_t = CUDNN_LSTM,
    biasMode::cudnnRNNBiasMode_t = CUDNN_RNN_DOUBLE_BIAS,
    dirMode::cudnnDirectionMode_t = CUDNN_UNIDIRECTIONAL,
    inputMode::cudnnRNNInputMode_t = CUDNN_LINEAR_INPUT,
    mathPrec::DataType = eltype(x),
    mathType::cudnnMathType_t = math_mode(),
    inputSize::Integer = size(x,1),
    projSize::Integer = hiddenSize,
    numLayers::Integer = 1,
    dropout::Real = 0,
    auxFlags::Integer = CUDNN_RNN_PADDED_IO_ENABLED,
)
    d = cudnnRNNDescriptor(algo, cellMode, biasMode, dirMode, inputMode,
                            cudnnDataType(eltype(x)), cudnnDataType(mathPrec), mathType,
                            Int32(inputSize), Int32(hiddenSize), Int32(projSize),
                            Int32(numLayers), cudnnDropoutDescriptor(Cfloat(dropout)),
                            UInt32(auxFlags))
    y = cudnnRNNForward(w, x; hx, cx, hy, cy, layout, seqLengthArray, fwdMode,
                        hiddenSize, algo, cellMode, biasMode, dirMode, inputMode,
                        mathPrec, mathType, inputSize, projSize, numLayers, dropout,
                        auxFlags)
    _y = copy(y)
    _hy = (hy === nothing ? hy : copy(hy[]))
    _cy = (cy === nothing ? cy : copy(cy[]))
    @test Array(_y) ≈ cudnnRNNForward!(y, w, x; hx, cx, hy, cy, layout, seqLengthArray, fwdMode,
                                        hiddenSize, algo, cellMode, biasMode, dirMode, inputMode,
                                        mathPrec, mathType, inputSize, projSize, numLayers, dropout, auxFlags) |> Array
    (_hy === hy === nothing || @test Array(_hy) ≈ Array(hy[]))
    (_cy === cy === nothing || @test Array(_cy) ≈ Array(cy[]))
    @test Array(_y) ≈ cudnnRNNForward(w, x, d; hx, cx, hy, cy, layout, seqLengthArray, fwdMode) |> Array
    (_hy === hy === nothing || @test Array(_hy) ≈ Array(hy[]))
    (_cy === cy === nothing || @test Array(_cy) ≈ Array(cy[]))
    @test Array(_y) ≈ cudnnRNNForward!(y, w, x, d; hx, cx, hy, cy, layout, seqLengthArray, fwdMode) |> Array
    (_hy === hy === nothing || @test Array(_hy) ≈ Array(hy[]))
    (_cy === cy === nothing || @test Array(_cy) ≈ Array(cy[]))
end

rnntest()
rnntest(hx=hx1)
rnntest(cx=cx1)
rnntest(hy=Ref{Any}())
rnntest(cy=Ref{Any}())
rnntest(layout=CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED)
rnntest(layout=CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED)
rnntest(seqLengthArray=Cint[1,2,1,2])
rnntest(fwdMode=CUDNN_FWD_MODE_TRAINING)
rnntest(hiddenSize=16)
# XXX: it's unclear which devices support this algorithm
if capability(device()) >= v"6.1"
    rnntest(algo=CUDNN_RNN_ALGO_PERSIST_STATIC)
end
#rnntest(algo=CUDNN_RNN_ALGO_PERSIST_DYNAMIC) # causes segfault
rnntest(cellMode=CUDNN_RNN_RELU)
rnntest(cellMode=CUDNN_RNN_TANH)
rnntest(cellMode=CUDNN_GRU)
rnntest(biasMode=CUDNN_RNN_NO_BIAS)
rnntest(biasMode=CUDNN_RNN_SINGLE_INP_BIAS)
rnntest(biasMode=CUDNN_RNN_SINGLE_REC_BIAS)
rnntest(dirMode=CUDNN_BIDIRECTIONAL)
rnntest(inputMode=CUDNN_SKIP_INPUT)
rnntest(mathPrec=Float32) # only possible option for F32 input
rnntest(mathType=CUDNN_DEFAULT_MATH)
rnntest(mathType=CUDNN_TENSOR_OP_MATH)
rnntest(mathType=CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)
rnntest(projSize=4)
rnntest(numLayers=2)
rnntest(dropout=0.5)
rnntest(auxFlags=CUDNN_RNN_PADDED_IO_DISABLED)
rnntest(auxFlags=CUDNN_RNN_PADDED_IO_ENABLED)
