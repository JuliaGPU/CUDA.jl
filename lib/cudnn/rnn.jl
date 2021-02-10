"""
    cudnnRNNForward(w, x; hiddenSize, o...)
    cudnnRNNForward!(y, w, x; hiddenSize, o...)
    cudnnRNNForward(w, x, d::cudnnRNNDescriptor; o...)
    cudnnRNNForward!(y, w, x, d::cudnnRNNDescriptor; o...)

Apply the RNN specified with weights `w` and configuration given by `d` or keyword options
to input `x`.

Keyword arguments for hidden input/output:
* `hx=nothing`: initialize the hidden vector if specified (by default initialized to 0).
* `cx=nothing`: initialize the cell vector (only in LSTMs) if specified (by default initialized to 0).
* `hy=nothing`: return the final hidden vector in hy if set to `Ref{Any}()`.
* `cy=nothing`: return the final cell vector in cy (only in LSTMs) if set to `Ref{Any}()`.

Keyword arguments specifying the RNN when `d::cudnnRNNDescriptor` is not given:
* `hiddenSize::Integer`: hidden vector size, which must be supplied when `d` is not given
* `algo::cudnnRNNAlgo_t = CUDNN_RNN_ALGO_STANDARD`: RNN algo (CUDNN_RNN_ALGO_STANDARD, CUDNN_RNN_ALGO_PERSIST_STATIC, or CUDNN_RNN_ALGO_PERSIST_DYNAMIC).
* `cellMode::cudnnRNNMode_t = CUDNN_LSTM`: Specifies the RNN cell type in the entire model (CUDNN_RNN_RELU, CUDNN_RNN_TANH, CUDNN_LSTM, CUDNN_GRU).
* `biasMode::cudnnRNNBiasMode_t = CUDNN_RNN_DOUBLE_BIAS`: Sets the number of bias vectors (CUDNN_RNN_NO_BIAS, CUDNN_RNN_SINGLE_INP_BIAS, CUDNN_RNN_SINGLE_REC_BIAS, CUDNN_RNN_DOUBLE_BIAS). The two single bias settings are functionally the same for RELU, TANH and LSTM cell types. For differences in GRU cells, see the description of CUDNN_GRU in cudnn docs.
* `dirMode::cudnnDirectionMode_t = CUDNN_UNIDIRECTIONAL`: Specifies the recurrence pattern: CUDNN_UNIDIRECTIONAL or CUDNN_BIDIRECTIONAL. In bidirectional RNNs, the hidden states passed between physical layers are concatenations of forward and backward hidden states.
* `inputMode::cudnnRNNInputMode_t = CUDNN_LINEAR_INPUT`: Specifies how the input to the RNN model is processed by the first layer. When inputMode is CUDNN_LINEAR_INPUT, original input vectors of size inputSize are multiplied by the weight matrix to obtain vectors of hiddenSize. When inputMode is CUDNN_SKIP_INPUT, the original input vectors to the first layer are used as is without multiplying them by the weight matrix.
* `mathPrec::DataType = eltype(x)`: This parameter is used to control the compute math precision in the RNN model. For Float16 input/output can be Float16 or Float32, for Float32 or Float64 input/output, must match the input/output type.
* `mathType::cudnnMathType_t = math_mode()`: Sets the preferred option to use NVIDIA Tensor Cores accelerators on Volta (SM 7.0) or higher GPU-s. When dataType is CUDNN_DATA_HALF, the mathType parameter can be CUDNN_DEFAULT_MATH or CUDNN_TENSOR_OP_MATH. The ALLOW_CONVERSION setting is treated the same CUDNN_TENSOR_OP_MATH for this data type. When dataType is CUDNN_DATA_FLOAT, the mathType parameter can be CUDNN_DEFAULT_MATH or CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION. When the latter settings are used, original weights and intermediate results will be down-converted to CUDNN_DATA_HALF before they are used in another recursive iteration. When dataType is CUDNN_DATA_DOUBLE, the mathType parameter can be CUDNN_DEFAULT_MATH.
* `inputSize::Integer = size(x,1)`: Size of the input vector in the RNN model. When the inputMode=CUDNN_SKIP_INPUT, the inputSize should match the hiddenSize value.
* `projSize::Integer = hiddenSize`: The size of the LSTM cell output after the recurrent projection. This value should not be larger than hiddenSize. It is legal to set projSize equal to hiddenSize, however, in this case, the recurrent projection feature is disabled. The recurrent projection is an additional matrix multiplication in the LSTM cell to project hidden state vectors ht into smaller vectors rt = Wrht, where Wr is a rectangular matrix with projSize rows and hiddenSize columns. When the recurrent projection is enabled, the output of the LSTM cell (both to the next layer and unrolled in-time) is rt instead of ht. The recurrent projection can be enabled for LSTM cells and CUDNN_RNN_ALGO_STANDARD only.
* `numLayers::Integer = 1`: Number of stacked, physical layers in the deep RNN model. When dirMode= CUDNN_BIDIRECTIONAL, the physical layer consists of two pseudo-layers corresponding to forward and backward directions. 
* `dropout::Real = 0`: When non-zero, dropout operation will be applied between physical layers. A single layer network will have no dropout applied. Dropout is used in the training mode only.
* `auxFlags::Integer = CUDNN_RNN_PADDED_IO_ENABLED`: Miscellaneous switches that do not require additional numerical values to configure the corresponding feature. In future cuDNN releases, this parameter will be used to extend the RNN functionality without adding new API functions (applicable options should be bitwise OR-ed). Currently, this parameter is used to enable or disable padded input/output (CUDNN_RNN_PADDED_IO_DISABLED, CUDNN_RNN_PADDED_IO_ENABLED). When the padded I/O is enabled, layouts CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED and CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED are permitted in RNN data descriptors. 

Other keyword arguments:
* `layout::cudnnRNNDataLayout_t = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED`: The memory layout of the RNN data tensor. Options are CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED: Data layout is padded, with outer stride from one time-step to the next; CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED: The sequence length is sorted and packed as in the basic RNN API; CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED: Data layout is padded, with outer stride from one batch to the next.
* `seqLengthArray::Vector{Cint} = nothing`: An integer array with batchSize number of elements. Describes the length (number of time-steps) of each sequence. Each element in seqLengthArray must be greater than or equal to 0 but less than or equal to maxSeqLength. In the packed layout, the elements should be sorted in descending order, similar to the layout required by the non-extended RNN compute functions. The default value `nothing` assumes uniform seqLengths, no padding.
* `devSeqLengths::CuVector{Cint} = nothing`: Device copy of seqLengthArray
* `fwdMode::cudnnForwardMode_t = CUDNN_FWD_MODE_INFERENCE`: set to `CUDNN_FWD_MODE_TRAINING` when training
"""
cudnnRNNForward, cudnnRNNForward!


# Public methods
cudnnRNNForward(w, x; hiddenSize, o...)     = cudnnRNNForwardWithDefaults(w, x; hiddenSize, o...)
cudnnRNNForward!(y, w, x; hiddenSize, o...) = cudnnRNNForwardWithDefaults(w, x; y, hiddenSize, o...)
cudnnRNNForward(w, x, d::cudnnRNNDescriptor; o...)     = cudnnRNNForwardWithDefaults(w, x; rnnDesc=d, o...)
cudnnRNNForward!(y, w, x, d::cudnnRNNDescriptor; o...) = cudnnRNNForwardWithDefaults(w, x; y, rnnDesc=d, o...)


# Private method
function cudnnRNNForwardWithDefaults(
    w, x;

    # input hidden vectors
    hx = nothing,
    cx = nothing,

    # output buffers
    y = nothing,
    hy = nothing,
    cy = nothing,

    # rnnDescriptor parameters
    # TODO: look into GetClip, SetClip
    algo::cudnnRNNAlgo_t = CUDNN_RNN_ALGO_STANDARD,
    cellMode::cudnnRNNMode_t = CUDNN_LSTM,
    biasMode::cudnnRNNBiasMode_t = CUDNN_RNN_DOUBLE_BIAS,
    dirMode::cudnnDirectionMode_t = CUDNN_UNIDIRECTIONAL,
    inputMode::cudnnRNNInputMode_t = CUDNN_LINEAR_INPUT,
    dataType::DataType = eltype(x),
    mathPrec::DataType = dataType, # has to match dataType with one extra possibility dt=Float16 => mp=Float16|Float32
    mathType::cudnnMathType_t = math_mode(),
    inputSize::Integer = size(x,1),
    hiddenSize::Integer = 0,
    projSize::Integer = hiddenSize,
    numLayers::Integer = 1,
    dropout::Real = 0,
    auxFlags::Integer = CUDNN_RNN_PADDED_IO_ENABLED, # When the padded I/O is enabled, layouts CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED and CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED are permitted in RNN data descriptors.

    # rnnDescriptor
    rnnDesc::cudnnRNNDescriptor = cudnnRNNDescriptor(algo, cellMode, biasMode, dirMode, inputMode, cudnnDataType(dataType), cudnnDataType(mathPrec), mathType, Int32(inputSize), checkHidden(hiddenSize), Int32(projSize), Int32(numLayers), cudnnDropoutDescriptor(Cfloat(dropout)), UInt32(auxFlags)),

    # rnnData parameters:
    layout::cudnnRNNDataLayout_t = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, # padded [X,B,T] array
    seqLengthArray::Union{Nothing,Vector{Cint}} = nothing,         # assume no padding by default
    paddingFill::Ptr{Cvoid} = C_NULL,

    # rnnForward parameters
    fwdMode::cudnnForwardMode_t = CUDNN_FWD_MODE_INFERENCE, # set to CUDNN_FWD_MODE_TRAINING when training
    devSeqLengths::Union{Nothing,CuArray{Cint,1}} = nothing,
    reserveSpace::Union{CuArray,Nothing} = nothing,
    workspace::Union{CuArray,Nothing} = nothing,

    # gradient buffers: layer designers may want to preallocate, so leave them as kwargs
    dw = Ref{Any}(nothing),
    dx = Ref{Any}(nothing),
    dhx = Ref{Any}(nothing),
    dcx = Ref{Any}(nothing),
)
    # Verify all inputs: they should be compatible with rnnDesc (in case it is supplied), not necessarily with kwargs:
    rd = cudnnGetRNNDescriptor_v8(rnnDesc)
    @assert rd.hiddenSize > 0  "hiddenSize > 0 must be provided"
    @assert cudnnDataType(eltype(x)) == rd.dataType "Input x type not compatible with RNN"
    @assert size(x,1) == rd.inputSize               "Input x size not compatible with RNN"
    ydims = (rd.projSize << (rd.dirMode === CUDNN_BIDIRECTIONAL), size(x)[2:end]...)
    if y !== nothing
        @assert cudnnDataType(eltype(y)) == rd.dataType  "Output y type not compatible with RNN"
        @assert size(y) == ydims                         "Output y size not compatible with RNN or input x"
    else
        y = similar(x, ydims)
    end
    if layout === CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED
        if seqLengthArray === nothing; seqLengthArray = fill(Cint(size(x,3)), size(x,2)); end
        @assert size(x,2) == length(seqLengthArray)  "Input x batchsize not compatible with seqLengthArray"
        @assert size(x,3) >= maximum(seqLengthArray) "Input x seqLength not compatible with seqLengthArray"
        xDesc = cudnnRNNDataDescriptor(rd.dataType, layout, Cint(size(x,3)), Cint(size(x,2)), Cint(size(x,1)), seqLengthArray, paddingFill)
        yDesc = cudnnRNNDataDescriptor(rd.dataType, layout, Cint(size(y,3)), Cint(size(y,2)), Cint(size(y,1)), seqLengthArray, paddingFill)
    elseif layout === CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED
        if seqLengthArray === nothing; seqLengthArray = fill(Cint(size(x,2)), size(x,3)); end
        @assert size(x,3) == length(seqLengthArray)  "Input x batchsize not compatible with seqLengthArray"
        @assert size(x,2) >= maximum(seqLengthArray) "Input x seqLength not compatible with seqLengthArray"
        xDesc = cudnnRNNDataDescriptor(rd.dataType, layout, Cint(size(x,2)), Cint(size(x,3)), Cint(size(x,1)), seqLengthArray, paddingFill)
        yDesc = cudnnRNNDataDescriptor(rd.dataType, layout, Cint(size(y,2)), Cint(size(y,3)), Cint(size(y,1)), seqLengthArray, paddingFill)
    elseif layout === CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED
        if seqLengthArray === nothing; seqLengthArray = fill(Cint(size(x,3)), size(x,2)); end
        @assert length(x)÷size(x,1) == sum(seqLengthArray) "Input x size not compatible with seqLengthArray"
        xDesc = cudnnRNNDataDescriptor(rd.dataType, layout, maximum(seqLengthArray), Cint(length(seqLengthArray)), Cint(size(x,1)), seqLengthArray, paddingFill)
        yDesc = cudnnRNNDataDescriptor(rd.dataType, layout, maximum(seqLengthArray), Cint(length(seqLengthArray)), Cint(size(y,1)), seqLengthArray, paddingFill)
    else
        error("Unknown layout $layout")
    end
    if devSeqLengths === nothing; devSeqLengths = CuArray(seqLengthArray); end

    hdims = (rd.projSize, length(seqLengthArray), rd.numLayers << (rd.dirMode === CUDNN_BIDIRECTIONAL))
    cdims = (rd.hiddenSize, length(seqLengthArray), rd.numLayers << (rd.dirMode === CUDNN_BIDIRECTIONAL))
    if hx !== nothing
        @assert cudnnDataType(eltype(hx)) == rd.dataType "Hidden hx type not compatible with RNN"
        @assert size(hx) == hdims                        "Hidden hx size not compatible with RNN"
    end
    if hy !== nothing
        @assert hy isa Ref{Any}
        if isassigned(hy) && hy[] !== nothing
            @assert cudnnDataType(eltype(hy[])) == rd.dataType "Hidden hy type not compatible with RNN"
            @assert size(hy[]) == hdims                        "Hidden hy size not compatible with RNN"
        else
            hy[] = similar(y, hdims)
        end
    end
    if rd.cellMode === CUDNN_LSTM
        if cx !== nothing
            @assert cudnnDataType(eltype(cx)) == rd.dataType "Hidden cx type not compatible with RNN"
            @assert size(cx) == cdims                        "Hidden cx size not compatible with RNN"
        end
        if cy !== nothing
            @assert cy isa Ref{Any}
            if isassigned(cy) && cy[] !== nothing
                @assert cudnnDataType(eltype(cy[])) == rd.dataType "Hidden cy type not compatible with RNN"
                @assert size(cy[]) == cdims                        "Hidden cy size not compatible with RNN"
            else
                cy[] = similar(y, cdims)
            end
        end
    end
    hDesc = cudnnTensorDescriptor(CUDNN_TENSOR_NCHW, rd.dataType, Cint(3), Cint[reverse(hdims)...])
    cDesc = cudnnTensorDescriptor(CUDNN_TENSOR_NCHW, rd.dataType, Cint(3), Cint[reverse(cdims)...])

    weightSpaceSize = cudnnRNNWeightSpaceSize(rnnDesc)
    @assert sizeof(w) >= weightSpaceSize "RNN weights should be at least $weightSpaceSize bytes."

    # Backward called separately on each variable. We will calculate all gradients on first call. Use `dready` to avoid subsequent calls.
    dready = Ref{Bool}(false)   # this will be turned to `true` by the first backward call.

    y_h_c = cudnnRNNForwardAD(w, x, hx, cx; rnnDesc, fwdMode, devSeqLengths, xDesc, yDesc, y, hDesc, hy=(hy isa Ref ? hy[] : hy), cDesc, cy=(cy isa Ref ? cy[] : cy), workspace, reserveSpace, dw, dx, dhx, dcx, dready)
    if hy isa Ref; hy[] = y_h_c[2]; end
    if cy isa Ref && rd.cellMode === CUDNN_LSTM; cy[] = y_h_c[3]; end
    return y_h_c[1]             # only return y; hy and cy can be accessed through keyword arguments. They still need to be in AutoGrad return value to be included in gradient calc.
end


# AD method

function cudnnRNNForwardAD(w, x, hx, cx; rnnDesc, fwdMode, devSeqLengths, xDesc, yDesc, y, hDesc, hy, cDesc, cy, workspace, reserveSpace, dw, dx, dhx, dcx, dready)
    (workspaceSize, reserveSpaceSize) = cudnnRNNTempSpaceSizes(rnnDesc, fwdMode, xDesc)
    if reserveSpaceSize > 0 && reserveSpace === nothing; reserveSpace = cudnnTempSpace(reserveSpaceSize); end
    @assert sizeof(reserveSpace) >= reserveSpaceSize  "reserveSpace should be at least $reserveSpaceSize bytes"
    # Cannot use @workspace here because it is shared between forw and back calls
    if workspaceSize > 0 && workspace === nothing; workspace = cudnnTempSpace(workspaceSize); end
    @assert sizeof(workspace) >= workspaceSize  "workspace should be at least $workspaceSize bytes"
    cudnnRNNForward(handle(), rnnDesc, fwdMode, devSeqLengths, xDesc, x, yDesc, y, hDesc, something(hx, CU_NULL), something(hy, CU_NULL), cDesc, something(cx, CU_NULL), something(cy, CU_NULL), sizeof(w), w, sizeof(workspace), something(workspace, CU_NULL), sizeof(reserveSpace), something(reserveSpace, CU_NULL))
    return (y, hy, cy)
end


# Helper methods

function cudnnRNNWeightSpaceSize(rnnDesc::cudnnRNNDescriptor)
    ws = Csize_t[0]
    cudnnGetRNNWeightSpaceSize(handle(), rnnDesc, ws)
    ws[1]
end

function cudnnRNNTempSpaceSizes(rnnDesc::cudnnRNNDescriptor, fwdMode::cudnnForwardMode_t, xDesc::cudnnRNNDataDescriptor)
    ws = Csize_t[0]; rs = Csize_t[0]
    cudnnGetRNNTempSpaceSizes(handle(), rnnDesc, fwdMode, xDesc, ws, rs)
    ws[1], rs[1]
end

function cudnnGetRNNDescriptor_v8(rnnDesc::cudnnRNNDescriptor)
    (algo, cellMode, biasMode, dirMode, inputMode, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropout, auxFlags) = (Ref{cudnnRNNAlgo_t}(), Ref{cudnnRNNMode_t}(), Ref{cudnnRNNBiasMode_t}(), Ref{cudnnDirectionMode_t}(), Ref{cudnnRNNInputMode_t}(), Ref{cudnnDataType_t}(), Ref{cudnnDataType_t}(), Ref{cudnnMathType_t}(), Ref{Int32}(), Ref{Int32}(), Ref{Int32}(), Ref{Int32}(), Ref{Ptr{Nothing}}(), Ref{UInt32}())
    cudnnGetRNNDescriptor_v8(rnnDesc, algo, cellMode, biasMode, dirMode, inputMode, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropout, auxFlags)
    (algo, cellMode, biasMode, dirMode, inputMode, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropout, auxFlags) = (algo[], cellMode[], biasMode[], dirMode[], inputMode[], dataType[], mathPrec[], mathType[], inputSize[], hiddenSize[], projSize[], numLayers[], dropout[], auxFlags[])
    (; rnnDesc, algo, cellMode, biasMode, dirMode, inputMode, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropout, auxFlags)
end

checkHidden(h) = (h > 0 ? Int32(h) : error("hiddenSize > 0 is required"))


"""
    cudnnGetRNNWeightParams(w, d::cudnnRNNDescriptor)
    cudnnGetRNNWeightParams(w; hiddenSize, o...)

Return an array of weight matrices and bias vectors of an RNN specified by `d` or keyword
options as views into `w`. The keyword arguments and defaults in the second form are the
same as those in cudnnRNNForward specifying the RNN.

In the returned array `a[1,l,p]` and `a[2,l,p]` give the weight matrix and bias vector for
the l'th layer and p'th parameter or `nothing` if the specified matrix/vector does not
exist. Note that the matrices should be transposed for left multiplication, e.g. `a[1,l,p]'
* x`

The `l` index refers to the pseudo-layer number. In uni-directional RNNs, a pseudo-layer is
the same as a physical layer (pseudoLayer=1 is the RNN input layer, pseudoLayer=2 is the
first hidden layer). In bi-directional RNNs, there are twice as many pseudo-layers in
comparison to physical layers:

    pseudoLayer=1 refers to the forward direction sub-layer of the physical input layer
    pseudoLayer=2 refers to the backward direction sub-layer of the physical input layer
    pseudoLayer=3 is the forward direction sub-layer of the first hidden layer, and so on

The `p` index refers to the weight matrix or bias vector linear ID index.

If cellMode in rnnDesc was set to CUDNN_RNN_RELU or CUDNN_RNN_TANH:

    Value 1 references the weight matrix or bias vector used in conjunction with the input from the previous layer or input to the RNN model.
    Value 2 references the weight matrix or bias vector used in conjunction with the hidden state from the previous time step or the initial hidden state.

If cellMode in rnnDesc was set to CUDNN_LSTM:

    Values 1, 2, 3 and 4 reference weight matrices or bias vectors used in conjunction with the input from the previous layer or input to the RNN model.
    Values 5, 6, 7 and 8 reference weight matrices or bias vectors used in conjunction with the hidden state from the previous time step or the initial hidden state.
    Value 9 corresponds to the projection matrix, if enabled (there is no bias in this operation).

Values and their LSTM gates:

    Values 1 and 5 correspond to the input gate.
    Values 2 and 6 correspond to the forget gate.
    Values 3 and 7 correspond to the new cell state calculations with hyperbolic tangent.
    Values 4 and 8 correspond to the output gate.

If cellMode in rnnDesc was set to CUDNN_GRU:

    Values 1, 2 and 3 reference weight matrices or bias vectors used in conjunction with the input from the previous layer or input to the RNN model.
    Values 4, 5 and 6 reference weight matrices or bias vectors used in conjunction with the hidden state from the previous time step or the initial hidden state.

Values and their GRU gates:

    Values 1 and 4 correspond to the reset gate.
    Values 2 and 5 reference to the update gate.
    Values 3 and 6 correspond to the new hidden state calculations with hyperbolic tangent.

"""
function cudnnGetRNNWeightParams(
    w;
    hiddenSize::Integer,
    inputSize::Integer = hiddenSize,
    projSize::Integer = hiddenSize,
    algo::cudnnRNNAlgo_t = CUDNN_RNN_ALGO_STANDARD,
    cellMode::cudnnRNNMode_t = CUDNN_LSTM,
    biasMode::cudnnRNNBiasMode_t = CUDNN_RNN_DOUBLE_BIAS,
    dirMode::cudnnDirectionMode_t = CUDNN_UNIDIRECTIONAL,
    inputMode::cudnnRNNInputMode_t = CUDNN_LINEAR_INPUT,
    dataType::DataType = Float32,
    mathPrec::DataType = dataType,
    mathType::cudnnMathType_t = math_mode(),
    numLayers::Integer = 1,
    dropout::Real = 0,
    auxFlags::Integer = CUDNN_RNN_PADDED_IO_ENABLED,
)
    cudnnGetRNNWeightParams(w, cudnnRNNDescriptor(algo, cellMode, biasMode, dirMode, inputMode, cudnnDataType(dataType), cudnnDataType(mathPrec), mathType, Int32(inputSize), checkHidden(hiddenSize), Int32(projSize), Int32(numLayers), cudnnDropoutDescriptor(Cfloat(dropout)), UInt32(auxFlags)))
end


function cudnnGetRNNWeightParams(w, rnnDesc::cudnnRNNDescriptor)
    d = cudnnGetRNNDescriptor_v8(rnnDesc)
    T = juliaDataType(d.dataType)
    weightSpace = reinterpret(T, w)
    nlayers = d.numLayers << (d.dirMode === CUDNN_BIDIRECTIONAL)
    nparams = (d.cellMode === CUDNN_RNN_RELU || d.cellMode === CUDNN_RNN_TANH ? 2 :
               d.cellMode === CUDNN_LSTM ? 9 : d.cellMode === CUDNN_GRU  ? 6 :
               error("Unknown cellMode $(d.cellMode)"))
    a = Array{Any}(undef, 2, nlayers, nparams)
    p = Ref{Ptr{Cvoid}}(0)
    cudnnCreateTensorDescriptor(p); mDesc = cudnnTensorDescriptor(p[])
    cudnnCreateTensorDescriptor(p); bDesc = cudnnTensorDescriptor(p[])
    mAddr = Ref{CuPtr{Cvoid}}(0)
    bAddr = Ref{CuPtr{Cvoid}}(0)
    for l in 1:nlayers, p in 1:nparams
        cudnnGetRNNWeightParams(handle(), rnnDesc, l-1, sizeof(weightSpace), weightSpace, p-1, mDesc, mAddr, bDesc, bAddr)
        mT,mD,mS = cudnnGetTensorDescriptor(mDesc)
        bT,bD,bS = cudnnGetTensorDescriptor(bDesc)
        @assert mT === bT === T
        if mAddr[] === CU_NULL
            a[1,l,p] = nothing
        else
            m0 = (mAddr[] - pointer(weightSpace)) ÷ sizeof(T) |> Int
            a[1,l,p] = reshape(view(weightSpace, (m0+1):(m0+prod(mD))), (mD[1],mD[2]))
        end
        if bAddr[] === CU_NULL
            a[2,l,p] = nothing
        else
            b0 = (bAddr[] - pointer(weightSpace)) ÷ sizeof(T) |> Int
            a[2,l,p] = view(weightSpace, (b0+1):(b0+prod(bD)))
        end
    end
    cudnnDestroyTensorDescriptor.((mDesc,bDesc))
    return a
end

