# descriptor types only used by the legacy fixed-function wrappers

"""
    cudnnActivationDescriptor(mode::cudnnActivationMode_t,
                              reluNanOpt::cudnnNanPropagation_t,
                              coef::Cfloat)
"""
@cudnnDescriptor(Activation)


"""
    cudnnAttnDescriptor(attnMode::Cuint,
                        nHeads::Cint,
                        smScaler::Cdouble,
                        dataType::cudnnDataType_t,
                        computePrec::cudnnDataType_t,
                        mathType::cudnnMathType_t,
                        attnDropoutDesc::cudnnDropoutDescriptor_t,
                        postDropoutDesc::cudnnDropoutDescriptor_t,
                        qSize::Cint,
                        kSize::Cint,
                        vSize::Cint,
                        qProjSize::Cint,
                        kProjSize::Cint,
                        vProjSize::Cint,
                        oProjSize::Cint,
                        qoMaxSeqLength::Cint,
                        kvMaxSeqLength::Cint,
                        maxBatchSize::Cint,
                        maxBeamSize::Cint)
"""
@cudnnDescriptor(Attn)


"""
    cudnnCTCLossDescriptor(compType::cudnnDataType_t,
                           normMode::cudnnLossNormalizationMode_t,
                           gradMode::cudnnNanPropagation_t,
                           maxLabelLength::Cint)
"""
@cudnnDescriptor(CTCLoss, cudnnSetCTCLossDescriptor_v8)


"""
cudnnConvolutionDescriptor(pad::Vector{Cint},
                           stride::Vector{Cint},
                           dilation::Vector{Cint},
                           mode::cudnnConvolutionMode_t,
                           dataType::cudnnDataType_t,
                           groupCount::Cint,
                           mathType::cudnnMathType_t,
                           reorderType::cudnnReorderType_t)
"""
@cudnnDescriptor(Convolution, cudnnSetConvolutionDescriptor)


"""
    cudnnLRNDescriptor(lrnN::Cuint,
                       lrnAlpha::Cdouble,
                       lrnBeta::Cdouble,
                       lrnK::Cdouble)
"""
@cudnnDescriptor(LRN)


"""
    cudnnOpTensorDescriptor(opTensorOp::cudnnOpTensorOp_t,
                            opTensorCompType::cudnnDataType_t,
                            opTensorNanOpt::cudnnNanPropagation_t)
"""
@cudnnDescriptor(OpTensor)


"""
    cudnnPoolingDescriptor(mode::cudnnPoolingMode_t,
                           maxpoolingNanOpt::cudnnNanPropagation_t,
                           nbDims::Cint,
                           windowDimA::Vector{Cint},
                           paddingA::Vector{Cint},
                           strideA::Vector{Cint})
"""
@cudnnDescriptor(Pooling, cudnnSetPoolingNdDescriptor)


"""
    cudnnReduceTensorDescriptor(reduceTensorOp::cudnnReduceTensorOp_t,
                                reduceTensorCompType::cudnnDataType_t,
                                reduceTensorNanOpt::cudnnNanPropagation_t,
                                reduceTensorIndices::cudnnReduceTensorIndices_t,
                                reduceTensorIndicesType::cudnnIndicesType_t)
"""
@cudnnDescriptor(ReduceTensor)


"""
    cudnnSeqDataDescriptor(dataType::cudnnDataType_t,
                           nbDims::Cint,
                           dimA::Vector{Cint},
                           axes::Vector{cudnnSeqDataAxis_t},
                           seqLengthArraySize::Csize_t,
                           seqLengthArray::Vector{Cint},
                           paddingFill::Ptr{Cvoid})
"""
@cudnnDescriptor(SeqData)


"""
    cudnnSpatialTransformerDescriptor(samplerType::cudnnSamplerType_t,
                                      dataType::cudnnDataType_t,
                                      nbDims::Cint,
                                      dimA::Vector{Cint})
"""
@cudnnDescriptor(SpatialTransformer, cudnnSetSpatialTransformerNdDescriptor)


"""
    cudnnTensorTransformDescriptor(nbDims::UInt32,
                                   destFormat::cudnnTensorFormat_t,
                                   padBeforeA::Vector{Int32},
                                   padAfterA::Vector{Int32},
                                   foldA::Vector{UInt32},
                                   direction::cudnnFoldingDirection_t)
"""
@cudnnDescriptor(TensorTransform)


function cudnnGetConvolutionDescriptor(d::cudnnConvolutionDescriptor)
    # we don't know the dimension of the convolution, so we start by
    # allocating the maximum size it can be.
    nbDimsRequested = CUDNN_DIM_MAX - 2
    # later, here we get the actual dimensionality of the convolution
    arrlen = Ref{Cint}(nbDimsRequested)
    padding = Array{Cint}(undef, nbDimsRequested)
    stride = Array{Cint}(undef, nbDimsRequested)
    dilation = Array{Cint}(undef, nbDimsRequested)
    mode = Ref{cuDNN.cudnnConvolutionMode_t}(CUDNN_CONVOLUTION)
    dataType = Ref{cuDNN.cudnnDataType_t}(cuDNN.CUDNN_DATA_FLOAT)

    cudnnGetConvolutionNdDescriptor(d, nbDimsRequested, arrlen, padding, stride, dilation,
                                    mode, dataType)
    T = juliaDataType(dataType[])
    SZ = arrlen[]
    P = (padding[1:SZ]..., )
    S = (stride[1:SZ]..., )
    D = (dilation[1:SZ]..., )
    return T, mode[], SZ, P, S, D
end

# Helper for cudnnConvolutionDescriptor
function cudnnSetConvolutionDescriptor(
    ptr::cudnnConvolutionDescriptor_t,
    padding::Vector{Cint},
    stride::Vector{Cint},
    dilation::Vector{Cint},
    mode::cudnnConvolutionMode_t,
    dataType::cudnnDataType_t,
    mathType::cudnnMathType_t,
    reorderType::cudnnReorderType_t,
    groupCount::Cint,
)
    cudnnSetConvolutionNdDescriptor(ptr, Cint(length(padding)), padding, stride, dilation, mode, dataType)
    mathType != CUDNN_DEFAULT_MATH       && cudnnSetConvolutionMathType(ptr, mathType)
    reorderType != CUDNN_DEFAULT_REORDER && cudnnSetConvolutionReorderType(ptr, reorderType)
    groupCount != 1                      && cudnnSetConvolutionGroupCount(ptr, groupCount)
end
