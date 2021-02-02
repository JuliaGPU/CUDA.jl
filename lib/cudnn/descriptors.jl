using Base: @__doc__


"""
    @cudnnDescriptor(XXX, setter=cudnnSetXXXDescriptor)

Defines a new type `cudnnXXXDescriptor` with a single field `ptr::cudnnXXXDescriptor_t` and
its constructor. The second optional argument is the function that sets the descriptor
fields and defaults to `cudnnSetXXXDescriptor`. The constructor is memoized, i.e. when
called with the same arguments it returns the same object rather than creating a new one.

The arguments of the constructor and thus the keys to the memoization cache depend on the
setter: If the setter has arguments `cudnnSetXXXDescriptor(ptr::cudnnXXXDescriptor_t,
args...)`, then the constructor has `cudnnXXXDescriptor(args...)`. The user can control
these arguments by defining a custom setter.
"""
macro cudnnDescriptor(x, set = Symbol("cudnnSet$(x)Descriptor"))
    sname = Symbol("cudnn$(x)Descriptor")
    tname = Symbol("cudnn$(x)Descriptor_t")
    cache = Symbol("cudnn$(x)DescriptorCache")
    create = Symbol("cudnnCreate$(x)Descriptor")
    destroy = Symbol("cudnnDestroy$(x)Descriptor")
    return quote
        @__doc__ mutable struct $sname                      # needs to be mutable for finalizer
            ptr::$tname
            $sname(p::$tname) = new(p)                      # prevent $sname(::Any) default constructor
        end
        Base.unsafe_convert(::Type{<:Ptr}, d::$sname)=d.ptr # needed for ccalls
        const $cache = Dict{Tuple,$sname}()                 # Dict is 3x faster than IdDict!
        function $sname(args...)
            get!($cache, args) do
                ptr = $tname[C_NULL]
                $create(ptr)
                $set(ptr[1], args...)
                d = $sname(ptr[1])
                finalizer(x->$destroy(x.ptr), d)
                return d
            end
        end
    end |> esc
end


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
    cudnnDropoutDescriptor(dropout::Real)
"""
@cudnnDescriptor(Dropout, cudnnSetDropoutDescriptorFromFloat)


"""
    cudnnFilterDescriptor(dataType::cudnnDataType_t,
                          format::cudnnTensorFormat_t,
                          nbDims::Cint,
                          filterDimA::Vector{Cint})
"""
@cudnnDescriptor(Filter, cudnnSetFilterNdDescriptor)


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
    cudnnRNNDescriptor(algo::cudnnRNNAlgo_t,
                       cellMode::cudnnRNNMode_t,
                       biasMode::cudnnRNNBiasMode_t,
                       dirMode::cudnnDirectionMode_t,
                       inputMode::cudnnRNNInputMode_t,
                       dataType::cudnnDataType_t,
                       mathPrec::cudnnDataType_t,
                       mathType::cudnnMathType_t,
                       inputSize::Int32,
                       hiddenSize::Int32,
                       projSize::Int32,
                       numLayers::Int32,
                       dropoutDesc::cudnnDropoutDescriptor_t,
                       auxFlags::UInt32)
"""
@cudnnDescriptor(RNN, cudnnSetRNNDescriptor_v8)


"""
    cudnnRNNDataDescriptor(dataType::cudnnDataType_t,
                           layout::cudnnRNNDataLayout_t,
                           maxSeqLength::Cint,
                           batchSize::Cint,
                           vectorSize::Cint,
                           seqLengthArray::Vector{Cint},
                           paddingFill::Ptr{Cvoid})
"""
@cudnnDescriptor(RNNData)


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
    cudnnTensorDescriptor(format::cudnnTensorFormat_t,
                          dataType::cudnnDataType_t,
                          nbDims::Cint,
                          dimA::Vector{Cint})
"""
@cudnnDescriptor(Tensor, cudnnSetTensorNdDescriptorEx)


"""
    cudnnTensorTransformDescriptor(nbDims::UInt32,
                                   destFormat::cudnnTensorFormat_t,
                                   padBeforeA::Vector{Int32},
                                   padAfterA::Vector{Int32},
                                   foldA::Vector{UInt32},
                                   direction::cudnnFoldingDirection_t)
"""
@cudnnDescriptor(TensorTransform)
