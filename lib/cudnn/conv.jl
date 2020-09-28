using NNlib: DenseConvDims


# descriptor

mutable struct ConvDesc
    ptr::cudnnConvolutionDescriptor_t
end

unsafe_free!(cd::ConvDesc) = cudnnDestroyConvolutionDescriptor(cd.ptr)

Base.unsafe_convert(::Type{cudnnConvolutionDescriptor_t}, cd::ConvDesc)=cd.ptr

function cdsize(w, nd)
    isa(w, Integer) && return Cint[fill(w,nd)...]
    length(w) == nd && return Cint[reverse(w)...]
    length(w) == 2*nd && return Cint[reverse(w[nd+1:end])...]
    throw(DimensionMismatch())
end

pdsize(w, nd)=Cint[reverse(psize(w,nd))...]
function psize(w, nd)
    isa(w, Integer) && return Cint[fill(w,nd)...]
    length(w) == nd && return w
    length(w) == 2*nd && return w[1:nd]
    throw(DimensionMismatch())
end

Base.cconvert(::Type{cudnnConvolutionMode_t}, x::Bool) = x ? CUDNN_CROSS_CORRELATION : CUDNN_CONVOLUTION

function ConvDesc(T, N, padding, stride, dilation, mode)
    cd = Ref{cudnnConvolutionDescriptor_t}()
    cudnnCreateConvolutionDescriptor(cd)
    if version() >= v"4"
        cudnnSetConvolutionNdDescriptor(cd[],N,cdsize(padding,N),cdsize(stride,N),cdsize(dilation,N),mode,cudnnDataType(T))
    elseif version() >= v"3"
        cudnnSetConvolutionNdDescriptor_v3(cd[],N,cdsize(padding,N),cdsize(stride,N),cdsize(dilation,N),mode,cudnnDataType(T))
    else
        cudnnSetConvolutionNdDescriptor(cd[],N,cdsize(padding,N),cdsize(stride,N),cdsize(dilation,N),mode)
    end
    cudnnSetConvolutionMathType(cd[], math_mode())
    this = ConvDesc(cd[])
    finalizer(unsafe_free!, this)
    return this
end

function ConvDesc(T, cdims::DenseConvDims)
    pd = NNlib.padding(cdims)
    if !all(pd[1:2:end] .== pd[2:2:end])
        @warn("CuDNN does not support asymmetric padding; defaulting to symmetric choice")
    end
    return ConvDesc(T, NNlib.spatial_dims(cdims), pd[1:2:end], NNlib.stride(cdims),
                       NNlib.dilation(cdims), NNlib.flipkernel(cdims))
end


# wrappers

# Forward

function cudnnGetConvolutionForwardAlgorithmMaxCount()
    count=@argout(
        cudnnGetConvolutionForwardAlgorithmMaxCount(
            handle(),
            out(Ref{Cint}()))
        )[]
    return count
end

# will be removed in cuDNN 8
function cudnnGetConvolutionForwardAlgorithm(y::DenseCuArray{T,N}, x::DenseCuArray{T,N}, w::DenseCuArray{T,N},
                                             cdims::DenseConvDims; preference=0, workspacesize=1<<32) where {T,N}
    algo=@argout(
        cudnnGetConvolutionForwardAlgorithm(
            handle(), TensorDesc(x),
            FilterDesc(w), ConvDesc(T, cdims),
            TensorDesc(y),
            cudnnConvolutionFwdPreference_t(preference),
            Csize_t(workspacesize),
            out(Ref{cudnnConvolutionFwdAlgo_t}()))
        )[]
    return algo
end

function cudnnGetConvolutionForwardAlgorithm_v7(y::DenseCuArray{T,N}, x::DenseCuArray{T,N}, w::DenseCuArray{T,N},
                                                cdims::DenseConvDims; count=-1) where {T,N}
    if count < 0
        count = cudnnGetConvolutionForwardAlgorithmMaxCount()
    end
    perfResults = Array{cudnnConvolutionFwdAlgoPerf_t, 1}(undef, count)
    returnedAlgoCount=@argout(
        cudnnGetConvolutionForwardAlgorithm_v7(
            handle(), TensorDesc(x),
            FilterDesc(w), ConvDesc(T, cdims),
            TensorDesc(y),
            Cint(count),
            out(Ref{Cint}()),
            perfResults)
        )[]
    return returnedAlgoCount, perfResults
end

function cudnnFindConvolutionForwardAlgorithm(y::DenseCuArray{T,N}, x::DenseCuArray{T,N}, w::DenseCuArray{T,N},
                                              cdims::DenseConvDims; count=-1) where {T,N}
    if count < 0
        count = cudnnGetConvolutionForwardAlgorithmMaxCount()
    end
    perfResults = Array{cudnnConvolutionFwdAlgoPerf_t, 1}(undef, count)
    returnedAlgoCount=@argout(
        cudnnFindConvolutionForwardAlgorithm(
            handle(), TensorDesc(x),
            FilterDesc(w), ConvDesc(T, cdims),
            TensorDesc(y),
            Cint(count),
            out(Ref{Cint}()),
            perfResults)
        )[]
    return returnedAlgoCount, perfResults
end

function cudnnFindConvolutionForwardAlgorithmEx(y::DenseCuArray{T,N}, x::DenseCuArray{T,N}, w::DenseCuArray{T,N},
                                              cdims::DenseConvDims; count=-1, workspacesize=1<<32) where {T,N}
    if count < 0
        count = cudnnGetConvolutionForwardAlgorithmMaxCount()
    end
    @workspace size=workspacesize workspace->begin
        perfResults = Array{cudnnConvolutionFwdAlgoPerf_t, 1}(undef, count)
        returnedAlgoCount=@argout(
            cudnnFindConvolutionForwardAlgorithmEx(
                handle(), TensorDesc(x), x,
                FilterDesc(w), w, ConvDesc(T, cdims),
                TensorDesc(y), y,
                Cint(count),
                out(Ref{Cint}()),
                perfResults,
                workspace,
                workspacesize)
            )[]
        return returnedAlgoCount, perfResults
    end
end

function cudnnConvolutionForward(y::DenseCuArray{T,N}, x::DenseCuArray{T,N}, w::DenseCuArray{T,N},
                                 cdims::DenseConvDims; algo=0, alpha=1, beta=0) where {T,N}
    @workspace size=@argout(
            cudnnGetConvolutionForwardWorkspaceSize(
                handle(), TensorDesc(x),
                FilterDesc(w), ConvDesc(T, cdims),
                TensorDesc(y),
                cudnnConvolutionFwdAlgo_t(algo),
                out(Ref{Csize_t}()))
        )[] workspace->begin
            cudnnConvolutionForward(
                handle(), scalingParameter(T, alpha), TensorDesc(x), x, FilterDesc(w), w,
                ConvDesc(T,cdims), cudnnConvolutionFwdAlgo_t(algo), workspace,
                sizeof(workspace), scalingParameter(T, beta), TensorDesc(y), y)
        end
    return y
end

function cudnnConvolutionBiasActivationForward(y::DenseCuArray{T,N}, x::DenseCuArray{T,N}, w::DenseCuArray{T,N}, z::DenseCuArray{T,N}, bias::DenseCuArray{T,N},
                                               cdims::DenseConvDims; algo=0, alpha1=1, alpha2=1,
                                               activationMode=CUDNN_ACTIVATION_RELU, activationCoeff=0.0, activationReluNanOpt=CUDNN_NOT_PROPAGATE_NAN) where {T,N}
    @workspace size=@argout(
            cudnnGetConvolutionForwardWorkspaceSize(
                handle(), TensorDesc(x),
                FilterDesc(w), ConvDesc(T, cdims),
                TensorDesc(y),
                cudnnConvolutionFwdAlgo_t(algo),
                out(Ref{Csize_t}()))
        )[] workspace->begin
            cudnnConvolutionBiasActivationForward(
                handle(), scalingParameter(T, alpha1), TensorDesc(x), x, FilterDesc(w), w,
                ConvDesc(T, cdims), cudnnConvolutionFwdAlgo_t(algo), workspace,
                sizeof(workspace), scalingParameter(T, alpha2), TensorDesc(z), z, TensorDesc(bias), bias, ActivationDesc(activationMode, activationCoeff, activationReluNanOpt), TensorDesc(y),y)
        end
    return y
end

# Backward data

function cudnnGetConvolutionBackwardDataAlgorithmMaxCount()
    count=@argout(
        cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
            handle(),
            out(Ref{Cint}()))
        )[]
    return count
end

# will be removed in cuDNN 8
function cudnnGetConvolutionBackwardDataAlgorithm(dx::DenseCuArray{T,N}, w::DenseCuArray{T,N}, dy::DenseCuArray{T,N},
                                                  cdims::DenseConvDims; preference=0, workspacesize=1<<32) where {T,N}
    algo=@argout(
        cudnnGetConvolutionBackwardDataAlgorithm(
            handle(), FilterDesc(w), TensorDesc(dy), ConvDesc(T, cdims),
            TensorDesc(dx), cudnnConvolutionBwdDataPreference_t(preference),
            Csize_t(workspacesize), out(Ref{cudnnConvolutionBwdDataAlgo_t}()))
        )[]
    return algo
end

function cudnnGetConvolutionBackwardDataAlgorithm_v7(dx::DenseCuArray{T,N}, w::DenseCuArray{T,N}, dy::DenseCuArray{T,N},
                                                     cdims::DenseConvDims; count=-1) where {T,N}
    if count < 0
        count = cudnnGetConvolutionBackwardDataAlgorithmMaxCount()
    end
    perfResults = Array{cudnnConvolutionBwdDataAlgoPerf_t, 1}(undef, count)
    returnedAlgoCount=@argout(
        cudnnGetConvolutionBackwardDataAlgorithm_v7(
            handle(), FilterDesc(w), TensorDesc(dy),
            ConvDesc(T, cdims), TensorDesc(dx),
            Cint(count),
            out(Ref{Cint}()), perfResults)
        )[]
    return returnedAlgoCount, perfResults
end

function cudnnFindConvolutionBackwardDataAlgorithm(dx::DenseCuArray{T,N}, w::DenseCuArray{T,N}, dy::DenseCuArray{T,N},
                                                   cdims::DenseConvDims; count=-1) where {T,N}
    if count < 0
        count = cudnnGetConvolutionBackwardDataAlgorithmMaxCount()
    end
    perfResults = Array{cudnnConvolutionBwdDataAlgoPerf_t, 1}(undef, count)
    returnedAlgoCount=@argout(
        cudnnFindConvolutionBackwardDataAlgorithm(
            handle(), FilterDesc(w), TensorDesc(dy),
            ConvDesc(T, cdims), TensorDesc(dx),
            Cint(count),
            out(Ref{Cint}()), perfResults)
        )[]
    return returnedAlgoCount, perfResults
end

function cudnnFindConvolutionBackwardDataAlgorithmEx(dx::DenseCuArray{T,N}, w::DenseCuArray{T,N}, dy::DenseCuArray{T,N},
                                                     cdims::DenseConvDims; count=-1, workspacesize=1<<32) where {T,N}
    if count < 0
        count = cudnnGetConvolutionBackwardDataAlgorithmMaxCount()
    end
    @workspace size=workspacesize workspace->begin
        perfResults = Array{cudnnConvolutionBwdDataAlgoPerf_t, 1}(undef, count)
        returnedAlgoCount=@argout(
            cudnnFindConvolutionBackwardDataAlgorithmEx(
                handle(), FilterDesc(w), w, TensorDesc(dy), dy,
                ConvDesc(T, cdims), TensorDesc(dx), dx,
                Cint(count),
                out(Ref{Cint}()),
                perfResults, workspace,
                workspacesize)
            )[]
        return returnedAlgoCount, perfResults
    end
end

function cudnnConvolutionBackwardData(dx::DenseCuArray{T,N}, w::DenseCuArray{T,N}, dy::DenseCuArray{T,N},
                                      cdims::DenseConvDims; algo=0, alpha=1, beta=0) where {T,N}
    @workspace size=@argout(
            cudnnGetConvolutionBackwardDataWorkspaceSize(
                handle(), FilterDesc(w),
                TensorDesc(dy), ConvDesc(T, cdims), TensorDesc(dx),
                cudnnConvolutionBwdDataAlgo_t(algo),
                out(Ref{Csize_t}()))
        )[] workspace->begin
            cudnnConvolutionBackwardData(
                handle(), scalingParameter(T, alpha), FilterDesc(w), w,
                TensorDesc(dy), dy, ConvDesc(T, cdims),
                cudnnConvolutionBwdDataAlgo_t(algo),
                workspace, sizeof(workspace),
                scalingParameter(T, beta), TensorDesc(dx), dx)
        end
    return dx
end

# Backward filter

function cudnnGetConvolutionBackwardFilterAlgorithmMaxCount()
    count=@argout(
        cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
            handle(),
            out(Ref{Cint}()))
        )[]
    return count
end

# will be removed in cuDNN 8
function cudnnGetConvolutionBackwardFilterAlgorithm(dw::DenseCuArray{T,N}, x::DenseCuArray{T,N}, dy::DenseCuArray{T,N},
                                                    cdims::DenseConvDims; preference=0, workspacesize=1<<32) where {T,N}
    algo=@argout(
        cudnnGetConvolutionBackwardFilterAlgorithm(
            handle(), TensorDesc(x), TensorDesc(dy),
            ConvDesc(T, cdims), FilterDesc(dw), cudnnConvolutionBwdFilterPreference_t(preference),
            Csize_t(workspacesize), out(Ref{cudnnConvolutionBwdFilterAlgo_t}()))
        )[]
    return algo
end

function cudnnGetConvolutionBackwardFilterAlgorithm_v7(dw::DenseCuArray{T,N}, x::DenseCuArray{T,N}, dy::DenseCuArray{T,N},
                                                     cdims::DenseConvDims; count=-1) where {T,N}
    if count < 0
        count = cudnnGetConvolutionBackwardFilterAlgorithmMaxCount()
    end
    perfResults = Array{cudnnConvolutionBwdFilterAlgoPerf_t, 1}(undef, count)
    returnedAlgoCount=@argout(
        cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            handle(), TensorDesc(x), TensorDesc(dy),
            ConvDesc(T, cdims), FilterDesc(dw),
            Cint(count),
            out(Ref{Cint}()),
            perfResults)
        )[]
    return returnedAlgoCount, perfResults
end

function cudnnFindConvolutionBackwardFilterAlgorithm(dw::DenseCuArray{T,N}, x::DenseCuArray{T,N}, dy::DenseCuArray{T,N},
                                                     cdims::DenseConvDims; count=-1) where {T,N}
    if count < 0
        count = cudnnGetConvolutionBackwardFilterAlgorithmMaxCount()
    end
    perfResults = Array{cudnnConvolutionBwdFilterAlgoPerf_t, 1}(undef, count)
    returnedAlgoCount=@argout(
        cudnnFindConvolutionBackwardFilterAlgorithm(
            handle(), TensorDesc(x), TensorDesc(dy),
            ConvDesc(T, cdims), FilterDesc(dw),
            Cint(count),
            out(Ref{Cint}()),
            perfResults)
        )[]
    return returnedAlgoCount, perfResults
end

function cudnnFindConvolutionBackwardFilterAlgorithmEx(dw::DenseCuArray{T,N}, x::DenseCuArray{T,N}, dy::DenseCuArray{T,N},
                                                     cdims::DenseConvDims; count=-1, workspacesize=1<<32) where {T,N}
    if count < 0
        count = cudnnGetConvolutionBackwardFilterAlgorithmMaxCount()
    end
    @workspace size=workspacesize workspace->begin
        perfResults = Array{cudnnConvolutionBwdFilterAlgoPerf_t, 1}(undef, count)
        returnedAlgoCount=@argout(
            cudnnFindConvolutionBackwardFilterAlgorithmEx(
                handle(), TensorDesc(x), x, TensorDesc(dy),
                dy, ConvDesc(T, cdims), FilterDesc(dw), dw,
                Cint(count),
                out(Ref{Cint}()),
                perfResults, workspace,
                workspacesize)
            )[]
        return returnedAlgoCount, perfResults
    end
end

function cudnnConvolutionBackwardFilter(dw::DenseCuArray{T,N}, x::DenseCuArray{T,N}, dy::DenseCuArray{T,N},
                                        cdims::DenseConvDims; algo=0, alpha=1, beta=0) where {T,N}
    @workspace size=@argout(
            cudnnGetConvolutionBackwardFilterWorkspaceSize(
                handle(), TensorDesc(x),
                TensorDesc(dy),
                ConvDesc(T, cdims),
                FilterDesc(dw),
                cudnnConvolutionBwdFilterAlgo_t(algo),
                out(Ref{Csize_t}()))
        )[] workspace->begin
            cudnnConvolutionBackwardFilter(
                handle(), scalingParameter(T, alpha), TensorDesc(x), x,
                TensorDesc(dy), dy, ConvDesc(T, cdims),
                cudnnConvolutionBwdFilterAlgo_t(algo), workspace,
                sizeof(workspace), scalingParameter(T, beta), FilterDesc(dw), dw)
        end
    return dw
end

# Backward bias

function cudnnConvolutionBackwardBias(db::DenseCuArray{T,N}, dy::DenseCuArray{T,N}; alpha=1, beta=0) where {T,N}
    cudnnConvolutionBackwardBias(handle(),
                                 scalingParameter(T, alpha), TensorDesc(dy), dy,
                                 scalingParameter(T, beta),  TensorDesc(db), db)
    return db
end
