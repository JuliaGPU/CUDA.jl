# For low level cudnn functions that require a pointer to a number
cptr(x,a::CuArray{Float64})=Float64[x]
cptr(x,a::CuArray{Float32})=Float32[x]
cptr(x,a::CuArray{Float16})=Float32[x]

# Conversion between Julia and CUDNN datatypes
cudnnDataType(::Type{Float16})=CUDNN_DATA_HALF
cudnnDataType(::Type{Float32})=CUDNN_DATA_FLOAT
cudnnDataType(::Type{Float64})=CUDNN_DATA_DOUBLE
juliaDataType(a)=(a==CUDNN_DATA_HALF ? Float16 :
                  a==CUDNN_DATA_FLOAT ? Float32 :
                  a==CUDNN_DATA_DOUBLE ? Float64 : error())

tuple_strides(A::Tuple) = _strides((1,), A)
_strides(out::Tuple{Int}, A::Tuple{}) = ()
_strides(out::NTuple{N,Int}, A::NTuple{N}) where {N} = out
function _strides(out::NTuple{M,Int}, A::Tuple) where M
    Base.@_inline_meta
    _strides((out..., out[M]*A[M]), A)
end

# Descriptors

mutable struct TensorDesc; ptr; end
free(td::TensorDesc) = cudnnDestroyTensorDescriptor(td.ptr)
Base.unsafe_convert(::Type{cudnnTensorDescriptor_t}, td::TensorDesc) = td.ptr
Base.unsafe_convert(::Type{Ptr{Nothing}}, td::TensorDesc) = convert(Ptr{Nothing}, td.ptr)

function TensorDesc(T::Type, size::NTuple{N,Integer}, strides::NTuple{N,Integer} = tuple_strides(size)) where N
    sz = Cint.(size) |> reverse |> collect
    st = Cint.(strides) |> reverse |> collect
    d = Ref{cudnnTensorDescriptor_t}()
    cudnnCreateTensorDescriptor(d)
    cudnnSetTensorNdDescriptor(d[], cudnnDataType(T), length(sz), sz, st)
    this = TensorDesc(d[])
    finalizer(free, this)
    return this
end

TensorDesc(a::CuArray) = TensorDesc(eltype(a), size(a), strides(a))

mutable struct FilterDesc
  ptr
end
free(fd::FilterDesc)=cudnnDestroyFilterDescriptor(fd.ptr)
Base.unsafe_convert(::Type{cudnnFilterDescriptor_t}, fd::FilterDesc)=fd.ptr
Base.unsafe_convert(::Type{Ptr{Nothing}}, fd::FilterDesc)=fd.ptr

function createFilterDesc()
  d = Ref{cudnnFilterDescriptor_t}()
  @check cudnnCreateFilterDescriptor(d)
  return d[]
end

function FilterDesc(T::Type, size::Tuple; format = CUDNN_TENSOR_NCHW)
    # The only difference of a FilterDescriptor is no strides.
    sz = Cint.(size) |> reverse |> collect
    d = createFilterDesc()
    version() >= v"5" ?
        cudnnSetFilterNdDescriptor(d, cudnnDataType(T), format, length(sz), sz) :
    version() >= v"4" ?
        cudnnSetFilterNdDescriptor_v4(d, cudnnDataType(T), format, length(sz), sz) :
        cudnnSetFilterNdDescriptor(d, cudnnDataType(T), length(sz), sz)
    this = FilterDesc(d)
    finalizer(free, this)
    return this
end

FilterDesc(a::CuArray; format = CUDNN_TENSOR_NCHW) = FilterDesc(eltype(a), size(a), format = format)

function Base.size(f::FilterDesc)
  typ = Ref{Cuint}()
  format = Ref{Cuint}()
  ndims = Ref{Cint}()
  dims = Vector{Cint}(undef, 8)
  cudnnGetFilterNdDescriptor(f, 8, typ, format, ndims, dims)
  @assert ndims[] ≤ 8
  return (dims[1:ndims[]]...,) |> reverse
end

mutable struct ConvDesc; ptr; end
free(cd::ConvDesc) = cudnnDestroyConvolutionDescriptor(cd.ptr)
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

function ConvDesc(T, N, padding, stride, dilation, mode)
    cd = Ref{cudnnConvolutionDescriptor_t}()
    cudnnCreateConvolutionDescriptor(cd)
    version() >= v"4" ? cudnnSetConvolutionNdDescriptor(cd[],N,cdsize(padding,N),cdsize(stride,N),cdsize(dilation,N),mode,cudnnDataType(T)) :
    version() >= v"3" ? cudnnSetConvolutionNdDescriptor_v3(cd[],N,cdsize(padding,N),cdsize(stride,N),cdsize(dilation,N),mode,cudnnDataType(T)) :
    cudnnSetConvolutionNdDescriptor(cd[],N,cdsize(padding,N),cdsize(stride,N),cdsize(dilation,N),mode)
    this = ConvDesc(cd[])
    finalizer(free, this)
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

mutable struct PoolDesc; ptr; end
free(pd::PoolDesc)=cudnnDestroyPoolingDescriptor(pd.ptr)
Base.unsafe_convert(::Type{cudnnPoolingDescriptor_t}, pd::PoolDesc)=pd.ptr

function PoolDesc(nd, window, padding, stride, mode, maxpoolingNanOpt=CUDNN_NOT_PROPAGATE_NAN)
    pd = Ref{cudnnPoolingDescriptor_t}()
    cudnnCreatePoolingDescriptor(pd)
    cudnnSetPoolingNdDescriptor(pd[],mode,maxpoolingNanOpt,nd,pdsize(window,nd),pdsize(padding,nd),pdsize(stride,nd))
    this = PoolDesc(pd[])
    finalizer(free, this)
    return this
end

function PoolDesc(pdims::PoolDims, mode, maxpoolingNanOpt=CUDNN_NOT_PROPAGATE_NAN)
    pd = NNlib.padding(pdims)
    if !all(pd[1:2:end] .== pd[2:2:end])
        @warn("CuDNN does not support asymmetric padding; defaulting to symmetric choice")
    end
    return PoolDesc(NNlib.spatial_dims(pdims), NNlib.kernel_size(pdims), pd[1:2:end],
                    NNlib.stride(pdims), mode, maxpoolingNanOpt)
end

mutable struct ActivationDesc; ptr; end
free(ad::ActivationDesc)=cudnnDestroyActivationDescriptor(ad.ptr)
Base.unsafe_convert(::Type{cudnnActivationDescriptor_t}, ad::ActivationDesc)=ad.ptr

function ActivationDesc(mode, coeff, reluNanOpt=CUDNN_NOT_PROPAGATE_NAN)
    ad = Ref{cudnnActivationDescriptor_t}()
    cudnnCreateActivationDescriptor(ad)
    cudnnSetActivationDescriptor(ad[],mode,reluNanOpt,coeff)
    this = ActivationDesc(ad[])
    finalizer(free, this)
    return this
end


# wrappers for low-level CUDNN functionality

function cudnnCreate()
    handle = Ref{cudnnHandle_t}()
    cudnnCreate(handle)
    return handle[]
end

function cudnnSoftmaxForward(src::CuArray{T,4}, dest::CuArray{T,4}=src;
                             algorithm=CUDNN_SOFTMAX_ACCURATE, # or CUDNN_SOFTMAX_FAST
                             mode=CUDNN_SOFTMAX_MODE_INSTANCE, # or CUDNN_SOFTMAX_MODE_CHANNEL
                             alpha=1.0, beta=0.0) where T
    cudnnSoftmaxForward(algorithm, mode,
                        cptr(alpha, src), TensorDesc(src), src,
                        cptr(beta, dest), TensorDesc(dest), dest)
    return dest
end

function cudnnSoftmaxBackward(src::CuArray{T,4}, srcDiff::CuArray{T,4}, destDiff::CuArray=srcDiff;
                              algorithm=CUDNN_SOFTMAX_ACCURATE, # or CUDNN_SOFTMAX_FAST
                              mode=CUDNN_SOFTMAX_MODE_INSTANCE, # or CUDNN_SOFTMAX_MODE_CHANNEL
                              alpha=1.0, beta=0.0) where T
    cudnnSoftmaxBackward(algorithm, mode,
                         cptr(alpha, src), TensorDesc(src), src,
                         TensorDesc(srcDiff), srcDiff,
                         cptr(beta, destDiff), TensorDesc(destDiff), destDiff)
    return destDiff
end

function cudnnConvolutionBiasActivationForward(y::CuArray{T,N}, x::CuArray{T,N}, w::CuArray{T,N}, bias::CuArray{T,N};
                                               alpha1=1, workspace=CU_NULL, workspace_size=0,
                                               algo=0, alpha2=0, padding=0, stride=1, dilation=1, mode=0,
                                               activationMode=CUDNN_ACTIVATION_IDENTITY, activationCoeff=0.0,
                                               activationReluNanOpt=CUDNN_NOT_PROPAGATE_NAN) where {T,N}
    cd = ConvDesc(T, N-2, padding, stride, dilation, mode)
    ad = ActivationDesc(activationMode, T(activationCoeff), activationReluNanOpt)
    cudnnConvolutionBiasActivationForward(Ref(T(alpha1)),TensorDesc(x),x,FilterDesc(w),w,cd,algo,workspace,
        workspace_size,Ref(T(alpha2)),TensorDesc(bias),bias,ad,TensorDesc(y),y)
    return y
end

function cudnnConvolutionForward(y::CuArray{T,N}, x::CuArray{T,N}, w::CuArray{T,N},
                                 cdims::DenseConvDims; algo=0, workspace=CU_NULL,
                                 workspace_size=0, alpha=1, beta=0) where {T,N}
    cudnnConvolutionForward(
      Ref(T(alpha)), TensorDesc(x), x, FilterDesc(w), w, ConvDesc(T,cdims),
      algo, workspace, workspace_size, Ref(T(beta)), TensorDesc(y), y
    )
    return y
end

function cudnnGetConvolutionForwardWorkspaceSize(y::CuArray{T,N}, x::CuArray{T,N}, w::CuArray{T,N},
                                                 cdims::DenseConvDims; algo=0) where {T,N}
    workspace_size = Ref{Cint}()
    cudnnGetConvolutionForwardWorkspaceSize(
        TensorDesc(x), FilterDesc(w), ConvDesc(T, cdims),
        TensorDesc(y), algo, workspace_size
    )
    return Int(workspace_size[])
end

function cudnnConvolutionBackwardData(dx::CuArray{T,N}, w::CuArray{T,N}, dy::CuArray{T,N},
                                      cdims::DenseConvDims; algo=0, workspace=CU_NULL,
                                      workspace_size=0, alpha=1, beta=0) where {T,N}
    cudnnConvolutionBackwardData(
      Ref(T(alpha)), FilterDesc(w), w, TensorDesc(dy), dy, ConvDesc(T, cdims),
      algo, workspace, workspace_size, Ref(T(beta)), TensorDesc(dx), dx
    )
    return dx
end

function cudnnGetConvolutionBackwardDataWorkspaceSize(dx::CuArray{T,N}, w::CuArray{T,N}, dy::CuArray{T,N},
                                                      cdims::DenseConvDims; algo=0) where {T,N}
    workspace_size = Ref{Cint}()
    cudnnGetConvolutionBackwardDataWorkspaceSize(
        FilterDesc(w), TensorDesc(dy), ConvDesc(T, cdims),
        TensorDesc(dx), algo, workspace_size
    )
    return Int(workspace_size[])
end

function cudnnConvolutionBackwardFilter(dw::CuArray{T,N}, x::CuArray{T,N}, dy::CuArray{T,N},
                                        cdims::DenseConvDims; algo=0, workspace=CU_NULL,
                                        workspace_size=0, alpha=1, beta=0) where {T,N}
    cudnnConvolutionBackwardFilter(
        Ref(T(alpha)), TensorDesc(x), x, TensorDesc(dy), dy, ConvDesc(T, cdims),
        algo, workspace, workspace_size, Ref(T(beta)), FilterDesc(dw), dw
    )
    return dw
end

function cudnnGetConvolutionBackwardFilterWorkspaceSize(dw::CuArray{T,N}, x::CuArray{T,N}, dy::CuArray{T,N},
                                                        cdims::DenseConvDims; algo=0) where {T,N}
    workspace_size = Ref{Cint}()
    cudnnGetConvolutionBackwardFilterWorkspaceSize(
        TensorDesc(x), TensorDesc(dy), ConvDesc(T, cdims),
        FilterDesc(dw), algo, workspace_size
    )
    return Int(workspace_size[])
end

function cudnnConvolutionBackwardBias(db::CuArray{T,N}, dy::CuArray{T,N}; alpha=1, beta=0) where {T,N}
    cudnnConvolutionBackwardBias(Ref(T(alpha)), TensorDesc(dy), dy, Ref(T(beta)), TensorDesc(db), db)
    return db
end

function cudnnPoolingForward(y::CuArray{T,N}, x::CuArray{T,N}, pdims::PoolDims;
                             alpha=1, mode=0) where {T,N}
    beta = 0
    cudnnPoolingForward(PoolDesc(pdims, mode), Ref(T(alpha)), TensorDesc(x), x, Ref(T(beta)), TensorDesc(y), y)
    return y
end

function cudnnPoolingBackward(dx::CuArray{T,N}, dy::CuArray{T,N}, x::CuArray{T,N}, y::CuArray{T,N},
                              pdims::PoolDims; alpha=1, mode=0) where {T,N}
    if alpha!=1 && mode==0; error("Gradient of pool(alpha!=1,mode=0) broken in CUDNN"); end
    beta = 0
    cudnnPoolingBackward(
        PoolDesc(pdims, mode), Ref(T(alpha)), TensorDesc(y), y,
        TensorDesc(dy), dy, TensorDesc(x), x, Ref(T(beta)), TensorDesc(dx), dx
    )
    return dx
end

function cudnnActivationForward(y::CuArray{T,N}, x::CuArray{T,N}; mode=CUDNN_ACTIVATION_RELU, #CUDNN_ACTIVATION_IDENTITY will not work
                                coeff=0.0, reluNanOpt=CUDNN_NOT_PROPAGATE_NAN, alpha=1, beta=0) where {T,N}
    ad = ActivationDesc(mode, T(coeff), reluNanOpt)
    cudnnActivationForward(ad, Ref(T(alpha)), TensorDesc(x), x, Ref(T(beta)), TensorDesc(y), y)
    return y
end

function cudnnActivationBackward(dx::CuArray{T,N}, x::CuArray{T,N}, y::CuArray{T,N}, dy::CuArray{T,N};
                                 mode=CUDNN_ACTIVATION_RELU, #CUDNN_ACTIVATION_IDENTITY will not work
                                 coeff=0.0, reluNanOpt=CUDNN_NOT_PROPAGATE_NAN, alpha=1, beta=0) where {T,N}
    ad = ActivationDesc(mode, T(coeff), reluNanOpt)
    cudnnActivationBackward(ad, Ref(T(alpha)), TensorDesc(y), y, TensorDesc(dy), dy, TensorDesc(x), x, Ref(T(beta)), TensorDesc(dx), dx)
    return dx
end

function cudnnAddTensor(A::CuArray{T,N}, C::CuArray{T,N}; alpha=1,
                        beta=1) where {T,N}
    aDesc = TensorDesc(A)
    cDesc = TensorDesc(C)
    cudnnAddTensor(Ref(T(alpha)), aDesc, A, Ref(T(beta)), cDesc, C)
    return C
end

function cudnnGetProperty(property::CUDAapi.libraryPropertyType)
  value_ref = Ref{Cint}()
  cudnnGetProperty(property, value_ref)
  value_ref[]
end
