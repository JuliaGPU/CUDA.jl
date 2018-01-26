# For low level cudnn functions that require a pointer to a number
cptr(x,a::CuArray{Float64})=Float64[x]
cptr(x,a::CuArray{Float32})=Float32[x]
cptr(x,a::CuArray{Float16})=Float32[x]

# Conversion between Julia and CUDNN datatypes
cudnnDataType(::CuArray{Float16})=CUDNN_DATA_HALF
cudnnDataType(::CuArray{Float32})=CUDNN_DATA_FLOAT
cudnnDataType(::CuArray{Float64})=CUDNN_DATA_DOUBLE
cudnnDataType(::Type{Float16})=CUDNN_DATA_HALF
cudnnDataType(::Type{Float32})=CUDNN_DATA_FLOAT
cudnnDataType(::Type{Float64})=CUDNN_DATA_DOUBLE
juliaDataType(a)=(a==CUDNN_DATA_HALF ? Float16 :
                  a==CUDNN_DATA_FLOAT ? Float32 :
                  a==CUDNN_DATA_DOUBLE ? Float64 : error())

# Descriptors

mutable struct TensorDesc; ptr; end
free(td::TensorDesc) = cudnnDestroyTensorDescriptor(td.ptr)
Base.unsafe_convert(::Type{cudnnTensorDescriptor_t}, td::TensorDesc) = td.ptr
Base.unsafe_convert(::Type{Ptr{Void}}, td::TensorDesc) = convert(Ptr{Void}, td.ptr)

function TensorDesc(a::CuArray)
    sz = Cint.(size(a)) |> reverse |> collect
    st = Cint.(strides(a)) |> reverse |> collect
    d = cudnnTensorDescriptor_t[0]
    cudnnCreateTensorDescriptor(d)
    cudnnSetTensorNdDescriptor(d[1], cudnnDataType(a), length(sz), sz, st)
    this = TensorDesc(d[1])
    finalizer(this, free)
    return this
end

mutable struct FilterDesc; ptr; end
free(fd::FilterDesc)=cudnnDestroyFilterDescriptor(fd.ptr)
Base.unsafe_convert(::Type{cudnnFilterDescriptor_t}, fd::FilterDesc)=fd.ptr

function FilterDesc(a::CuArray, format=CUDNN_TENSOR_NCHW)
    # The only difference of a FilterDescriptor is no strides.
    sz = Cint.(size(a)) |> reverse |> collect
    d = cudnnFilterDescriptor_t[0]
    cudnnCreateFilterDescriptor(d)
    CUDNN_VERSION >= 5000 ?
        cudnnSetFilterNdDescriptor(d[1], cudnnDataType(a), format, length(sz), sz) :
    CUDNN_VERSION >= 4000 ?
        cudnnSetFilterNdDescriptor_v4(d[1], cudnnDataType(a), format, length(sz), sz) :
        cudnnSetFilterNdDescriptor(d[1], cudnnDataType(a), length(sz), sz)
    this = FilterDesc(d[1])
    finalizer(this, free)
    return this
end

mutable struct ConvDesc; ptr; end
free(cd::ConvDesc) = cudnnDestroyConvolutionDescriptor(cd.ptr)
Base.unsafe_convert(::Type{cudnnConvolutionDescriptor_t}, cd::ConvDesc)=cd.ptr

function cdsize(w, nd)
    isa(w, Integer) ? Cint[fill(w,nd)...] :
    length(w)!=nd ? error("Dimension mismatch") :
    Cint[reverse(w)...]
end

pdsize(w, nd)=Cint[reverse(psize(w,nd))...]
psize(w, nd)=(isa(w,Integer)  ? fill(w,nd) : length(w) != nd ? error("Dimension mismatch") : w)

function ConvDesc(T, N, padding, stride, upscale, mode)
    cd = cudnnConvolutionDescriptor_t[0]
    cudnnCreateConvolutionDescriptor(cd)
    CUDNN_VERSION >= 4000 ? cudnnSetConvolutionNdDescriptor(cd[1],N,cdsize(padding,N),cdsize(stride,N),cdsize(upscale,N),mode,cudnnDataType(T)) :
    CUDNN_VERSION >= 3000 ? cudnnSetConvolutionNdDescriptor_v3(cd[1],N,cdsize(padding,N),cdsize(stride,N),cdsize(upscale,N),mode,cudnnDataType(T)) :
    cudnnSetConvolutionNdDescriptor(cd[1],N,cdsize(padding,N),cdsize(stride,N),cdsize(upscale,N),mode)
    this = ConvDesc(cd[1])
    finalizer(this, free)
    return this
end

mutable struct PoolDesc; ptr; end
free(pd::PoolDesc)=cudnnDestroyPoolingDescriptor(pd.ptr)
Base.unsafe_convert(::Type{cudnnPoolingDescriptor_t}, pd::PoolDesc)=pd.ptr

function PoolDesc(nd, window, padding, stride, mode, maxpoolingNanOpt=CUDNN_NOT_PROPAGATE_NAN)
    pd = cudnnPoolingDescriptor_t[0]
    cudnnCreatePoolingDescriptor(pd)
    cudnnSetPoolingNdDescriptor(pd[1],mode,maxpoolingNanOpt,nd,pdsize(window,nd),pdsize(padding,nd),pdsize(stride,nd))
    this = PoolDesc(pd[1])
    finalizer(this, free)
    return this
end
