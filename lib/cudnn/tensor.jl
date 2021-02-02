# Alternative constructors for cudnnTensorDescriptor and cudnnFilterDescriptor

function cudnnTensorDescriptor(   # alternative constructor from array; main one in descriptors.jl
    array;
    format::cudnnTensorFormat_t=CUDNN_TENSOR_NCHW,
    dims::Vector{Cint}=dim4(size(array),Val(format))
)
    @assert length(dims) <= CUDNN_DIM_MAX # length(dims) may not be N
    cudnnTensorDescriptor(format, cudnnDataType(eltype(array)), Cint(length(dims)), dims)
end


function cudnnFilterDescriptor(  # alternative constructor from array; main one in descriptors.jl
    array;
    format::cudnnTensorFormat_t=CUDNN_TENSOR_NCHW,
    dims::Vector{Cint}=dim4(size(array),Val(format))
)
    @assert length(dims) <= CUDNN_DIM_MAX # length(dims) may not be N
    cudnnFilterDescriptor(cudnnDataType(eltype(array)), format, Cint(length(dims)), dims)
end


# From cuDNN docs: Due to historical reasons, the minimum number of dimensions in the filter
# descriptor is three, and at most CUDNN_DIM_MAX dimensions (defined in cudnn.h = 8). 
# However many operations only support 4 and 5. So we will pad dims to 4.
# Note also the order of dims reverse from Julia to cuDNN.
# RNN and multiHeadAttn do use 3D descriptors so they do not use dim4.
# Note on formats: even when using the NHWC format the dims are given in NCHW order!

dim4(s::Dims{0}, ::Val{CUDNN_TENSOR_NCHW}) = Cint[1,1,1,1]
dim4(s::Dims{0}, ::Val{CUDNN_TENSOR_NHWC}) = Cint[1,1,1,1]
dim4(s::Dims{1}, ::Val{CUDNN_TENSOR_NCHW}) = Cint[s[1],1,1,1]    # Cy -> Cy,1,1,1
dim4(s::Dims{1}, ::Val{CUDNN_TENSOR_NHWC}) = Cint[s[1],1,1,1]    # Cy -> Cy,1,1,1
dim4(s::Dims{2}, ::Val{CUDNN_TENSOR_NCHW}) = Cint[s[2],s[1],1,1] # Cx,Cy -> Cy,Cx,1,1
dim4(s::Dims{2}, ::Val{CUDNN_TENSOR_NHWC}) = Cint[s[2],s[1],1,1] # Cx,Cy -> Cy,Cx,1,1
dim4(s::Dims{3}, ::Val{CUDNN_TENSOR_NCHW}) = Cint[s[3],s[2],s[1],1] # Xn,Cx,Cy -> Cy,Cx,Xn,1
dim4(s::Dims{3}, ::Val{CUDNN_TENSOR_NHWC}) = Cint[s[3],s[1],s[2],1] # Cx,Xn,Cy -> Cy,Cx,Xn,1
dim4(s::Dims{N}, ::Val{CUDNN_TENSOR_NCHW}) where {N} = Cint[reverse(s)...] # X1,...,Xn,Cx,Cy -> Cy,Cx,Xn,...,X1
dim4(s::Dims{N}, ::Val{CUDNN_TENSOR_NHWC}) where {N} = Cint[s[N],s[1],s[N-1:-1:2]...] # Cx,X1,...,Xn,Cy -> Cy,Cx,Xn,...,X1


# If array is nothing, return nothing for descriptor
cudnnTensorDescriptor(::Nothing; o...) = nothing
cudnnFilterDescriptor(::Nothing; o...) = nothing


# In case we need to get info about a descriptor

function cudnnGetTensorDescriptor(d::cudnnTensorDescriptor)
    nbDimsRequested = CUDNN_DIM_MAX
    dataType = Ref{cudnnDataType_t}(CUDNN_DATA_FLOAT)
    nbDims = Ref{Cint}(0)
    dimA = Array{Cint}(undef, CUDNN_DIM_MAX)
    strideA = Array{Cint}(undef, CUDNN_DIM_MAX)
    cudnnGetTensorNdDescriptor(d, nbDimsRequested, dataType, nbDims, dimA, strideA)
    T = juliaDataType(dataType[])
    D = (dimA[nbDims[]:-1:1]...,)
    S = (strideA[nbDims[]:-1:1]...,)
    return T,D,S
end

function cudnnGetFilterDescriptor(d::cudnnFilterDescriptor)
    nbDimsRequested = CUDNN_DIM_MAX
    dataType = Ref{cudnnDataType_t}(CUDNN_DATA_FLOAT)
    format = Ref{cudnnTensorFormat_t}(CUDNN_TENSOR_NCHW)
    nbDims = Ref{Cint}(0)
    dimA = Array{Cint}(undef, CUDNN_DIM_MAX)
    cudnnGetFilterNdDescriptor(d, nbDimsRequested, dataType, format, nbDims, dimA)
    T = juliaDataType(dataType[])
    D = (dimA[nbDims[]:-1:1]...,)
    return T,D,format[]
end


# Deprecated
function TensorDesc(ptr::cudnnTensorDescriptor_t)
    @warn "TensorDesc is deprecated, use cudnnTensorDescriptor instead." maxlog=1
    cudnnTensorDescriptor(ptr)
end

function TensorDesc(a::DenseCuArray)
    @warn "TensorDesc is deprecated, use cudnnTensorDescriptor instead." maxlog=1
    cudnnTensorDescriptor(a)
end

function TensorDesc(T::Type, size::NTuple{N,Integer}, strides::NTuple{N,Integer} = tuple_strides(size)) where N
    @warn "TensorDesc is deprecated, use cudnnTensorDescriptor instead." maxlog=1
    cudnnTensorDescriptor(CUDNN_TENSOR_NCHW, cudnnDataType(T), Cint(N), dim4(size,Val(CUDNN_TENSOR_NCHW)))
end

function FilterDesc(ptr::cudnnFilterDescriptor_t)
    @warn "FilterDesc is deprecated, use cudnnFilterDescriptor instead." maxlog=1
    cudnnFilterDescriptor(ptr)
end

function FilterDesc(a::DenseCuArray; format = CUDNN_TENSOR_NCHW)
    @warn "FilterDesc is deprecated, use cudnnFilterDescriptor instead." maxlog=1
    cudnnFilterDescriptor(a; format)
end

function FilterDesc(T::Type, size::Tuple; format = CUDNN_TENSOR_NCHW)
    @warn "FilterDesc is deprecated, use cudnnFilterDescriptor instead." maxlog=1
    dims = dim4(size, Val(format))
    cudnnFilterDescriptor(cudnnDataType(T), format, Cint(length(dims)), dims)
end
