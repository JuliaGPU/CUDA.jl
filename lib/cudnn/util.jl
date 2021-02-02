# For low level cudnn functions that require a pointer to a number
cptr(x,a::DenseCuArray{Float64})=Float64[x]
cptr(x,a::DenseCuArray{Float32})=Float32[x]
cptr(x,a::DenseCuArray{Float16})=Float32[x]

# Conversion between Julia and CUDNN datatypes
cudnnDataType(::Type{Float16})=CUDNN_DATA_HALF
cudnnDataType(::Type{Float32})=CUDNN_DATA_FLOAT
cudnnDataType(::Type{Float64})=CUDNN_DATA_DOUBLE
cudnnDataType(::Type{Int8}) = CUDNN_DATA_INT8
cudnnDataType(::Type{UInt8}) = CUDNN_DATA_UINT8
cudnnDataType(::Type{Int32}) = CUDNN_DATA_INT32
# The following are 32-bit elements each composed of 4 8-bit integers, only supported with CUDNN_TENSOR_NCHW_VECT_C
# CUDNN_DATA_INT8x4,
# CUDNN_DATA_UINT8x4,
# CUDNN_DATA_INT8x32,
juliaDataType(a)=(a==CUDNN_DATA_HALF ? Float16 :
                  a==CUDNN_DATA_FLOAT ? Float32 :
                  a==CUDNN_DATA_DOUBLE ? Float64 :
                  a==CUDNN_DATA_INT8 ? Int8 :
                  a==CUDNN_DATA_UINT8 ? UInt8 :
                  a==CUDNN_DATA_INT32 ? Int32 : error())

tuple_strides(A::Tuple) = _strides((1,), A)
_strides(out::Tuple{Int}, A::Tuple{}) = ()
_strides(out::NTuple{N,Int}, A::NTuple{N}) where {N} = out
function _strides(out::NTuple{M,Int}, A::Tuple) where M
    Base.@_inline_meta
    _strides((out..., out[M]*A[M]), A)
end

# The storage data types for alpha and beta are:
#     float for HALF and FLOAT tensors, and
#     double for DOUBLE tensors.
scalingParameter(T, val) = error("Unknown tensor type $T")
scalingParameter(::Type{Float16}, val) = Ref{Float32}(val)
scalingParameter(::Type{Float32}, val) = Ref{Float32}(val)
scalingParameter(::Type{Float64}, val) = Ref{Float64}(val)


# Create temporary reserveSpace. Use 128 to avoid alignment issues.
function cudnnTempSpace(nbytes)
    nbytes == 0 ? nothing : CuArray{Int128}(undef, (nbytes-1)÷sizeof(Int128)+1)
end


function nnlibPadding(dims)
    pd = NNlib.padding(dims)
    if !all(pd[1:2:end] .== pd[2:2:end])
        @warn "cuDNN does not support asymmetric padding; defaulting to symmetric choice" maxlog=1
    end
    return pd[1:2:end]
end
