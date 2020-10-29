# For low level cudnn functions that require a pointer to a number
cptr(x,a::DenseCuArray{Float64})=Float64[x]
cptr(x,a::DenseCuArray{Float32})=Float32[x]
cptr(x,a::DenseCuArray{Float16})=Float32[x]

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

# The storage data types for alpha and beta are:
#     float for HALF and FLOAT tensors, and
#     double for DOUBLE tensors.
scalingParameter(T, val) = error("Unknown tensor type $T")
scalingParameter(::Type{Float16}, val) = Ref{Float32}(val)
scalingParameter(::Type{Float32}, val) = Ref{Float32}(val)
scalingParameter(::Type{Float64}, val) = Ref{Float64}(val)
