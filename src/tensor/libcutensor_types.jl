"""
Status messages from CUTENSOR's C API.
"""
const cutensorStatus_t = UInt32
const CUTENSOR_STATUS_SUCCESS                = UInt32(0)
const CUTENSOR_STATUS_NOT_INITIALIZED        = UInt32(1)
const CUTENSOR_STATUS_ALLOC_FAILED           = UInt32(3)
const CUTENSOR_STATUS_INVALID_VALUE          = UInt32(7)
const CUTENSOR_STATUS_ARCH_MISMATCH          = UInt32(8)
const CUTENSOR_STATUS_MAPPING_ERROR          = UInt32(11)
const CUTENSOR_STATUS_EXECUTION_FAILED       = UInt32(13)
const CUTENSOR_STATUS_INTERNAL_ERROR         = UInt32(14)
const CUTENSOR_STATUS_NOT_SUPPORTED          = UInt32(15)
const CUTENSOR_STATUS_LICENSE_ERROR          = UInt32(16)
const CUTENSOR_STATUS_CUBLAS_ERROR           = UInt32(17)
const CUTENSOR_STATUS_CUDA_ERROR             = UInt32(18)
const CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE = UInt32(19)

#enum cutensorOperator_t
const cutensorOperator_t = UInt32
# Unary   
const CUTENSOR_OP_IDENTITY = UInt32(1)
const CUTENSOR_OP_SQRT     = UInt32(2)
const CUTENSOR_OP_RELU     = UInt32(8)
const CUTENSOR_OP_CONJ     = UInt32(9)
const CUTENSOR_OP_RCP      = UInt32(10)
# Binary
const CUTENSOR_OP_ADD      = UInt32(3)
const CUTENSOR_OP_MUL      = UInt32(5)
const CUTENSOR_OP_MAX      = UInt32(6)
const CUTENSOR_OP_MIN      = UInt32(7)
const CUTENSOR_OP_UNKNOWN  = UInt32(126)

#enum cutensorWorksizePreference_t
const cutensorWorksizePreference_t = UInt32
const CUTENSOR_WORKSPACE_MIN         = UInt32(1)
const CUTENSOR_WORKSPACE_RECOMMENDED = UInt32(2)
const CUTENSOR_WORKSPACE_MAX         = UInt32(3)

#enum cutensorPaddingType_t
const cutensorPaddingType_t = UInt32
const CUTENSOR_PADDING_NONE = UInt32(0)
const CUTENSOR_PADDING_ZERO = UInt32(1)

#enum cutensorAlgo_t
const cutensorAlgo_t = Int32
const CUTENSOR_ALGO_TGETT          = Int32(-7)
const CUTENSOR_ALGO_GETT           = Int32(-6)
const CUTENSOR_ALGO_LOG_TENSOR_OP  = Int32(-5)
const CUTENSOR_ALGO_LOG            = Int32(-4)
const CUTENSOR_ALGO_TTGT_TENSOR_OP = Int32(-3)
const CUTENSOR_ALGO_TTGT           = Int32(-2)
const CUTENSOR_ALGO_DEFAULT        = Int32(-1)

const cutensorTensorDescriptor_t = Ptr{Cvoid}
const cutensorContext = Cvoid
const cutensorHandle_t = Ptr{cutensorContext}
    
const cudaDataType_t = UInt32
const CUDA_R_16F = UInt32(2)
const CUDA_C_16F = UInt32(6)
const CUDA_R_32F = UInt32(0)
const CUDA_C_32F = UInt32(4)
const CUDA_R_64F = UInt32(1)
const CUDA_C_64F = UInt32(5)
const CUDA_R_8I  = UInt32(3)
const CUDA_C_8I  = UInt32(7)
const CUDA_R_8U  = UInt32(8)
const CUDA_C_8U  = UInt32(9)
const CUDA_R_32I = UInt32(10)
const CUDA_C_32I = UInt32(11)
const CUDA_R_32U = UInt32(12)
const CUDA_C_32U = UInt32(13)


mutable struct CuTensor{T, N} <: AbstractArray{T, N}
    data::CuArray{T, N}
    inds::Vector{Cwchar_t}
    function CuTensor{T, N}(data::CuArray{T, N}, inds::Vector{Cwchar_t}) where {T<:Number, N}
        new(data, inds)
    end
    function CuTensor{T, N}(data::CuArray{N, T}, inds::Vector{<:AbstractChar}) where {T<:Number, N}
        new(data, Cwchar_t.(inds))
    end
end
CuTensor(data::CuArray{T, N}, inds::Vector{<:AbstractChar}) where {T<:Number, N} = CuTensor{T, N}(data, convert(Vector{Cwchar_t}, inds))
CuTensor(data::CuArray{T, N}, inds::Vector{Cwchar_t}) where {T<:Number, N} = CuTensor{T, N}(data, inds)

Base.size(T::CuTensor) = size(T.data)
Base.size(T::CuTensor, i) = size(T.data, i)
Base.length(T::CuTensor) = length(T.data)
Base.ndims(T::CuTensor) = length(T.inds)
Base.strides(T::CuTensor) = strides(T.data)
Base.eltype(T::CuTensor) = eltype(T.data)
Base.similar(T::CuTensor{Tv, N}) where {Tv, N} = CuTensor{Tv, N}(similar(T.data), copy(T.inds))
Base.collect(T::CuTensor) = (collect(T.data), T.inds)
