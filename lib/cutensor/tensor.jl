export CuTensor

mutable struct CuTensor{T, N}
    data::CuArray{T, N}
    inds::Vector{Cwchar_t}
    function CuTensor{T, N}(data::CuArray{T, N}, inds::Vector{Cwchar_t}) where {T<:Number, N}
        new(data, inds)
    end
    function CuTensor{T, N}(data::CuArray{N, T}, inds::Vector{<:AbstractChar}) where {T<:Number, N}
        new(data, Cwchar_t.(inds))
    end
end

CuTensor(data::CuArray{T, N}, inds::Vector{<:AbstractChar}) where {T<:Number, N} =
    CuTensor{T, N}(data, convert(Vector{Cwchar_t}, inds))

CuTensor(data::CuArray{T, N}, inds::Vector{Cwchar_t}) where {T<:Number, N} =
    CuTensor{T, N}(data, inds)

Base.size(T::CuTensor) = size(T.data)
Base.size(T::CuTensor, i) = size(T.data, i)
Base.length(T::CuTensor) = length(T.data)
Base.ndims(T::CuTensor) = length(T.inds)
Base.strides(T::CuTensor) = strides(T.data)
Base.eltype(T::CuTensor) = eltype(T.data)
Base.similar(T::CuTensor{Tv, N}) where {Tv, N} = CuTensor{Tv, N}(similar(T.data), copy(T.inds))
Base.collect(T::CuTensor) = (collect(T.data), T.inds)
