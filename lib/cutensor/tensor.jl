export CuTensor

mutable struct CuTensor{T, N}
    data::DenseCuArray{T, N}
    inds::Vector{Char}
    function CuTensor{T, N}(data::DenseCuArray{T, N}, inds::Vector{Char}) where {T<:Number, N}
        new(data, inds)
    end
    function CuTensor{T, N}(data::DenseCuArray{N, T}, inds::Vector{<:AbstractChar}) where {T<:Number, N}
        new(data, Char.(inds))
    end
end

CuTensor(data::DenseCuArray{T, N}, inds::Vector{<:AbstractChar}) where {T<:Number, N} =
    CuTensor{T, N}(data, convert(Vector{Char}, inds))

CuTensor(data::DenseCuArray{T, N}, inds::Vector{Char}) where {T<:Number, N} =
    CuTensor{T, N}(data, inds)

Base.size(T::CuTensor) = size(T.data)
Base.size(T::CuTensor, i) = size(T.data, i)
Base.length(T::CuTensor) = length(T.data)
Base.ndims(T::CuTensor) = length(T.inds)
Base.strides(T::CuTensor) = strides(T.data)
Base.eltype(T::CuTensor) = eltype(T.data)
Base.similar(T::CuTensor{Tv, N}) where {Tv, N} = CuTensor{Tv, N}(similar(T.data), copy(T.inds))
Base.collect(T::CuTensor) = (collect(T.data), T.inds)
