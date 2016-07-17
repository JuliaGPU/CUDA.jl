# Contiguous on-device arrays

import Base: length, size, getindex, setindex!, convert

export
    CuDeviceArray


immutable CuDeviceArray{T,N} <: AbstractArray{T,N}
    ptr::DevicePtr{T}
    shape::NTuple{N,Int}
    len::Int

    function CuDeviceArray(::Type{T}, shape::NTuple{N,Int}, ptr::DevicePtr{T})
        len = prod(shape)
        new(ptr, shape, len)
    end
end

cudaconvert{T,N}(::Type{CuArray{T,N}}) = CuDeviceArray{T,N}

# Define outer constructors for parameter-less construction
CuDeviceArray{T}(::Type{T}, ptr::DevicePtr{T}, len::Int) = CuDeviceArray{T,1}(T, (len,), ptr)
CuDeviceArray{T,N}(::Type{T}, ptr::DevicePtr{T}, shape::NTuple{N,Int}) = CuDeviceArray{T,N}(T, shape, ptr)

convert{T,N}(::Type{CuDeviceArray{T,N}}, a::CuArray{T,N}) = CuDeviceArray{T,N}(T, a.shape, a.ptr)

length(g::CuDeviceArray) = g.len
size(g::CuDeviceArray) = g.shape

@target ptx getindex{T}(a::CuDeviceArray{T}, i0::Real) =
    Base.pointerref(a.ptr.inner, Base.to_index(i0), 8)::T
@target ptx setindex!{T}(a::CuDeviceArray{T}, x::T, i0::Real) =
    Base.pointerset(a.ptr.inner, convert(T, x)::T, Base.to_index(i0), 8)
