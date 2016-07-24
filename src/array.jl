# Contiguous on-device arrays

import Base: length, size, getindex, setindex!, convert, unsafe_convert

export
    CuDeviceArray, CuBoundsError


immutable CuDeviceArray{T,N} <: AbstractArray{T,N}
    ptr::Ptr{T}
    shape::NTuple{N,Int}
    len::Int

    function CuDeviceArray(shape::NTuple{N,Int}, ptr::Ptr{T})
        len = prod(shape)
        new(ptr, shape, len)
    end
end

CuDeviceArray{T,N}(shape::NTuple{N,Int}, p::Ptr{T}) = CuDeviceArray{T,N}(shape, p)
CuDeviceArray{T}(len::Int, p::Ptr{T})               = CuDeviceArray{T,1}((len,), p)

# deprecated
CuDeviceArray{T,N}(::Type{T}, shape::NTuple{N,Int}, p::Ptr{T}) = CuDeviceArray{T,N}(shape, p)
CuDeviceArray{T}(::Type{T}, len::Int, p::Ptr{T})               = CuDeviceArray{T,1}((len,), p)

cudaconvert{T,N}(::Type{CuArray{T,N}}) = CuDeviceArray{T,N}

convert{T,N}(::Type{CuDeviceArray{T,N}}, a::CuArray{T,N}) =
    CuDeviceArray{T,N}(a.shape, unsafe_convert(Ptr{T}, a.ptr))

@target ptx length(g::CuDeviceArray) = g.len
@target ptx size(g::CuDeviceArray) = g.shape

import Base: to_index, pointerref, pointerset

@target ptx @inline function getindex{T}(A::CuDeviceArray{T}, I::Real)
    @boundscheck checkbounds(A, I)
    pointerref(unsafe_convert(Ptr{T}, A), to_index(I), 8)::T
end

@target ptx @inline function setindex!{T}(A::CuDeviceArray{T}, x, I::Real)
    @boundscheck checkbounds(A, I)
    pointerset(unsafe_convert(Ptr{T}, A), convert(T, x)::T, to_index(I), 8)
end

@target ptx @inline unsafe_convert{T}(::Type{Ptr{T}}, A::CuDeviceArray{T}) = A.ptr::Ptr{T}

# TODO: remove this hack as soon as immutables with heap references (such as BoundsError)
#       can be stack-allocated
import Base: throw_boundserror
immutable CuBoundsError <: Exception end
@target ptx throw_boundserror{T,N}(A::CuDeviceArray{T,N}, I) = (Base.@_noinline_meta; throw(CuBoundsError()))

# TODO: same for SubArray, although it might be too complex to ever be non-allocating
import Base: unsafe_view, ViewIndex
@target ptx function unsafe_view{T}(A::CuDeviceArray{T,1}, I::Vararg{ViewIndex,1})
    Base.@_inline_meta
    ptr = unsafe_convert(Ptr{T}, A) + (I[1].start-1)*sizeof(T)
    len = I[1].stop - I[1].start + 1
    CuDeviceArray(len, ptr)
end
