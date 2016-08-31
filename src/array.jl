# Contiguous on-device arrays

import Base: convert, unsafe_convert

export
    CuDeviceArray, CuBoundsError


## construction

immutable CuDeviceArray{T,N} <: AbstractArray{T,N}
    ptr::Ptr{T}
    shape::NTuple{N,Int}
    len::Int

    function CuDeviceArray(shape::NTuple{N,Int}, ptr::Ptr{T})
        len = prod(shape)
        new(ptr, shape, len)
    end
end

(::Type{CuDeviceArray{T}}){T,N}(shape::NTuple{N,Int}, p::Ptr{T}) = CuDeviceArray{T,N}(shape, p)
(::Type{CuDeviceArray{T}}){T}(len::Int, p::Ptr{T})               = CuDeviceArray{T,1}((len,), p)

# deprecated
CuDeviceArray{T,N}(::Type{T}, shape::NTuple{N,Int}, p::Ptr{T}) = CuDeviceArray{T,N}(shape, p)
CuDeviceArray{T}(::Type{T}, len::Int, p::Ptr{T})               = CuDeviceArray{T,1}((len,), p)

cudaconvert{T,N}(::Type{CuArray{T,N}}) = CuDeviceArray{T,N}

convert{T,N}(::Type{CuDeviceArray{T,N}}, a::CuArray{T,N}) =
    CuDeviceArray{T,N}(a.shape, unsafe_convert(Ptr{T}, a.ptr))


## array interface

import Base: length, size,
             linearindexing, LinearFast, getindex, setindex!,
             pointerref, pointerset

@target ptx length(g::CuDeviceArray) = g.len
@target ptx size(g::CuDeviceArray) = g.shape

linearindexing{A<:CuDeviceArray}(::Type{A}) = LinearFast()

@target ptx setindex!{T}(A::CuDeviceArray{T}, x, i::Int) =
    pointerset(unsafe_convert(Ptr{T}, A), convert(T, x)::T, i, 8)

@target ptx getindex{T}(A::CuDeviceArray{T}, i::Int) =
    pointerref(unsafe_convert(Ptr{T}, A), i, 8)::T

@target ptx @inline unsafe_convert{T}(::Type{Ptr{T}}, A::CuDeviceArray{T}) = A.ptr::Ptr{T}


## compatibility fixes

# TODO: remove this hack as soon as immutables with heap references (such as BoundsError)
#       can be stack-allocated
import Base: throw_boundserror
immutable CuBoundsError <: Exception end
@target ptx throw_boundserror{T,N}(A::CuDeviceArray{T,N}, I) =
    (Base.@_noinline_meta; throw(CuBoundsError()))

# TODO: same for SubArray, although it might be too complex to ever be non-allocating
import Base: unsafe_view, ViewIndex
@target ptx function unsafe_view{T}(A::CuDeviceArray{T,1}, I::Vararg{ViewIndex,1})
    Base.@_inline_meta
    ptr = unsafe_convert(Ptr{T}, A) + (I[1].start-1)*sizeof(T)
    len = I[1].stop - I[1].start + 1
    CuDeviceArray{T}(len, ptr)
end
