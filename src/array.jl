# Contiguous on-device arrays

import Base: length, size, getindex, setindex!, convert

export
    CuDeviceArray, CuBoundsError


immutable CuDeviceArray{T,N} <: AbstractArray{T,N}
    ptr::DevicePtr{T}
    shape::NTuple{N,Int}
    len::Int

    function CuDeviceArray(::Type{T}, shape::NTuple{N,Int}, ptr::DevicePtr{T})
        len = prod(shape)
        new(ptr, shape, len)
    end

    function CuDeviceArray(::Type{T}, len::Int, ptr::DevicePtr{T})
        shape = (len,)
        new(ptr, shape, len)
    end
end

cudaconvert{T,N}(::Type{CuArray{T,N}}) = CuDeviceArray{T,N}

# Define outer constructors for parameter-less construction
CuDeviceArray{T}(::Type{T}, ptr::DevicePtr{T}, len::Int) = CuDeviceArray{T,1}(T, len, ptr)
CuDeviceArray{T,N}(::Type{T}, ptr::DevicePtr{T}, shape::NTuple{N,Int}) = CuDeviceArray{T,N}(T, shape, ptr)

convert{T,N}(::Type{CuDeviceArray{T,N}}, a::CuArray{T,N}) = CuDeviceArray{T,N}(T, a.shape, a.ptr)

length(g::CuDeviceArray) = g.len
size(g::CuDeviceArray) = g.shape

@target ptx getindex{T}(a::CuDeviceArray{T}, i0::Real) =
    Base.pointerref(a.ptr.inner, Base.to_index(i0), 8)::T
@target ptx setindex!{T}(a::CuDeviceArray{T}, x::T, i0::Real) =
    Base.pointerset(a.ptr.inner, convert(T, x)::T, Base.to_index(i0), 8)


# TODO: remove this hack as soon as immutables with heap references (such as BoundsError)
#       can be stack-allocated
import Base: throw_boundserror
immutable CuBoundsError <: Exception end
@target ptx throw_boundserror{T,N}(A::CuDeviceArray{T,N}, I) = (Base.@_noinline_meta; throw(CuBoundsError()))

# TODO: same for SubArray, although it might be too complex to every be non-allocating
import Base: unsafe_view, ViewIndex
function unsafe_view{T}(A::CuDeviceArray{T,1}, I::Vararg{ViewIndex,1})
    Base.@_inline_meta
    ptr = DevicePtr{T}(A.ptr.inner + I[1].start*sizeof(T), true)
    len = I[1].stop - I[1].start
    CuDeviceArray{T,1}(T, len, ptr)
end
