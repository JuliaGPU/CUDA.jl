# Contiguous on-device arrays

export
    CuDeviceArray, CuBoundsError


## construction

struct CuDeviceArray{T,N} <: AbstractArray{T,N}
    shape::NTuple{N,Int}
    ptr::Ptr{T}

    CuDeviceArray{T,N}(shape::NTuple{N,Int}, ptr::Ptr{T}) where {T,N} = new(shape, ptr)
end

(::Type{CuDeviceArray{T}}){T,N}(shape::NTuple{N,Int}, p::Ptr{T}) = CuDeviceArray{T,N}(shape, p)
(::Type{CuDeviceArray{T}}){T}(len::Int, p::Ptr{T})               = CuDeviceArray{T,1}((len,), p)

CuDeviceArray{T,N}(shape::NTuple{N,Int}, p::Ptr{T}) = CuDeviceArray{T,N}(shape, p)
CuDeviceArray{T}(len::Int, p::Ptr{T})               = CuDeviceArray{T,1}((len,), p)

Base.convert{T,N}(::Type{CuDeviceArray{T,N}}, a::CuArray{T,N}) =
    CuDeviceArray{T,N}(a.shape, Base.unsafe_convert(Ptr{T}, a.devptr))

Base.unsafe_convert{T}(::Type{Ptr{T}}, a::CuDeviceArray{T}) = a.ptr::Ptr{T}


## array interface

Base.size(g::CuDeviceArray) = g.shape
Base.length(g::CuDeviceArray) = prod(g.shape)

@inline function Base.getindex{T}(A::CuDeviceArray{T}, index::Int)
    # FIXME: disabled due to PTX assembler issue (see #4)
    # @boundscheck checkbounds(A, index)
    Base.pointerref(Base.unsafe_convert(Ptr{T}, A), index, 8)::T
end

@inline function Base.setindex!{T}(A::CuDeviceArray{T}, x, index::Int)
    # FIXME: disabled due to PTX assembler issue (see #4)
    # @boundscheck checkbounds(A, index)
    Base.pointerset(Base.unsafe_convert(Ptr{T}, A), convert(T, x)::T, index, 8)
end

Base.IndexStyle{A<:CuDeviceArray}(::Type{A}) = Base.IndexLinear()

Base.show{T}(io::IO, a::CuDeviceArray{T,1}) =
    print(io, "$(length(a))-element device array at $(pointer(a))")
Base.show{T,N}(io::IO, a::CuDeviceArray{T,N}) =
    print(io, "$(join(a.shape, '×')) device array at $(pointer(a))")


## compatibility fixes

# TODO: remove this hack as soon as immutables with heap references (such as BoundsError)
#       can be stack-allocated
struct CuBoundsError <: Exception end
@inline Base.throw_boundserror{T,N}(A::CuDeviceArray{T,N}, I) =
    (Base.@_noinline_meta; throw(CuBoundsError()))

# TODO: same for SubArray, although it might be too complex to ever be non-allocating
function Base.unsafe_view{T}(A::CuDeviceArray{T,1}, I::Vararg{Base.ViewIndex,1})
    Base.@_inline_meta
    ptr = Base.unsafe_convert(Ptr{T}, A) + (I[1].start-1)*sizeof(T)
    len = I[1].stop - I[1].start + 1
    return CuDeviceArray(len, ptr)
end
