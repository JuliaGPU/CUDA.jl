# Contiguous on-device arrays

export
    CuDeviceArray, CuDeviceVector, CuDeviceMatrix, CuBoundsError


## construction

"""
    CuDeviceArray(dims, ptr)
    CuDeviceArray{T}(dims, ptr)
    CuDeviceArray{T,N}(dims, ptr)

Construct an `N`-dimensional dense CUDA device array with element type `T` wrapping a
pointer, where `N` is determined from the length of `dims` and `T` is determined from the
type of `ptr`. `dims` may be a single scalar, or a tuple of integers corresponding to the
lengths in each dimension). If the rank `N` is supplied explicitly as in `Array{T,N}(dims)`,
then it must match the length of `dims`. The same applies to the element type `T`, which
should match the type of the pointer `ptr`.
"""
CuDeviceArray

# NOTE: we can't support the typical `tuple or series of integer` style construction,
#       because we're currently requiring a trailing pointer argument.

struct CuDeviceArray{T,N} <: AbstractArray{T,N}
    shape::NTuple{N,Int}
    ptr::Ptr{T}

    # inner constructors (exact types, ie. Int not <:Integer)
    CuDeviceArray{T,N}(shape::NTuple{N,Int}, ptr::Ptr{T}) where {T,N} = new(shape, ptr)
end

const CuDeviceVector = CuDeviceArray{T,1} where {T}
const CuDeviceMatrix = CuDeviceArray{T,2} where {T}

# outer constructors, non-parameterized
CuDeviceArray(dims::NTuple{N,<:Integer}, p::Ptr{T})                where {T,N} = CuDeviceArray{T,N}(dims, p)
CuDeviceArray(len::Integer, p::Ptr{T})                             where {T}   = CuDeviceVector{T}((len,), p)

# outer constructors, partially parameterized
(::Type{CuDeviceArray{T}})(dims::NTuple{N,<:Integer}, p::Ptr{T})   where {T,N} = CuDeviceArray{T,N}(dims, p)
(::Type{CuDeviceArray{T}})(len::Integer, p::Ptr{T})                where {T}   = CuDeviceVector{T}((len,), p)

# outer constructors, fully parameterized
(::Type{CuDeviceArray{T,N}})(dims::NTuple{N,<:Integer}, p::Ptr{T}) where {T,N} = CuDeviceArray{T,N}(Int.(dims), p)
(::Type{CuDeviceVector{T}})(len::Integer, p::Ptr{T})               where {T}   = CuDeviceVector{T}((Int(len),), p)

Base.convert(::Type{CuDeviceArray{T,N}}, a::CuArray{T,N}) where {T,N} =
    CuDeviceArray{T,N}(a.shape, Base.unsafe_convert(Ptr{T}, a.devptr))

Base.unsafe_convert(::Type{Ptr{T}}, a::CuDeviceArray{T}) where {T} = a.ptr::Ptr{T}


## array interface

Base.size(g::CuDeviceArray) = g.shape
Base.length(g::CuDeviceArray) = prod(g.shape)

@inline function Base.getindex(A::CuDeviceArray{T}, index::Int) where {T}
    @boundscheck checkbounds(A, index)
    align = datatype_align(T)
    Base.pointerref(Base.unsafe_convert(Ptr{T}, A), index, align)::T
end

@inline function Base.setindex!(A::CuDeviceArray{T}, x, index::Int) where {T}
    @boundscheck checkbounds(A, index)
    align = datatype_align(T)
    Base.pointerset(Base.unsafe_convert(Ptr{T}, A), convert(T, x)::T, index, align)
end

Base.IndexStyle(::Type{<:CuDeviceArray}) = Base.IndexLinear()

Base.show(io::IO, a::CuDeviceVector{T}) where {T} =
    print(io, "$(length(a))-element device array at $(pointer(a))")
Base.show(io::IO, a::CuDeviceArray{T,N}) where {T,N} =
    print(io, "$(join(a.shape, '×')) device array at $(pointer(a))")


## quirks

# bounds checking is currently broken due to a PTX assembler issue (see #4)
Base.checkbounds(::CuDeviceArray, I...) = nothing

# replace boundserror-with-arguments to a non-allocating, argumentless version
# TODO: can this be fixed by stack-allocating immutables with heap references?
struct CuBoundsError <: Exception end
@inline Base.throw_boundserror(A::CuDeviceArray, I) =
    (Base.@_noinline_meta; throw(CuBoundsError()))

# idem
function Base.unsafe_view(A::CuDeviceVector{T}, I::Vararg{Base.ViewIndex,1}) where {T}
    Base.@_inline_meta
    ptr = Base.unsafe_convert(Ptr{T}, A) + (I[1].start-1)*sizeof(T)
    len = I[1].stop - I[1].start + 1
    return CuDeviceArray(len, ptr)
end
