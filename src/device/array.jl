# Contiguous on-device arrays

export
    CuDeviceArray, CuDeviceVector, CuDeviceMatrix, CuBoundsError


## construction

"""
    CuDeviceArray(dims, ptr)
    CuDeviceArray{T}(dims, ptr)
    CuDeviceArray{T,A}(dims, ptr)
    CuDeviceArray{T,A,N}(dims, ptr)

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

struct CuDeviceArray{T,N,A} <: AbstractArray{T,N}
    shape::NTuple{N,Int}
    ptr::DevicePtr{T,A}

    # inner constructors, fully parameterized, exact types (ie. Int not <:Integer)
    CuDeviceArray{T,N,A}(shape::NTuple{N,Int}, ptr::DevicePtr{T,A}) where {T,A,N} = new(shape,ptr)
end

const CuDeviceVector = CuDeviceArray{T,1,A} where {T,A}
const CuDeviceMatrix = CuDeviceArray{T,2,A} where {T,A}

# outer constructors, non-parameterized
CuDeviceArray(dims::NTuple{N,<:Integer}, p::DevicePtr{T,A})                where {T,A,N} = CuDeviceArray{T,N,A}(dims, p)
CuDeviceArray(len::Integer,              p::DevicePtr{T,A})                where {T,A}   = CuDeviceVector{T,A}((len,), p)

# outer constructors, partially parameterized
(::Type{CuDeviceArray{T}})(dims::NTuple{N,<:Integer},   p::DevicePtr{T,A}) where {T,A,N} = CuDeviceArray{T,N,A}(dims, p)
(::Type{CuDeviceArray{T}})(len::Integer,                p::DevicePtr{T,A}) where {T,A}   = CuDeviceVector{T,A}((len,), p)
(::Type{CuDeviceArray{T,N}})(dims::NTuple{N,<:Integer}, p::DevicePtr{T,A}) where {T,A,N} = CuDeviceArray{T,N,A}(dims, p)
(::Type{CuDeviceVector{T}})(len::Integer,               p::DevicePtr{T,A}) where {T,A}   = CuDeviceVector{T,A}((len,), p)

# outer constructors, fully parameterized
(::Type{CuDeviceArray{T,N,A}})(dims::NTuple{N,<:Integer}, p::DevicePtr{T,A}) where {T,A,N} = CuDeviceArray{T,N,A}(Int.(dims), p)
(::Type{CuDeviceVector{T,A}})(len::Integer,               p::DevicePtr{T,A}) where {T,A}   = CuDeviceVector{T,A}((Int(len),), p)


## getters

Base.pointer(a::CuDeviceArray) = a.ptr

Base.size(g::CuDeviceArray) = g.shape
Base.length(g::CuDeviceArray) = prod(g.shape)


## conversions

Base.unsafe_convert(::Type{DevicePtr{T,A}}, a::CuDeviceArray{T,N,A}) where {T,A,N} = pointer(a)

# from CuArray
function Base.convert(::Type{CuDeviceArray{T,N,AS.Global}}, a::CuArray{T,N}) where {T,N}
    owned_ptr = pointer(a)
    ptr = Base.unsafe_convert(Ptr{T}, owned_ptr)
    CuDeviceArray{T,N,AS.Global}(a.shape, DevicePtr{T,AS.Global}(ptr))
end
cudaconvert(::Type{CuArray{T,N}}) where {T,N} = CuDeviceArray{T,N,AS.Global}


## indexing

@inline function Base.getindex(A::CuDeviceArray{T}, index::Integer) where {T}
    @boundscheck checkbounds(A, index)
    align = datatype_align(T)
    Base.unsafe_load(pointer(A), index, Val{align})::T
end

@inline function Base.setindex!(A::CuDeviceArray{T}, x, index::Integer) where {T}
    @boundscheck checkbounds(A, index)
    align = datatype_align(T)
    Base.unsafe_store!(pointer(A), x, index, Val{align})
end

Base.IndexStyle(::Type{<:CuDeviceArray}) = Base.IndexLinear()

# bounds checking is currently broken due to a PTX assembler issue (see #4)
Base.checkbounds(::CuDeviceArray, I...) = nothing

# replace boundserror-with-arguments to a non-allocating, argumentless version
# TODO: can this be fixed by stack-allocating immutables with heap references?
struct CuBoundsError <: Exception end
@inline Base.throw_boundserror(A::CuDeviceArray, I) =
    (Base.@_noinline_meta; throw(CuBoundsError()))


## other

Base.show(io::IO, a::CuDeviceVector) =
    print(io, "$(length(a))-element device array at $(pointer(a))")
Base.show(io::IO, a::CuDeviceArray) =
    print(io, "$(join(a.shape, '×')) device array at $(pointer(a))")

Base.show(io::IO, mime::MIME"text/plain", a::CuDeviceArray) = show(io, a)

function Base.unsafe_view(A::CuDeviceVector{T}, I::Vararg{Base.ViewIndex,1}) where {T}
    Base.@_inline_meta
    ptr = pointer(A) + (I[1].start-1)*sizeof(T)
    len = I[1].stop - I[1].start + 1
    return CuDeviceArray(len, ptr)
end
