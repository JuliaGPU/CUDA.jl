# Contiguous on-device arrays

export CuDeviceArray, CuDeviceVector, CuDeviceMatrix, ldg


## construction

"""
    CuDeviceArray(dims, ptr)
    CuDeviceArray{T}(dims, ptr)
    CuDeviceArray{T,N}(dims, ptr)
    CuDeviceArray{T,N,A}(dims, ptr)

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
    shape::Dims{N}
    ptr::LLVMPtr{T,A}

    # inner constructors, fully parameterized, exact types (ie. Int not <:Integer)
    CuDeviceArray{T,N,A}(shape::Dims{N}, ptr::LLVMPtr{T,A}) where {T,A,N} = new(shape,ptr)
end

const CuDeviceVector = CuDeviceArray{T,1,A} where {T,A}
const CuDeviceMatrix = CuDeviceArray{T,2,A} where {T,A}

# outer constructors, non-parameterized
CuDeviceArray(dims::NTuple{N,<:Integer}, p::LLVMPtr{T,A})                where {T,A,N} = CuDeviceArray{T,N,A}(dims, p)
CuDeviceArray(len::Integer,              p::LLVMPtr{T,A})                where {T,A}   = CuDeviceVector{T,A}((len,), p)

# outer constructors, partially parameterized
CuDeviceArray{T}(dims::NTuple{N,<:Integer},   p::LLVMPtr{T,A}) where {T,A,N} = CuDeviceArray{T,N,A}(dims, p)
CuDeviceArray{T}(len::Integer,                p::LLVMPtr{T,A}) where {T,A}   = CuDeviceVector{T,A}((len,), p)
CuDeviceArray{T,N}(dims::NTuple{N,<:Integer}, p::LLVMPtr{T,A}) where {T,A,N} = CuDeviceArray{T,N,A}(dims, p)
CuDeviceVector{T}(len::Integer,               p::LLVMPtr{T,A}) where {T,A}   = CuDeviceVector{T,A}((len,), p)

# outer constructors, fully parameterized
CuDeviceArray{T,N,A}(dims::NTuple{N,<:Integer}, p::LLVMPtr{T,A}) where {T,A,N} = CuDeviceArray{T,N,A}(Int.(dims), p)
CuDeviceVector{T,A}(len::Integer,               p::LLVMPtr{T,A}) where {T,A}   = CuDeviceVector{T,A}((Int(len),), p)


## getters

Base.pointer(a::CuDeviceArray) = a.ptr
Base.pointer(a::CuDeviceArray, i::Integer) =
    pointer(a) + (i - 1) * Base.elsize(a)

Base.elsize(::Type{<:CuDeviceArray{T}}) where {T} = sizeof(T)
Base.size(g::CuDeviceArray) = g.shape
Base.length(g::CuDeviceArray) = prod(g.shape)


## conversions

Base.unsafe_convert(::Type{LLVMPtr{T,A}}, a::CuDeviceArray{T,N,A}) where {T,A,N} = pointer(a)


## indexing intrinsics

# NOTE: these intrinsics are now implemented using plain and simple pointer operations;
#       when adding support for isbits union arrays we will need to implement that here.

# TODO: arrays as allocated by the CUDA APIs are 256-byte aligned. we should keep track of
#       this information, because it enables optimizations like Load Store Vectorization
#       (cfr. shared memory and its wider-than-datatype alignment)

@inline function arrayref(A::CuDeviceArray{T}, index::Int) where {T}
    @boundscheck checkbounds(A, index)
    align = Base.datatype_alignment(T)
    unsafe_load(pointer(A), index, Val(align))
end

@inline function arrayset(A::CuDeviceArray{T}, x::T, index::Int) where {T}
    @boundscheck checkbounds(A, index)
    align = Base.datatype_alignment(T)
    unsafe_store!(pointer(A), x, index, Val(align))
    return A
end

@inline function const_arrayref(A::CuDeviceArray{T}, index::Int) where {T}
    @boundscheck checkbounds(A, index)
    align = Base.datatype_alignment(T)
    unsafe_cached_load(pointer(A), index, Val(align))
end


## indexing

Base.@propagate_inbounds Base.getindex(A::CuDeviceArray{T}, i1::Int) where {T} =
    arrayref(A, i1)
Base.@propagate_inbounds Base.setindex!(A::CuDeviceArray{T}, x, i1::Int) where {T} =
    arrayset(A, convert(T,x)::T, i1)

Base.IndexStyle(::Type{<:CuDeviceArray}) = Base.IndexLinear()


## const indexing

"""
    Const(A::CuDeviceArray)

Mark a CuDeviceArray as constant/read-only. The invariant guaranteed is that you will not
modify an CuDeviceArray for the duration of the current kernel.

This API can only be used on devices with compute capability 3.5 or higher.

!!! warning
    Experimental API. Subject to change without deprecation.
"""
struct Const{T,N,AS} <: DenseArray{T,N}
    a::CuDeviceArray{T,N,AS}
end
Base.Experimental.Const(A::CuDeviceArray) = Const(A)

Base.IndexStyle(::Type{<:Const}) = IndexLinear()
Base.size(C::Const) = size(C.a)
Base.axes(C::Const) = axes(C.a)
Base.@propagate_inbounds Base.getindex(A::Const, i1::Int) = const_arrayref(A.a, i1)

# deprecated
Base.@propagate_inbounds ldg(A::CuDeviceArray, i1::Integer) = const_arrayref(A, Int(i1))


## other

Base.show(io::IO, a::CuDeviceVector) =
    print(io, "$(length(a))-element device array at $(pointer(a))")
Base.show(io::IO, a::CuDeviceArray) =
    print(io, "$(join(a.shape, '×')) device array at $(pointer(a))")

Base.show(io::IO, mime::MIME"text/plain", a::CuDeviceArray) = show(io, a)

@inline function Base.unsafe_view(A::CuDeviceVector{T}, I::Vararg{Base.ViewIndex,1}) where {T}
    ptr = pointer(A, I[1].start)
    len = I[1].stop - I[1].start + 1
    return CuDeviceArray(len, ptr)
end

@inline function Base.iterate(A::CuDeviceArray, i=1)
    if (i % UInt) - 1 < length(A)
        (@inbounds A[i], i + 1)
    else
        nothing
    end
end
