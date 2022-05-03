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

struct CuDeviceArray{T,N,A} <: DenseArray{T,N}
    ptr::LLVMPtr{T,A}
    maxsize::Int

    dims::Dims{N}
    len::Int

    # inner constructors, fully parameterized, exact types (ie. Int not <:Integer)
    # TODO: deprecate; put `ptr` first like CuArray
    CuDeviceArray{T,N,A}(dims::Dims{N}, ptr::LLVMPtr{T,A},
                         maxsize::Int=prod(dims)*sizeof(T)) where {T,A,N} =
        new(ptr, maxsize, dims, prod(dims))
end

const CuDeviceVector = CuDeviceArray{T,1,A} where {T,A}
const CuDeviceMatrix = CuDeviceArray{T,2,A} where {T,A}

# outer constructors, non-parameterized
CuDeviceArray(dims::NTuple{N,Integer}, p::LLVMPtr{T,A})        where {T,A,N} = CuDeviceArray{T,N,A}(dims, p)
CuDeviceVector(dims::NTuple{1,Integer}, p::LLVMPtr{T,A})       where {T,A}   = CuDeviceVector{T,A}(dims, p)
CuDeviceMatrix(dims::NTuple{2,Integer}, p::LLVMPtr{T,A})       where {T,A}   = CuDeviceMatrix{T,A}(dims, p)
CuDeviceArray(len::Integer,            p::LLVMPtr{T,A})        where {T,A}   = CuDeviceVector{T,A}((len,), p)
CuDeviceArray(m::Integer, n::Integer,  p::LLVMPtr{T,A})        where {T,A}   = CuDeviceMatrix{T,A}((m,n), p)
CuDeviceVector(len::Integer,           p::LLVMPtr{T,A})        where {T,A}   = CuDeviceVector{T,A}((len,), p)
CuDeviceMatrix(m::Integer, n::Integer, p::LLVMPtr{T,A})        where {T,A}   = CuDeviceMatrix{T,A}((m,n), p)

# outer constructors, partially parameterized
CuDeviceArray{T}(dims::NTuple{N,Integer},     p::LLVMPtr{T,A}) where {T,A,N} = CuDeviceArray{T,N,A}(dims, p)
CuDeviceVector{T}(dims::NTuple{1,Integer},    p::LLVMPtr{T,A}) where {T,A}   = CuDeviceVector{T,A}(dims, p)
CuDeviceMatrix{T}(dims::NTuple{2,Integer},    p::LLVMPtr{T,A}) where {T,A}   = CuDeviceMatrix{T,A}(dims, p)
CuDeviceArray{T}(len::Integer,                p::LLVMPtr{T,A}) where {T,A}   = CuDeviceVector{T,A}((len,), p)
CuDeviceArray{T}(m::Integer, n::Integer,      p::LLVMPtr{T,A}) where {T,A}   = CuDeviceMatrix{T,A}((m,n), p)
CuDeviceArray{T,N}(dims::NTuple{N,Integer},   p::LLVMPtr{T,A}) where {T,A,N} = CuDeviceArray{T,N,A}(dims, p)
CuDeviceVector{T}(len::Integer,               p::LLVMPtr{T,A}) where {T,A}   = CuDeviceVector{T,A}((len,), p)
CuDeviceMatrix{T}(m::Integer, n::Integer,     p::LLVMPtr{T,A}) where {T,A}   = CuDeviceMatrix{T,A}((m,n), p)

# outer constructors, fully parameterized
CuDeviceArray{T,N,A}(dims::NTuple{N,Integer}, p::LLVMPtr{T,A}) where {T,A,N} = CuDeviceArray{T,N,A}(convert(Tuple{Vararg{Int}}, dims), p)
CuDeviceVector{T,A}(len::Integer,             p::LLVMPtr{T,A}) where {T,A}   = CuDeviceVector{T,A}((len,), p)
CuDeviceMatrix{T,A}(m::Integer, n::Integer,   p::LLVMPtr{T,A}) where {T,A}   = CuDeviceMatrix{T,A}((m,n), p)


## array interface

Base.elsize(::Type{<:CuDeviceArray{T}}) where {T} = sizeof(T)

Base.size(g::CuDeviceArray) = g.dims
Base.sizeof(x::CuDeviceArray) = Base.elsize(x) * length(x)

# we store the array length too; computing prod(size) is expensive
Base.length(g::CuDeviceArray) = g.len

Base.pointer(x::CuDeviceArray{T,<:Any,A}) where {T,A} = Base.unsafe_convert(LLVMPtr{T,A}, x)
@inline function Base.pointer(x::CuDeviceArray{T,<:Any,A}, i::Integer) where {T,A}
    Base.unsafe_convert(LLVMPtr{T,A}, x) + Base._memory_offset(x, i)
end

typetagdata(a::CuDeviceArray{<:Any,<:Any,A}, i=1) where {A} =
  reinterpret(LLVMPtr{UInt8,A}, a.ptr + a.maxsize) + i - one(i)


## conversions

Base.unsafe_convert(::Type{LLVMPtr{T,A}}, x::CuDeviceArray{T,<:Any,A}) where {T,A} =
  x.ptr


## indexing intrinsics

# TODO: arrays as allocated by the CUDA APIs are 256-byte aligned. we should keep track of
#       this information, because it enables optimizations like Load Store Vectorization
#       (cfr. shared memory and its wider-than-datatype alignment)

@generated function alignment(::CuDeviceArray{T}) where {T}
    if Base.isbitsunion(T)
        _, sz, al = Base.uniontype_layout(T)
        al
    else
        Base.datatype_alignment(T)
    end
end

@device_function @inline function arrayref(A::CuDeviceArray{T}, index::Integer) where {T}
    @boundscheck checkbounds(A, index)
    if isbitstype(T)
        arrayref_bits(A, index)
    else #if isbitsunion(T)
        arrayref_union(A, index)
    end
end

@inline function arrayref_bits(A::CuDeviceArray{T}, index::Integer) where {T}
    align = alignment(A)
    unsafe_load(pointer(A), index, Val(align))
end

@inline @generated function arrayref_union(A::CuDeviceArray{T,<:Any,AS}, index::Integer) where {T,AS}
    typs = Base.uniontypes(T)

    # generate code that conditionally loads a value based on the selector value.
    # lacking noreturn, we return T to avoid inference thinking this can return Nothing.
    ex = :(Base.llvmcall("unreachable", $T, Tuple{}))
    for (sel, typ) in Iterators.reverse(enumerate(typs))
        ex = quote
            if selector == $(sel-1)
                ptr = reinterpret(LLVMPtr{$typ,AS}, data_ptr)
                unsafe_load(ptr, 1, Val(align))
            else
                $ex
            end
        end
    end

    quote
        selector_ptr = typetagdata(A, index)
        selector = unsafe_load(selector_ptr)

        align = alignment(A)
        data_ptr = pointer(A, index)

        return $ex
    end
end

@device_function @inline function arrayset(A::CuDeviceArray{T}, x::T, index::Integer) where {T}
    @boundscheck checkbounds(A, index)
    if isbitstype(T)
        arrayset_bits(A, x, index)
    else #if isbitsunion(T)
        arrayset_union(A, x, index)
    end
    return A
end

@inline function arrayset_bits(A::CuDeviceArray{T}, x::T, index::Integer) where {T}
    align = alignment(A)
    unsafe_store!(pointer(A), x, index, Val(align))
end

@inline @generated function arrayset_union(A::CuDeviceArray{T,<:Any,AS}, x::T, index::Integer) where {T,AS}
    typs = Base.uniontypes(T)
    sel = findfirst(isequal(x), typs)

    quote
        selector_ptr = typetagdata(A, index)
        unsafe_store!(selector_ptr, $(UInt8(sel-1)))

        align = alignment(A)
        data_ptr = pointer(A, index)

        unsafe_store!(reinterpret(LLVMPtr{$x,AS}, data_ptr), x, 1, Val(align))
        return
    end
end

@device_function @inline function const_arrayref(A::CuDeviceArray{T}, index::Integer) where {T}
    @boundscheck checkbounds(A, index)
    align = alignment(A)
    unsafe_cached_load(pointer(A), index, Val(align))
end


## indexing

Base.IndexStyle(::Type{<:CuDeviceArray}) = Base.IndexLinear()

Base.@propagate_inbounds Base.getindex(A::CuDeviceArray{T}, i1::Integer) where {T} =
    arrayref(A, i1)
Base.@propagate_inbounds Base.setindex!(A::CuDeviceArray{T}, x, i1::Integer) where {T} =
    arrayset(A, convert(T,x)::T, i1)

# preserve the specific integer type when indexing device arrays,
# to avoid extending 32-bit hardware indices to 64-bit.
Base.to_index(::CuDeviceArray, i::Integer) = i

# Base doesn't like Integer indices, so we need our own ND get and setindex! routines.
# See also: https://github.com/JuliaLang/julia/pull/42289
Base.@propagate_inbounds Base.getindex(A::CuDeviceArray,
                                       I::Union{Integer, CartesianIndex}...) =
    A[Base._to_linear_index(A, to_indices(A, I)...)]
Base.@propagate_inbounds Base.setindex!(A::CuDeviceArray, x,
                                        I::Union{Integer, CartesianIndex}...) =
    A[Base._to_linear_index(A, to_indices(A, I)...)] = x


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
Base.@propagate_inbounds Base.getindex(A::Const, i1::Integer) = const_arrayref(A.a, i1)

# deprecated
Base.@propagate_inbounds ldg(A::CuDeviceArray, i1::Integer) = const_arrayref(A, i1)


## other

Base.show(io::IO, a::CuDeviceVector) =
    @printf(io, "%g-element device array at %p", length(a), Int(pointer(a)))
Base.show(io::IO, a::CuDeviceArray) =
    @printf(io, "%s device array at %p", join(a.dims, '×'), Int(pointer(a)))

Base.show(io::IO, mime::MIME"text/plain", a::CuDeviceArray) = show(io, a)

@inline function Base.iterate(A::CuDeviceArray, i=1)
    if (i % UInt) - 1 < length(A)
        (@inbounds A[i], i + 1)
    else
        nothing
    end
end

function Base.reinterpret(::Type{T}, a::CuDeviceArray{S,N,A}) where {T,S,N,A}
  err = _reinterpret_exception(T, a)
  err === nothing || throw(err)

  if sizeof(T) == sizeof(S) # fast case
    return CuDeviceArray{T,N,A}(size(a), reinterpret(LLVMPtr{T,A}, a.ptr), a.maxsize)
  end

  isize = size(a)
  size1 = div(isize[1]*sizeof(S), sizeof(T))
  osize = tuple(size1, Base.tail(isize)...)
  return CuDeviceArray{T,N,A}(osize, reinterpret(LLVMPtr{T,A}, a.ptr), a.maxsize)
end
