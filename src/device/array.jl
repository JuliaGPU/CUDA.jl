# Contiguous on-device arrays

export CuDeviceArray, CuDeviceVector, CuDeviceMatrix,
       DenseCuDeviceArray, DenseCuDeviceVector, DenseCuDeviceMatrix,
       StridedCuDeviceArray, StridedCuDeviceVector, StridedCuDeviceMatrix,
       ldg


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

struct CuDeviceArray{T,N,A,S} <: AbstractArray{T,N}
    ptr::LLVMPtr{T,A}
    maxsize::Int

    dims::Dims{N}
    len::Int

    # S here is the actual stride type, as opposed to a boolean as with the CuArray type,
    # to reduce the size of the device array object when using contiguous memory.
    strides::S

    # inner constructors, fully parameterized, exact types (ie. Int not <:Integer)
    # TODO: deprecate; put `ptr` first like CuArray
    CuDeviceArray{T,N,A}(dims::Dims{N}, ptr::LLVMPtr{T,A},
                         maxsize::Int=prod(dims)*sizeof(T)) where {T,A,N} =
        new{T,N,A,Nothing}(ptr, maxsize, dims, prod(dims), nothing)
    CuDeviceArray{T,N,A}(dims::Dims{N}, strides::Dims{N}, ptr::LLVMPtr{T,A},
                         maxsize::Int=prod(dims)*sizeof(T)) where {T,A,N} =
        new{T,N,A,Dims{N}}(ptr, maxsize, dims, prod(dims), strides)
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


const DenseCuDeviceArray{T,N,A} = CuDeviceArray{T,N,A,Nothing}
const DenseCuDeviceVector{T,N,A} = CuDeviceArray{T,1,A,Nothing}
const DenseCuDeviceMatrix{T,N,A} = CuDeviceArray{T,2,A,Nothing}

# XXX: as opposed to StridedArray, strided does not include dense.
const StridedCuDeviceArray{T,N,A} = CuDeviceArray{T,N,A,Dims{N}}
const StridedCuDeviceVector{T,N,A} = CuDeviceArray{T,1,A,Dims{N}}
const StridedCuDeviceMatrix{T,N,A} = CuDeviceArray{T,2,A,Dims{N}}


## array interface

Base.elsize(::Type{<:CuDeviceArray{T}}) where {T} = sizeof(T)

Base.size(g::CuDeviceArray) = g.dims
Base.sizeof(x::CuDeviceArray) = Base.elsize(x) * length(x)

# we store the array length too; computing prod(size) is expensive
Base.length(g::CuDeviceArray) = g.len

Base.strides(x::DenseCuDeviceArray) = Base.size_to_strides(1, size(x)...)
Base.strides(x::StridedCuDeviceArray) = x.strides

function Base.stride(a::DenseCuDeviceArray, i::Int)
    if i > ndims(a)
        return length(a)
    end
    s = 1
    for n = 1:(i-1)
        s *= size(a, n)
    end
    return s
end
function Base.stride(a::StridedCuDeviceArray, i::Int)
    if i > ndims(a)
        i = ndims(a)
        return size(a,i) * @inbounds strides(a)[i]
    end
    @inbounds strides(a)[i]
end

Base.pointer(x::CuDeviceArray{T,<:Any,A}) where {T,A} = Base.unsafe_convert(LLVMPtr{T,A}, x)
typetagdata(a::CuDeviceArray{<:Any,<:Any,A}) where {A} =
  reinterpret(LLVMPtr{UInt8,A}, a.ptr + a.maxsize)

Base.pointer(x::CuDeviceArray{<:Any,<:Any,<:Any,Nothing}, i::Integer) =
    pointer(x) + (i - one(i)) * Base.elsize(x)
typetagdata(a::CuDeviceArray{<:Any,<:Any,<:Any,Nothing}, i::Integer) =
  typetagdata(a) + Base._memory_offset(x, i) ÷ Base.elsize(a)

# memory_offset on CuDeviceArray (which isn't <:DenseArray) is slow, so optimize it
Base.pointer(x::StridedCuDeviceArray, i::Integer) =
    pointer(x) + Base._memory_offset(x, i)
typetagdata(a::StridedCuDeviceArray, i::Integer) =
  typetagdata(a) + Base._memory_offset(x, i) ÷ Base.elsize(a)


## conversions

Base.unsafe_convert(::Type{LLVMPtr{T,A}}, x::CuDeviceArray{T,<:Any,A}) where {T,A} =
  x.ptr


## indexing intrinsics

# for strided arrays, these should be invoked with a set of integer indices.
# in the case of contiguous arrays, a single linear index should be passed.
# this matches what Base passes to get/setindex with resp. IndexLinear and IndexCartesian.

# from Strided.jl
@inline _computeind(indices::Tuple{}, strides::Tuple{}) = 1
@inline function _computeind(indices::NTuple{N,Integer}, strides::NTuple{N,Integer}) where {N}
    (indices[1]-1)*strides[1] + _computeind(Base.tail(indices), Base.tail(strides))
end

convert_indices(A::DenseCuDeviceArray, i::Integer) = i
convert_indices(A::StridedCuDeviceArray{<:Any,N}, I::Vararg{Int,N}) where {N} =
    _computeind(I, strides(A))

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

@device_function @inline function arrayref(A::CuDeviceArray{T}, I...) where {T}
    index = convert_indices(A, I...)
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
                unsafe_load(ptr, index, Val(align))
            else
                $ex
            end
        end
    end

    quote
        selector_ptr = typetagdata(A)
        selector = unsafe_load(selector_ptr, index)

        align = alignment(A)
        data_ptr = pointer(A)

        return $ex
    end
end

@device_function @inline function arrayset(A::CuDeviceArray{T}, x::T, I...) where {T}
    index = convert_indices(A, I...)
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
        selector_ptr = typetagdata(A)
        unsafe_store!(selector_ptr, $(UInt8(sel-1)), index)

        align = alignment(A)
        data_ptr = pointer(A)

        unsafe_store!(reinterpret(LLVMPtr{$x,AS}, data_ptr), x, index, Val(align))
        return
    end
end

@device_function @inline function const_arrayref(A::CuDeviceArray{T}, I...) where {T}
    index = convert_indices(A, I...)
    align = alignment(A)
    unsafe_cached_load(pointer(A), index, Val(align))
end


## indexing

Base.IndexStyle(::Type{<:DenseCuDeviceArray}) = Base.IndexLinear()

function Base.getindex(A::DenseCuDeviceArray{T}, i1::Integer) where {T}
    @boundscheck checkbounds(A, i1)
    arrayref(A, i1)
end
function Base.setindex!(A::DenseCuDeviceArray{T}, x, i1::Integer) where {T}
    @boundscheck checkbounds(A, i1)
    arrayset(A, convert(T,x)::T, i1)
end

Base.IndexStyle(::Type{<:StridedCuDeviceArray}) = Base.IndexCartesian()

function Base.getindex(A::StridedCuDeviceArray{T,N},I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, I...)
    arrayref(A, I...)
end
function Base.setindex!(A::StridedCuDeviceArray{T,N}, x, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, I...)
    arrayset(A, convert(T,x)::T, I...)
end

# when we need cartesian indices, optimize the conversion from a linear index
function Base.getindex(A::StridedCuDeviceArray{T}, i::Int) where {T}
    @boundscheck checkbounds(A, i)
    assume.(size(A) .> 0)   # avoids checked conversion
    arrayref(A, Base._to_subscript_indices(A, i)...)
end
function Base.setindex!(A::StridedCuDeviceArray{T}, x, i::Int) where {T}
    @boundscheck checkbounds(A, i)
    assume.(size(A) .> 0)   # avoids checked conversion
    arrayset(A, convert(T,x)::T, Base._to_subscript_indices(A, i)...)
end

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
struct Const{T,N,A,S} <: DenseArray{T,N}
    a::CuDeviceArray{T,N,A,S}
end
Base.Experimental.Const(A::CuDeviceArray) = Const(A)

Base.IndexStyle(::Type{<:Const}) = IndexLinear()
Base.size(C::Const) = size(C.a)
Base.axes(C::Const) = axes(C.a)
function Base.getindex(A::Const, i1::Integer)
    @boundscheck checkbounds(A, i1)
    const_arrayref(A.a, i1)
end

# deprecated
function ldg(A::CuDeviceArray, i1::Integer)
    @boundscheck checkbounds(A, i1)
    const_arrayref(A, i1)
end


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
