# Contiguous on-device arrays

export CuDeviceArray, CuDeviceVector, CuDeviceMatrix, ldg


## construction

"""
    CuDeviceArray{T,N,A}(ptr, dims, [maxsize])

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
    CuDeviceArray{T,N,A}(ptr::LLVMPtr{T,A}, dims::Tuple,
                         maxsize::Int=prod(dims)*aligned_sizeof(T)) where {T,A,N} =
        new(ptr, maxsize, dims, prod(dims))
end

const CuDeviceVector = CuDeviceArray{T,1,A} where {T,A}
const CuDeviceMatrix = CuDeviceArray{T,2,A} where {T,A}


## array interface

Base.elsize(::Type{<:CuDeviceArray{T}}) where {T} = aligned_sizeof(T)

Base.size(g::CuDeviceArray) = g.dims
Base.sizeof(x::CuDeviceArray) = Base.elsize(x) * length(x)

# we store the array length too; computing prod(size) is expensive
Base.size(g::CuDeviceArray{<:Any,1}) = (g.len,)
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
    # simplified bounds check to avoid the OneTo construction, which calls `max`
    # and breaks elimination of redundant bounds checks in the generated code.
    #@boundscheck checkbounds(A, index)
    @boundscheck index <= length(A) || Base.throw_boundserror(A, index)

    if Base.isbitsunion(T)
        arrayref_union(A, index)
    else
        arrayref_bits(A, index)
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
    # simplified bounds check (see `arrayref`)
    #@boundscheck checkbounds(A, index)
    @boundscheck index <= length(A) || Base.throw_boundserror(A, index)

    if Base.isbitsunion(T)
        arrayset_union(A, x, index)
    else
        arrayset_bits(A, x, index)
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
    # simplified bounds check (see `arrayset`)
    #@boundscheck checkbounds(A, index)
    @boundscheck index <= length(A) || Base.throw_boundserror(A, index)

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
  err = GPUArrays._reinterpret_exception(T, a)
  err === nothing || throw(err)

  if aligned_sizeof(T) == aligned_sizeof(S) # fast case
    return CuDeviceArray{T,N,A}(reinterpret(LLVMPtr{T,A}, a.ptr), size(a), a.maxsize)
  end

  isize = size(a)
  size1 = div(isize[1]*aligned_sizeof(S), aligned_sizeof(T))
  osize = tuple(size1, Base.tail(isize)...)
  return CuDeviceArray{T,N,A}(reinterpret(LLVMPtr{T,A}, a.ptr), osize, a.maxsize)
end


## reshape

function Base.reshape(a::CuDeviceArray{T,M,A}, dims::NTuple{N,Int}) where {T,N,M,A}
  if prod(dims) != length(a)
      throw(DimensionMismatch("new dimensions (argument `dims`) must be consistent with array size (`size(a)`)"))
  end
  if N == M && dims == size(a)
      return a
  end
  _derived_array(a, T, dims)
end

# create a derived device array (reinterpreted or reshaped) that's still a CuDeviceArray
@inline function _derived_array(a::CuDeviceArray{<:Any,<:Any,A}, ::Type{T},
                                osize::Dims{N}) where {T, N, A}
  return CuDeviceArray{T,N,A}(a.ptr, osize, a.maxsize)
end
