# Contiguous on-device arrays

export CuDeviceArray, CuDeviceVector, CuDeviceMatrix, ldg


## construction

"""
    CuDeviceArray{T,N,A,I}(ptr, dims, [maxsize])

Construct an `N`-dimensional dense CUDA device array with element type `T` wrapping a
pointer, where `N` is determined from the length of `dims` and `T` is determined from the
type of `ptr`. `dims` may be a single scalar, or a tuple of integers corresponding to the
lengths in each dimension). If the rank `N` is supplied explicitly as in `Array{T,N}(dims)`,
then it must match the length of `dims`. The same applies to the element type `T`, which
should match the type of the pointer `ptr`.

The index type `I` (`Int32` or `Int64`) is used to store the dimensions of the array, and
to perform index computations. Every dimension, as well as the length of the array, needs
to be representable by `I`. If `I` is omitted, it defaults to `Int`. Host arrays converted
for use in a kernel use `Int32` when possible.

Regardless of the index type, the dimensions are exposed as `Int` (i.e., `size` and
`length` return `Int`s).
"""
CuDeviceArray

# NOTE: we can't support the typical `tuple or series of integer` style construction,
#       because we're currently requiring a trailing pointer argument.

# marker for constructing a device array without checking that its dimensions fit
struct Unchecked end

# GPU-compatible version in quirks.jl
@noinline throw_index_type_error(::Type{I}, dims) where {I} =
    throw(ArgumentError("dimensions $dims do not fit index type $I"))

struct CuDeviceArray{T,N,A,I<:Union{Int32,Int64}} <: DenseArray{T,N}
    ptr::LLVMPtr{T,A}
    maxsize::Int

    dims::NTuple{N,I}
    len::I

    # inner constructors, fully parameterized
    @inline function CuDeviceArray{T,N,A,I}(ptr::LLVMPtr{T,A}, dims::Tuple,
                                            maxsize::Int=prod(dims)*aligned_sizeof(T)) where {T,A,N,I}
        fits_index_type(I, dims) || throw_index_type_error(I, dims)
        new(ptr, maxsize, map(d -> d % I, dims), prod(dims) % I)
    end

    # for when the dimensions are known to fit the index type
    @inline CuDeviceArray{T,N,A,I}(::Unchecked, ptr::LLVMPtr{T,A}, dims::Tuple,
                                   maxsize::Int=prod(dims)*aligned_sizeof(T)) where {T,A,N,I} =
        new(ptr, maxsize, map(d -> d % I, dims), prod(dims) % I)
end

@inline CuDeviceArray{T,N,A}(ptr::LLVMPtr{T,A}, dims::Tuple,
                             maxsize::Int=prod(dims)*aligned_sizeof(T)) where {T,A,N} =
    CuDeviceArray{T,N,A,Int}(ptr, dims, maxsize)

const CuDeviceVector = CuDeviceArray{T,1,A,I} where {T,A,I<:Union{Int32,Int64}}
const CuDeviceMatrix = CuDeviceArray{T,2,A,I} where {T,A,I<:Union{Int32,Int64}}

# the index type for a device array with dimensions `dims`: `Int32` if every dimension and
# the length can be represented using 32-bit integers, `Int64` otherwise.
index_type(dims::Tuple) = fits_index_type(Int32, dims) ? Int32 : Int64

@inline function fits_index_type(::Type{I}, dims::Tuple) where {I}
    all(d -> 0 <= d <= typemax(I), dims) || return false
    # the length of an empty array fits, regardless of the other dimensions
    any(iszero, dims) && return true
    return fits_length(I, 1, dims...)
end
# recursive, so that the check folds away for constant dimensions
@inline fits_length(::Type, len::Int) = true
@inline function fits_length(::Type{I}, len::Int, d, ds...) where {I}
    len, overflow = Base.mul_with_overflow(len, d % Int)
    (overflow || len > typemax(I)) && return false
    return fits_length(I, len, ds...)
end


## array interface

Base.elsize(::Type{<:CuDeviceArray{T}}) where {T} = aligned_sizeof(T)

Base.size(g::CuDeviceArray) = map(widen_index, g.dims)
Base.sizeof(x::CuDeviceArray) = Base.elsize(x) * length(x)

# we store the array length too; computing prod(size) is expensive
Base.size(g::CuDeviceArray{<:Any,1}) = (widen_index(g.len),)
Base.length(g::CuDeviceArray) = widen_index(g.len)

Base.pointer(x::CuDeviceArray{T,<:Any,A}) where {T,A} = Base.unsafe_convert(LLVMPtr{T,A}, x)
@inline function Base.pointer(x::CuDeviceArray{T,<:Any,A}, i::Integer) where {T,A}
    Base.unsafe_convert(LLVMPtr{T,A}, x) + Base._memory_offset(x, i)
end

typetagdata(a::CuDeviceArray{<:Any,<:Any,A}, i=1) where {A} =
  reinterpret(LLVMPtr{UInt8,A}, a.ptr + a.maxsize) + i - one(i)


## conversions

Base.unsafe_convert(::Type{LLVMPtr{T,A}}, x::CuDeviceArray{T,<:Any,A}) where {T,A} =
  x.ptr


## index arithmetic

# Index computations are performed using the array's index type. For in-bounds indices, the
# result and every intermediate value fit that type, so the narrow computation is exact.
# These helpers should therefore only be used after bounds checking (or in an `@inbounds`
# context, where out-of-bounds accesses are undefined behavior already).
#
# NOTE: these operations could be annotated for LLVM (`zext nneg`, `trunc nuw nsw`, `nuw nsw`
#       arithmetic) using `llvmcall`, but the inliner considers `llvmcall` expensive, which
#       prevented inlining of code that indexes arrays. The annotations also did not result
#       in measurably better code.

# sizes are exposed as `Int`, like for any other array. dimensions are non-negative.
@inline widen_index(i::Int64) = i
@inline widen_index(i::Int32) = Core.Intrinsics.zext_int(Int64, i)

# convert an in-bounds index to the index type
@inline trunc_index(::Type{I}, i::I) where {I} = i
@inline trunc_index(::Type{Int64}, i::Int32) = widen_index(i)
@inline trunc_index(::Type{I}, i::Integer) where {I} = i % I

# the index used for pointer arithmetic, which uses `Int`. `Int64` indices are passed
# through as-is: truncating them only to widen them again would force LLVM to mask the
# value, which prevents loop strength reduction.
@inline element_index(::Type, i::Int64) = i
@inline element_index(::Type, i::Integer) = i
# `Int32` offsets are in bounds, so non-negative. telling LLVM lets loop strength reduction
# turn the address computation of N-d indexing in loops into a pointer increment.
@inline function element_index(::Type, i::Int32)
    assume(i >= Int32(0))
    widen_index(i)
end

# convert in-bounds N-d indices to a linear one
@inline linearize(::Tuple{}, ::Tuple{}) = 1
@inline linearize(::Tuple{T}, I::Tuple{T}) where {T} = I[1]
@inline function linearize(dims::Tuple{T,T,Vararg{T}}, I::Tuple{T,T,Vararg{T}}) where {T}
    rest = linearize(Base.tail(dims), Base.tail(I))
    I[1] + dims[1] * (rest - one(T))
end
@inline function linear_index(A::CuDeviceArray{<:Any,N,<:Any,I}, J::Tuple) where {N,I}
    if length(J) == N
        linearize(A.dims, map(j -> trunc_index(I, j), J))
    else
        # fewer or more indices than dimensions
        Base._to_linear_index(A, J...)
    end
end


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

# check that `i` indexes into `1:n`. the explicit check avoids the OneTo construction of
# `checkbounds`, which calls `max` and breaks elimination of redundant bounds checks in the
# generated code. the two bounds are checked with separate signed comparisons: folding them
# into a single unsigned one (`(i-1) % UInt < n % UInt`) keeps LLVM from recognizing the
# check as redundant after the kernel's own `i <= length(A)`.
@inline in_bounds(i::Integer, n::Int) = (one(i) <= i) & (i <= n)

# unchecked element accessors; the bounds are checked by the callers (see `getindex`)
@device_function @inline function arrayref(A::CuDeviceArray{T}, index::Integer) where {T}
    if Base.isbitsunion(T)
        arrayref_union(A, index)
    else
        arrayref_bits(A, index)
    end
end

@inline function arrayref_bits(A::CuDeviceArray{T,<:Any,<:Any,I}, index::Integer) where {T,I}
    align = alignment(A)
    unsafe_load(pointer(A), element_index(I, index), Val(align))
end

# `load` is the function used to read the selector and the value (`unsafe_load` or
# `unsafe_cached_load`)
@inline @generated function arrayref_union(A::CuDeviceArray{T,<:Any,AS}, index::Integer,
                                           load=unsafe_load) where {T,AS}
    typs = Base.uniontypes(T)

    # generate code that conditionally loads a value based on the selector value.
    # lacking noreturn, we return T to avoid inference thinking this can return Nothing.
    ex = :(Base.llvmcall("unreachable", $T, Tuple{}))
    for (sel, typ) in Iterators.reverse(enumerate(typs))
        ex = quote
            if selector == $(sel-1)
                ptr = reinterpret(LLVMPtr{$typ,AS}, data_ptr)
                load(ptr, 1, Val(align))
            else
                $ex
            end
        end
    end

    quote
        selector_ptr = typetagdata(A, index)
        selector = load(selector_ptr)

        align = alignment(A)
        data_ptr = pointer(A, index)

        return $ex
    end
end

@device_function @inline function arrayset(A::CuDeviceArray{T}, x::T, index::Integer) where {T}
    if Base.isbitsunion(T)
        arrayset_union(A, x, index)
    else
        arrayset_bits(A, x, index)
    end
    return A
end

@inline function arrayset_bits(A::CuDeviceArray{T,<:Any,<:Any,I}, x::T, index::Integer) where {T,I}
    align = alignment(A)
    unsafe_store!(pointer(A), x, element_index(I, index), Val(align))
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

@device_function @inline function const_arrayref(A::CuDeviceArray{T,<:Any,<:Any,I}, index::Integer) where {T,I}
    @boundscheck in_bounds(index, length(A)) || Base.throw_boundserror(A, index)

    if Base.isbitsunion(T)
        arrayref_union(A, index, unsafe_cached_load)
    else
        align = alignment(A)
        unsafe_cached_load(pointer(A), element_index(I, index), Val(align))
    end
end


## indexing

Base.IndexStyle(::Type{<:CuDeviceArray}) = Base.IndexLinear()

Base.@propagate_inbounds function Base.getindex(A::CuDeviceArray, i1::Integer)
    @boundscheck in_bounds(i1, length(A)) || Base.throw_boundserror(A, i1)
    arrayref(A, i1)
end
Base.@propagate_inbounds function Base.setindex!(A::CuDeviceArray{T}, x, i1::Integer) where {T}
    @boundscheck in_bounds(i1, length(A)) || Base.throw_boundserror(A, i1)
    arrayset(A, convert(T,x)::T, i1)
end

# preserve the specific integer type when indexing device arrays,
# to avoid extending 32-bit hardware indices to 64-bit.
Base.to_index(::CuDeviceArray, i::Integer) = i

# Base doesn't like Integer indices, so we need our own ND get and setindex! routines.
# See also: https://github.com/JuliaLang/julia/pull/42289
#
# Like Base, every index is checked against its dimension. Checking only the linear index
# would accept out-of-bounds indices that happen to linearize into the array. It also makes
# it possible to linearize using the array's index type (see `linear_index`). The linear
# index is then accessed without a check of its own, rather than relying on `@inbounds`,
# which `--check-bounds=yes` ignores (and LLVM cannot prove the linear index in bounds).
Base.@propagate_inbounds function Base.getindex(A::CuDeviceArray,
                                                I::Union{Integer, CartesianIndex}...)
    J = to_indices(A, I)
    @boundscheck checkbounds_nd(A, J)
    arrayref(A, linear_index(A, J))
end
Base.@propagate_inbounds function Base.setindex!(A::CuDeviceArray{T}, x,
                                                 I::Union{Integer, CartesianIndex}...) where {T}
    J = to_indices(A, I)
    @boundscheck checkbounds_nd(A, J)
    arrayset(A, convert(T,x)::T, linear_index(A, J))
end

@inline function checkbounds_nd(A::CuDeviceArray{<:Any,N}, I::Tuple) where {N}
    if length(I) == N
        inbounds = reduce(&, map(in_bounds, I, size(A)); init=true)
        inbounds || Base.throw_boundserror(A, I)
    else
        # fewer or more indices than dimensions
        checkbounds(A, I...)
    end
    return
end


## const indexing

@public Const

"""
    Const(A::CuDeviceArray)

Mark a CuDeviceArray as constant/read-only. The invariant guaranteed is that you will not
modify an CuDeviceArray for the duration of the current kernel.

This API can only be used on devices with compute capability 3.5 or higher.

!!! warning
    Experimental API. Subject to change without deprecation.
"""
struct Const{T,N,AS,I} <: DenseArray{T,N}
    a::CuDeviceArray{T,N,AS,I}
end
Base.Experimental.Const(A::CuDeviceArray) = Const(A)

Base.IndexStyle(::Type{<:Const}) = IndexLinear()
Base.size(C::Const) = size(C.a)
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

function Base.reinterpret(::Type{T}, a::CuDeviceArray{S,N,A,I}) where {T,S,N,A,I}
  err = GPUArrays._reinterpret_exception(T, a)
  err === nothing || throw(err)

  if aligned_sizeof(T) == aligned_sizeof(S) # fast case
    return CuDeviceArray{T,N,A,I}(Unchecked(), reinterpret(LLVMPtr{T,A}, a.ptr), size(a),
                                  a.maxsize)
  end

  isize = size(a)
  size1 = div(isize[1]*aligned_sizeof(S), aligned_sizeof(T))
  osize = tuple(size1, Base.tail(isize)...)
  # reinterpreting to a larger type shrinks the first dimension, so the result still fits
  # the index type. reinterpreting to a smaller type grows it, which may not fit anymore.
  J = aligned_sizeof(T) > aligned_sizeof(S) ? I : Int
  return CuDeviceArray{T,N,A,J}(Unchecked(), reinterpret(LLVMPtr{T,A}, a.ptr), osize,
                                a.maxsize)
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
@inline function _derived_array(a::CuDeviceArray{<:Any,<:Any,A,I}, ::Type{T},
                                osize::Dims{N}) where {T, N, A, I}
  # the dimensions of a non-empty array are bounded by its length, which fits `I`.
  # only empty arrays can have dimensions that don't fit.
  if length(a) > 0 && all(>(0), osize)
    return CuDeviceArray{T,N,A,I}(Unchecked(), a.ptr, osize, a.maxsize)
  else
    return CuDeviceArray{T,N,A,I}(a.ptr, osize, a.maxsize)
  end
end


## index type conversions

# kernels are compiled for specific index types, but can be called with arrays of which
# the index type differs (e.g., when reusing a kernel compiled with `@cuda launch=false`)
Base.convert(::Type{CuDeviceArray{T,N,A,I}}, a::CuDeviceArray{T,N,A}) where {T,N,A,I} =
    CuDeviceArray{T,N,A,I}(a.ptr, size(a), a.maxsize)

# device arrays that need to have the same type, e.g. because they are stored in a struct
# with a single type parameter for both, can use this to agree on an index type.
unify_index_types(as::CuDeviceArray{<:Any,<:Any,<:Any,I}...) where {I<:Union{Int32,Int64}} = as
unify_index_types(as::CuDeviceArray...) =
    map(a -> convert(CuDeviceArray{eltype(a),ndims(a),addrspace(a),Int}, a), as)
addrspace(::CuDeviceArray{<:Any,<:Any,A}) where {A} = A
