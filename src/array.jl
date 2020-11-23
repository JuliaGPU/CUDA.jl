export CuArray, CuVector, CuMatrix, CuVecOrMat, cu

@enum ArrayState begin
  ARRAY_UNMANAGED
  ARRAY_MANAGED
  ARRAY_FREED
end

mutable struct CuArray{T,N} <: AbstractGPUArray{T,N}
  baseptr::CuPtr{Nothing}
  offset::Int
  dims::Dims{N}

  state::ArrayState
  ctx::CuContext

  function CuArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N}
    Base.isbitsunion(T) && error("CuArray does not yet support union bits types")
    Base.isbitstype(T)  || error("CuArray only supports bits types") # allocatedinline on 1.3+
    ptr = alloc(prod(dims) * sizeof(T))
    obj = new{T,N}(ptr, 0, dims, ARRAY_MANAGED, context())
    finalizer(unsafe_free!, obj)
    return obj
  end

  function CuArray{T,N}(ptr::CuPtr{Nothing}, dims::Dims{N}, ctx=context(); offset::Int=0) where {T,N}
    Base.isbitsunion(T) && error("CuArray does not yet support union bits types")
    Base.isbitstype(T)  || error("CuArray only supports bits types") # allocatedinline on 1.3+
    return new{T,N}(ptr, offset, dims, ARRAY_UNMANAGED, ctx)
  end
end

function unsafe_free!(xs::CuArray)
  # this call should only have an effect once, becuase both the user and the GC can call it
  if xs.state == ARRAY_FREED
    return
  elseif xs.state == ARRAY_UNMANAGED
    throw(ArgumentError("Cannot free an unmanaged buffer."))
  end

  if isvalid(xs.ctx)
    free(xs.baseptr)
  end
  xs.state = ARRAY_FREED

  # the object is dead, so we can also wipe the pointer
  xs.baseptr = CU_NULL

  return
end


## alias detection

Base.dataids(A::CuArray) = (UInt(A.baseptr),)

Base.unaliascopy(A::CuArray) = copy(A)

function Base.mightalias(A::CuArray, B::CuArray)
  rA = pointer(A):pointer(A)+sizeof(A)
  rB = pointer(B):pointer(B)+sizeof(B)
  return first(rA) <= first(rB) < last(rA) || first(rB) <= first(rA) < last(rB)
end


## convenience constructors

CuVector{T} = CuArray{T,1}
CuMatrix{T} = CuArray{T,2}
CuVecOrMat{T} = Union{CuVector{T},CuMatrix{T}}

# type and dimensionality specified, accepting dims as series of Ints
CuArray{T,N}(::UndefInitializer, dims::Integer...) where {T,N} = CuArray{T,N}(undef, dims)

# type but not dimensionality specified
CuArray{T}(::UndefInitializer, dims::Dims{N}) where {T,N} = CuArray{T,N}(undef, dims)
CuArray{T}(::UndefInitializer, dims::Integer...) where {T} =
  CuArray{T}(undef, convert(Tuple{Vararg{Int}}, dims))

# empty vector constructor
CuArray{T,1}() where {T} = CuArray{T,1}(undef, 0)

# do-block constructors
for (ctor, tvars) in (:CuArray => (), :(CuArray{T}) => (:T,), :(CuArray{T,N}) => (:T, :N))
  @eval begin
    function $ctor(f::Function, args...) where {$(tvars...)}
      xs = $ctor(args...)
      try
        f(xs)
      finally
        unsafe_free!(xs)
      end
    end
  end
end

Base.similar(a::CuArray{T,N}) where {T,N} = CuArray{T,N}(undef, size(a))
Base.similar(a::CuArray{T}, dims::Base.Dims{N}) where {T,N} = CuArray{T,N}(undef, dims)
Base.similar(a::CuArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} = CuArray{T,N}(undef, dims)

function Base.copy(a::CuArray{T,N}) where {T,N}
  b = similar(a)
  @inbounds copyto!(b, a)
end


"""
  unsafe_wrap(::CuArray, ptr::CuPtr{T}, dims; own=false, ctx=context())

Wrap a `CuArray` object around the data at the address given by `ptr`. The pointer
element type `T` determines the array element type. `dims` is either an integer (for a 1d
array) or a tuple of the array dimensions. `own` optionally specified whether Julia should
take ownership of the memory, calling `cudaFree` when the array is no longer referenced. The
`ctx` argument determines the CUDA context where the data is allocated in.
"""
function Base.unsafe_wrap(::Union{Type{CuArray},Type{CuArray{T}},Type{CuArray{T,N}}},
                          ptr::CuPtr{T}, dims::NTuple{N,Int};
                          own::Bool=false, ctx::CuContext=context()) where {T,N}
  xs = CuArray{T, length(dims)}(convert(CuPtr{Cvoid}, ptr), dims, ctx)
  if own
    finalizer(xs) do obj
      if isvalid(obj.ctx)
        buf = Mem.DeviceBuffer(obj.baseptr, sizeof(obj))
        Mem.free(buf)
      end
    end
  end
  return xs
end

function Base.unsafe_wrap(Atype::Union{Type{CuArray},Type{CuArray{T}},Type{CuArray{T,1}}},
                          p::CuPtr{T}, dim::Integer;
                          own::Bool=false, ctx::CuContext=context()) where {T}
  unsafe_wrap(Atype, p, (dim,); own=own, ctx=ctx)
end

Base.unsafe_wrap(T::Type{<:CuArray}, ::Ptr, dims::NTuple{N,Int}; kwargs...) where {N} =
  throw(ArgumentError("cannot wrap a CPU pointer with a $T"))


## array interface

Base.elsize(::Type{<:CuArray{T}}) where {T} = sizeof(T)

Base.size(x::CuArray) = x.dims
Base.sizeof(x::CuArray) = Base.elsize(x) * length(x)


## derived types

export DenseCuArray, DenseCuVector, DenseCuMatrix, DenseCuVecOrMat,
       StridedCuArray, StridedCuVector, StridedCuMatrix, StridedCuVecOrMat,
       AnyCuArray, AnyCuVector, AnyCuMatrix, AnyCuVecOrMat

# dense arrays: stored contiguously in memory
#
# all common dense wrappers are currently represented as CuArray objects.
# this simplifies common use cases, and greatly improves load time.
# CUDA.jl 2.0 experimented with using ReshapedArray/ReinterpretArray/SubArray,
# but that proved much too costly. TODO: revisit when we have better Base support.
DenseCuArray{T,N} = CuArray{T,N}
DenseCuVector{T} = DenseCuArray{T,1}
DenseCuMatrix{T} = DenseCuArray{T,2}
DenseCuVecOrMat{T} = Union{DenseCuVector{T}, DenseCuMatrix{T}}

# strided arrays
StridedSubCuArray{T,N,I<:Tuple{Vararg{Union{Base.RangeIndex, Base.ReshapedUnitRange,
                                            Base.AbstractCartesianIndex}}}} =
  SubArray{T,N,<:CuArray,I}
StridedCuArray{T,N} = Union{CuArray{T,N}, StridedSubCuArray{T,N}}
StridedCuVector{T} = StridedCuArray{T,1}
StridedCuMatrix{T} = StridedCuArray{T,2}
StridedCuVecOrMat{T} = Union{StridedCuVector{T}, StridedCuMatrix{T}}

Base.pointer(x::StridedCuArray{T}) where {T} = Base.unsafe_convert(CuPtr{T}, x)
@inline function Base.pointer(x::StridedCuArray{T}, i::Integer) where T
    Base.unsafe_convert(CuPtr{T}, x) + Base._memory_offset(x, i)
end

# anything that's (secretly) backed by a CuArray
AnyCuArray{T,N} = Union{CuArray{T,N}, WrappedArray{T,N,CuArray,CuArray{T,N}}}
AnyCuVector{T} = AnyCuArray{T,1}
AnyCuMatrix{T} = AnyCuArray{T,2}
AnyCuVecOrMat{T} = Union{AnyCuVector{T}, AnyCuMatrix{T}}


## interop with other arrays

@inline function CuArray{T,N}(xs::AbstractArray{<:Any,N}) where {T,N}
  A = CuArray{T,N}(undef, size(xs))
  copyto!(A, convert(Array{T}, xs))
  return A
end

# underspecified constructors
CuArray{T}(xs::AbstractArray{S,N}) where {T,N,S} = CuArray{T,N}(xs)
(::Type{CuArray{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = CuArray{S,N}(x)
CuArray(A::AbstractArray{T,N}) where {T,N} = CuArray{T,N}(A)

# idempotency
CuArray{T,N}(xs::CuArray{T,N}) where {T,N} = xs


## conversions

Base.convert(::Type{T}, x::T) where T <: CuArray = x


## interop with C libraries

Base.unsafe_convert(::Type{Ptr{T}}, x::CuArray{T}) where {T} =
  throw(ArgumentError("cannot take the CPU address of a $(typeof(x))"))
Base.unsafe_convert(::Type{CuPtr{T}}, x::CuArray{T}) where {T} =
  convert(CuPtr{T}, x.baseptr) + x.offset


## interop with device arrays

function Base.unsafe_convert(::Type{CuDeviceArray{T,N,AS.Global}}, a::DenseCuArray{T,N}) where {T,N}
  CuDeviceArray{T,N,AS.Global}(size(a), reinterpret(LLVMPtr{T,AS.Global}, pointer(a)))
end

Adapt.adapt_storage(::Adaptor, xs::CuArray{T,N}) where {T,N} =
  Base.unsafe_convert(CuDeviceArray{T,N,AS.Global}, xs)

# we materialize ReshapedArray/ReinterpretArray/SubArray/... directly as a device array
Adapt.adapt_structure(::Adaptor, xs::DenseCuArray{T,N}) where {T,N} =
  Base.unsafe_convert(CuDeviceArray{T,N,AS.Global}, xs)


## interop with CPU arrays

# We don't convert isbits types in `adapt`, since they are already
# considered GPU-compatible.

Adapt.adapt_storage(::Type{CuArray}, xs::AT) where {AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(CuArray, xs)

# if an element type is specified, convert to it
Adapt.adapt_storage(::Type{<:CuArray{T}}, xs::AT) where {T, AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(CuArray{T}, xs)

Adapt.adapt_storage(::Type{Array}, xs::CuArray) = convert(Array, xs)

Base.collect(x::CuArray{T,N}) where {T,N} = copyto!(Array{T,N}(undef, size(x)), x)

function Base.copyto!(dest::DenseCuArray{T}, doffs::Integer, src::Array{T}, soffs::Integer,
                      n::Integer) where T
  n==0 && return dest
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(dest, doffs, src, soffs, n)
  return dest
end

Base.copyto!(dest::DenseCuArray{T}, src::Array{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.copyto!(dest::Array{T}, doffs::Integer, src::DenseCuArray{T}, soffs::Integer,
                      n::Integer) where T
  n==0 && return dest
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(dest, doffs, src, soffs, n)
  return dest
end

Base.copyto!(dest::Array{T}, src::DenseCuArray{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.copyto!(dest::DenseCuArray{T}, doffs::Integer, src::DenseCuArray{T}, soffs::Integer,
                      n::Integer) where T
  n==0 && return dest
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(dest, doffs, src, soffs, n)
  return dest
end

Base.copyto!(dest::DenseCuArray{T}, src::DenseCuArray{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.unsafe_copyto!(dest::DenseCuArray{T}, doffs, src::Array{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("Not implemented")
  end
  return dest
end

function Base.unsafe_copyto!(dest::Array{T}, doffs, src::DenseCuArray{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("Not implemented")
  end
  return dest
end

function Base.unsafe_copyto!(dest::DenseCuArray{T}, doffs, src::DenseCuArray{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n;
                                       async=true, stream=CuDefaultStream())
  if Base.isbitsunion(T)
    # copy selector bytes
    error("Not implemented")
  end
  return dest
end

function Base.deepcopy_internal(x::CuArray, dict::IdDict)
  haskey(dict, x) && return dict[x]::typeof(x)
  return dict[x] = copy(x)
end


## Float32-preferring conversion

struct Float32Adaptor end

Adapt.adapt_storage(::Float32Adaptor, xs::AbstractArray) =
  isbits(xs) ? xs : convert(CuArray, xs)

Adapt.adapt_storage(::Float32Adaptor, xs::AbstractArray{<:AbstractFloat}) =
  isbits(xs) ? xs : convert(CuArray{Float32}, xs)

Adapt.adapt_storage(::Float32Adaptor, xs::AbstractArray{<:Complex{<:AbstractFloat}}) =
  isbits(xs) ? xs : convert(CuArray{ComplexF32}, xs)

# not for Float16
Adapt.adapt_storage(::Float32Adaptor, xs::AbstractArray{Float16}) =
  isbits(xs) ? xs : convert(CuArray, xs)
Adapt.adapt_storage(::Float32Adaptor, xs::AbstractArray{BFloat16}) =
  isbits(xs) ? xs : convert(CuArray, xs)

cu(xs) = adapt(Float32Adaptor(), xs)
Base.getindex(::typeof(cu), xs...) = CuArray([xs...])


## utilities

zeros(T::Type, dims...) = fill!(CuArray{T}(undef, dims...), 0)
ones(T::Type, dims...) = fill!(CuArray{T}(undef, dims...), 1)
zeros(dims...) = zeros(Float32, dims...)
ones(dims...) = ones(Float32, dims...)
fill(v, dims...) = fill!(CuArray{typeof(v)}(undef, dims...), v)
fill(v, dims::Dims) = fill!(CuArray{typeof(v)}(undef, dims...), v)

# optimized implementation of `fill!` for types that are directly supported by memset
memsettype(T::Type) = T
memsettype(T::Type{<:Signed}) = unsigned(T)
memsettype(T::Type{<:AbstractFloat}) = Base.uinttype(T)
const MemsetCompatTypes = Union{UInt8, Int8,
                                UInt16, Int16, Float16,
                                UInt32, Int32, Float32}
function Base.fill!(A::DenseCuArray{T}, x) where T <: MemsetCompatTypes
  U = memsettype(T)
  y = reinterpret(U, convert(T, x))
  Mem.set!(convert(CuPtr{U}, pointer(A)), y, length(A))
  A
end


## views

# optimize view to return a CuArray when contiguous

struct Contiguous end
struct NonContiguous end

# NOTE: this covers more cases than the I<:... in Base.FastContiguousSubArray
CuIndexStyle() = Contiguous()
CuIndexStyle(I...) = NonContiguous()
CuIndexStyle(::Base.ScalarIndex...) = Contiguous()
CuIndexStyle(i1::Colon, ::Base.ScalarIndex...) = Contiguous()
CuIndexStyle(i1::AbstractUnitRange, ::Base.ScalarIndex...) = Contiguous()
CuIndexStyle(i1::Colon, I...) = CuIndexStyle(I...)

cuviewlength() = ()
@inline cuviewlength(::Real, I...) = cuviewlength(I...) # skip scalar
@inline cuviewlength(i1::AbstractUnitRange, I...) = (Base.unsafe_length(i1), cuviewlength(I...)...)
@inline cuviewlength(i1::AbstractUnitRange, ::Base.ScalarIndex...) = (Base.unsafe_length(i1),)

@inline function Base.view(A::CuArray, I::Vararg{Any,N}) where {N}
    J = to_indices(A, I)
    @boundscheck begin
        # Base's boundscheck accesses the indices, so make sure they reside on the CPU.
        # this is expensive, but it's a bounds check after all.
        J_cpu = map(j->adapt(Array, j), J)
        checkbounds(A, J_cpu...)
    end
    J_gpu = map(j->adapt(CuArray, j), J)
    unsafe_view(A, J_gpu, CuIndexStyle(I...))
end

@inline function unsafe_view(A, I, ::Contiguous)
    unsafe_contiguous_view(Base._maybe_reshape_parent(A, Base.index_ndims(I...)), I, cuviewlength(I...))
end
@inline function unsafe_contiguous_view(a::CuArray{T}, I::NTuple{N,Base.ViewIndex}, dims::NTuple{M,Integer}) where {T,N,M}
    offset = Base.compute_offset1(a, 1, I) * sizeof(T)

    alias(a.baseptr)
    b = CuArray{T,M}(a.baseptr, dims, a.ctx; offset=a.offset+offset)
    finalizer(unsafe_free!, b)
    b.state = ARRAY_MANAGED
    return b
end

@inline function unsafe_view(A, I, ::NonContiguous)
    Base.unsafe_view(Base._maybe_reshape_parent(A, Base.index_ndims(I...)), I...)
end

# pointer conversions
## contiguous
function Base.unsafe_convert(::Type{CuPtr{T}}, V::SubArray{T,N,P,<:Tuple{Vararg{Base.RangeIndex}}}) where {T,N,P}
    return Base.unsafe_convert(CuPtr{T}, parent(V)) +
           Base._memory_offset(V.parent, map(first, V.indices)...)
end
## reshaped
function Base.unsafe_convert(::Type{CuPtr{T}}, V::SubArray{T,N,P,<:Tuple{Vararg{Union{Base.RangeIndex,Base.ReshapedUnitRange}}}}) where {T,N,P}
   return Base.unsafe_convert(CuPtr{T}, parent(V)) +
          (Base.first_index(V)-1)*sizeof(T)
end


## PermutedDimsArray

Base.unsafe_convert(::Type{CuPtr{T}}, A::PermutedDimsArray) where {T} =
    Base.unsafe_convert(CuPtr{T}, parent(A))


## reshape

# optimize reshape to return a CuArray

function Base.reshape(a::CuArray{T,M}, dims::NTuple{N,Int}) where {T,N,M}
  if prod(dims) != length(a)
      throw(DimensionMismatch("new dimensions $(dims) must be consistent with array size $(size(a))"))
  end

  if N == M && dims == size(a)
      return a
  end

  alias(a.baseptr)
  b = CuArray{T,N}(a.baseptr, dims, a.ctx; offset=a.offset)
  finalizer(unsafe_free!, b)
  b.state = ARRAY_MANAGED
  return b
end

# allow missing dimensions with Colon()
if VERSION < v"1.6.0-DEV.1358"
Base.reshape(parent::CuArray, dims::Tuple{Vararg{Union{Int,Colon}}}) =
  Base.reshape(parent, Base._reshape_uncolon(parent, dims))
end


## reinterpret

# optimize reshape to return a CuArray

function Base.reinterpret(::Type{T}, a::CuArray{S,N}) where {T,S,N}
  if N == 0 && sizeof(T) == sizeof(S)
    throw(ArgumentError("cannot reinterpret a zero-dimensional `$(S)` array to `$(T)` which is of a different size"))
  end
  if N != 0 && sizeof(S) != sizeof(T)
      ax1 = axes(a)[1]
      dim = length(ax1)
      if Base.rem(dim*sizeof(S),sizeof(T)) != 0
        throw(ArgumentError("""
            cannot reinterpret an `$(S)` array to `$(T)` whose first dimension has size `$(dim)`.
            The resulting array would have non-integral first dimension.
            """))
      end
      if first(ax1) != 1
        throw(ArgumentError("cannot reinterpret a `$(S)` array to `$(T)` when the first axis is $ax1. Try reshaping first."))
      end
  end

  isize = size(a)
  size1 = div(isize[1]*sizeof(S), sizeof(T))
  osize = tuple(size1, Base.tail(isize)...)

  alias(a.baseptr)
  b = CuArray{T,N}(a.baseptr, osize, a.ctx; offset=a.offset)
  finalizer(unsafe_free!, b)
  b.state = ARRAY_MANAGED
  return b
end


## resizing

"""
  resize!(a::CuVector, n::Int)

Resize `a` to contain `n` elements. If `n` is smaller than the current collection length,
the first `n` elements will be retained. If `n` is larger, the new elements are not
guaranteed to be initialized.

Note that this operation is only supported on managed buffers, i.e., not on arrays that are
created by `unsafe_wrap` with `own=false`.
"""
function Base.resize!(A::CuVector{T}, n::Int) where T
  ptr = alloc(n * sizeof(T))
  m = Base.min(length(A), n)
  unsafe_copyto!(convert(CuPtr{T}, ptr), pointer(A), m)

  unsafe_free!(A)

  A.state = ARRAY_MANAGED
  A.dims = (n,)
  A.baseptr = ptr
  A.offset = 0

  A
end
