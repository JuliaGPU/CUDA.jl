export CuUnifiedArray, CuUnifiedVector, CuUnifiedMatrix, CuUnifiedVecOrMat

mutable struct CuUnifiedArray{T,N} <: AbstractGPUArray{T,N}
  buf::Mem.UnifiedBuffer
  baseptr::CuPtr{T}
  dims::Dims{N}
  ctx::CuContext

  function CuUnifiedArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N}
    Base.isbitsunion(T) && error("CuUnifiedArray does not yet support union bits types")
    Base.isbitstype(T)  || error("CuUnifiedArray only supports bits types") # allocatedinline on 1.3+
    buf = Mem.alloc(Mem.Unified, prod(dims) * sizeof(T))
    ptr = convert(CuPtr{T}, buf)
    obj = new{T,N}(buf, ptr, dims, context())
    finalizer(t -> Mem.free(t.buf), obj)
  end
end

## convenience constructors
CuUnifiedVector{T} = CuUnifiedArray{T,1}
CuUnifiedMatrix{T} = CuUnifiedArray{T,2}
CuUnifiedVecOrMat{T} = Union{CuUnifiedVector{T},CuUnifiedMatrix{T}}

# type and dimensionality specified, accepting dims as series of Ints
CuUnifiedArray{T,N}(::UndefInitializer, dims::Integer...) where {T,N} = CuUnifiedArray{T,N}(undef, dims)

# type but not dimensionality specified
CuUnifiedArray{T}(::UndefInitializer, dims::Dims{N}) where {T,N} = CuUnifiedArray{T,N}(undef, dims)

CuUnifiedArray{T}(::UndefInitializer, dims::Integer...) where {T} = CuUnifiedArray{T}(undef, convert(Tuple{Vararg{Int}}, dims))

# empty vector constructor
CuUnifiedArray{T,1}() where {T} = CuUnifiedArray{T,1}(undef, 0)

## array interface

Base.elsize(::Type{<:CuUnifiedArray{T}}) where {T} = sizeof(T)

Base.size(x::CuUnifiedArray) = x.dims
Base.sizeof(x::CuUnifiedArray) = Base.elsize(x) * length(x)


## alias detection

Base.dataids(A::CuUnifiedArray) = (UInt(A.ptr),)

Base.unaliascopy(A::CuUnifiedArray) = copy(A)

function Base.mightalias(A::CuUnifiedArray, B::CuUnifiedArray)
  rA = pointer(A):pointer(A)+sizeof(A)
  rB = pointer(B):pointer(B)+sizeof(B)
  return first(rA) <= first(rB) < last(rA) || first(rB) <= first(rA) < last(rB)
end

Base.pointer(x::CuUnifiedArray) = x.baseptr

Base.similar(a::CuUnifiedArray{T,N}) where {T,N} = CuUnifiedArray{T,N}(undef, size(a))
Base.similar(a::CuUnifiedArray{T}, dims::Base.Dims{N}) where {T,N} = CuUnifiedArray{T,N}(undef, dims)
Base.similar(a::CuUnifiedArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} = CuUnifiedArray{T,N}(undef, dims)

function Base.copy(a::CuUnifiedArray{T,N}) where {T,N}
  b = similar(a)
  @inbounds copyto!(b, a)
end

function Base.copyto!(dest::Array{T}, doffs::Integer, src::CuUnifiedArray{T,N}, soffs::Integer,
                      n::Integer) where {T,N}
  # rely on CuArray, first wrap unsafely, and use CuArray's interface.
  src_cuarray = unsafe_wrap(CuArray{T,N}, src.baseptr, src.dims; own=false)
  copyto!(dest, doffs, src_cuarray, soffs, n)
  return dest
end

# underspecified constructors
CuUnifiedArray{T}(xs::AbstractArray{S,N}) where {T,N,S} = CuUnifiedArray{T,N}(xs)
(::Type{CuUnifiedArray{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = CuUnifiedArray{S,N}(x)
CuUnifiedArray(A::AbstractArray{T,N}) where {T,N} = CuUnifiedArray{T,N}(A)

# idempotency
CuUnifiedArray{T,N}(xs::CuUnifiedArray{T,N}) where {T,N} = xs

## conversions
Base.convert(::Type{T}, x::T) where T <: CuUnifiedArray = x

## interop with C libraries

Base.unsafe_convert(::Type{Ptr{T}}, x::CuUnifiedArray{T}) where {T} =
  throw(ArgumentError("cannot take the CPU address of a $(typeof(x))"))

Base.unsafe_convert(::Type{CuPtr{T}}, x::CuUnifiedArray{T}) where {T} =
  convert(CuPtr{T}, x.baseptr)

## interop with device arrays

function Base.unsafe_convert(::Type{CuDeviceArray{T,N,AS.Global}}, a::CuUnifiedArray{T,N}) where {T,N}
  CuDeviceArray{T,N,AS.Global}(size(a), reinterpret(LLVMPtr{T,AS.Global}, pointer(a)))
end

## interop with CPU arrays

# We don't convert isbits types in `adapt`, since they are already
# considered GPU-compatible.

Adapt.adapt_storage(::Type{CuUnifiedArray}, xs::AT) where {AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(CuUnifiedArray, xs)

# if an element type is specified, convert to it
Adapt.adapt_storage(::Type{<:CuUnifiedArray{T}}, xs::AT) where {T, AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(CuUnifiedArray{T}, xs)
