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
    finalizer(Mem.free, obj)
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
