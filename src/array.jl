using CUDAnative: DevicePtr
using GPUArrays

mutable struct CuArray{T,N} <: GPUArray{T,N}
  ptr::OwnedPtr{Void}
  dims::NTuple{N,Int}
  function CuArray{T,N}(ptr::OwnedPtr{Void}, dims::NTuple{N,Integer}) where {T,N}
    xs = new{T,N}(ptr, dims)
    Mem.retain(ptr)
    finalizer(xs, unsafe_free!)
    return xs
  end
end

CuVector{T} = CuArray{T,1}
CuMatrix{T} = CuArray{T,2}
CuVecOrMat{T} = Union{CuVector{T},CuMatrix{T}}

function unsafe_free!(xs::CuArray)
  Mem.release(xs.ptr) && dealloc(xs.ptr, prod(xs.dims)*sizeof(eltype(xs)))
  return
end

Base.unsafe_convert(::Type{Ptr{T}}, x::CuArray{T}) where T =
  Base.unsafe_convert(Ptr{T}, x.ptr)

CuArray{T,N}(dims::NTuple{N,Integer}) where {T,N} =
  CuArray{T,N}(alloc(prod(dims)*sizeof(T)), dims)

CuArray{T}(dims::NTuple{N,Integer}) where {T,N} =
  CuArray{T,N}(dims)

CuArray(dims::NTuple{N,Integer}) where N = CuArray{Float32,N}(dims)

(T::Type{<:CuArray})(dims::Integer...) = T(dims)

Base.similar(a::CuArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} =
  CuArray{T,N}(dims)

Base.size(x::CuArray) = x.dims
Base.sizeof(x::CuArray) = Base.elsize(x) * length(x)

function Base._reshape(parent::CuArray, dims::Dims)
  n = Base._length(parent)
  prod(dims) == n || throw(DimensionMismatch("parent has $n elements, which is incompatible with size $dims"))
  return CuArray{eltype(parent),length(dims)}(parent.ptr, dims)
end

# Interop with CPU array

function Base.copy!(dst::CuArray{T}, src::DenseArray{T}) where T
    @assert length(dst) == length(src)
    Mem.upload(dst.ptr, pointer(src), length(src) * sizeof(T))
    return dst
end

function Base.copy!(dst::DenseArray{T}, src::CuArray{T}) where T
    @assert length(dst) == length(src)
    Mem.download(pointer(dst), src.ptr, length(src) * sizeof(T))
    return dst
end

function Base.copy!(dst::CuArray{T}, src::CuArray{T}) where T
    @assert length(dst) == length(src)
    Mem.transfer(dst.ptr, src.ptr, length(src) * sizeof(T))
    return dst
end

Base.collect(x::CuArray{T,N}) where {T,N} =
  copy!(Array{T,N}(size(x)), x)

Base.convert(::Type{T}, x::T) where T <: CuArray = x

Base.convert(::Type{CuArray{T,N}}, xs::DenseArray{T,N}) where {T,N} =
    copy!(CuArray{T,N}(size(xs)), xs)

Base.convert(::Type{CuArray{T}}, xs::DenseArray{T,N}) where {T,N} =
    copy!(CuArray{T}(size(xs)), xs)

Base.convert(::Type{CuArray}, xs::DenseArray{T,N}) where {T,N} =
  convert(CuArray{T,N}, xs)

Base.convert(T::Type{<:CuArray}, xs::DenseArray) =
  convert(T, convert(AbstractArray{eltype(T)}, xs))

# Interop with CUDAdrv native array

Base.convert(::Type{CUDAdrv.CuArray{T,N}}, xs::CuArray{T,N}) where {T,N} =
  CUDAdrv.CuArray{T,N}(xs.dims, xs.ptr)

Base.convert(::Type{CUDAdrv.CuArray}, xs::CuArray{T,N}) where {T,N} =
  convert(CUDAdrv.CuArray{T,N}, xs)

Base.convert(::Type{CuArray{T,N}}, xs::CUDAdrv.CuArray{T,N}) where {T,N} =
  CuArray{T,N}(xs.ptr, xs.shape)

Base.convert(::Type{CuArray}, xs::CUDAdrv.CuArray{T,N}) where {T,N} =
  convert(CuArray{T,N}, xs)

# Interop with CUDAnative device array

function Base.convert(::Type{CuDeviceArray{T,N,AS.Global}}, a::CuArray{T,N}) where {T,N}
    ptr = Base.unsafe_convert(Ptr{T}, a.ptr)
    CuDeviceArray{T,N,AS.Global}(a.dims, DevicePtr{T,AS.Global}(ptr))
end

cudaconvert(a::CuArray{T,N}) where {T,N} = convert(CuDeviceArray{T,N,AS.Global}, a)

# Utils

Base.show(io::IO, ::Type{CuArray{T,N}}) where {T,N} =
  print(io, "CuArray{$T,$N}")

function Base.showarray(io::IO, X::CuArray, repr::Bool = true; header = true)
  if repr
    print(io, "CuArray(")
    Base.showarray(io, collect(X), true)
    print(io, ")")
  else
    header && println(io, summary(X), ":")
    Base.showarray(io, collect(X), false, header = false)
  end
end

import NNlib: adapt, adapt_

adapt_(::Type{<:CuArray}, xs::AbstractArray) =
  isbits(xs) ? xs : convert(CuArray, xs)

adapt_(::Type{<:CuArray{T}}, xs::AbstractArray{<:Real}) where T <: AbstractFloat =
  isbits(xs) ? xs : convert(CuArray{T}, xs)

cu(xs) = adapt(CuArray{Float32}, xs)

Base.getindex(::typeof(cu), xs...) = CuArray([xs...])
