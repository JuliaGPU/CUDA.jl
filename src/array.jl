using GPUArrays

import CUDAnative: DevicePtr

mutable struct CuArray{T,N} <: GPUArray{T,N}
  buf::Mem.Buffer
  offset::Int
  dims::NTuple{N,Int}
  function CuArray{T,N}(buf::Mem.Buffer, offset::Integer, dims::NTuple{N,Integer}) where {T,N}
    xs = new{T,N}(buf, offset, dims)
    Mem.retain(buf)
    finalizer(xs, unsafe_free!)
    return xs
  end
end

CuArray{T,N}(buf::Mem.Buffer, dims::NTuple{N,Integer}) where {T,N} = CuArray{T,N}(buf, 0, dims)

CuVector{T} = CuArray{T,1}
CuMatrix{T} = CuArray{T,2}
CuVecOrMat{T} = Union{CuVector{T},CuMatrix{T}}

function unsafe_free!(xs::CuArray)
  Mem.release(xs.buf) && dealloc(xs.buf, prod(xs.dims)*sizeof(eltype(xs)))
  return
end

unsafe_buffer(xs::CuArray) =
  Mem.Buffer(xs.buf.ptr+xs.offset, sizeof(xs), xs.buf.ctx)

Base.cconvert(::Type{Ptr{T}}, x::CuArray{T}) where T = unsafe_buffer(x)
Base.cconvert(::Type{Ptr{Void}}, x::CuArray) = unsafe_buffer(x)

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
  return CuArray{eltype(parent),length(dims)}(parent.buf, parent.offset, dims)
end

# Interop with CPU array

function Base.copy!(dst::CuArray{T}, src::Array{T}) where T
    @assert length(dst) == length(src)
    Mem.upload!(unsafe_buffer(dst), src)
    return dst
end

function Base.copy!(dst::Array{T}, src::CuArray{T}) where T
    @assert length(dst) == length(src)
    Mem.download!(dst, unsafe_buffer(src))
    return dst
end

function Base.copy!(dst::CuArray{T}, src::CuArray{T}) where T
    @assert length(dst) == length(src)
    Mem.transfer!(unsafe_buffer(dst), unsafe_buffer(src))
    return dst
end

Base.collect(x::CuArray{T,N}) where {T,N} =
  copy!(Array{T,N}(size(x)), x)

function Base.deepcopy_internal(x::CuArray, dict::ObjectIdDict)
  haskey(dict, x) && return dict[x]::typeof(x)
  return dict[x] = copy(x)
end

Base.convert(::Type{T}, x::T) where T <: CuArray = x

Base.convert(::Type{CuArray{T,N}}, xs::Array{T,N}) where {T,N} =
    copy!(CuArray{T,N}(size(xs)), xs)

Base.convert(::Type{CuArray{T}}, xs::Array{T,N}) where {T,N} =
    copy!(CuArray{T}(size(xs)), xs)

Base.convert(::Type{CuArray}, xs::Array{T,N}) where {T,N} =
  convert(CuArray{T,N}, xs)

Base.convert(T::Type{<:CuArray}, xs::Array) =
  convert(T, convert(AbstractArray{eltype(T)}, xs))

# Interop with CUDAnative device array

function Base.convert(::Type{CuDeviceArray{T,N,AS.Global}}, a::CuArray{T,N}) where {T,N}
    ptr = Base.unsafe_convert(Ptr{T}, a.buf)
    CuDeviceArray{T,N,AS.Global}(a.dims, DevicePtr{T,AS.Global}(ptr+a.offset))
end

cudaconvert(a::CuArray{T,N}) where {T,N} = convert(CuDeviceArray{T,N,AS.Global}, a)

# Utils

cuzeros(T::Type, dims...) = fill!(CuArray{T}(dims...), 0)
cuones(T::Type, dims...) = fill!(CuArray{T}(dims...), 1)
cuzeros(dims...) = cuzeros(Float32, dims...)
cuones(dims...) = cuones(Float32, dims...)

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

import Adapt: adapt, adapt_

adapt_(::Type{<:CuArray}, xs::AbstractArray) =
  isbits(xs) ? xs : convert(CuArray, xs)

adapt_(::Type{<:CuArray{T}}, xs::AbstractArray{<:Real}) where T <: AbstractFloat =
  isbits(xs) ? xs : convert(CuArray{T}, xs)

cu(xs) = adapt(CuArray{Float32}, xs)

Base.getindex(::typeof(cu), xs...) = CuArray([xs...])
