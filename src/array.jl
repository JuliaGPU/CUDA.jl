using GPUArrays

import CUDAnative: DevicePtr

mutable struct CuArray{T,N} <: GPUArray{T,N}
  buf::Mem.Buffer
  offset::Int
  dims::NTuple{N,Int}

  function CuArray{T,N}(buf::Mem.Buffer, offset::Integer, dims::NTuple{N,Integer}) where {T,N}
    xs = new{T,N}(buf, offset, dims)
    Mem.retain(buf)
    finalizer(unsafe_free!, xs)
    return xs
  end
end

CuArray{T,N}(buf::Mem.Buffer, dims::NTuple{N,Integer}) where {T,N} = CuArray{T,N}(buf, 0, dims)

CuVector{T} = CuArray{T,1}
CuMatrix{T} = CuArray{T,2}
CuVecOrMat{T} = Union{CuVector{T},CuMatrix{T}}

Base.elsize(::CuArray{T}) where T = sizeof(T)

function unsafe_free!(xs::CuArray)
  Mem.release(xs.buf) && dealloc(xs.buf, prod(xs.dims)*sizeof(eltype(xs)))
  return
end

"""
  buffer(array::CuArray [, index])

Get the native address of a CuArray, optionally at a given location `index`.
Equivalent of `Base.pointer` on `Array`s.
"""
function buffer(xs::CuArray, index=1)
  extra_offset = (index-1) * Base.elsize(xs)
  Mem.Buffer(xs.buf.ptr + xs.offset + extra_offset,
             sizeof(xs) - extra_offset,
             xs.buf.ctx)
end

Base.cconvert(::Type{Ptr{T}}, x::CuArray{T}) where T = buffer(x)
Base.cconvert(::Type{Ptr{Nothing}}, x::CuArray) = buffer(x)

CuArray{T,N}(dims::NTuple{N,Integer}) where {T,N} =
  CuArray{T,N}(alloc(prod(dims)*sizeof(T)), dims)

CuArray{T}(dims::NTuple{N,Integer}) where {T,N} =
  CuArray{T,N}(dims)

CuArray(dims::NTuple{N,Integer}) where N = CuArray{Float32,N}(dims)

(T::Type{<:CuArray})(dims::Integer...) = T(dims)

Base.similar(a::CuArray{T,N}) where {T,N} = CuArray{T,N}(size(a))
Base.similar(a::CuArray{T}, dims::Base.Dims{N}) where {T,N} = CuArray{T,N}(dims)
Base.similar(a::CuArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} =
  CuArray{T,N}(dims)

Base.size(x::CuArray) = x.dims
Base.sizeof(x::CuArray) = Base.elsize(x) * length(x)

function Base._reshape(parent::CuArray, dims::Dims)
  n = length(parent)
  prod(dims) == n || throw(DimensionMismatch("parent has $n elements, which is incompatible with size $dims"))
  return CuArray{eltype(parent),length(dims)}(parent.buf, parent.offset, dims)
end

# Interop with CPU array

function Base.unsafe_copyto!(dest::CuArray{T}, doffs, src::Array{T}, soffs, n) where T
    Mem.upload!(buffer(dest, doffs), pointer(src, soffs), n*sizeof(T))
    return dest
end

function Base.unsafe_copyto!(dest::Array{T}, doffs, src::CuArray{T}, soffs, n) where T
    Mem.download!(pointer(dest, doffs), buffer(src, soffs), n*sizeof(T))
    return dest
end

function Base.unsafe_copyto!(dest::CuArray{T}, doffs, src::CuArray{T}, soffs, n) where T
    Mem.transfer!(buffer(dest, doffs), buffer(src, soffs), sizeof(src), n*sizeof(T))
    return dest
end

Base.collect(x::CuArray{T,N}) where {T,N} = copyto!(Array{T,N}(undef, size(x)), x)

function Base.deepcopy_internal(x::CuArray, dict::IdDict)
  haskey(dict, x) && return dict[x]::typeof(x)
  return dict[x] = copy(x)
end

Base.convert(::Type{T}, x::T) where T <: CuArray = x

Base.convert(::Type{CuArray{T,N}}, xs::Array{T,N}) where {T,N} =
  copyto!(CuArray{T,N}(size(xs)), xs)

Base.convert(::Type{CuArray{T}}, xs::Array{T,N}) where {T,N} =
  copyto!(CuArray{T}(size(xs)), xs)

Base.convert(::Type{CuArray}, xs::Array{T,N}) where {T,N} =
  convert(CuArray{T,N}, xs)

# Generic methods

Base.convert(::Type{CuArray{T,N}}, xs::AbstractArray{T,N}) where {T,N} =
  isbits(xs) ?
    (CuArray{T,N}(size(xs)) .= xs) :
    convert(CuArray{T,N}, collect(xs))

Base.convert(::Type{CuArray{T,N}}, xs::AbstractArray{S,N}) where {S,T,N} =
  convert(CuArray{T,N}, (x -> T(x)).(xs))

Base.convert(::Type{CuArray{T}}, xs::AbstractArray) where T =
  convert(CuArray{T,ndims(xs)},xs)

Base.convert(::Type{CuArray}, xs::AbstractArray) = convert(CuArray{eltype(xs)}, xs)

# Work around GPUArrays ambiguity
Base.convert(AT::Type{CuArray{T1,N}}, A::DenseArray{T2, N}) where {T1, T2, N} =
  invoke(convert, Tuple{Type{CuArray{T1,N}},AbstractArray{T2,N}}, AT, A)

Base.convert(AT::Type{CuArray{T1}}, A::DenseArray{T2, N}) where {T1, T2, N} =
  invoke(convert, Tuple{Type{CuArray{T1}},AbstractArray{T2,N}}, AT, A)

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

Base.show(io::IO, x::CuArray) = show(io, collect(x))
Base.show(io::IO, x::LinearAlgebra.Adjoint{<:Any,<:CuArray}) = show(io, LinearAlgebra.adjoint(collect(x.parent)))
Base.show(io::IO, x::LinearAlgebra.Transpose{<:Any,<:CuArray}) = show(io, LinearAlgebra.transpose(collect(x.parent)))

Base.show(io::IO, ::MIME"text/plain", x::CuArray) = show(io, collect(x))
Base.show(io::IO, ::MIME"text/plain", x::LinearAlgebra.Adjoint{<:Any,<:CuArray}) = show(io, LinearAlgebra.adjoint(collect(x.parent)))
Base.show(io::IO, ::MIME"text/plain", x::LinearAlgebra.Transpose{<:Any,<:CuArray}) = show(io, LinearAlgebra.transpose(collect(x.parent)))

import Adapt: adapt, adapt_

adapt_(::Type{<:CuArray}, xs::AbstractArray) =
  isbits(xs) ? xs : convert(CuArray, xs)

adapt_(::Type{<:CuArray{T}}, xs::AbstractArray{<:Real}) where T <: AbstractFloat =
  isbits(xs) ? xs : convert(CuArray{T}, xs)

adapt_(::Type{<:Array}, xs::CuArray) = collect(xs)

cu(xs) = adapt(CuArray{Float32}, xs)

Base.getindex(::typeof(cu), xs...) = CuArray([xs...])
