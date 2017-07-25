module CuArrays

using CUDAdrv, CUDAnative

export CuArray

mutable struct CuArray{T,N} <: DenseArray{T,N}
  ptr::DevicePtr{T}
  dims::NTuple{N,Int}
end

function CuArray{T,N}(dims::NTuple{N,Integer}) where {T,N}
  xs = CuArray{T,N}(Mem.alloc(Float64, prod(dims)), dims)
  finalizer(xs, unsafe_free!)
  return xs
end

function unsafe_free!(xs::CuArray)
  CUDAdrv.isvalid(xs.ptr.ctx) && Mem.free(xs.ptr)
  return
end

Base.size(x::CuArray) = x.dims

Base.sizeof(x::CuArray) = Base.elsize(x) * length(x)

Base.IndexStyle(::Type{<:CuArray}) = IndexLinear()

function Base.getindex{T}(xs::CuArray{T}, i::Integer)
  x = Array{T}(1)
  ptr = DevicePtr{T}(xs.ptr.ptr + (i-1)*sizeof(T), xs.ptr.ctx)
  Mem.download(pointer(x), ptr, sizeof(T))
  return x[1]
end

function Base.setindex!{T}(xs::CuArray{T}, v::T, i::Integer)
  x = T[v]
  ptr = DevicePtr{T}(xs.ptr.ptr + (i-1)*sizeof(T), xs.ptr.ctx)
  Mem.upload(ptr, pointer(x), sizeof(T))
  return x[1]
end

Base.setindex!(xs::CuArray, v, i::Integer) = xs[i] = convert(eltype(xs), v)

end # module
