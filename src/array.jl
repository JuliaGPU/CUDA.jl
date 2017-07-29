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

function Base.copy!{T}(dst::CuArray{T}, src::DenseArray{T})
    @assert length(dst) == length(src)
    Mem.upload(dst.ptr, pointer(src), length(src) * sizeof(T))
    return dst
end

function Base.copy!{T}(dst::DenseArray{T}, src::CuArray{T})
    @assert length(dst) == length(src)
    Mem.download(pointer(dst), src.ptr, length(src) * sizeof(T))
    return dst
end

function Base.copy!{T}(dst::CuArray{T}, src::CuArray{T})
    @assert length(dst) == length(src)
    Mem.transfer(dst.ptr, src.ptr, length(src) * sizeof(T))
    return dst
end
