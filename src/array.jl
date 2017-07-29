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
