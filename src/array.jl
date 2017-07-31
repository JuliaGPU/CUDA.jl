mutable struct CuArray{T,N} <: DenseArray{T,N}
  ptr::DevicePtr{T}
  dims::NTuple{N,Int}
end

function CuArray{T,N}(dims::NTuple{N,Integer}) where {T,N}
  xs = CuArray{T,N}(Mem.alloc(T, prod(dims)), dims)
  finalizer(xs, unsafe_free!)
  return xs
end

CuArray{T}(dims::NTuple{N,Integer}) where {T,N} =
  CuArray{T,N}(dims)

CuArray(dims::NTuple{N,Integer}) where N = CuArray{Float64,N}(dims)

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

Base.convert(::Type{CuArray{T,N}}, xs::DenseArray{T,N}) where {T,N} =
  copy!(CuArray{T,N}(size(xs)), xs)

Base.convert(::Type{CuArray}, xs::DenseArray{T,N}) where {T,N} =
  convert(CuArray{T,N}, xs)

Base.convert(::Type{CuDeviceArray{T,N}}, a::CuArray{T,N}) where {T,N} =
    CuDeviceArray{T,N}(a.dims, Base.unsafe_convert(Ptr{T}, a.ptr))

Base.convert(::Type{CuDeviceArray}, a::CuArray{T,N}) where {T,N} =
  convert(CuDeviceArray{T,N}, a)

# TODO: auto conversions in CUDAnative
todevice(x) = x
todevice(x::CuArray) = convert(CuDeviceArray, x)
