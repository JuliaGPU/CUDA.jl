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
