import GPUArrays: allowscalar, @allowscalar

Base.IndexStyle(::Type{<:CuArray}) = IndexLinear()

function _getindex(xs::CuArray{T}, i::Integer) where T
  buf = Mem.view(buffer(xs), (i-1)*sizeof(T))
  return Mem.download(T, buf)[1]
end

function Base.getindex(xs::CuArray{T}, i::UnitRange) where T
  CuVector{T}(xs.buf, xs.offset+(i.start-1)*sizeof(T), (i.stop-i.start+1,))
end

function _setindex!(xs::CuArray{T}, v::T, i::Integer) where T
  buf = Mem.view(buffer(xs), (i-1)*sizeof(T))
  Mem.upload!(buf, T[v])
end
