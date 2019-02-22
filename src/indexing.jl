import GPUArrays: allowscalar, @allowscalar

function _getindex(xs::CuArray{T}, i::Integer) where T
  buf = Mem.view(buffer(xs), (i-1)*sizeof(T))
  return Mem.download(T, buf)[1]
end

function _setindex!(xs::CuArray{T}, v::T, i::Integer) where T
  buf = Mem.view(buffer(xs), (i-1)*sizeof(T))
  Mem.upload!(buf, T[v])
end


## logical indexing

Base.getindex(xs::CuArray, bools::AbstractArray{Bool}) = getindex(xs, CuArray(bools))

function Base.getindex(xs::CuArray{T}, bools::CuArray{Bool}) where {T}
  bools = reshape(bools, prod(size(bools)))
  indices = cumsum(bools)  # unique indices for elements that are true

  n = _getindex(indices, length(indices))  # number that are true
  ys = CuArray{T}(undef, n)

  if n > 0
    num_threads = min(n, 256)
    num_blocks = ceil(Int, length(indices) / num_threads)

    function kernel(ys::CuDeviceArray{T}, xs::CuDeviceArray{T}, bools, indices)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        if i <= length(xs) && bools[i]
            b = indices[i]   # new position
            ys[b] = xs[i]

        end

        return
    end

    @cuda blocks=num_blocks threads=num_threads kernel(ys, xs, bools, indices)
  end

  return ys
end
