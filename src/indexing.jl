import GPUArrays: allowscalar, @allowscalar

function _getindex(xs::CuArray{T}, i::Integer) where T
  buf = Mem.view(buffer(xs), (i-1)*sizeof(T))
  return Mem.download(T, buf)[1]
end

function _setindex!(xs::CuArray{T}, v::T, i::Integer) where T
  buf = Mem.view(buffer(xs), (i-1)*sizeof(T))
  Mem.upload!(buf, T[v])
end


# logical indexing
function getindex(xs::CuVector{T}, bools::CuVector{Bool}) where {T}

    indices = cumsum(bools)  # unique indices for elements that are true

    n = _getindex(indices, length(indices))  # number that are true
    ys = CuVector{T}(undef, n)

    num_threads = min(n, 256)
    num_blocks = ceil(Int, total_number / threads)

    @cuda blocks=num_blocks threads=num_threads extract!(ys, xs, bools, indices)

    return ys

end

"Extract those elements of xs where indices[i] is true into ys"
function _extract!(ys::CuVector{T}, xs::CuVector{T}, bools, indices)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i <= length(xs) && bools[i]
        which = indices[i]   # new position
        ys[which] = xs[i]

    end

    return nothing
end
