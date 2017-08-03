function Base.fill!(xs::CuArray, x)
  function kernel(xs, x)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    xs[i] = x
    return
  end
  @cuda (1, length(xs)) kernel(xs, convert(eltype(xs), x))
  return xs
end
