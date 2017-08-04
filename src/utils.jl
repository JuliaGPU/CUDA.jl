using Base.Cartesian

function nextdivisor(n, d)
  while n % d ≠ 0
    d += 1
  end
  return d
end

function cudims(n::Integer)
  warp = 32
  max = 1024
  if n % warp == 0
    n ÷= warp
    blocks = nextdivisor(n, Base.ceil(Int, n / (max ÷ warp)))
    (blocks, warp*n÷blocks)
  else
    blocks = nextdivisor(n, Base.ceil(Int, n / max))
    (blocks, n ÷ blocks)
  end
end

function Base.fill!(xs::CuArray, x)
  function kernel(xs, x)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    xs[i] = x
    return
  end
  blk, thr = cudims(length(xs))
  @cuda (blk, thr) kernel(xs, convert(eltype(xs), x))
  return xs
end

allequal(x) = true
allequal(x, y, z...) = x == y && allequal(y, z...)

function Base.map!(f, y::CuArray, xs::CuArray...)
  @assert allequal(size.((y, xs...))...)
  y .= f.(xs...)
end

# Break ambiguities with base
Base.map!(f, y::CuArray) =
  invoke(map!, Tuple{Any,CuArray,Vararg{CuArray}}, f, y)
Base.map!(f, y::CuArray, x::CuArray) =
  invoke(map!, Tuple{Any,CuArray,Vararg{CuArray}}, f, y, x)
Base.map!(f, y::CuArray, x1::CuArray, x2::CuArray) =
  invoke(map!, Tuple{Any,CuArray,Vararg{CuArray}}, f, y, x1, x2)

Base.map(f, y::CuArray, xs::CuArray...) = map!(f, similar(y), y, xs...)
