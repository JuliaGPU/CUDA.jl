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

using Base.PermutedDimsArrays: genperm

function Base.permutedims!(dest::CuArray, src::CuArray, perm)
  function kernel(dest, src, perm)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    I = ind2sub(dest, i)
    @inbounds dest[I...] = src[genperm(I, perm)...]
    return
  end
  blk, thr = cudims(length(dest))
  @cuda (blk, thr) kernel(dest, src, perm)
  return dest
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

# Concatenation

@generated function nindex(i::Int, ls::NTuple{N}) where N
  quote
    Base.@_inline_meta
    $(foldr((n, els) -> :(i ≤ ls[$n] ? ($n, i) : (i -= ls[$n]; $els)), :(-1, -1), 1:N))
  end
end

function catindex(dim, I::NTuple{N}, shapes) where N
  @inbounds x, i = nindex(I[dim], getindex.(shapes, dim))
  x, ntuple(n -> n == dim ? i : I[n], Val{N})
end

function _cat(dim, dest, xs...)
  function kernel(dim, dest, xs)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    I = ind2sub(dest, i)
    n, I′ = catindex(dim, I, size.(xs))
    @inbounds dest[I...] = xs[n][I′...]
    return
  end
  blk, thr = cudims(length(dest))
  @cuda (blk, thr) kernel(dim, dest, xs)
  return dest
end

function Base.cat_t(dims::Integer, T::Type, x::CuArray, xs::CuArray...)
  catdims = Base.dims2cat(dims)
  shape = Base.cat_shape(catdims, (), size.((x, xs...))...)
  dest = Base.cat_similar(x, T, shape)
  _cat(dims, dest, x, xs...)
end

Base.vcat(xs::CuArray...) = cat(1, xs...)
Base.hcat(xs::CuArray...) = cat(2, xs...)
