using Base.Cartesian

function cudims(n::Integer)
  threads = min(n, 256)
  ceil(Int, n / threads), threads
end

cudims(a::AbstractArray) = cudims(length(a))

@inline ind2sub_(a::AbstractArray{T,0}, i) where T = ()
@inline ind2sub_(a, i) = Tuple(CartesianIndices(a)[i])

macro cuindex(A)
  quote
    A = $(esc(A))
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    i > length(A) && return
    ind2sub_(A, i)
  end
end


@generated function nindex(i::T, ls::NTuple{N,T}) where {N,T}
  na = one(i)
  quote
    Base.@_inline_meta
    $(foldr((n, els) -> :(i ≤ ls[$n] ? ($n, i) : (i -= ls[$n]; $els)), :($na, $na), one(i):i(N)))
  end
end

@inline function catindex(dim, I::NTuple{N}, shapes) where N
  @inbounds x, i = nindex(I[dim], getindex.(shapes, dim))
  x, ntuple(n -> n == dim ? i : I[n], Val{N})
end

function growdims(dim, x)
  if ndims(x) >= dim
    x
  else
    reshape(x, size.((x,), 1:dim)...)
  end
end

function _cat(dim, dest, xs...)
  function kernel(dim, dest, xs)
    I = @cuindex dest
    @inbounds n, I′ = catindex(dim, Int.(I), size.(xs))
    @inbounds dest[I...] = xs[n][I′...]
    return
  end
  xs = growdims.(dim, xs)
  blk, thr = cudims(dest)
  @cuda blocks=blk threads=thr kernel(dim, dest, xs)
  return dest
end

function Base.cat_t(dims::Integer, T::Type, x::CuArray, xs::CuArray...)
  catdims = Base.dims2cat(dims)
  shape = Base.cat_shape(catdims, (), size.((x, xs...))...)
  dest = Base.cat_similar(x, T, shape)
  _cat(dims, dest, x, xs...)
end

Base.vcat(xs::CuArray...) = cat(xs..., dims=1)
Base.hcat(xs::CuArray...) = cat(xs..., dims=2)


"""
    @sync ex

Run expressions and synchronize the GPU afterwards.

This is useful for timing code that executes asynchronously.
"""
macro sync(ex)
    quote
        local ret = $(esc(ex))
        CUDAdrv.synchronize()
        ret
    end
end
