using Base.Cartesian

function Base.fill!(xs::CuArray, x)
  function kernel(xs, x)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    xs[i] = x
    return
  end
  @cuda (1, length(xs)) kernel(xs, convert(eltype(xs), x))
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
