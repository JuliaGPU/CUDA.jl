# Reduction across dimension

function mapreducedim_kernel(f, op, R, A, range)
  I = @cuindex R
  newrange = map((r, i) -> r == nothing ? i : r, range, I)
  for I′ in CartesianRange(newrange)
    @inbounds R[I...] = op(R[I...], f(A[I′]))
  end
end

function Base._mapreducedim!(f, op, R::CuArray, A::CuArray)
  range = ifelse.(length.(indices(R)) .== 1, indices(A), nothing)
  blk, thr = cudims(R)
  @cuda (blk, thr) mapreducedim_kernel(f, op, R, A, range)
  return R
end

# Reduction to scalar

@inline function reduce_warp(op, val::T)::T where {T}
  offset = CUDAnative.warpsize() ÷ UInt32(2)
  while offset > 0
    val = op(val, shfl_down(val, offset))
    offset ÷= UInt32(2)
  end
  return val
end

@inline function reduce_block(op, v0::T, val::T)::T where {T}
  shared = @cuStaticSharedMem(T, 32)
  wid  = div(threadIdx().x-UInt32(1), CUDAnative.warpsize()) + UInt32(1)
  lane = rem(threadIdx().x-UInt32(1), CUDAnative.warpsize()) + UInt32(1)
  val = reduce_warp(op, val)
  if lane == 1
    @inbounds shared[wid] = val
  end
  sync_threads()
  @inbounds val = (threadIdx().x <= fld(blockDim().x, CUDAnative.warpsize())) ? shared[lane] : v0
  if wid == 1
    val = reduce_warp(op, val)
  end
  return val
end

function reduce_grid(op, v0::T, input::CuDeviceArray{T}, output::CuDeviceArray{T},
                     len::Integer) where {T}
  val = v0
  i = (blockIdx().x-UInt32(1)) * blockDim().x + threadIdx().x
  step = blockDim().x * gridDim().x
  while i <= len
    @inbounds val = op(val, input[i])
    i += step
  end
  val = reduce_block(op, v0, val)
  if threadIdx().x == UInt32(1)
    @inbounds output[blockIdx().x] = val
  end
  return
end

function reduce_cudim(n)
  threads = 512
  blocks = Base.min((n+ threads - 1) ÷ threads, 1024)
  return threads, blocks
end

function _reduce(op, v0, input, output,
                 dim = reduce_cudim(length(input)))
  threads, blocks = dim
  if length(output) < blocks
    throw(ArgumentError("output array too small, should be at least $blocks elements"))
  end
  @cuda (blocks,threads) reduce_grid(op, v0, input, output, Int32(length(input)))
  @cuda (1,1024) reduce_grid(op, v0, output, output, Int32(blocks))
  return
end

# TODO: first elem as v0

function Base.reduce(f, v0::T, xs::CuArray{T}) where T
  dim = reduce_cudim(length(xs))
  scratch = similar(xs, dim[2])
  _reduce(f, v0, xs, scratch, dim)
  return _getindex(scratch, 1)
end

Base.reduce(f, v0, xs::CuArray) =
  reduce(f, convert(eltype(xs), v0), xs)

Base.sum(xs::CuArray) = reduce(+, 0, xs)
Base.prod(xs::CuArray) = reduce(*, 1, xs)
