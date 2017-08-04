# Reduction across dimension

function mapreducedim_kernel(f, op, R, A, range)
  i = (blockIdx().x-1) * blockDim().x + threadIdx().x
  I = ind2sub(R, i)
  newrange = map((r, i) -> r == nothing ? i : r, range, I)
  for I′ in CartesianRange(newrange)
    @inbounds R[I...] = op(R[I...], f(A[I′]))
  end
end

function Base._mapreducedim!(f, op, R::CuArray, A::CuArray)
  range = ifelse.(length.(indices(R)) .== 1, indices(A), nothing)
  blk, thr = cudims(length(R))
  @cuda (blk, thr) mapreducedim_kernel(f, op, R, A, range)
  return R
end
