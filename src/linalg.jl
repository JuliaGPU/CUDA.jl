## generic linear algebra routines

function LinearAlgebra.tril!(A::CuMatrix{T}, d::Integer = 0) where T
  function kernel!(_A, _d)
    li = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    m, n = size(_A)
    if 0 < li <= m*n
      i, j = Tuple(CartesianIndices(_A)[li])
      if i < j - _d
        _A[i, j] = 0
      end
    end
    return nothing
  end

  blk, thr = cudims(A)
  @cuda blocks=blk threads=thr kernel!(A, d)
  return A
end

function LinearAlgebra.triu!(A::CuMatrix{T}, d::Integer = 0) where T
  function kernel!(_A, _d)
    li = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    m, n = size(_A)
    if 0 < li <= m*n
      i, j = Tuple(CartesianIndices(_A)[li])
      if j < i + _d
        _A[i, j] = 0
      end
    end
    return nothing
  end

  blk, thr = cudims(A)
  @cuda blocks=blk threads=thr kernel!(A, d)
  return A
end

function LinearAlgebra.copy_transpose!(dst::CuArray, src::CuArray)
  function kernel(dst, src)
    I = @cuindex dst
    dst[I...] = src[reverse(I)...]
    return
  end
  blk, thr = cudims(dst)
  @cuda blocks=blk threads=thr kernel(dst, src)
  return dst
end

# Matrix division

const CuOrAdj = Union{CuVecOrMat, LinearAlgebra.Adjoint{T, CuVecOrMat{T}}, LinearAlgebra.Transpose{T, CuVecOrMat{T}}} where {T<:AbstractFloat}

function Base.:\(_A::AT1, _B::AT2) where {AT1<:CuOrAdj, AT2<:CuOrAdj}
    A, B = copy(_A), copy(_B)
    A, ipiv = CuArrays.CUSOLVER.getrf!(A)
    return CuArrays.CUSOLVER.getrs!('N', A, ipiv, B)
end
