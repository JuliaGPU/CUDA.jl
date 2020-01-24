# integration with LinearAlgebra.jl

CuMatOrAdj{T} = Union{CuMatrix, LinearAlgebra.Adjoint{T, <:CuMatrix{T}}, LinearAlgebra.Transpose{T, <:CuMatrix{T}}}
CuOrAdj{T} = Union{CuVecOrMat, LinearAlgebra.Adjoint{T, <:CuVecOrMat{T}}, LinearAlgebra.Transpose{T, <:CuVecOrMat{T}}}

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


# matrix division

function Base.:\(_A::CuMatOrAdj, _B::CuOrAdj)
    A, B = copy(_A), copy(_B)
    A, ipiv = CuArrays.CUSOLVER.getrf!(A)
    return CuArrays.CUSOLVER.getrs!('N', A, ipiv, B)
end


# matrix multiplication

function generic_matmatmul!(C::AbstractVecOrMat{R}, A::AbstractVecOrMat{T}, B::AbstractVecOrMat{S}) where {T,S,R}
    if size(A,2) != size(B,1)
        throw(DimensionMismatch("matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))"))
    end
    if size(C,1) != size(A,1) || size(C,2) != size(B,2)
        throw(DimensionMismatch("result C has dimensions $(size(C)), needs $((size(A,1),size(B,2)))"))
    end
    if isempty(A) || isempty(B)
        return fill!(C, zero(R))
    end

    function kernel(C, A, B)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        j = (blockIdx().y-1) * blockDim().y + threadIdx().y

        if i <= size(A,1) && j <= size(B,2)
            z2 = zero(A[i, 1]*B[1, j] + A[i, 1]*B[1, j])
            Ctmp = convert(promote_type(R, typeof(z2)), z2)
            for k in 1:size(A,2)
                Ctmp += A[i, k]*B[k, j]
            end
            C[i,j] = Ctmp
        end

        return
    end

    max_threads = 256
    threads_x = min(max_threads, size(C,1))
    threads_y = min(max_threads ÷ threads_x, size(C,2))
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (size(C,1), size(C,2)) ./ threads)

    @cuda threads=threads blocks=blocks kernel(C, A, B)

    C
end

LinearAlgebra.mul!(C::CuVecOrMat, A::CuVecOrMat, B::CuVecOrMat) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::CuVecOrMat, A::CuVecOrMat, B::LinearAlgebra.Adjoint{<:Any, <:CuVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::CuVecOrMat, A::CuVecOrMat, B::LinearAlgebra.Transpose{<:Any, <:CuVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::CuVecOrMat, A::LinearAlgebra.Adjoint{<:Any, <:CuVecOrMat}, B::CuVecOrMat) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::CuVecOrMat, A::LinearAlgebra.Transpose{<:Any, <:CuVecOrMat}, B::CuVecOrMat) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::CuVecOrMat, A::LinearAlgebra.Transpose{<:Any, <:CuVecOrMat}, B::LinearAlgebra.Adjoint{<:Any, <:CuVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::CuVecOrMat, A::LinearAlgebra.Adjoint{<:Any, <:CuVecOrMat}, B::LinearAlgebra.Transpose{<:Any, <:CuVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::CuVecOrMat, A::LinearAlgebra.Adjoint{<:Any, <:CuVecOrMat}, B::LinearAlgebra.Adjoint{<:Any, <:CuVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::CuVecOrMat, A::LinearAlgebra.Transpose{<:Any, <:CuVecOrMat}, B::LinearAlgebra.Transpose{<:Any, <:CuVecOrMat}) = generic_matmatmul!(C, A, B)

function generic_rmul!(X::CuArray, s::Number)
    function kernel(X, s)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        @inbounds X[i] *= s
        return
    end
    @cuda blocks=length(X) kernel(X, s)
    X
end

LinearAlgebra.rmul!(A::CuArray, b::Number) = generic_rmul!(A, b)

function generic_lmul!(s::Number, X::CuArray)
    function kernel(s, X)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        @inbounds X[i] = s*X[i]
        return
    end
    @cuda blocks=length(X) kernel(s, X)
    X
end

LinearAlgebra.lmul!(a::Number, B::CuArray) = generic_lmul!(a, B)
