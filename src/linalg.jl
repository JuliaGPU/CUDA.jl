# integration with LinearAlgebra.jl

CuMatOrAdj{T} = Union{CuMatrix, LinearAlgebra.Adjoint{T, <:CuMatrix{T}}, LinearAlgebra.Transpose{T, <:CuMatrix{T}}}
CuOrAdj{T} = Union{CuVecOrMat, LinearAlgebra.Adjoint{T, <:CuVecOrMat{T}}, LinearAlgebra.Transpose{T, <:CuVecOrMat{T}}}


# matrix division

function Base.:\(_A::CuMatOrAdj, _B::CuOrAdj)
    A, B = copy(_A), copy(_B)
    A, ipiv = CUSOLVER.getrf!(A)
    return CUSOLVER.getrs!('N', A, ipiv, B)
end

# patch JuliaLang/julia#40899 to create a CuArray
# (see https://github.com/JuliaLang/julia/pull/41331#issuecomment-868374522)
if VERSION >= v"1.7-"
_zeros(::Type{T}, b::AbstractVector, n::Integer) where {T} = CUDA.zeros(T, max(length(b), n))
_zeros(::Type{T}, B::AbstractMatrix, n::Integer) where {T} = CUDA.zeros(T, max(size(B, 1), n), size(B, 2))
function Base.:\(F::Union{LinearAlgebra.LAPACKFactorizations{<:Any,<:CuArray},
                          Adjoint{<:Any,<:LinearAlgebra.LAPACKFactorizations{<:Any,<:CuArray}}},
                 B::AbstractVecOrMat)
    m, n = size(F)
    if m != size(B, 1)
        throw(DimensionMismatch("arguments must have the same number of rows"))
    end

    TFB = typeof(oneunit(eltype(B)) / oneunit(eltype(F)))
    FF = Factorization{TFB}(F)

    # For wide problem we (often) compute a minimum norm solution. The solution
    # is larger than the right hand side so we use size(F, 2).
    BB = _zeros(TFB, B, n)

    if n > size(B, 1)
        # Underdetermined
        copyto!(view(BB, 1:m, :), B)
    else
        copyto!(BB, B)
    end

    ldiv!(FF, BB)

    # For tall problems, we compute a least squares solution so only part
    # of the rhs should be returned from \ while ldiv! uses (and returns)
    # the complete rhs
    return LinearAlgebra._cut_B(BB, 1:n)
end
end


# qr

using LinearAlgebra: AbstractQ

# AbstractQ's `size` is the size of the full matrix,
# while `Matrix(Q)` only gives the compact Q.
# See JuliaLang/julia#26591 and JuliaGPU/CUDA.jl#969.
CuMatrix{T}(Q::AbstractQ{S}) where {T,S} = convert(CuArray, Matrix{T}(Q))
CuMatrix(Q::AbstractQ{T}) where {T} = CuMatrix{T}(Q)
CuArray{T}(Q::AbstractQ) where {T} = CuMatrix{T}(Q)
CuArray(Q::AbstractQ) = CuMatrix(Q)

function LinearAlgebra.ldiv!(_qr::CUDA.CUSOLVER.CuQR,b::CUDA.CuArray)
  _x = UpperTriangular(_qr.R) \ (_qr.Q' * reshape(b,length(b),1))
  b .= vec(_x)
  CUDA.unsafe_free!(_x)
  return b
end

function LinearAlgebra.ldiv!(x::CUDA.CuArray,_qr::CUDA.CUSOLVER.CuQR,b::CUDA.CuArray)
  _x = UpperTriangular(_qr.R) \ (_qr.Q' * reshape(b,length(b),1))
  x .= vec(_x)
  CUDA.unsafe_free!(_x)
  return x
end
