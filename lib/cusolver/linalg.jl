# implementation of LinearAlgebra interfaces

using LinearAlgebra


# matrix division

CuMatOrAdj{T} = Union{CuMatrix,
                      LinearAlgebra.Adjoint{T, <:CuMatrix{T}},
                      LinearAlgebra.Transpose{T, <:CuMatrix{T}}}
CuOrAdj{T} = Union{CuVecOrMat,
                   LinearAlgebra.Adjoint{T, <:CuVecOrMat{T}},
                   LinearAlgebra.Transpose{T, <:CuVecOrMat{T}}}

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


# QR factorization

using LinearAlgebra: AbstractQ

struct CuQR{T,S<:AbstractMatrix} <: LinearAlgebra.Factorization{T}
    factors::S
    τ::CuVector{T}
    CuQR{T,S}(factors::AbstractMatrix{T}, τ::CuVector{T}) where {T,S<:AbstractMatrix} = new(factors, τ)
end

struct CuQRPackedQ{T,S<:AbstractMatrix} <: AbstractQ{T}
    factors::CuMatrix{T}
    τ::CuVector{T}
    CuQRPackedQ{T,S}(factors::AbstractMatrix{T}, τ::CuVector{T}) where {T,S<:AbstractMatrix} = new(factors, τ)
end

CuQR(factors::AbstractMatrix{T}, τ::CuVector{T}) where {T} = CuQR{T,typeof(factors)}(factors, τ)
CuQRPackedQ(factors::AbstractMatrix{T}, τ::CuVector{T}) where {T} = CuQRPackedQ{T,typeof(factors)}(factors, τ)

LinearAlgebra.qr!(A::CuMatrix{T}) where T = CuQR(geqrf!(A::CuMatrix{T})...)
Base.size(A::CuQR) = size(A.factors)
Base.size(A::CuQRPackedQ, dim::Integer) = 0 < dim ? (dim <= 2 ? size(A.factors, 1) : 1) : throw(BoundsError())
CUDA.CuMatrix(A::CuQRPackedQ) = orgqr!(copy(A.factors), A.τ)
CUDA.CuArray(A::CuQRPackedQ) = CuMatrix(A)
Base.Matrix(A::CuQRPackedQ) = Matrix(CuMatrix(A))

function Base.getproperty(A::CuQR, d::Symbol)
    m, n = size(getfield(A, :factors))
    if d == :R
        return triu!(A.factors[1:min(m, n), 1:n])
    elseif d == :Q
        return CuQRPackedQ(A.factors, A.τ)
    else
        getfield(A, d)
    end
end

# iteration for destructuring into components
Base.iterate(S::CuQR) = (S.Q, Val(:R))
Base.iterate(S::CuQR, ::Val{:R}) = (S.R, Val(:done))
Base.iterate(S::CuQR, ::Val{:done}) = nothing

# Apply changes Q from the left
LinearAlgebra.lmul!(A::CuQRPackedQ{T,S}, B::CuVecOrMat{T}) where {T<:Number, S<:CuMatrix} =
    ormqr!('L', 'N', A.factors, A.τ, B)
LinearAlgebra.lmul!(adjA::Adjoint{T,<:CuQRPackedQ{T,S}}, B::CuVecOrMat{T}) where {T<:Real, S<:CuMatrix} =
    ormqr!('L', 'T', parent(adjA).factors, parent(adjA).τ, B)
LinearAlgebra.lmul!(adjA::Adjoint{T,<:CuQRPackedQ{T,S}}, B::CuVecOrMat{T}) where {T<:Complex, S<:CuMatrix} =
    ormqr!('L', 'C', parent(adjA).factors, parent(adjA).τ, B)
LinearAlgebra.lmul!(trA::Transpose{T,<:CuQRPackedQ{T,S}}, B::CuVecOrMat{T}) where {T<:Number, S<:CuMatrix} =
    ormqr!('L', 'T', parent(trA).factors, parent(trA).τ, B)

function Base.getindex(A::CuQRPackedQ{T, S}, i::Integer, j::Integer) where {T, S}
    assertscalar("CuQRPackedQ getindex")
    x = CUDA.zeros(T, size(A, 2))
    x[j] = 1
    lmul!(A, x)
    return x[i]
end

function Base.show(io::IO, F::CuQR)
    println(io, "$(typeof(F)) with factors Q and R:")
    show(io, F.Q)
    println(io)
    show(io, F.R)
end

# https://github.com/JuliaLang/julia/pull/32887
LinearAlgebra.det(Q::CuQRPackedQ{<:Real}) = isodd(count(!iszero, Q.τ)) ? -1 : 1
LinearAlgebra.det(Q::CuQRPackedQ) = prod(τ -> iszero(τ) ? one(τ) : -sign(τ)^2, Q.τ)

# AbstractQ's `size` is the size of the full matrix,
# while `Matrix(Q)` only gives the compact Q.
# See JuliaLang/julia#26591 and JuliaGPU/CUDA.jl#969.
CuMatrix{T}(Q::AbstractQ{S}) where {T,S} = convert(CuArray, Matrix{T}(Q))
CuMatrix(Q::AbstractQ{T}) where {T} = CuMatrix{T}(Q)
CuArray{T}(Q::AbstractQ) where {T} = CuMatrix{T}(Q)
CuArray(Q::AbstractQ) = CuMatrix(Q)

function LinearAlgebra.ldiv!(_qr::CuQR, b::CuArray)
    _x = UpperTriangular(_qr.R) \ (_qr.Q' * reshape(b,length(b),1))
    b .= vec(_x)
    unsafe_free!(_x)
    return b
end

function LinearAlgebra.ldiv!(x::CuArray,_qr::CuQR, b::CuArray)
    _x = UpperTriangular(_qr.R) \ (_qr.Q' * reshape(b,length(b),1))
    x .= vec(_x)
    unsafe_free!(_x)
    return x
end


# Singular Value Decomposition

struct CuSVD{T,Tr,A<:AbstractMatrix{T}} <: LinearAlgebra.Factorization{T}
    U::CuMatrix{T}
    S::CuVector{Tr}
    V::A
end

# iteration for destructuring into components
Base.iterate(S::CuSVD) = (S.U, Val(:S))
Base.iterate(S::CuSVD, ::Val{:S}) = (S.S, Val(:V))
Base.iterate(S::CuSVD, ::Val{:V}) = (S.V, Val(:done))
Base.iterate(S::CuSVD, ::Val{:done}) = nothing

@inline function Base.getproperty(S::CuSVD, s::Symbol)
    if s === :Vt
        return getfield(S, :V)'
    else
        return getfield(S, s)
    end
end

abstract type SVDAlgorithm end
struct QRAlgorithm <: SVDAlgorithm end
struct JacobiAlgorithm <: SVDAlgorithm end

LinearAlgebra.svd!(A::CuMatrix{T}; full::Bool=false,
                   alg::SVDAlgorithm=JacobiAlgorithm()) where {T} =
    _svd!(A, full, alg)
LinearAlgebra.svd(A::CuMatrix; full=false, alg::SVDAlgorithm=JacobiAlgorithm()) =
    _svd!(copy(A), full, alg)

_svd!(A::CuMatrix{T}, full::Bool, alg::SVDAlgorithm) where T =
    throw(ArgumentError("Unsupported value for `alg` keyword."))
function _svd!(A::CuMatrix{T}, full::Bool, alg::QRAlgorithm) where T
    U, s, Vt = gesvd!(full ? 'A' : 'S', full ? 'A' : 'S', A::CuMatrix{T})
    return CuSVD(U, s, Vt')
end
function _svd!(A::CuMatrix{T}, full::Bool, alg::JacobiAlgorithm) where T
    return CuSVD(gesvdj!('V', Int(!full), A::CuMatrix{T})...)
end

LinearAlgebra.svdvals!(A::CuMatrix{T}; alg::SVDAlgorithm=JacobiAlgorithm()) where {T} =
    _svdvals!(A, alg)
LinearAlgebra.svdvals(A::CuMatrix; alg::SVDAlgorithm=JacobiAlgorithm()) =
    _svdvals!(copy(A), alg)

_svdvals!(A::CuMatrix{T}, alg::SVDAlgorithm) where T =
    throw(ArgumentError("Unsupported value for `alg` keyword."))
_svdvals!(A::CuMatrix{T}, alg::QRAlgorithm) where T = gesvd!('N', 'N', A::CuMatrix{T})[2]
_svdvals!(A::CuMatrix{T}, alg::JacobiAlgorithm) where T = gesvdj!('N', 1, A::CuMatrix{T})[2]

if VERSION >= v"1.8-"
    function LinearAlgebra.cholesky(A::LinearAlgebra.RealHermSymComplexHerm{<:Real,<:CuMatrix},
             ::Val{false}=Val(false); check::Bool = true)
        C, info = LinearAlgebra._chol!(copy(parent(A)), A.uplo == 'U' ? UpperTriangular : LowerTriangular)
        return Cholesky(C.data, A.uplo, info)
    end
end


# LU decomposition

function LinearAlgebra.lu!(A::StridedCuMatrix{T}, ::RowMaximum; check::Bool = true) where {T<:CublasFloat}
    lpt = getrf!(A)
    check && LinearAlgebra.checknonsingular(lpt[3])
    return LU{T,typeof(A)}(lpt[1], lpt[2], lpt[3])
end

# GPU-compatible accessors of the LU decomposition properties
# TODO: upstream a GPU-compatible (broadcasting) way to set the diagonal
function Base.getproperty(F::LU{T,<:StridedCuMatrix}, d::Symbol) where T
    m, n = size(F)
    if d === :L
        L = tril!(getfield(F, :factors)[1:m, 1:min(m,n)])
        L[1:min(m,n)+1:end] .= one(T)   # set the diagonal (linear indexing trick)
        return L
    elseif d === :U
        return triu!(getfield(F, :factors)[1:min(m,n), 1:n])
    elseif d === :p
        return LinearAlgebra.ipiv2perm(getfield(F, :ipiv), m)
    elseif d === :P
        return Matrix{T}(I, m, m)[:,invperm(F.p)]
    else
        getfield(F, d)
    end
end
