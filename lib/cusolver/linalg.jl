# implementation of LinearAlgebra interfaces

using LinearAlgebra
using LinearAlgebra: BlasComplex, BlasFloat, BlasReal, checksquare
using ..CUBLAS: CublasFloat

function copy_cublasfloat(As...)
    eltypes = eltype.(As)
    promoted_eltype = reduce(promote_type, eltypes)
    cublasfloat = promote_type(Float32, promoted_eltype)
    if !(cublasfloat <: CublasFloat)
        throw(ArgumentError("cannot promote eltypes $eltypes to a CUBLAS float type"))
    end
    out = _copywitheltype(cublasfloat, As...)
    length(out) == 1 && return first(out)
    return out
end

_copywitheltype(::Type{T}, As...) where {T} = map(A -> copyto!(similar(A, T), A), As)

# matrix division

const CuMatOrAdj{T} = Union{CuMatrix{T},
                            LinearAlgebra.Adjoint{T, <:CuMatrix{T}},
                            LinearAlgebra.Transpose{T, <:CuMatrix{T}}}
const CuOrAdj{T} = Union{CuVecOrMat{T},
                         LinearAlgebra.Adjoint{T, <:CuVecOrMat{T}},
                         LinearAlgebra.Transpose{T, <:CuVecOrMat{T}}}

function Base.:\(_A::CuMatOrAdj, _B::CuOrAdj)
    A, B = copy_cublasfloat(_A, _B)
    T = eltype(A)
    n,m = size(A)
    if n < m
        # LQ decomposition
        At = CuMatrix(A')
        F, tau = geqrf!(At)  # A = RᴴQᴴ
        if B isa CuVector{T}
            CUBLAS.trsv!('U', 'C', 'N', view(F,1:n,1:n), B)
            X = CUDA.zeros(T, m)
            view(X, 1:n) .= B
        else
            CUBLAS.trsm!('L', 'U', 'C', 'N', one(T), view(F,1:n,1:n), B)
            p = size(B, 2)
            X = CUDA.zeros(T, m, p)
            view(X, 1:n, :) .= B
        end
        ormqr!('L', 'N', F, tau, X)
    elseif n == m
        # LU decomposition with partial pivoting
        F, p, info = getrf!(A)  # PA = LU
        X = getrs!('N', F, p, B)
    else
        # QR decomposition
        F, tau = geqrf!(A)  # A = QR
        ormqr!('L', 'C', F, tau, B)
        if B isa CuVector{T}
            X = B[1:m]
            CUBLAS.trsv!('U', 'N', 'N', view(F,1:m,1:m), X)
        else
            X = B[1:m,:]
            CUBLAS.trsm!('L', 'U', 'N', 'N', one(T), view(F,1:m,1:m), X)
        end
    end
    return X
end

function Base.:\(_A::Symmetric{<:Any,<:CuMatOrAdj}, _B::CuOrAdj)
    uplo = A.uplo
    A, B = copy_cublasfloat(_A.data, _B)

    # LDLᴴ decomposition with partial pivoting
    factors, ipiv, info = sytrf!(uplo, A)
    ipiv = CuVector{Int64}(ipiv)
    X = sytrs!(uplo, factors, ipiv, B)
    return X
end

# patch JuliaLang/julia#40899 to create a CuArray
# (see https://github.com/JuliaLang/julia/pull/41331#issuecomment-868374522)
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

# eigenvalues

function LinearAlgebra.eigen(A::Symmetric{T,<:CuMatrix}) where {T<:BlasReal}
    A2 = copy(A.data)
    Eigen(syevd!('V', 'U', A2)...)
end
function LinearAlgebra.eigen(A::Hermitian{T,<:CuMatrix}) where {T<:BlasComplex}
    A2 = copy(A.data)
    Eigen(heevd!('V', 'U', A2)...)
end
function LinearAlgebra.eigen(A::Hermitian{T,<:CuMatrix}) where {T<:BlasReal}
    eigen(Symmetric(A))
end

function LinearAlgebra.eigen(A::CuMatrix{T}) where {T<:BlasReal}
    A2 = copy(A)
    issymmetric(A) ? Eigen(syevd!('V', 'U', A2)...) : error("GPU eigensolver supports only Hermitian or Symmetric matrices.")
end
function LinearAlgebra.eigen(A::CuMatrix{T}) where {T<:BlasComplex}
    A2 = copy(A)
    ishermitian(A) ? Eigen(heevd!('V', 'U', A2)...) : error("GPU eigensolver supports only Hermitian or Symmetric matrices.")
end


# factorizations

using LinearAlgebra: Factorization, AbstractQ, QRCompactWY, QRCompactWYQ, QRPackedQ

## QR

LinearAlgebra.qr!(A::CuMatrix{T}) where T = QR(geqrf!(A::CuMatrix{T})...)

# conversions
CuMatrix(F::Union{QR,QRCompactWY}) = CuArray(AbstractArray(F))
CuArray(F::Union{QR,QRCompactWY}) = CuMatrix(F)
CuMatrix(F::QRPivoted) = CuArray(AbstractArray(F))
CuArray(F::QRPivoted) = CuMatrix(F)

function LinearAlgebra.ldiv!(_qr::QR, b::CuVector)
    m,n = size(_qr)
    _x = UpperTriangular(_qr.R[1:min(m,n), 1:n]) \ ((_qr.Q' * b)[1:n])
    b[1:n] .= _x
    unsafe_free!(_x)
    return b[1:n]
end

function LinearAlgebra.ldiv!(_qr::QR, B::CuMatrix)
    m,n = size(_qr)
    _x = UpperTriangular(_qr.R[1:min(m,n), 1:n]) \ ((_qr.Q' * B)[1:n, 1:size(B, 2)])
    B[1:n, 1:size(B, 2)] .= _x
    unsafe_free!(_x)
    return B[1:n, 1:size(B, 2)]
end

function LinearAlgebra.ldiv!(x::CuArray, _qr::QR, b::CuArray)
    _x = ldiv!(_qr, b)
    x .= vec(_x)
    unsafe_free!(_x)
    return x
end

# AbstractQ's `size` is the size of the full matrix,
# while `Matrix(Q)` only gives the compact Q.
# See JuliaLang/julia#26591 and JuliaGPU/CUDA.jl#969.
CuArray(Q::AbstractQ) = CuMatrix(Q)
CuArray{T}(Q::AbstractQ) where {T} = CuMatrix{T}(Q)
CuMatrix(Q::AbstractQ{T}) where {T} = CuMatrix{T}(Q)
CuMatrix{T}(Q::QRPackedQ{S}) where {T,S} =
    CuMatrix{T}(lmul!(Q, CuMatrix{S}(I, size(Q, 1), min(size(Q.factors)...))))
CuMatrix{T, B}(Q::QRPackedQ{S}) where {T, B, S} = CuMatrix{T}(Q)
CuMatrix{T}(Q::QRCompactWYQ) where {T} = error("QRCompactWY format is not supported")
# avoid the CPU array in the above mul!
Matrix{T}(Q::QRPackedQ{S,<:CuArray,<:CuArray}) where {T,S} = Array(CuMatrix{T}(Q))
Matrix{T}(Q::QRCompactWYQ{S,<:CuArray,<:CuArray}) where {T,S} = Array(CuMatrix{T}(Q))

function Base.getindex(Q::QRPackedQ{<:Any, <:CuArray}, ::Colon, j::Int)
    y = CUDA.zeros(eltype(Q), size(Q, 2))
    y[j] = 1
    lmul!(Q, y)
end

# multiplication by Q
LinearAlgebra.lmul!(A::QRPackedQ{T,<:CuArray,<:CuArray},
                    B::CuVecOrMat{T}) where {T<:BlasFloat} =
    ormqr!('L', 'N', A.factors, A.τ, B)
LinearAlgebra.lmul!(adjA::Adjoint{T,<:QRPackedQ{T,<:CuArray,<:CuArray}},
                    B::CuVecOrMat{T}) where {T<:BlasReal} =
    ormqr!('L', 'T', parent(adjA).factors, parent(adjA).τ, B)
LinearAlgebra.lmul!(adjA::Adjoint{T,<:QRPackedQ{T,<:CuArray,<:CuArray}},
                    B::CuVecOrMat{T}) where {T<:BlasComplex} =
    ormqr!('L', 'C', parent(adjA).factors, parent(adjA).τ, B)
LinearAlgebra.lmul!(trA::Transpose{T,<:QRPackedQ{T,<:CuArray,<:CuArray}},
                    B::CuVecOrMat{T}) where {T<:BlasFloat} =
    ormqr!('L', 'T', parent(trA).factors, parent(trA).τ, B)

LinearAlgebra.rmul!(A::CuVecOrMat{T},
                    B::QRPackedQ{T,<:CuArray,<:CuArray}) where {T<:BlasFloat} =
    ormqr!('R', 'N', B.factors, B.τ, A)
LinearAlgebra.rmul!(A::CuVecOrMat{T},
                    adjB::Adjoint{<:Any,<:QRPackedQ{T,<:CuArray,<:CuArray}}) where {T<:BlasReal} =
    ormqr!('R', 'T', parent(adjB).factors, parent(adjB).τ, A)
LinearAlgebra.rmul!(A::CuVecOrMat{T},
                    adjB::Adjoint{<:Any,<:QRPackedQ{T,<:CuArray,<:CuArray}}) where {T<:BlasComplex} =
    ormqr!('R', 'C', parent(adjB).factors, parent(adjB).τ, A)
LinearAlgebra.rmul!(A::CuVecOrMat{T},
                    trA::Transpose{<:Any,<:QRPackedQ{T,<:CuArray,<:CuArray}}) where {T<:BlasFloat} =
    ormqr!('R', 'T', parent(trA).factors, parent(adjB).τ, A)


## SVD

abstract type SVDAlgorithm end
struct QRAlgorithm <: SVDAlgorithm end
struct JacobiAlgorithm <: SVDAlgorithm end
struct ApproximateAlgorithm <: SVDAlgorithm end

const CuMatOrBatched{T} = Union{CuMatrix{T}, CuArray{T,3}} where T

LinearAlgebra.svd!(A::CuMatOrBatched{T}; full::Bool=false,
                   alg::SVDAlgorithm=JacobiAlgorithm()) where {T} =
    _svd!(A, full, alg)
LinearAlgebra.svd(A::CuMatOrBatched; full=false, alg::SVDAlgorithm=JacobiAlgorithm()) =
    _svd!(copy_cublasfloat(A), full, alg)

_svd!(A::CuMatOrBatched, full::Bool, alg::SVDAlgorithm) =
    throw(ArgumentError("Unsupported value for `alg` keyword."))
function _svd!(A::CuMatrix{T}, full::Bool, alg::QRAlgorithm) where T
    U, S, Vt = gesvd!(full ? 'A' : 'S', full ? 'A' : 'S', A)
    return SVD(U, S, Vt)
end
function _svd!(A::CuMatrix{T}, full::Bool, alg::JacobiAlgorithm) where T
    U, S, V = gesvdj!('V', Int(!full), A)
    return SVD(U, S, V')
end
function _svd!(A::CuArray{T,3}, full::Bool, alg::JacobiAlgorithm) where T
    U, S, V = gesvdj!('V', A)
    return CuSVDBatched(U, S, V)
end

function _svd!(A::CuArray{T,3}, full::Bool, alg::ApproximateAlgorithm; rank::Int=min(size(A,1), size(A,2))) where T
    U, S, V = gesvda!('V', A; rank=rank)
    return CuSVDBatched(U, S, V)
end

struct CuSVDBatched{T,Tr,A<:AbstractArray{T,3}} <: LinearAlgebra.Factorization{T}
    U::CuArray{T,3}
    S::CuMatrix{Tr}
    V::A
end

# iteration for destructuring into components
Base.iterate(S::CuSVDBatched) = (S.U, Val(:S))
Base.iterate(S::CuSVDBatched, ::Val{:S}) = (S.S, Val(:V))
Base.iterate(S::CuSVDBatched, ::Val{:V}) = (S.V, Val(:done))
Base.iterate(S::CuSVDBatched, ::Val{:done}) = nothing

LinearAlgebra.svdvals!(A::CuMatOrBatched{T}; alg::SVDAlgorithm=JacobiAlgorithm()) where {T} =
    _svdvals!(A, alg)
LinearAlgebra.svdvals(A::CuMatOrBatched; alg::SVDAlgorithm=JacobiAlgorithm()) =
    _svdvals!(copy_cublasfloat(A), alg)

_svdvals!(A::CuMatOrBatched{T}, alg::SVDAlgorithm) where T =
    throw(ArgumentError("Unsupported value for `alg` keyword."))
_svdvals!(A::CuMatrix{T}, alg::QRAlgorithm) where T = gesvd!('N', 'N', A::CuMatrix{T})[2]
_svdvals!(A::CuMatrix{T}, alg::JacobiAlgorithm) where T = gesvdj!('N', 1, A::CuMatOrBatched{T})[2]
_svdvals!(A::CuArray{T,3}, alg::JacobiAlgorithm) where T = gesvdj!('N', A::CuArray{T,3})[2]
_svdvals!(A::CuArray{T,3}, alg::ApproximateAlgorithm; rank=min(size(A,1), size(A,2))) where T = gesvda!('N', A::CuArray{T,3}; rank=rank)[2]

### opnorm2, enabled by svdvals

function LinearAlgebra.opnorm2(A::CuMatrix{T}) where {T}
    # The implementation in Base.LinearAlgebra can be reused verbatim, but it uses a scalar
    # index to access the larges singular value, so must be wrapped in @allowscalar
    return @allowscalar invoke(LinearAlgebra.opnorm2, Tuple{AbstractMatrix{T}}, A)
end


## LU

function _check_lu_success(info, allowsingular)
    if info < 0 # zero pivot error from unpivoted LU
        LinearAlgebra.checknozeropivot(-info)
    else
        allowsingular || LinearAlgebra.checknonsingular(info)
    end
end

function LinearAlgebra.lu!(A::StridedCuMatrix{T}, ::RowMaximum;
                           check::Bool=true, allowsingular::Bool=false) where {T}
    lpt = getrf!(A)
    check && _check_lu_success(lpt[3], allowsingular)
    return LU(lpt[1], lpt[2], Int(lpt[3]))
end

# GPU-compatible accessors of the LU decomposition properties
function Base.getproperty(F::LU{T,<:StridedCuMatrix}, d::Symbol) where T
    m, n = size(F)
    if d === :L
        L = tril!(getfield(F, :factors)[1:m, 1:min(m,n)])
        L[1:m+1:end] .= one(T)   # set the diagonal (linear indexing trick)
        return L
    else
        invoke(getproperty, Tuple{LU{T}, Symbol}, F, d)
    end
end

# LAPACK's pivoting sequence needs to be iterated sequentially...
# TODO: figure out a GPU-compatible way to get the permutation matrix
LinearAlgebra.ipiv2perm(v::CuVector{T}, maxi::Integer) where T =
    LinearAlgebra.ipiv2perm(Array(v), maxi)

function LinearAlgebra.ldiv!(F::LU{T,<:StridedCuMatrix{T}}, B::CuVecOrMat{T}) where {T}
    return getrs!('N', F.factors, F.ipiv, B)
end

# LinearAlgebra.jl defines a comparable method with these restrictions on T, so we need
#   to define with the same type parameters to resolve method ambiguity for these cases
function LinearAlgebra.ldiv!(F::LU{T,<:StridedCuMatrix{T}}, B::CuVecOrMat{T}) where T <: BlasFloat
    return getrs!('N', F.factors, F.ipiv, B)
end


## cholesky

function LinearAlgebra.cholesky(A::LinearAlgebra.RealHermSymComplexHerm{<:Real,<:CuMatrix},
                                ::Val{false}=Val(false); check::Bool = true)
    C, info = LinearAlgebra._chol!(copy(parent(A)), A.uplo == 'U' ? UpperTriangular : LowerTriangular)
    return Cholesky(C.data, A.uplo, info)
end

LinearAlgebra.cholcopy(A::LinearAlgebra.RealHermSymComplexHerm{<:Any,<:CuArray}) =
    copyto!(similar(A, LinearAlgebra.choltype(A)), A)

## inv

function LinearAlgebra.inv(A::StridedCuMatrix{T}) where T <: BlasFloat
    n = checksquare(A)
    F = copy(A)
    factors, ipiv, info = getrf!(F)
    B = CuMatrix{T}(I, n, n)
    getrs!('N', factors, ipiv, B)
    return B
end

function LinearAlgebra.inv(A::Symmetric{T,<:StridedCuMatrix{T}}) where T <: BlasFloat
    n = checksquare(A)
    F = copy(A.data)
    factors, ipiv, info = sytrf!(A.uplo, F)
    ipiv = CuVector{Int64}(ipiv)
    B = CuMatrix{T}(I, n, n)
    sytrs!(A.uplo, factors, ipiv, B)
    return B
end

for (triangle, uplo, diag) in ((:LowerTriangular, 'L', 'N'),
                               (:UnitLowerTriangular, 'L', 'U'),
                               (:UpperTriangular, 'U', 'N'),
                               (:UnitUpperTriangular, 'U', 'U'))
    @eval begin
        function LinearAlgebra.inv(A::$triangle{T,<:StridedCuMatrix{T}}) where T <: BlasFloat
            n = checksquare(A)
            B = copy(A.data)
            trtri!(uplo, diag, B)
            return B
        end
    end
end
