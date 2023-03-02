# implementation of LinearAlgebra interfaces

using LinearAlgebra
using LinearAlgebra: BlasComplex, BlasFloat, BlasReal
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

const CuMatOrAdj{T} = Union{CuMatrix,
                            LinearAlgebra.Adjoint{T, <:CuMatrix{T}},
                            LinearAlgebra.Transpose{T, <:CuMatrix{T}}}
const CuOrAdj{T} = Union{CuVecOrMat,
                         LinearAlgebra.Adjoint{T, <:CuVecOrMat{T}},
                         LinearAlgebra.Transpose{T, <:CuVecOrMat{T}}}

function Base.:\(_A::CuMatOrAdj, _B::CuOrAdj)
    A, B = copy_cublasfloat(_A, _B)
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

if VERSION >= v"1.8-"

LinearAlgebra.qr!(A::CuMatrix{T}) where T = CuQR(geqrf!(A::CuMatrix{T})...)
LinearAlgebra.qr!(A::StridedCuMatrix{T}) where T = CuQR(geqrf!(A::StridedCuMatrix{T})...)

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

# extracting the full matrix can be done with `collect` (which defaults to `Array`)
function Base.collect(src::Union{QRPackedQ{<:Any,<:CuArray,<:CuArray},
                                 QRCompactWYQ{<:Any,<:CuArray,<:CuArray}})
    dest = similar(src)
    copyto!(dest, I)
    lmul!(src, dest)
    collect(dest)
end

# avoid the generic similar fallback that returns a CPU array
Base.similar(Q::Union{QRPackedQ{<:Any,<:CuArray,<:CuArray},
                      QRCompactWYQ{<:Any,<:CuArray,<:CuArray}},
             ::Type{T}, dims::Dims{N}) where {T,N} =
    CuArray{T,N}(undef, dims)

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

else

struct CuQR{T} <: Factorization{T}
    factors::CuMatrix
    τ::CuVector{T}
    CuQR{T}(factors::CuMatrix{T}, τ::CuVector{T}) where {T} = new(factors, τ)
end

struct CuQRPackedQ{T} <: AbstractQ{T}
    factors::CuMatrix{T}
    τ::CuVector{T}
    CuQRPackedQ{T}(factors::CuMatrix{T}, τ::CuVector{T}) where {T} = new(factors, τ)
end

CuQR(factors::CuMatrix{T}, τ::CuVector{T}) where {T} =
    CuQR{T}(factors, τ)
CuQRPackedQ(factors::CuMatrix{T}, τ::CuVector{T}) where {T} =
    CuQRPackedQ{T}(factors, τ)

# AbstractQ's `size` is the size of the full matrix,
# while `Matrix(Q)` only gives the compact Q.
# See JuliaLang/julia#26591 and JuliaGPU/CUDA.jl#969.
CuMatrix{T}(Q::AbstractQ{S}) where {T,S} = convert(CuArray{T}, Matrix(Q))
CuMatrix{T, B}(Q::AbstractQ{S}) where {T, B, S} = CuMatrix{T}(Q)
CuMatrix(Q::AbstractQ{T}) where {T} = CuMatrix{T}(Q)
CuArray{T}(Q::AbstractQ) where {T} = CuMatrix{T}(Q)
CuArray(Q::AbstractQ) = CuMatrix(Q)

# extracting the full matrix can be done with `collect` (which defaults to `Array`)
function Base.collect(src::CuQRPackedQ)
    dest = similar(src)
    copyto!(dest, I)
    lmul!(src, dest)
    collect(dest)
end

# avoid the generic similar fallback that returns a CPU array
Base.similar(Q::CuQRPackedQ, ::Type{T}, dims::Dims{N}) where {T,N} =
    CuArray{T,N}(undef, dims)


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
LinearAlgebra.lmul!(A::CuQRPackedQ{T}, B::CuVecOrMat{T}) where {T<:BlasFloat} =
    ormqr!('L', 'N', A.factors, A.τ, B)
LinearAlgebra.lmul!(adjA::Adjoint{T,<:CuQRPackedQ{T}}, B::CuVecOrMat{T}) where {T<:BlasReal} =
    ormqr!('L', 'T', parent(adjA).factors, parent(adjA).τ, B)
LinearAlgebra.lmul!(adjA::Adjoint{T,<:CuQRPackedQ{T}}, B::CuVecOrMat{T}) where {T<:BlasComplex} =
    ormqr!('L', 'C', parent(adjA).factors, parent(adjA).τ, B)
LinearAlgebra.lmul!(trA::Transpose{T,<:CuQRPackedQ{T}}, B::CuVecOrMat{T}) where {T<:BlasFloat} =
    ormqr!('L', 'T', parent(trA).factors, parent(trA).τ, B)

# Apply changes Q from the right
LinearAlgebra.rmul!(A::CuVecOrMat{T}, B::CuQRPackedQ{T}) where {T<:BlasFloat} =
    ormqr!('R', 'N', B.factors, B.τ, A)
LinearAlgebra.rmul!(A::CuVecOrMat{T},
                    adjB::Adjoint{<:Any,<:CuQRPackedQ{T}}) where {T<:BlasReal} =
    ormqr!('R', 'T', parent(adjB).factors, parent(adjB).τ, A)
LinearAlgebra.rmul!(A::CuVecOrMat{T},
                    adjB::Adjoint{<:Any,<:CuQRPackedQ{T}}) where {T<:BlasComplex} =
    ormqr!('R', 'C', parent(adjB).factors, parent(adjB).τ, A)
LinearAlgebra.rmul!(A::CuVecOrMat{T},
                    trA::Transpose{<:Any,<:CuQRPackedQ{T}}) where {T<:BlasFloat} =
    ormqr!('R', 'T', parent(trA).factors, parent(adjB).τ, A)

function Base.getindex(A::CuQRPackedQ{T}, i::Int, j::Int) where {T}
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

function LinearAlgebra.ldiv!(_qr::CuQR, b::CuVector)
    m,n = size(_qr)
    _x = UpperTriangular(_qr.R[1:min(m,n), 1:n]) \ ((_qr.Q' * b)[1:n])
    b[1:n] .= _x
    unsafe_free!(_x)
    return b[1:n]
end

function LinearAlgebra.ldiv!(_qr::CuQR, B::CuMatrix)
    m,n = size(_qr)
    _x = UpperTriangular(_qr.R[1:min(m,n), 1:n]) \ ((_qr.Q' * B)[1:n, 1:size(B, 2)])
    B[1:n, 1:size(B, 2)] .= _x
    unsafe_free!(_x)
    return B[1:n, 1:size(B, 2)]
end

function LinearAlgebra.ldiv!(x::CuArray,_qr::CuQR, b::CuArray)
    _x = ldiv!(_qr, b)
    x .= vec(_x)
    unsafe_free!(_x)
    return x
end

end

## SVD

abstract type SVDAlgorithm end
struct QRAlgorithm <: SVDAlgorithm end
struct JacobiAlgorithm <: SVDAlgorithm end

if VERSION >= v"1.8-"

LinearAlgebra.svd!(A::CuMatrix{T}; full::Bool=false,
                   alg::SVDAlgorithm=JacobiAlgorithm()) where {T} =
    _svd!(A, full, alg)
LinearAlgebra.svd(A::CuMatrix; full=false, alg::SVDAlgorithm=JacobiAlgorithm()) =
    _svd!(copy_cublasfloat(A), full, alg)

_svd!(A::CuMatrix{T}, full::Bool, alg::SVDAlgorithm) where T =
    throw(ArgumentError("Unsupported value for `alg` keyword."))
function _svd!(A::CuMatrix{T}, full::Bool, alg::QRAlgorithm) where T
    U, S, Vt = gesvd!(full ? 'A' : 'S', full ? 'A' : 'S', A)
    return SVD(U, S, Vt)
end
function _svd!(A::CuMatrix{T}, full::Bool, alg::JacobiAlgorithm) where T
    U, S, V = gesvdj!('V', Int(!full), A)
    return SVD(U, S, V')
end

else


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

LinearAlgebra.svd!(A::CuMatrix{T}; full::Bool=false,
                   alg::SVDAlgorithm=JacobiAlgorithm()) where {T} =
    _svd!(A, full, alg)
LinearAlgebra.svd(A::CuMatrix; full=false, alg::SVDAlgorithm=JacobiAlgorithm()) =
    _svd!(copy_cublasfloat(A), full, alg)

_svd!(A::CuMatrix{T}, full::Bool, alg::SVDAlgorithm) where T =
    throw(ArgumentError("Unsupported value for `alg` keyword."))
function _svd!(A::CuMatrix{T}, full::Bool, alg::QRAlgorithm) where T
    U, s, Vt = gesvd!(full ? 'A' : 'S', full ? 'A' : 'S', A::CuMatrix{T})
    return CuSVD(U, s, Vt')
end
function _svd!(A::CuMatrix{T}, full::Bool, alg::JacobiAlgorithm) where T
    return CuSVD(gesvdj!('V', Int(!full), A::CuMatrix{T})...)
end

end

LinearAlgebra.svdvals!(A::CuMatrix{T}; alg::SVDAlgorithm=JacobiAlgorithm()) where {T} =
    _svdvals!(A, alg)
LinearAlgebra.svdvals(A::CuMatrix; alg::SVDAlgorithm=JacobiAlgorithm()) =
    _svdvals!(copy_cublasfloat(A), alg)

_svdvals!(A::CuMatrix{T}, alg::SVDAlgorithm) where T =
    throw(ArgumentError("Unsupported value for `alg` keyword."))
_svdvals!(A::CuMatrix{T}, alg::QRAlgorithm) where T = gesvd!('N', 'N', A::CuMatrix{T})[2]
_svdvals!(A::CuMatrix{T}, alg::JacobiAlgorithm) where T = gesvdj!('N', 1, A::CuMatrix{T})[2]

### opnorm2, enabled by svdvals

function LinearAlgebra.opnorm2(A::CuMatrix{T}) where {T}
    # The implementation in Base.LinearAlgebra can be reused verbatim, but it uses a scalar
    # index to access the larges singular value, so must be wrapped in @allowscalar
    return @allowscalar invoke(LinearAlgebra.opnorm2, Tuple{AbstractMatrix{T}}, A)
end

## LU

if VERSION >= v"1.8-"

function LinearAlgebra.lu!(A::StridedCuMatrix{T}, ::RowMaximum; check::Bool = true) where {T}
    lpt = getrf!(A)
    check && LinearAlgebra.checknonsingular(lpt[3])
    return LU(lpt[1], lpt[2], Int(lpt[3]))
end

# GPU-compatible accessors of the LU decomposition properties
function Base.getproperty(F::LU{T,<:StridedCuMatrix}, d::Symbol) where T
    m, n = size(F)
    if d === :L
        L = tril!(getfield(F, :factors)[1:m, 1:min(m,n)])
        L[1:min(m,n)+1:end] .= one(T)   # set the diagonal (linear indexing trick)
        return L
    elseif VERSION >= v"1.9.0-DEV.1775"
        invoke(getproperty, Tuple{LU{T}, Symbol}, F, d)
    else
        invoke(getproperty, Tuple{LU{T,<:StridedMatrix}, Symbol}, F, d)
    end
end

# LAPACK's pivoting sequence needs to be iterated sequentially...
# TODO: figure out a GPU-compatible way to get the permutation matrix
LinearAlgebra.ipiv2perm(v::CuVector{T}, maxi::Integer) where T =
    LinearAlgebra.ipiv2perm(Array(v), maxi)

end

function LinearAlgebra.ldiv!(F::LU{T,<:StridedCuMatrix{T}}, B::CuVecOrMat{T}) where {T}
    return getrs!('N', F.factors, F.ipiv, B)
end

# LinearAlgebra.jl defines a comparable method with these restrictions on T, so we need
#   to define with the same type parameters to resolve method ambiguity for these cases
function LinearAlgebra.ldiv!(F::LU{T,<:StridedCuMatrix{T}}, B::CuVecOrMat{T}) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}}
    return getrs!('N', F.factors, F.ipiv, B)
end

## cholesky

if VERSION >= v"1.8-"
    function LinearAlgebra.cholesky(A::LinearAlgebra.RealHermSymComplexHerm{<:Real,<:CuMatrix},
             ::Val{false}=Val(false); check::Bool = true)
        C, info = LinearAlgebra._chol!(copy(parent(A)), A.uplo == 'U' ? UpperTriangular : LowerTriangular)
        return Cholesky(C.data, A.uplo, info)
    end

    LinearAlgebra.cholcopy(A::LinearAlgebra.RealHermSymComplexHerm{<:Any,<:CuArray}) =
        copyto!(similar(A, LinearAlgebra.choltype(A)), A)
end
