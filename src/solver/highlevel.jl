# QR factorization

struct CuQR{T,S<:AbstractMatrix} <: LinearAlgebra.Factorization{T}
    factors::S
    τ::CuVector{T}
    CuQR{T,S}(factors::AbstractMatrix{T}, τ::CuVector{T}) where {T,S<:AbstractMatrix} = new(factors, τ)
end

struct CuQRPackedQ{T,S<:AbstractMatrix} <: LinearAlgebra.AbstractQ{T}
    factors::CuMatrix{T}
    τ::CuVector{T}
    CuQRPackedQ{T,S}(factors::AbstractMatrix{T}, τ::CuVector{T}) where {T,S<:AbstractMatrix} = new(factors, τ)
end

CuQR(factors::AbstractMatrix{T}, τ::CuVector{T}) where {T} = CuQR{T,typeof(factors)}(factors, τ)
CuQRPackedQ(factors::AbstractMatrix{T}, τ::CuVector{T}) where {T} = CuQRPackedQ{T,typeof(factors)}(factors, τ)

LinearAlgebra.qr!(A::CuMatrix{T}) where T = CuQR(geqrf!(A::CuMatrix{T})...)
Base.size(A::CuQR) = size(A.factors)
Base.size(A::CuQRPackedQ, dim::Integer) = 0 < dim ? (dim <= 2 ? size(A.factors, 1) : 1) : throw(BoundsError())
CuArrays.CuMatrix(A::CuQRPackedQ) = orgqr!(copy(A.factors), A.τ)
CuArrays.CuArray(A::CuQRPackedQ) = CuMatrix(A)
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
    x = CuArray{T}(undef, size(A, 2)) .= 0
    x[j] = 1
    lmul!(A, x)
    return _getindex(x, i)
end

function Base.show(io::IO, F::CuQR)
    println(io, "$(typeof(F)) with factors Q and R:")
    show(io, F.Q)
    println(io)
    show(io, F.R)
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

@enum SVDAlgorithm QRAlgorithm JacobiAlgorithm
function LinearAlgebra.svd!(A::CuMatrix{T}, method::SVDAlgorithm=JacobiAlgorithm; full::Bool=false) where T
    if method === QRAlgorithm
        U, s, Vt = gesvd!(full ? 'A' : 'S', full ? 'A' : 'S', A::CuMatrix{T})
        return CuSVD(U, s, Vt')
    elseif method === JacobiAlgorithm
        return CuSVD(gesvdj!('V', Int(!full), A::CuMatrix{T})...)
    end
end
# Once LinearAlgebra.svd(::AbstractMatrix) accepts kwargs this method can be deleted
LinearAlgebra.svd(A::CuMatrix, method::SVDAlgorithm=JacobiAlgorithm; full=false) = svd!(copy(A), method, full=full)

function LinearAlgebra.svdvals!(A::CuMatrix{T}, method::SVDAlgorithm=JacobiAlgorithm) where T
    if method === QRAlgorithm
        return gesvd!('N', 'N', A::CuMatrix{T})[2]
    elseif method === JacobiAlgorithm
        return gesvdj!('N', 1, A::CuMatrix{T})[2]
    end
end
# Once LinearAlgebra.svdvals(::AbstractMatrix) accepts kwargs this method can be deleted
LinearAlgebra.svdvals(A::CuMatrix, method::SVDAlgorithm=JacobiAlgorithm) = svdvals!(copy(A), method)
