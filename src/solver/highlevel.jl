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
Base.convert(::Type{CuMatrix}, A::CuQRPackedQ) = orgqr!(copy(A.factors), A.τ)
Base.convert(::Type{CuArray}, A::CuQRPackedQ) = convert(CuMatrix, A)

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

LinearAlgebra.lmul!(A::CuQRPackedQ{T,S}, B::CuVecOrMat{T}) where {T<:Number, S<:CuMatrix} =
    ormqr!('L', 'N', A.factors, A.τ, B)
LinearAlgebra.lmul!(adjA::Adjoint{T,<:CuQRPackedQ{T,S}}, B::CuVecOrMat{T}) where {T<:Number, S<:CuMatrix} =
    ormqr!('L', 'C', parent(adjA).factors, parent(adjA).τ, B)
LinearAlgebra.lmul!(trA::Transpose{T,<:CuQRPackedQ{T,S}}, B::CuVecOrMat{T}) where {T<:Number, S<:CuMatrix} =
    ormqr!('L', 'T', parent(trA).factors, parent(trA).τ, B)

function Base.getindex(A::CuQRPackedQ{T, S}, i::Integer, j::Integer) where {T, S}
    x = CuArray{T}(size(A, 2)) .= 0
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
