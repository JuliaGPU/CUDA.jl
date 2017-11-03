import Base: qr, qrfact, qrfact!, getindex, A_mul_B!
export qrq!

# QR factorization

struct CuQR{T,S<:AbstractMatrix} <: Factorization{T}
    factors::S
    τ::CuVector{T}
    CuQR{T,S}(factors::AbstractMatrix{T}, τ::CuVector{T}) where {T,S<:AbstractMatrix} = new(factors, τ)
end

struct CuQRPackedQ{T,S<:AbstractMatrix} <: AbstractMatrix{T}
    factors::CuMatrix{T}
    τ::CuVector{T}
    CuQRPackedQ{T,S}(factors::AbstractMatrix{T}, τ::CuVector{T}) where {T,S<:AbstractMatrix} = new(factors, τ)
end

CuQR(factors::AbstractMatrix{T}, τ::CuVector{T}) where {T} = CuQR{T,typeof(factors)}(factors, τ)
CuQRPackedQ(factors::AbstractMatrix{T}, τ::CuVector{T}) where {T} = CuQRPackedQ{T,typeof(factors)}(factors, τ)

qrfact!(A::CuMatrix{T}) where T = CuQR(geqrf!(A::CuMatrix{T})...)
qrfact(A::CuMatrix) = qrfact!(copy(A))
getq(A::CuQR) = CuQRPackedQ(A.factors, A.τ)
Base.size(A::CuQR) = size(A.factors)
Base.size(A::CuQRPackedQ, dim::Integer) = 0 < dim ? (dim <= 2 ? size(A.factors, 1) : 1) : throw(BoundsError())
Base.size(A::CuQRPackedQ) = size(A, 1), size(A, 2)
Base.convert(::Type{CuMatrix}, A::CuQRPackedQ) = orgqr!(copy(A.factors), A.τ)
Base.convert(::Type{CuArray}, A::CuQRPackedQ) = convert(CuMatrix, A)

function getindex(A::CuQR, d::Symbol)
    m, n = size(A)
    if d == :R
        return triu!(A.factors[1:min(m, n), 1:n])
    elseif d == :Q
        return getq(A)
    else
        throw(KeyError(d))
    end
end

function getindex(A::CuQRPackedQ{T, S}, i::Integer, j::Integer) where {T, S}
    B = CuArray{T}(size(A, 2)) .= 0
    B[j] = 1
    B = A_mul_B!(A, B)
    _getindex(B, i)
end

function qr(A::CuMatrix)
    F = qrfact(A)
    Q, R = getq(F), F[:R]
    return CuArray(Q), R
end


"""
Returns the matrix `Q` from the QR factorization of the matrix `A` where
`A = Q R`. `A` is overwritten in the process.
"""
function qrq!(A::CuMatrix)
    orgqr!(geqrf!(A::CuMatrix{T})...)
end

A_mul_B!(A::CuQRPackedQ{T,S}, B::CuVecOrMat{T}) where {T<:Number, S<:CuMatrix} =
    ormqr!('L', 'N', A.factors, A.τ, B)

function Base.show(io::IO, F::CuQR)
    println(io, "$(typeof(F)) with factors Q and R:")
    show(io, F[:Q])
    println(io)
    show(io, F[:R])
end
