import Base: A_mul_B!, At_mul_B!, A_mul_Bt!, Ac_mul_B!, A_mul_Bc!, At_mul_Bt!, Ac_mul_Bc!, At_mul_Bt!

cublas_size(t::Char, M::CuVecOrMat) = (size(M, t=='N' ? 1:2), size(M, t=='N' ? 2:1))

CublasArray{T<:CublasFloat} = CuArray{T}

###########
#
# BLAS 1
#
###########

Base.scale!(x::CuArray{T}, k::Number) where T<:CublasFloat =
  scal!(length(x), convert(eltype(x), k), x, 1)

# Work around ambiguity with GPUArrays wrapper
Base.scale!(x::CuArray{T}, k::Real) where T<:CublasFloat =
  invoke(scale!, Tuple{typeof(x), Number}, x, k)

function Base.BLAS.dot(DX::CuArray{T}, DY::CuArray{T}) where T<:Union{Float32,Float64}
    n = length(DX)
    n==length(DY) || throw(DimensionMismatch("dot product arguments have lengths $(length(DX)) and $(length(DY))"))
    dot(n, DX, 1, DY, 1)
end
function Base.BLAS.dotc(DX::CuArray{T}, DY::CuArray{T}) where T<:Union{Complex64,Complex128}
    n = length(DX)
    n==length(DY) || throw(DimensionMismatch("dot product arguments have lengths $(length(DX)) and $(length(DY))"))
    dotc(n, DX, 1, DY, 1)
end
function Base.BLAS.dot(DX::CuArray{T}, DY::CuArray{T}) where T<:Union{Complex64,Complex128}
    Base.BLAS.dotc(DX, DY)
end
function Base.BLAS.dotu(DX::CuArray{T}, DY::CuArray{T}) where T<:Union{Complex64,Complex128}
    n = length(DX)
    n==length(DY) || throw(DimensionMismatch("dot product arguments have lengths $(length(DX)) and $(length(DY))"))
    dotu(n, DX, 1, DY, 1)
end

Base.At_mul_B(x::CuVector{T}, y::CuVector{T}) where T<:CublasReal = Base.BLAS.dot(x, y)

Base.norm(x::CublasArray) = nrm2(x)
Base.BLAS.asum(x::CublasArray) = asum(length(x), x, 1)

function Base.axpy!(alpha::Number, x::CuArray{T}, y::CuArray{T}) where T<:CublasFloat
    length(x)==length(y) || throw(DimensionMismatch(""))
    axpy!(length(x), convert(T,alpha), x, 1, y, 1)
end

Base.indmin(xs::CublasArray{T}) where T <: CublasReal = iamin(xs)
Base.indmax(xs::CublasArray{T}) where T <: CublasReal = iamax(xs)

############
#
# BLAS 2
#
############


#########
# GEMV
##########
function gemv_wrapper!(y::CuVector{T}, tA::Char, A::CuMatrix{T}, x::CuVector{T},
                       alpha = one(T), beta = zero(T)) where T<:CublasFloat
    mA, nA = cublas_size(tA, A)
    if nA != length(x)
        throw(DimensionMismatch("second dimension of A, $nA, does not match length of x, $(length(x))"))
    end
    if mA != length(y)
        throw(DimensionMismatch("first dimension of A, $mA, does not match length of y, $(length(y))"))
    end
    if mA == 0
        return y
    end
    if nA == 0
        return scale!(y, 0)
    end
    gemv!(tA, alpha, A, x, beta, y)
end

A_mul_B!(y::CuVector{T}, A::CuMatrix{T}, x::CuVector{T}) where T<:CublasFloat = gemv_wrapper!(y, 'N', A,  x)
At_mul_B!(y::CuVector{T}, A::CuMatrix{T}, x::CuVector{T}) where T<:CublasFloat = gemv_wrapper!(y, 'T', A, x)
Ac_mul_B!(y::CuVector{T}, A::CuMatrix{T}, x::CuVector{T}) where T<:CublasFloat = gemv_wrapper!(y, 'T', A, x)
Ac_mul_B!(y::CuVector{T}, A::CuMatrix{T}, x::CuVector{T}) where T<:CublasComplex = gemv_wrapper!(y, 'C', A, x)

############
#
# BLAS 3
#
############


########
# GEMM
########
function gemm_wrapper!(C::CuVecOrMat{T}, tA::Char, tB::Char,
                   A::CuVecOrMat{T},
                   B::CuVecOrMat{T},
                   alpha = one(T),
                   beta = zero(T)) where T <: CublasFloat
    mA, nA = cublas_size(tA, A)
    mB, nB = cublas_size(tB, B)

    if nA != mB
        throw(DimensionMismatch("A has dimensions ($mA,$nA) but B has dimensions ($mB,$nB)"))
    end

    if C === A || B === C
        throw(ArgumentError("output matrix must not be aliased with input matrix"))
    end

    if mA == 0 || nA == 0 || nB == 0
        if size(C) != (mA, nB)
            throw(DimensionMismatch("C has dimensions $(size(C)), should have ($mA,$nB)"))
        end
        return scale!(C, 0)
    end

    gemm!(tA, tB, alpha, A, B, beta, C)
end

# Mutating
A_mul_B!(C::CuMatrix{T}, A::CuMatrix{T}, B::CuMatrix{T}) where T<:CublasFloat = gemm_wrapper!(C, 'N', 'N', A, B)
At_mul_B!(C::CuMatrix, A::CuMatrix, B::CuMatrix) = gemm_wrapper!(C, 'T', 'N', A, B)
A_mul_Bt!(C::CuMatrix, A::CuMatrix, B::CuMatrix) = gemm_wrapper!(C, 'N', 'T', A, B)
At_mul_Bt!(C::CuMatrix, A::CuMatrix, B::CuMatrix) = gemm_wrapper!(C, 'T', 'T', A, B)
Ac_mul_B!(C::CuMatrix{T}, A::CuMatrix{T}, B::CuMatrix{T}) where T<:CublasReal = At_mul_B!(C, A, B)
Ac_mul_B!(C::CuMatrix, A::CuMatrix, B::CuMatrix) = gemm_wrapper!(C, 'C', 'N', A, B)
A_mul_Bc!(C::CuMatrix{T}, A::CuMatrix{T}, B::CuMatrix{T}) where T<:CublasReal = A_mul_Bt!(C, A, B)
A_mul_Bc!(C::CuMatrix, A::CuMatrix, B::CuMatrix) = gemm_wrapper!(C, 'N', 'C', A, B)
Ac_mul_Bc!(C::CuMatrix{T}, A::CuMatrix{T}, B::CuMatrix{T}) where T<:CublasReal = At_mul_Bt!(C, A, B)
Ac_mul_Bc!(C::CuMatrix, A::CuMatrix, B::CuMatrix) = gemm_wrapper!(C, 'C', 'C', A, B)

function A_mul_B!(C::CuMatrix{T}, A::CuVecOrMat{T}, B::CuVecOrMat{T}) where T
    gemm_wrapper!(C, 'N', 'N', A, B)
end
