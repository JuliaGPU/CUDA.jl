cublas_size(t::Char, M::CuVecOrMat) = (size(M, t=='N' ? 1 : 2), size(M, t=='N' ? 2 : 1))

CublasArray{T<:CublasFloat} = CuArray{T}

###########
#
# BLAS 1
#
###########

LinearAlgebra.rmul!(x::CuArray{<:CublasFloat}, k::Number) =
  scal!(length(x), convert(eltype(x), k), x, 1)

# Work around ambiguity with GPUArrays wrapper
LinearAlgebra.rmul!(x::CuArray{<:CublasFloat}, k::Real) =
  invoke(rmul!, Tuple{typeof(x), Number}, x, k)

function LinearAlgebra.BLAS.dot(DX::CuArray{T}, DY::CuArray{T}) where T<:Union{Float32,Float64}
    n = length(DX)
    n==length(DY) || throw(DimensionMismatch("dot product arguments have lengths $(length(DX)) and $(length(DY))"))
    dot(n, DX, 1, DY, 1)
end

function LinearAlgebra.BLAS.dotc(DX::CuArray{T}, DY::CuArray{T}) where T<:Union{ComplexF32,ComplexF64}
    n = length(DX)
    n==length(DY) || throw(DimensionMismatch("dot product arguments have lengths $(length(DX)) and $(length(DY))"))
    dotc(n, DX, 1, DY, 1)
end

function LinearAlgebra.BLAS.dot(DX::CuArray{T}, DY::CuArray{T}) where T<:Union{ComplexF32,ComplexF64}
    dotc(DX, DY)
end

function LinearAlgebra.BLAS.dotu(DX::CuArray{T}, DY::CuArray{T}) where T<:Union{ComplexF32,ComplexF64}
    n = length(DX)
    n==length(DY) || throw(DimensionMismatch("dot product arguments have lengths $(length(DX)) and $(length(DY))"))
    dotu(n, DX, 1, DY, 1)
end

LinearAlgebra.norm(x::CublasArray) = nrm2(x)
LinearAlgebra.BLAS.asum(x::CublasArray) = asum(length(x), x, 1)

function LinearAlgebra.axpy!(alpha::Number, x::CuArray{T}, y::CuArray{T}) where T<:CublasFloat
    length(x)==length(y) || throw(DimensionMismatch(""))
    axpy!(length(x), convert(T,alpha), x, 1, y, 1)
end

Base.argmin(xs::CublasArray{<:CublasReal}) = iamin(xs)
Base.argmax(xs::CublasArray{<:CublasReal}) = iamax(xs)

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
        return rmul!(y, 0)
    end
    gemv!(tA, alpha, A, x, beta, y)
end

LinearAlgebra.mul!(Y::CuVector{T}, A::CuMatrix{T}, B::CuVector{T}) where T<:CublasFloat = gemv_wrapper!(Y, 'N', A,  B)
LinearAlgebra.lmul!(Y::CuVector{T}, A::LinearAlgebra.Transpose{<:Any, CuMatrix{T}}, B::CuVector{T}) where T<:CublasFloat = gemv_wrapper!(Y, 'T', A.parent, B)
LinearAlgebra.lmul!(Y::CuVector{T}, A::LinearAlgebra.Adjoint{<:Any, CuMatrix{T}}, B::CuVector{T}) where T<:CublasFloat = gemv_wrapper!(Y, 'T', A.parent, B)
LinearAlgebra.lmul!(Y::CuVector{T}, A::LinearAlgebra.Adjoint{<:Any, CuMatrix{T}}, B::CuVector{T}) where T<:CublasComplex = gemv_wrapper!(Y, 'C', A.parent, B)

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
        return LinearAlgebra.rmul!(C, 0)
    end

    gemm!(tA, tB, alpha, A, B, beta, C)
end

# Mutating
LinearAlgebra.mul!(C::CuMatrix{T}, A::CuVecOrMat{T}, B::CuVecOrMat{T}) where T<:CublasFloat = gemm_wrapper!(C, 'N', 'N', A, B)
LinearAlgebra.mul!(C::CuMatrix{T}, trA::LinearAlgebra.Transpose{<:Any, <:CuMatrix{T}}, B::CuMatrix{T}) where T<:CublasFloat =
    gemm_wrapper!(C, 'T', 'N', parent(trA), B)
LinearAlgebra.mul!(C::CuMatrix{T}, A::CuMatrix{T}, trB::LinearAlgebra.Transpose{<:Any, <:CuMatrix{T}}) where T<:CublasFloat =
    gemm_wrapper!(C, 'N', 'T', A, parent(trB))
LinearAlgebra.mul!(C::CuMatrix{T}, trA::LinearAlgebra.Transpose{<:Any, <:CuMatrix{T}}, trB::LinearAlgebra.Transpose{<:Any, <:CuMatrix{T}}) where T<:CublasFloat =
    gemm_wrapper!(C, 'T', 'T', parent(trA), parent(trB))
LinearAlgebra.mul!(C::CuMatrix{T}, adjA::LinearAlgebra.Adjoint{<:Any, CuMatrix{T}}, B::CuMatrix{T}) where T<:CublasReal =
    gemm_wrapper!(C, 'T', 'N', parent(adjA), B)
LinearAlgebra.mul!(C::CuMatrix{T}, adjA::LinearAlgebra.Adjoint{<:Any, <:CuMatrix{T}}, B::CuMatrix{T}) where T<:CublasFloat =
    gemm_wrapper!(C, 'C', 'N', parent(adjA), B)
LinearAlgebra.mul!(C::CuMatrix{T}, A::CuMatrix{T}, adjB::LinearAlgebra.Adjoint{<:Any, CuMatrix{T}}) where T<:CublasReal =
    gemm_wrapper!(C, 'N', 'T', A, parent(adjB))
LinearAlgebra.mul!(C::CuMatrix{T}, A::CuMatrix{T}, adjB::LinearAlgebra.Adjoint{<:Any, <:CuMatrix{T}}) where T<:CublasFloat =
    gemm_wrapper!(C, 'N', 'C', A, parent(adjB))
LinearAlgebra.mul!(C::CuMatrix{T}, adjA::LinearAlgebra.Adjoint{<:Any, CuMatrix{T}}, adjB::LinearAlgebra.Adjoint{<:Any, CuMatrix{T}}) where T<:CublasReal =
    gemm_wrapper!(C, 'T', 'T', parent(adjA), parent(adjB))
LinearAlgebra.mul!(C::CuMatrix{T}, adjA::LinearAlgebra.Adjoint{<:Any, <:CuMatrix{T}}, adjB::LinearAlgebra.Adjoint{<:Any, <:CuMatrix{T}}) where T<:CublasFloat =
    gemm_wrapper!(C, 'C', 'C', parent(adjA), parent(adjB))
LinearAlgebra.mul!(C::CuMatrix{T}, trA::LinearAlgebra.Transpose{<:Any, <:CuMatrix{T}}, adjB::LinearAlgebra.Adjoint{T, <:CuMatrix{T}}) where T<:CublasReal =
    gemm_wrapper!(C, 'T', 'T', parent(trA), parent(adjB))
LinearAlgebra.mul!(C::CuMatrix{T}, trA::LinearAlgebra.Transpose{<:Any, <:CuMatrix{T}}, adjB::LinearAlgebra.Adjoint{<:Any, <:CuMatrix{T}}) where T<:CublasFloat =
    gemm_wrapper!(C, 'T', 'C', parent(trA), parent(adjB))
LinearAlgebra.mul!(C::CuMatrix{T}, adjA::LinearAlgebra.Adjoint{T, <:CuMatrix{T}}, trB::LinearAlgebra.Transpose{<:Any, <:CuMatrix{T}}) where T<:CublasReal =
    gemm_wrapper!(C, 'T', 'T', parent(adjA), parent(trB))
LinearAlgebra.mul!(C::CuMatrix{T}, adjA::LinearAlgebra.Adjoint{<:Any, <:CuMatrix{T}}, trB::LinearAlgebra.Transpose{<:Any, <:CuMatrix{T}}) where T <: CublasFloat =
    gemm_wrapper!(C, 'C', 'T', parent(adjA), parent(trB))
