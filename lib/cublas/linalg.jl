# interfacing with LinearAlgebra standard library

cublas_size(t::Char, M::CuVecOrMat) = (size(M, t=='N' ? 1 : 2), size(M, t=='N' ? 2 : 1))

CublasArray{T<:CublasFloat} = CuArray{T}


#
# BLAS 1
#

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
    BLAS.dotc(DX, DY)
end

function LinearAlgebra.BLAS.dotu(DX::CuArray{T}, DY::CuArray{T}) where T<:Union{ComplexF32,ComplexF64}
    n = length(DX)
    n==length(DY) || throw(DimensionMismatch("dot product arguments have lengths $(length(DX)) and $(length(DY))"))
    dotu(n, DX, 1, DY, 1)
end

LinearAlgebra.norm(x::CublasArray) = nrm2(x)
LinearAlgebra.BLAS.asum(x::CublasArray) = asum(length(x), x, 1)

function LinearAlgebra.axpy!(alpha::Number, x::CuArray{T}, y::CuArray{T}) where T<:CublasFloat
    length(x)==length(y) || throw(DimensionMismatch("axpy arguments have lengths $(length(x)) and $(length(y))"))
    axpy!(length(x), convert(T,alpha), x, 1, y, 1)
end

function LinearAlgebra.axpby!(alpha::Number, x::CuArray{T}, beta::Number, y::CuArray{T}) where T<:CublasFloat
    length(x)==length(y) || throw(DimensionMismatch("axpy arguments have lengths $(length(x)) and $(length(y))"))
    axpby!(length(x), convert(T,alpha), x, 1, convert(T,beta), y, 1)
end


#
# BLAS 2
#

# GEMV

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

function promote_alpha_beta(a, b, ::Type{T}) where {T}
    a_prom, b_prom = promote(a, b, zero(T))
    a_prom, b_prom
end

LinearAlgebra.mul!(Y::CuVector{T}, A::CuMatrix{T}, B::CuVector{T}, a::Number, b::Number) where T<:CublasFloat =
    gemv_wrapper!(Y, 'N', A, B, promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(Y::CuVector{T}, A::Transpose{<:Any, <:CuVecOrMat{T}}, B::CuVector{T}, a::Number, b::Number) where T<:CublasFloat =
    gemv_wrapper!(Y, 'T', A.parent, B, promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(Y::CuVector{T}, A::Adjoint{<:Any, <:CuVecOrMat{T}}, B::CuVector{T}, a::Real, b::Real) where T<:CublasReal =
    gemv_wrapper!(Y, 'T', A.parent, B, promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(Y::CuVector{T}, A::Adjoint{<:Any, <:CuVecOrMat{T}}, B::CuVector{T}, a::Number, b::Number) where T<:CublasComplex =
    gemv_wrapper!(Y, 'C', A.parent, B, promote_alpha_beta(a, b, T)...)

# specificity hacks: Base and GPUArrays has mul! with a::Real, b::Real
LinearAlgebra.mul!(Y::CuVector{T}, A::CuMatrix{T}, B::CuVector{T}, a::Real, b::Real) where T<:CublasFloat =
    gemv_wrapper!(Y, 'N', A, B, promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(Y::CuVector{T}, A::Transpose{<:Any, <:CuVecOrMat{T}}, B::CuVector{T}, a::Real, b::Real) where T<:CublasFloat =
    gemv_wrapper!(Y, 'T', A.parent, B, promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(Y::CuVector{T}, A::Adjoint{<:Any, <:CuVecOrMat{T}}, B::CuVector{T}, a::Real, b::Real) where T<:CublasComplex =
    gemv_wrapper!(Y, 'C', A.parent, B, promote_alpha_beta(a, b, T)...)

# Fix Julia <= 1.3.1 ambiguities... they're fixed in 1.4.x thanks to https://github.com/JuliaLang/julia/pull/33743
@static if v"1.3.0" <= VERSION <= v"1.3.1"
    LinearAlgebra.mul!(Y::CuVector{T}, A::CuMatrix{T}, B::CuVector{T}, a::Union{T,Bool}, b::Union{T,Bool}) where T<:CublasFloat =
        gemv_wrapper!(Y, 'N', A, B, promote_alpha_beta(a, b, T)...)
    LinearAlgebra.mul!(Y::CuVector{T}, A::Transpose{<:Any, <:CuVecOrMat{T}}, B::CuVector{T}, a::Union{T,Bool}, b::Union{T,Bool}) where T<:CublasFloat =
        gemv_wrapper!(Y, 'T', A.parent, B, promote_alpha_beta(a, b, T)...)
    LinearAlgebra.mul!(Y::CuVector{T}, A::Adjoint{<:Any, <:CuVecOrMat{T}}, B::CuVector{T}, a::Union{T,Bool}, b::Union{T,Bool}) where T<:CublasReal =
        gemv_wrapper!(Y, 'T', A.parent, B, promote_alpha_beta(a, b, T)...)
    LinearAlgebra.mul!(Y::CuVector{T}, A::Adjoint{<:Any, <:CuVecOrMat{T}}, B::CuVector{T}, a::Union{T,Bool}, b::Union{T,Bool}) where T<:CublasComplex =
        gemv_wrapper!(Y, 'C', A.parent, B, promote_alpha_beta(a, b, T)...)
end

# TRSV

function LinearAlgebra.ldiv!(
    A::UpperTriangular{T, <:CuMatrix{T}},
    x::CuVector{T},
) where T<:CublasFloat
    return CUBLAS.trsv!('U', 'N', 'N', parent(A), x)
end

function LinearAlgebra.ldiv!(
    A::Adjoint{T, <:UpperTriangular{T, <:CuMatrix{T}}},
    x::CuVector{T},
) where {T<:CUBLAS.CublasFloat}
    return CUBLAS.trsv!('U', 'C', 'N', parent(parent(A)), x)
end

function LinearAlgebra.ldiv!(
    A::Transpose{T, <:UpperTriangular{T, <:CuMatrix{T}}},
    x::CuVector{T},
) where {T<:CUBLAS.CublasFloat}
    return CUBLAS.trsv!('U', 'T', 'N', parent(parent(A)), x)
end

function LinearAlgebra.ldiv!(
    A::LowerTriangular{T, <:CuMatrix{T}},
    x::CuVector{T},
) where T<:CublasFloat
    return CUBLAS.trsv!('L', 'N', 'N', parent(A), x)
end

function LinearAlgebra.ldiv!(
    A::Adjoint{T, <:LowerTriangular{T, <:CuMatrix{T}}},
    x::CuVector{T},
) where {T<:CUBLAS.CublasFloat}
    return CUBLAS.trsv!('L', 'C', 'N', parent(parent(A)), x)
end

function LinearAlgebra.ldiv!(
    A::Transpose{T, <:LowerTriangular{T, <:CuMatrix{T}}},
    x::CuVector{T},
) where {T<:CUBLAS.CublasFloat}
    return CUBLAS.trsv!('L', 'T', 'N', parent(parent(A)), x)
end



#
# BLAS 3
#

# GEMM

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
LinearAlgebra.mul!(C::CuMatrix{T}, A::CuVecOrMat{T}, B::CuVecOrMat{T}, a::Number, b::Number) where T<:CublasFloat =
    gemm_wrapper!(C, 'N', 'N', A, B, promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, trA::Transpose{<:Any, <:CuVecOrMat{T}}, B::CuMatrix{T}, a::Number, b::Number) where T<:CublasFloat =
    gemm_wrapper!(C, 'T', 'N', parent(trA), B, promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, A::CuMatrix{T}, trB::Transpose{<:Any, <:CuVecOrMat{T}}, a::Number, b::Number) where T<:CublasFloat =
    gemm_wrapper!(C, 'N', 'T', A, parent(trB), promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, trA::Transpose{<:Any, <:CuVecOrMat{T}}, trB::Transpose{<:Any, <:CuVecOrMat{T}}, a::Number, b::Number) where T<:CublasFloat =
    gemm_wrapper!(C, 'T', 'T', parent(trA), parent(trB), promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, adjA::Adjoint{<:Any, <:CuVecOrMat{T}}, B::CuMatrix{T}, a::Real, b::Real) where T<:CublasReal =
    gemm_wrapper!(C, 'T', 'N', parent(adjA), B, promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, adjA::Adjoint{<:Any, <:CuVecOrMat{T}}, B::CuMatrix{T}, a::Number, b::Number) where T<:CublasComplex =
    gemm_wrapper!(C, 'C', 'N', parent(adjA), B, promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, A::CuMatrix{T}, adjB::Adjoint{<:Any, <:CuVecOrMat{T}}, a::Real, b::Real) where T<:CublasReal =
    gemm_wrapper!(C, 'N', 'T', A, parent(adjB), promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, A::CuMatrix{T}, adjB::Adjoint{<:Any, <:CuVecOrMat{T}}, a::Number, b::Number) where T<:CublasComplex =
    gemm_wrapper!(C, 'N', 'C', A, parent(adjB), promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, adjA::Adjoint{<:Any, <:CuVecOrMat{T}}, adjB::Adjoint{<:Any, <:CuVecOrMat{T}}, a::Real, b::Real) where T<:CublasReal =
    gemm_wrapper!(C, 'T', 'T', parent(adjA), parent(adjB), promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, adjA::Adjoint{<:Any, <:CuVecOrMat{T}}, adjB::Adjoint{<:Any, <:CuVecOrMat{T}}, a::Number, b::Number) where T<:CublasComplex =
    gemm_wrapper!(C, 'C', 'C', parent(adjA), parent(adjB), promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, trA::Transpose{<:Any, <:CuVecOrMat{T}}, adjB::Adjoint{T, <:CuVecOrMat{T}}, a::Real, b::Real) where T<:CublasReal =
    gemm_wrapper!(C, 'T', 'T', parent(trA), parent(adjB), promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, trA::Transpose{<:Any, <:CuVecOrMat{T}}, adjB::Adjoint{<:Any, <:CuVecOrMat{T}}, a::Number, b::Number) where T<:CublasComplex =
    gemm_wrapper!(C, 'T', 'C', parent(trA), parent(adjB), promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, adjA::Adjoint{T, <:CuVecOrMat{T}}, trB::Transpose{<:Any, <:CuVecOrMat{T}}, a::Real, b::Real) where T<:CublasReal =
    gemm_wrapper!(C, 'T', 'T', parent(adjA), parent(trB), promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, adjA::Adjoint{<:Any, <:CuVecOrMat{T}}, trB::Transpose{<:Any, <:CuVecOrMat{T}}, a::Number, b::Number) where T <: CublasComplex =
    gemm_wrapper!(C, 'C', 'T', parent(adjA), parent(trB), promote_alpha_beta(a, b, T)...)

# specificity hacks: Base and GPUArrays has mul! with a::Real, b::Real
LinearAlgebra.mul!(C::CuMatrix{T}, A::CuVecOrMat{T}, B::CuVecOrMat{T}, a::Real, b::Real) where T<:CublasFloat =
    gemm_wrapper!(C, 'N', 'N', A, B, promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, trA::Transpose{<:Any, <:CuVecOrMat{T}}, B::CuMatrix{T}, a::Real, b::Real) where T<:CublasFloat =
    gemm_wrapper!(C, 'T', 'N', parent(trA), B, promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, A::CuMatrix{T}, trB::Transpose{<:Any, <:CuVecOrMat{T}}, a::Real, b::Real) where T<:CublasFloat =
    gemm_wrapper!(C, 'N', 'T', A, parent(trB), promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, trA::Transpose{<:Any, <:CuVecOrMat{T}}, trB::Transpose{<:Any, <:CuVecOrMat{T}}, a::Real, b::Real) where T<:CublasFloat =
    gemm_wrapper!(C, 'T', 'T', parent(trA), parent(trB), promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, adjA::Adjoint{<:Any, <:CuVecOrMat{T}}, B::CuMatrix{T}, a::Real, b::Real) where T<:CublasComplex =
    gemm_wrapper!(C, 'C', 'N', parent(adjA), B, promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, A::CuMatrix{T}, adjB::Adjoint{<:Any, <:CuVecOrMat{T}}, a::Real, b::Real) where T<:CublasComplex =
    gemm_wrapper!(C, 'N', 'C', A, parent(adjB), promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, adjA::Adjoint{<:Any, <:CuVecOrMat{T}}, adjB::Adjoint{<:Any, <:CuVecOrMat{T}}, a::Real, b::Real) where T<:CublasComplex =
    gemm_wrapper!(C, 'C', 'C', parent(adjA), parent(adjB), promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, trA::Transpose{<:Any, <:CuVecOrMat{T}}, adjB::Adjoint{<:Any, <:CuVecOrMat{T}}, a::Real, b::Real) where T<:CublasComplex =
    gemm_wrapper!(C, 'T', 'C', parent(trA), parent(adjB), promote_alpha_beta(a, b, T)...)
LinearAlgebra.mul!(C::CuMatrix{T}, adjA::Adjoint{<:Any, <:CuVecOrMat{T}}, trB::Transpose{<:Any, <:CuVecOrMat{T}}, a::Real, b::Real) where T <: CublasComplex =
    gemm_wrapper!(C, 'C', 'T', parent(adjA), parent(trB), promote_alpha_beta(a, b, T)...)

# Fix Julia <= 1.3.1 ambiguities... they're fixed in 1.4.x thanks to https://github.com/JuliaLang/julia/pull/33743
@static if v"1.3.0" <= VERSION <= v"1.3.1"
    LinearAlgebra.mul!(C::CuMatrix{T}, A::CuVecOrMat{T}, B::CuVecOrMat{T}, a::Union{T,Bool}, b::Union{T,Bool}) where T<:CublasFloat =
        gemm_wrapper!(C, 'N', 'N', A, B, promote_alpha_beta(a, b, T)...)
    LinearAlgebra.mul!(C::CuMatrix{T}, trA::Transpose{<:Any, <:CuVecOrMat{T}}, B::CuMatrix{T}, a::Union{T,Bool}, b::Union{T,Bool}) where T<:CublasFloat =
        gemm_wrapper!(C, 'T', 'N', parent(trA), B, promote_alpha_beta(a, b, T)...)
    LinearAlgebra.mul!(C::CuMatrix{T}, A::CuMatrix{T}, trB::Transpose{<:Any, <:CuVecOrMat{T}}, a::Union{T,Bool}, b::Union{T,Bool}) where T<:CublasFloat =
        gemm_wrapper!(C, 'N', 'T', A, parent(trB), promote_alpha_beta(a, b, T)...)
    LinearAlgebra.mul!(C::CuMatrix{T}, trA::Transpose{<:Any, <:CuVecOrMat{T}}, trB::Transpose{<:Any, <:CuVecOrMat{T}}, a::Union{T,Bool}, b::Union{T,Bool}) where T<:CublasFloat =
        gemm_wrapper!(C, 'T', 'T', parent(trA), parent(trB), promote_alpha_beta(a, b, T)...)
    LinearAlgebra.mul!(C::CuMatrix{T}, adjA::Adjoint{<:Any, <:CuVecOrMat{T}}, B::CuMatrix{T}, a::Union{T,Bool}, b::Union{T,Bool}) where T<:CublasReal =
        gemm_wrapper!(C, 'T', 'N', parent(adjA), B, promote_alpha_beta(a, b, T)...)
    LinearAlgebra.mul!(C::CuMatrix{T}, adjA::Adjoint{<:Any, <:CuVecOrMat{T}}, B::CuMatrix{T}, a::Union{T,Bool}, b::Union{T,Bool}) where T<:CublasComplex =
        gemm_wrapper!(C, 'C', 'N', parent(adjA), B, promote_alpha_beta(a, b, T)...)
    LinearAlgebra.mul!(C::CuMatrix{T}, A::CuMatrix{T}, adjB::Adjoint{<:Any, <:CuVecOrMat{T}}, a::Union{T,Bool}, b::Union{T,Bool}) where T<:CublasReal =
        gemm_wrapper!(C, 'N', 'T', A, parent(adjB), promote_alpha_beta(a, b, T)...)
    LinearAlgebra.mul!(C::CuMatrix{T}, A::CuMatrix{T}, adjB::Adjoint{<:Any, <:CuVecOrMat{T}}, a::Union{T,Bool}, b::Union{T,Bool}) where T<:CublasComplex =
        gemm_wrapper!(C, 'N', 'C', A, parent(adjB), promote_alpha_beta(a, b, T)...)
    LinearAlgebra.mul!(C::CuMatrix{T}, adjA::Adjoint{<:Any, <:CuVecOrMat{T}}, adjB::Adjoint{<:Any, <:CuVecOrMat{T}}, a::Union{T,Bool}, b::Union{T,Bool}) where T<:CublasReal =
        gemm_wrapper!(C, 'T', 'T', parent(adjA), parent(adjB), promote_alpha_beta(a, b, T)...)
    LinearAlgebra.mul!(C::CuMatrix{T}, adjA::Adjoint{<:Any, <:CuVecOrMat{T}}, adjB::Adjoint{<:Any, <:CuVecOrMat{T}}, a::Union{T,Bool}, b::Union{T,Bool}) where T<:CublasComplex =
        gemm_wrapper!(C, 'C', 'C', parent(adjA), parent(adjB), promote_alpha_beta(a, b, T)...)
    LinearAlgebra.mul!(C::CuMatrix{T}, trA::Transpose{<:Any, <:CuVecOrMat{T}}, adjB::Adjoint{T, <:CuVecOrMat{T}}, a::Union{T,Bool}, b::Union{T,Bool}) where T<:CublasReal =
        gemm_wrapper!(C, 'T', 'T', parent(trA), parent(adjB), promote_alpha_beta(a, b, T)...)
    LinearAlgebra.mul!(C::CuMatrix{T}, trA::Transpose{<:Any, <:CuVecOrMat{T}}, adjB::Adjoint{<:Any, <:CuVecOrMat{T}}, a::Union{T,Bool}, b::Union{T,Bool}) where T<:CublasComplex =
        gemm_wrapper!(C, 'T', 'C', parent(trA), parent(adjB), promote_alpha_beta(a, b, T)...)
    LinearAlgebra.mul!(C::CuMatrix{T}, adjA::Adjoint{T, <:CuVecOrMat{T}}, trB::Transpose{<:Any, <:CuVecOrMat{T}}, a::Union{T,Bool}, b::Union{T,Bool}) where T<:CublasReal =
        gemm_wrapper!(C, 'T', 'T', parent(adjA), parent(trB), promote_alpha_beta(a, b, T)...)
    LinearAlgebra.mul!(C::CuMatrix{T}, adjA::Adjoint{<:Any, <:CuVecOrMat{T}}, trB::Transpose{<:Any, <:CuVecOrMat{T}}, a::Union{T,Bool}, b::Union{T,Bool}) where T <: CublasComplex =
        gemm_wrapper!(C, 'C', 'T', parent(adjA), parent(trB), promote_alpha_beta(a, b, T)...)
end

# TRSM

# ldiv!
## No transpose/adjoint
LinearAlgebra.ldiv!(A::UpperTriangular{T, <:CuMatrix{T}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trsm!('L', 'U', 'N', 'N', one(T), parent(A), B)
LinearAlgebra.ldiv!(A::UnitUpperTriangular{T, <:CuMatrix{T}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trsm!('L', 'U', 'N', 'U', one(T), parent(A), B)
LinearAlgebra.ldiv!(A::LowerTriangular{T, <:CuMatrix{T}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trsm!('L', 'L', 'N', 'N', one(T), parent(A), B)
LinearAlgebra.ldiv!(A::UnitLowerTriangular{T, <:CuMatrix{T}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trsm!('L', 'L', 'N', 'U', one(T), parent(A), B)
## Adjoint
LinearAlgebra.ldiv!(A::Adjoint{T,<:UpperTriangular{T, <:CuMatrix{T}}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trsm!('L', 'U', 'C', 'N', one(T), parent(parent(A)), B)
LinearAlgebra.ldiv!(A::Adjoint{T,<:UnitUpperTriangular{T, <:CuMatrix{T}}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trsm!('L', 'U', 'C', 'U', one(T), parent(parent(A)), B)
LinearAlgebra.ldiv!(A::Adjoint{T,<:LowerTriangular{T, <:CuMatrix{T}}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trsm!('L', 'L', 'C', 'N', one(T), parent(parent(A)), B)
LinearAlgebra.ldiv!(A::Adjoint{T,<:UnitLowerTriangular{T, <:CuMatrix{T}}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trsm!('L', 'L', 'C', 'U', one(T), parent(parent(A)), B)
## Transpose
LinearAlgebra.ldiv!(A::Transpose{T,<:UpperTriangular{T, <:CuMatrix{T}}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trsm!('L', 'U', 'T', 'N', one(T), parent(parent(A)), B)
LinearAlgebra.ldiv!(A::Transpose{T,<:UnitUpperTriangular{T, <:CuMatrix{T}}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trsm!('L', 'U', 'T', 'U', one(T), parent(parent(A)), B)
LinearAlgebra.ldiv!(A::Transpose{T,<:LowerTriangular{T, <:CuMatrix{T}}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trsm!('L', 'L', 'T', 'N', one(T), parent(parent(A)), B)
LinearAlgebra.ldiv!(A::Transpose{T,<:UnitLowerTriangular{T, <:CuMatrix{T}}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trsm!('L', 'L', 'T', 'U', one(T), parent(parent(A)), B)

# rdiv!
## No transpose/adjoint
LinearAlgebra.rdiv!(A::CuMatrix{T}, B::UpperTriangular{T, <:CuMatrix{T}}) where T<:CublasFloat =
    CUBLAS.trsm!('R', 'U', 'N', 'N', one(T), parent(B), A)
LinearAlgebra.rdiv!(A::CuMatrix{T}, B::UnitUpperTriangular{T, <:CuMatrix{T}}) where T<:CublasFloat =
    CUBLAS.trsm!('R', 'U', 'N', 'U', one(T), parent(B), A)
LinearAlgebra.rdiv!(A::CuMatrix{T}, B::LowerTriangular{T, <:CuMatrix{T}}) where T<:CublasFloat =
    CUBLAS.trsm!('R', 'L', 'N', 'N', one(T), parent(B), A)
LinearAlgebra.rdiv!(A::CuMatrix{T}, B::UnitLowerTriangular{T, <:CuMatrix{T}}) where T<:CublasFloat =
    CUBLAS.trsm!('R', 'L', 'N', 'U', one(T), parent(B), A)
## Adjoint
LinearAlgebra.rdiv!(A::CuMatrix{T}, B::Adjoint{T,<:UpperTriangular{T, <:CuMatrix{T}}}) where T<:CublasFloat =
    CUBLAS.trsm!('R', 'U', 'C', 'N', one(T), parent(parent(B)), A)
LinearAlgebra.rdiv!(A::CuMatrix{T}, B::Adjoint{T,<:UnitUpperTriangular{T, <:CuMatrix{T}}}) where T<:CublasFloat =
    CUBLAS.trsm!('R', 'U', 'C', 'U', one(T), parent(parent(B)), A)
LinearAlgebra.rdiv!(A::CuMatrix{T}, B::Adjoint{T,<:LowerTriangular{T, <:CuMatrix{T}}}) where T<:CublasFloat =
    CUBLAS.trsm!('R', 'L', 'C', 'N', one(T), parent(parent(B)), A)
LinearAlgebra.rdiv!(A::CuMatrix{T}, B::Adjoint{T,<:UnitLowerTriangular{T, <:CuMatrix{T}}}) where T<:CublasFloat =
    CUBLAS.trsm!('R', 'L', 'C', 'U', one(T), parent(parent(B)), A)
## Transpose
LinearAlgebra.rdiv!(A::CuMatrix{T}, B::Transpose{T,<:UpperTriangular{T, <:CuMatrix{T}}}) where T<:CublasFloat =
    CUBLAS.trsm!('R', 'U', 'T', 'N', one(T), parent(parent(B)), A)
LinearAlgebra.rdiv!(A::CuMatrix{T}, B::Transpose{T,<:UnitUpperTriangular{T, <:CuMatrix{T}}}) where T<:CublasFloat =
    CUBLAS.trsm!('R', 'U', 'T', 'U', one(T), parent(parent(B)), A)
LinearAlgebra.rdiv!(A::CuMatrix{T}, B::Transpose{T,<:LowerTriangular{T, <:CuMatrix{T}}}) where T<:CublasFloat =
    CUBLAS.trsm!('R', 'L', 'T', 'N', one(T), parent(parent(B)), A)
LinearAlgebra.rdiv!(A::CuMatrix{T}, B::Transpose{T,<:UnitLowerTriangular{T, <:CuMatrix{T}}}) where T<:CublasFloat =
    CUBLAS.trsm!('R', 'L', 'T', 'U', one(T), parent(parent(B)), A)


# TRMM

# Left mul!
## No transpose/adjoint
LinearAlgebra.mul!(X::CuMatrix{T}, A::UpperTriangular{T, <:CuMatrix{T}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trmm!('L', 'U', 'N', 'N', one(T), parent(A), B, X)
LinearAlgebra.mul!(X::CuMatrix{T}, A::UnitUpperTriangular{T, <:CuMatrix{T}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trmm!('L', 'U', 'N', 'U', one(T), parent(A), B, X)
LinearAlgebra.mul!(X::CuMatrix{T}, A::LowerTriangular{T, <:CuMatrix{T}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trmm!('L', 'L', 'N', 'N', one(T), parent(A), B, X)
LinearAlgebra.mul!(X::CuMatrix{T}, A::UnitLowerTriangular{T, <:CuMatrix{T}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trmm!('L', 'L', 'N', 'U', one(T), parent(A), B, X)
## Adjoint
LinearAlgebra.mul!(X::CuMatrix{T}, A::Adjoint{T,<:UpperTriangular{T, <:CuMatrix{T}}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trmm!('L', 'U', 'C', 'N', one(T), parent(parent(A)), B, X)
LinearAlgebra.mul!(X::CuMatrix{T}, A::Adjoint{T,<:UnitUpperTriangular{T, <:CuMatrix{T}}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trmm!('L', 'U', 'C', 'U', one(T), parent(parent(A)), B, X)
LinearAlgebra.mul!(X::CuMatrix{T}, A::Adjoint{T,<:LowerTriangular{T, <:CuMatrix{T}}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trmm!('L', 'L', 'C', 'N', one(T), parent(parent(A)), B, X)
LinearAlgebra.mul!(X::CuMatrix{T}, A::Adjoint{T,<:UnitLowerTriangular{T, <:CuMatrix{T}}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trmm!('L', 'L', 'C', 'U', one(T), parent(parent(A)), B, X)
## Transpose
LinearAlgebra.mul!(X::CuMatrix{T}, A::Transpose{T,<:UpperTriangular{T, <:CuMatrix{T}}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trmm!('L', 'U', 'T', 'N', one(T), parent(parent(A)), B, X)
LinearAlgebra.mul!(X::CuMatrix{T}, A::Transpose{T,<:UnitUpperTriangular{T, <:CuMatrix{T}}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trmm!('L', 'U', 'T', 'U', one(T), parent(parent(A)), B, X)
LinearAlgebra.mul!(X::CuMatrix{T}, A::Transpose{T,<:LowerTriangular{T, <:CuMatrix{T}}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trmm!('L', 'L', 'T', 'N', one(T), parent(parent(A)), B, X)
LinearAlgebra.mul!(X::CuMatrix{T}, A::Transpose{T,<:UnitLowerTriangular{T, <:CuMatrix{T}}}, B::CuMatrix{T}) where T<:CublasFloat =
    CUBLAS.trmm!('L', 'L', 'T', 'U', one(T), parent(parent(A)), B, X)

# Right mul!
## No transpose/adjoint
LinearAlgebra.mul!(X::CuMatrix{T}, A::CuMatrix{T}, B::UpperTriangular{T, <:CuMatrix{T}}) where T<:CublasFloat =
    CUBLAS.trmm!('R', 'U', 'N', 'N', one(T), parent(B), A, X)
LinearAlgebra.mul!(X::CuMatrix{T}, A::CuMatrix{T}, B::UnitUpperTriangular{T, <:CuMatrix{T}}) where T<:CublasFloat =
    CUBLAS.trmm!('R', 'U', 'N', 'U', one(T), parent(B), A, X)
LinearAlgebra.mul!(X::CuMatrix{T}, A::CuMatrix{T}, B::LowerTriangular{T, <:CuMatrix{T}}) where T<:CublasFloat =
    CUBLAS.trmm!('R', 'L', 'N', 'N', one(T), parent(B), A, X)
LinearAlgebra.mul!(X::CuMatrix{T}, A::CuMatrix{T}, B::UnitLowerTriangular{T, <:CuMatrix{T}}) where T<:CublasFloat =
    CUBLAS.trmm!('R', 'L', 'N', 'U', one(T), parent(B), A, X)
## Adjoint
LinearAlgebra.mul!(X::CuMatrix{T}, A::CuMatrix{T}, B::Adjoint{T,<:UpperTriangular{T, <:CuMatrix{T}}}) where T<:CublasFloat =
    CUBLAS.trmm!('R', 'U', 'C', 'N', one(T), parent(parent(B)), A, X)
LinearAlgebra.mul!(X::CuMatrix{T}, A::CuMatrix{T}, B::Adjoint{T,<:UnitUpperTriangular{T, <:CuMatrix{T}}}) where T<:CublasFloat =
    CUBLAS.trmm!('R', 'U', 'C', 'U', one(T), parent(parent(B)), A, X)
LinearAlgebra.mul!(X::CuMatrix{T}, A::CuMatrix{T}, B::Adjoint{T,<:LowerTriangular{T, <:CuMatrix{T}}}) where T<:CublasFloat =
    CUBLAS.trmm!('R', 'L', 'C', 'N', one(T), parent(parent(B)), A, X)
LinearAlgebra.mul!(X::CuMatrix{T}, A::CuMatrix{T}, B::Adjoint{T,<:UnitLowerTriangular{T, <:CuMatrix{T}}}) where T<:CublasFloat =
    CUBLAS.trmm!('R', 'L', 'C', 'U', one(T), parent(parent(B)), A, X)
## Transpose
LinearAlgebra.mul!(X::CuMatrix{T}, A::CuMatrix{T}, B::Transpose{T,<:UpperTriangular{T, <:CuMatrix{T}}}) where T<:CublasFloat =
    CUBLAS.trmm!('R', 'U', 'T', 'N', one(T), parent(parent(B)), A, X)
LinearAlgebra.mul!(X::CuMatrix{T}, A::CuMatrix{T}, B::Transpose{T,<:UnitUpperTriangular{T, <:CuMatrix{T}}}) where T<:CublasFloat =
    CUBLAS.trmm!('R', 'U', 'T', 'U', one(T), parent(parent(B)), A, X)
LinearAlgebra.mul!(X::CuMatrix{T}, A::CuMatrix{T}, B::Transpose{T,<:LowerTriangular{T, <:CuMatrix{T}}}) where T<:CublasFloat =
    CUBLAS.trmm!('R', 'L', 'T', 'N', one(T), parent(parent(B)), A, X)
LinearAlgebra.mul!(X::CuMatrix{T}, A::CuMatrix{T}, B::Transpose{T,<:UnitLowerTriangular{T, <:CuMatrix{T}}}) where T<:CublasFloat =
    CUBLAS.trmm!('R', 'L', 'T', 'U', one(T), parent(parent(B)), A, X)


# Direct BLAS calls
for T in Base.uniontypes(CublasFloat) # needed to avoid ambiguous method error
    @eval LinearAlgebra.BLAS.trmm!(side::AbstractChar, uplo::AbstractChar, transa::AbstractChar, diag::AbstractChar, alpha::$T, A::CuMatrix{$T}, B::CuMatrix{$T}) =
        trmm!(side, uplo, transa, diag, alpha, A, B, B)
    @eval LinearAlgebra.BLAS.trsm!(side::AbstractChar, uplo::AbstractChar, transa::AbstractChar, diag::AbstractChar, alpha::$T, A::CuMatrix{$T}, B::CuMatrix{$T}) =
        trsm!(side, uplo, transa, diag, alpha, A, B)
end
