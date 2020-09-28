# interfacing with LinearAlgebra standard library


#
# BLAS 1
#

LinearAlgebra.rmul!(x::StridedCuArray{<:CublasFloat}, k::Number) =
  scal!(length(x), k, x)

# Work around ambiguity with GPUArrays wrapper
LinearAlgebra.rmul!(x::DenseCuArray{<:CublasFloat}, k::Real) =
  invoke(rmul!, Tuple{typeof(x), Number}, x, k)

function LinearAlgebra.BLAS.dot(DX::StridedCuArray{T}, DY::StridedCuArray{T}) where T<:Union{Float32,Float64}
    n = length(DX)
    n==length(DY) || throw(DimensionMismatch("dot product arguments have lengths $(length(DX)) and $(length(DY))"))
    dot(n, DX, DY)
end

function LinearAlgebra.BLAS.dotc(DX::StridedCuArray{T}, DY::StridedCuArray{T}) where T<:Union{ComplexF32,ComplexF64}
    n = length(DX)
    n==length(DY) || throw(DimensionMismatch("dot product arguments have lengths $(length(DX)) and $(length(DY))"))
    dotc(n, DX, DY)
end

function LinearAlgebra.BLAS.dot(DX::DenseCuArray{T}, DY::DenseCuArray{T}) where T<:Union{ComplexF32,ComplexF64}
    BLAS.dotc(DX, DY)
end

function LinearAlgebra.BLAS.dotu(DX::StridedCuArray{T}, DY::StridedCuArray{T}) where T<:Union{ComplexF32,ComplexF64}
    n = length(DX)
    n==length(DY) || throw(DimensionMismatch("dot product arguments have lengths $(length(DX)) and $(length(DY))"))
    dotu(n, DX, DY)
end

LinearAlgebra.norm(x::DenseCuArray{<:CublasFloat}) = nrm2(x)
LinearAlgebra.BLAS.asum(x::StridedCuArray{<:CublasFloat}) = asum(length(x), x)

function LinearAlgebra.axpy!(alpha::Number, x::StridedCuArray{T}, y::StridedCuArray{T}) where T<:CublasFloat
    length(x)==length(y) || throw(DimensionMismatch("axpy arguments have lengths $(length(x)) and $(length(y))"))
    axpy!(length(x), alpha, x, y)
end

function LinearAlgebra.axpby!(alpha::Number, x::StridedCuArray{T}, beta::Number, y::StridedCuArray{T}) where T<:CublasFloat
    length(x)==length(y) || throw(DimensionMismatch("axpby arguments have lengths $(length(x)) and $(length(y))"))
    axpby!(length(x), alpha, x, beta, y)
end

function LinearAlgebra.rotate!(x::StridedCuArray{T}, y::StridedCuArray{T}, c, s) where T<:CublasFloat
    nx = length(x)
    ny = length(y)
    nx==ny || throw(DimensionMismatch("rotate arguments have lengths $nx and $ny"))
    rot!(nx, x, y, c, s)
end

function LinearAlgebra.reflect!(x::StridedCuArray{T}, y::StridedCuArray{T}, c, s) where T<:CublasFloat
    nx = length(x)
    ny = length(y)
    nx==ny || throw(DimensionMismatch("reflect arguments have lengths $nx and $ny"))
    rot!(nx, x, y, c, s)
    scal!(ny, -real(one(T)), y)
    x, y
end



#
# BLAS 2
#

# GEMV

function gemv_dispatch!(Y::CuVector{T}, A, B, alpha::Number=true, beta::Number=false) where T
    mA, nA = size(A)

    if nA != length(B)
        throw(DimensionMismatch("second dimension of A, $nA, does not match length of B, $(length(B))"))
    end

    if mA != length(Y)
        throw(DimensionMismatch("first dimension of A, $mA, does not match length of Y, $(length(Y))"))
    end

    if mA == 0
        return Y
    end

    if nA == 0
        return rmul!(Y, 0)
    end

    tA, dA = if A isa Transpose
        'T', parent(A)
    elseif A isa Adjoint
        'C', parent(A)
    else
        'N', A
    end

    if T <: CublasFloat && A isa StridedCuArray{T} && B isa StridedCuArray{T}
        gemv!(tA, alpha, dA, B, beta, Y)
    else
        # XXX: re-use gemm_dispatch! here? it currently doesn't support strided inputs,
        #      in part because cublasGemmEx doesn't.
        GPUArrays.generic_matmatmul!(Y, A, B, alpha, beta)
    end
end

for NT in (Number, Real)
    # NOTE: alpha/beta also ::Real to avoid ambiguities with certain Base methods
    @eval begin

        LinearAlgebra.mul!(Y::CuVector, A::StridedCuMatrix, B::StridedCuVector, a::$NT, b::$NT) =
            gemv_dispatch!(Y, A, B, a, b)
        LinearAlgebra.mul!(Y::CuVector, A::Transpose{<:Any, <:StridedCuVecOrMat}, B::StridedCuVector, a::$NT, b::$NT) =
            gemv_dispatch!(Y, A, B, a, b)
        LinearAlgebra.mul!(Y::CuVector, A::Adjoint{<:Any, <:StridedCuVecOrMat}, B::StridedCuVector, a::$NT, b::$NT) =
            gemv_dispatch!(Y, A, B, a, b)
    end
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

function gemm_dispatch!(C::CuVecOrMat, A, B, alpha::Number=true, beta::Number=false)
    mA, nA = size(A)
    mB, nB = size(B)

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

    tA, dA = if A isa Transpose
        'T', parent(A)
    elseif A isa Adjoint
        'C', parent(A)
    else
        'N', A
    end

    tB, dB = if B isa Transpose
        'T', parent(B)
    elseif B isa Adjoint
        'C', parent(B)
    else
        'N', B
    end

    if gemmExComputeType(eltype(A), eltype(B), eltype(C), mA, nA, nB) !== nothing
        gemmEx!(tA, tB, alpha, dA, dB, beta, C)
    elseif eltype(A) === eltype(B) === eltype(C) && eltype(A) <: CublasFloat
        gemm!(tA, tB, alpha, dA, dB, beta, C)
    else
        GPUArrays.generic_matmatmul!(C, A, B, alpha, beta)
    end
end

for NT in (Number, Real)
    # NOTE: alpha/beta also ::Real to avoid ambiguities with certain Base methods
    @eval begin
        LinearAlgebra.mul!(C::CuMatrix, A::CuVecOrMat, B::CuVecOrMat, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)

        LinearAlgebra.mul!(C::CuMatrix, A::Transpose{<:Any, <:CuVecOrMat}, B::CuMatrix, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::CuMatrix, A::CuMatrix, B::Transpose{<:Any, <:CuVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::CuMatrix, A::Transpose{<:Any, <:CuVecOrMat}, B::Transpose{<:Any, <:CuVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)

        LinearAlgebra.mul!(C::CuMatrix, A::Adjoint{<:Any, <:CuVecOrMat}, B::CuMatrix, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::CuMatrix, A::CuMatrix, B::Adjoint{<:Any, <:CuVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::CuMatrix, A::Adjoint{<:Any, <:CuVecOrMat}, B::Adjoint{<:Any, <:CuVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)

        LinearAlgebra.mul!(C::CuMatrix, A::Transpose{<:Any, <:CuVecOrMat}, B::Adjoint{<:Any, <:CuVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::CuMatrix, A::Adjoint{<:Any, <:CuVecOrMat}, B::Transpose{<:Any, <:CuVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
    end
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
