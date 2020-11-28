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

function gemv_dispatch!(Y::CuVector, A, B, alpha::Number=true, beta::Number=false)
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

    T = eltype(Y)
    if T <: CublasFloat && A isa StridedCuArray{T} && B isa StridedCuArray{T}
        gemv!(tA, alpha, dA, B, beta, Y)
    else
        gemm_dispatch!(Y, A, B, alpha, beta)
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

# triangular

## direct multiplication/division
for (t, uploc, isunitc) in ((:LowerTriangular, 'L', 'N'),
                            (:UnitLowerTriangular, 'L', 'U'),
                            (:UpperTriangular, 'U', 'N'),
                            (:UnitUpperTriangular, 'U', 'U'))
    @eval begin
        # Multiplication
        LinearAlgebra.lmul!(A::$t{T,<:DenseCuMatrix},
                            b::StridedCuVector{T}) where {T<:CublasFloat} =
            CUBLAS.trmv!($uploc, 'N', $isunitc, parent(A), b)

        # Left division
        LinearAlgebra.ldiv!(A::$t{T,<:DenseCuMatrix},
                            B::StridedCuVector{T}) where {T<:CublasFloat} =
            CUBLAS.trsv!($uploc, 'N', $isunitc, parent(A), B)
    end
end

## adjoint/transpose multiplication ('uploc' reversed)
for (t, uploc, isunitc) in ((:LowerTriangular, 'U', 'N'),
                            (:UnitLowerTriangular, 'U', 'U'),
                            (:UpperTriangular, 'L', 'N'),
                            (:UnitUpperTriangular, 'L', 'U'))
    @eval begin
        # Multiplication
        LinearAlgebra.lmul!(A::$t{<:Any,<:Transpose{T,<:DenseCuMatrix}},
                            b::DenseCuVector{T}) where {T<:CublasFloat} =
            CUBLAS.trmv!($uploc, 'T', $isunitc, parent(parent(A)), b)
        LinearAlgebra.lmul!(A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                            b::DenseCuVector{T}) where {T<:CublasReal} =
            CUBLAS.trmv!($uploc, 'T', $isunitc, parent(parent(A)), b)
        LinearAlgebra.lmul!(A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                            b::DenseCuVector{T}) where {T<:CublasComplex} =
            CUBLAS.trmv!($uploc, 'C', $isunitc, parent(parent(A)), b)

        # Left division
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Transpose{T,<:DenseCuMatrix}},
                            B::StridedCuVector{T}) where {T<:CublasFloat} =
            CUBLAS.trsv!($uploc, 'T', $isunitc, parent(parent(A)), B)
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                            B::StridedCuVector{T}) where {T<:CublasReal} =
            CUBLAS.trsv!($uploc, 'T', $isunitc, parent(parent(A)), B)
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                            B::StridedCuVector{T}) where {T<:CublasComplex} =
            CUBLAS.trsv!($uploc, 'C', $isunitc, parent(parent(A)), B)
    end
end



#
# BLAS 3
#

# GEMM

function gemm_dispatch!(C::CuVecOrMat, A, B, alpha::Number=true, beta::Number=false)
    if ndims(A) > 2
        throw(ArgumentError("A has more than 2 dimensions"))
    elseif ndims(B) > 2
        throw(ArgumentError("B has more than 2 dimensions"))
    end
    mA, nA = size(A,1), size(A,2)
    mB, nB = size(B,1), size(B,2)

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

    T = eltype(C)
    if dA isa DenseCuArray && dB isa DenseCuArray &&
       gemmExComputeType(eltype(A), eltype(B), eltype(C), mA, nA, nB) !== nothing
        gemmEx!(tA, tB, alpha, dA, dB, beta, C)
    elseif T <: CublasFloat && dA isa DenseCuArray{T} && dB isa DenseCuArray{T}
        gemm!(tA, tB, alpha, dA, dB, beta, C)
    else
        GPUArrays.generic_matmatmul!(C, A, B, alpha, beta)
    end
end

for NT in (Number, Real)
    # NOTE: alpha/beta also ::Real to avoid ambiguities with certain Base methods
    @eval begin
        LinearAlgebra.mul!(C::CuMatrix, A::StridedCuVecOrMat, B::StridedCuVecOrMat, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)

        LinearAlgebra.mul!(C::CuMatrix, A::Transpose{<:Any, <:StridedCuVecOrMat}, B::StridedCuMatrix, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::CuMatrix, A::StridedCuMatrix, B::Transpose{<:Any, <:StridedCuVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::CuMatrix, A::Transpose{<:Any, <:StridedCuVecOrMat}, B::Transpose{<:Any, <:StridedCuVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)

        LinearAlgebra.mul!(C::CuMatrix, A::Adjoint{<:Any, <:StridedCuVecOrMat}, B::StridedCuMatrix, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::CuMatrix, A::StridedCuMatrix, B::Adjoint{<:Any, <:StridedCuVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::CuMatrix, A::Adjoint{<:Any, <:StridedCuVecOrMat}, B::Adjoint{<:Any, <:StridedCuVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)

        LinearAlgebra.mul!(C::CuMatrix, A::Transpose{<:Any, <:StridedCuVecOrMat}, B::Adjoint{<:Any, <:StridedCuVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::CuMatrix, A::Adjoint{<:Any, <:StridedCuVecOrMat}, B::Transpose{<:Any, <:StridedCuVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
    end
end

# triangular

## direct multiplication/division
for (t, uploc, isunitc) in ((:LowerTriangular, 'L', 'N'),
                            (:UnitLowerTriangular, 'L', 'U'),
                            (:UpperTriangular, 'U', 'N'),
                            (:UnitUpperTriangular, 'U', 'U'))
    @eval begin
        # Multiplication
        LinearAlgebra.lmul!(A::$t{T,<:DenseCuMatrix},
                            B::DenseCuMatrix{T}) where {T<:CublasFloat} =
            CUBLAS.trmm!('L', $uploc, 'N', $isunitc, one(T), parent(A), B, B)
        LinearAlgebra.rmul!(A::DenseCuMatrix{T},
                            B::$t{T,<:DenseCuMatrix}) where {T<:CublasFloat} =
            CUBLAS.trmm!('R', $uploc, 'N', $isunitc, one(T), parent(B), A, A)

        # optimization: Base.mul! uses lmul!/rmul! with a copy (because of BLAS)
        LinearAlgebra.mul!(X::DenseCuMatrix{T}, A::$t{T,<:DenseCuMatrix},
                           B::DenseCuMatrix{T}) where {T<:CublasFloat} =
            CUBLAS.trmm!('L', $uploc, 'N', $isunitc, one(T), parent(A), B, X)
        LinearAlgebra.mul!(X::DenseCuMatrix{T}, A::DenseCuMatrix{T},
                           B::$t{T,<:DenseCuMatrix}) where {T<:CublasFloat} =
            CUBLAS.trmm!('R', $uploc, 'N', $isunitc, one(T), parent(B), A, X)

        # Left division
        LinearAlgebra.ldiv!(A::$t{T,<:DenseCuMatrix},
                            B::DenseCuMatrix{T}) where {T<:CublasFloat} =
            CUBLAS.trsm!('L', $uploc, 'N', $isunitc, one(T), parent(A), B)

        # Right division
        LinearAlgebra.rdiv!(A::DenseCuMatrix{T},
                            B::$t{T,<:DenseCuMatrix}) where {T<:CublasFloat} =
            CUBLAS.trsm!('R', $uploc, 'N', $isunitc, one(T), parent(B), A)

        # Matrix inverse
        function LinearAlgebra.inv(x::$t{T, <:CuMatrix{T}}) where T<:CublasFloat
            out = CuArray{T}(I(size(x,1)))
            $t(LinearAlgebra.ldiv!(x, out))
        end
    end
end

## adjoint/transpose multiplication ('uploc' reversed)
for (t, uploc, isunitc) in ((:LowerTriangular, 'U', 'N'),
                            (:UnitLowerTriangular, 'U', 'U'),
                            (:UpperTriangular, 'L', 'N'),
                            (:UnitUpperTriangular, 'L', 'U'))
    @eval begin
        # Multiplication
        LinearAlgebra.lmul!(A::$t{<:Any,<:Transpose{T,<:DenseCuMatrix}},
                            B::DenseCuMatrix{T}) where {T<:CublasFloat} =
            CUBLAS.trmm!('L', $uploc, 'T', $isunitc, one(T), parent(parent(A)), B, B)
        LinearAlgebra.lmul!(A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                            B::DenseCuMatrix{T}) where {T<:CublasComplex} =
            CUBLAS.trmm!('L', $uploc, 'C', $isunitc, one(T), parent(parent(A)), B, B)
        LinearAlgebra.lmul!(A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                            B::DenseCuMatrix{T}) where {T<:CublasReal} =
            CUBLAS.trmm!('L', $uploc, 'T', $isunitc, one(T), parent(parent(A)), B, B)

        LinearAlgebra.rmul!(A::DenseCuMatrix{T},
                            B::$t{<:Any,<:Transpose{T,<:DenseCuMatrix}}) where {T<:CublasFloat} =
            CUBLAS.trmm!('R', $uploc, 'T', $isunitc, one(T), parent(parent(B)), A, A)
        LinearAlgebra.rmul!(A::DenseCuMatrix{T},
                            B::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}}) where {T<:CublasComplex} =
            CUBLAS.trmm!('R', $uploc, 'C', $isunitc, one(T), parent(parent(B)), A, A)
        LinearAlgebra.rmul!(A::DenseCuMatrix{T},
                            B::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}}) where {T<:CublasReal} =
            CUBLAS.trmm!('R', $uploc, 'T', $isunitc, one(T), parent(parent(B)), A, A)

        # optimization: Base.mul! uses lmul!/rmul! with a copy (because of BLAS)
        LinearAlgebra.mul!(X::DenseCuMatrix{T}, A::$t{<:Any,<:Transpose{T,<:DenseCuMatrix}},
                           B::DenseCuMatrix{T}) where {T<:CublasFloat} =
            CUBLAS.trmm!('L', $uploc, 'T', $isunitc, one(T), parent(parent(A)), B, X)
        LinearAlgebra.mul!(X::DenseCuMatrix{T}, A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                           B::DenseCuMatrix{T}) where {T<:CublasComplex} =
            CUBLAS.trmm!('L', $uploc, 'C', $isunitc, one(T), parent(parent(A)), B, X)
        LinearAlgebra.mul!(X::DenseCuMatrix{T}, A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                           B::DenseCuMatrix{T}) where {T<:CublasReal} =
            CUBLAS.trmm!('L', $uploc, 'T', $isunitc, one(T), parent(parent(A)), B, X)
        LinearAlgebra.mul!(X::DenseCuMatrix{T}, A::DenseCuMatrix{T},
                           B::$t{<:Any,<:Transpose{T,<:DenseCuMatrix}}) where {T<:CublasFloat} =
            CUBLAS.trmm!('R', $uploc, 'T', $isunitc, one(T), parent(parent(B)), A, X)
        LinearAlgebra.mul!(X::DenseCuMatrix{T}, A::DenseCuMatrix{T},
                           B::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}}) where {T<:CublasComplex} =
            CUBLAS.trmm!('R', $uploc, 'C', $isunitc, one(T), parent(parent(B)), A, X)
        LinearAlgebra.mul!(X::DenseCuMatrix{T}, A::DenseCuMatrix{T},
                           B::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}}) where {T<:CublasReal} =
            CUBLAS.trmm!('R', $uploc, 'T', $isunitc, one(T), parent(parent(B)), A, X)

        # Left division
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Transpose{T,<:DenseCuMatrix}},
                            B::DenseCuMatrix{T}) where {T<:CublasFloat} =
            CUBLAS.trsm!('L', $uploc, 'T', $isunitc, one(T), parent(parent(A)), B)
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                            B::DenseCuMatrix{T}) where {T<:CublasReal} =
            CUBLAS.trsm!('L', $uploc, 'T', $isunitc, one(T), parent(parent(A)), B)
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}},
                            B::DenseCuMatrix{T}) where {T<:CublasComplex} =
            CUBLAS.trsm!('L', $uploc, 'C', $isunitc, one(T), parent(parent(A)), B)

        # Right division
        LinearAlgebra.rdiv!(A::DenseCuMatrix{T},
                            B::$t{<:Any,<:Transpose{T,<:DenseCuMatrix}}) where {T<:CublasFloat} =
            CUBLAS.trsm!('R', $uploc, 'T', $isunitc, one(T), parent(parent(B)), A)
        LinearAlgebra.rdiv!(A::DenseCuMatrix{T},
                            B::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}}) where {T<:CublasReal} =
            CUBLAS.trsm!('R', $uploc, 'T', $isunitc, one(T), parent(parent(B)), A)
        LinearAlgebra.rdiv!(A::DenseCuMatrix{T},
                            B::$t{<:Any,<:Adjoint{T,<:DenseCuMatrix}}) where {T<:CublasComplex} =
            CUBLAS.trsm!('R', $uploc, 'C', $isunitc, one(T), parent(parent(B)), A)
    end
end
