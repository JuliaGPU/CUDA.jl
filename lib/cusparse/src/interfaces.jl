# interfacing with other packages
#
# GPUArrays implements LinearAlgebra and SparseArrays for the canonical sparse formats
# with portable kernels. The methods here override the products and triangular solves
# where cuSPARSE applies: the element types it supports, 32- or 64-bit indices (32-bit
# for the legacy routines), and contiguous dense operands. Other cases fall through to
# GPUArrays' generic implementations (`GPUArrays.generic_spmv!` etc.).

using LinearAlgebra
using LinearAlgebra: BlasComplex, BlasFloat, BlasReal, AdjOrTrans, HermOrSym

function mv_wrapper(transa::SparseChar, alpha::Number, A::CuSparseMatrix, X::DenseCuVector{T},
                    beta::Number, Y::CuVector{T}) where {T}
    mv!(transa, alpha, A, X, beta, Y, 'O')
end

# element types accepted by cuSPARSE's generic SpMV/SpMM API (`mv!`, `mm!`, `bmm!`).
# half-precision types are not in `BlasFloat` because CPU BLAS doesn't support them;
# cuSPARSE handles them by accumulating in Float32/ComplexF32 (see `mv!` in generic.jl).
# Routines based on the classical BLAS bindings (`gemvi!`, `gemm!`, `geam`, ...) stay
# restricted to plain `BlasFloat`.
const SpMatMulFloat = Union{Float16, ComplexF16, BlasFloat}

# index types of the generic API; the legacy routines only take `Cint`
const SpIndex = Union{Int32, Int64}

# the canonical formats cuSPARSE's generic API takes
const CuVendorMatrix{Tv} = Union{CuSparseMatrixCSR{Tv,<:SpIndex},CuSparseMatrixCSC{Tv,<:SpIndex},
                                 CuSparseMatrixCOO{Tv,<:SpIndex}}

# convert the element type of a sparse matrix or vector (issue #3128: mixed-eltype
# matmul), sharing the index buffers
sparse_with_eltype(A::GPUArrays.GPUSparseArray, ::Type{T}) where {T} =
    GPUArrays.with_eltypes(A, T, SparseArrays.indtype(A))
sparse_with_eltype(A::CuSparseMatrixBSR{<:Any, Ti}, ::Type{T}) where {T, Ti} =
    CuSparseMatrixBSR{T, Ti}(A.rowPtr, A.colVal, convert(CuVector{T}, nonzeros(A)), size(A),
                             A.blockDim, A.dir, A.nnzb)

# LinearAlgebra's storage-level `mul!` passes a transposed, adjoint, `Symmetric` or
# `Hermitian` operand as its parent and a character. cuSPARSE has no symmetric or
# Hermitian products, so materialize the full matrix that the wrapper represents.
# (issue #2572)
function unwrap_symmherm(t::AbstractChar, A)
    uppercase(t) in ('S', 'H') || return t, A
    A isa CuSparseMatrixBSR && (A = CuSparseMatrixCSR(A))
    return 'N', materialize_symmherm(LinearAlgebra.wrap(A, t))
end
materialize_symmherm(A::HermOrSym{<:Any,<:GPUArrays.GPUSparseMatrix}) = GPUArrays.materialize(A)
function materialize_symmherm(A::HermOrSym{<:Any,<:DenseCuMatrix})
    # `CuMatrix(A)` would go through the CPU
    B = LinearAlgebra.copytri!(copy(parent(A)), A.uplo, A isa Hermitian)
    if A isa Hermitian
        d = view(B, diagind(B))
        d .= real.(d)
    end
    return B
end

# SpGEMM only multiplies untransposed operands
vendor_operand(t::AbstractChar, A) = t == 'N' ? A : GPUArrays.materialize(LinearAlgebra.wrap(A, t))


## products

check_dot_dims(x, y) = length(x) == length(y) ||
    throw(DimensionMismatch("x has length $(length(x)), y has length $(length(y))"))
LinearAlgebra.dot(x::CuSparseVector{T,<:SpIndex}, y::DenseCuVector{T}) where {T <: BlasReal} =
    (check_dot_dims(x, y); vv!('N', x, y, 'O'))
LinearAlgebra.dot(x::DenseCuVector{T}, y::CuSparseVector{T,<:SpIndex}) where {T <: BlasReal} = dot(y, x)

LinearAlgebra.dot(x::CuSparseVector{T,<:SpIndex}, y::DenseCuVector{T}) where {T <: BlasComplex} =
    (check_dot_dims(x, y); vv!('C', x, y, 'O'))
LinearAlgebra.dot(x::DenseCuVector{T}, y::CuSparseVector{T,<:SpIndex}) where {T <: BlasComplex} = conj(dot(y,x))

# The products below accept inputs of any numeric eltype (e.g. integer or Bool matrices,
# issue #2072), converting those cuSPARSE doesn't support to the eltype of the output.

# mv! tolerates A.eltype ≠ X.eltype for floating-point eltypes, but requires
# X.eltype == Y.eltype.
function sparse_mv!(C::CuVector{Tc}, tA::AbstractChar, A, B::DenseCuVector{Tb}, alpha::Number, beta::Number) where {Tc, Tb}
    tA, A = unwrap_symmherm(tA, A)
    A′ = eltype(A) <: SpMatMulFloat ? A : sparse_with_eltype(A, Tc)
    B′ = Tb === Tc ? B : convert(CuVector{Tc}, B)
    mv_wrapper(tA, alpha, A′, B′, beta, C)
end
LinearAlgebra.mul!(C::CuVector{Tc}, tA::AbstractChar, A::CuVendorMatrix{<:Number}, B::DenseCuVector{<:Number}, alpha::Number, beta::Number) where {Tc <: SpMatMulFloat} =
    sparse_mv!(C, tA, A, B, alpha, beta)
LinearAlgebra.mul!(C::CuVector{Tc}, tA::AbstractChar, A::CuSparseMatrixBSR{<:Number}, B::DenseCuVector{<:Number}, alpha::Number, beta::Number) where {Tc <: SpMatMulFloat} =
    sparse_mv!(C, tA, cint_view(A), B, alpha, beta)

# mm! requires all three operands to share an eltype, so promote A and B to match C.
function sparse_mm!(C::CuMatrix{Tc}, tA, tB, A, B::DenseCuMatrix{Tb}, alpha::Number, beta::Number) where {Tc, Tb}
    tA, A = unwrap_symmherm(tA, A)
    tB, B = unwrap_symmherm(tB, B)
    A′ = eltype(A) === Tc ? A : sparse_with_eltype(A, Tc)
    B′ = Tb === Tc ? B : convert(CuMatrix{Tc}, B)
    mm!(tA, tB, alpha, A′, B′, beta, C, 'O')
end
# The half-precision SpMM of cuSPARSE (which accumulates in single precision) supports
# fewer layouts than the other element types; the generic implementation covers the rest.
effective_char(t::AbstractChar) = uppercase(t) in ('S', 'H') ? 'N' : t
function half_spmm_supported(::Type{Tc}, sparse_left::Bool, S, ts, td) where {Tc}
    Tc <: Union{Float16, ComplexF16} || return true
    # (ComplexF16 fails for layouts that work with Float16)
    Tc === ComplexF16 && return false
    ts, td = effective_char(ts), effective_char(td)
    return sparse_left ? !(S isa CuSparseMatrixCSC) && ts == 'N' :
           S isa CuSparseMatrixCSC ? ts == 'N' : S isa CuSparseMatrixCOO && ts != 'N'
end

function LinearAlgebra.mul!(C::CuMatrix{Tc}, tA::AbstractChar, tB::AbstractChar, A::CuVendorMatrix{<:Number}, B::DenseCuMatrix{<:Number}, alpha::Number, beta::Number) where {Tc <: SpMatMulFloat}
    half_spmm_supported(Tc, true, A, tA, tB) ||
        return GPUArrays.generic_spmm!(C, tA, tB, A, B, alpha, beta)
    sparse_mm!(C, tA, tB, A, B, alpha, beta)
end
LinearAlgebra.mul!(C::CuMatrix{Tc}, tA::AbstractChar, tB::AbstractChar, A::CuSparseMatrixBSR{<:Number}, B::DenseCuMatrix{<:Number}, alpha::Number, beta::Number) where {Tc <: SpMatMulFloat} =
    sparse_mm!(C, tA, tB, cint_view(A), B, alpha, beta)

# mm! requires matching eltypes, so promote A and B to match C.
function LinearAlgebra.mul!(C::CuMatrix{Tc}, tA::AbstractChar, tB::AbstractChar, A::DenseCuMatrix{Ta}, B::CuVendorMatrix{Tb}, alpha::Number, beta::Number) where {Tc <: SpMatMulFloat, Ta <: Number, Tb <: Number}
    half_spmm_supported(Tc, false, B, tB, tA) ||
        return GPUArrays.generic_spmm!(C, tA, tB, A, B, alpha, beta)
    tA, A = unwrap_symmherm(tA, A)
    tB, B = unwrap_symmherm(tB, B)
    A′ = Ta === Tc ? A : convert(CuMatrix{Tc}, A)
    B′ = eltype(B) === Tc ? B : sparse_with_eltype(B, Tc)
    mm!(tA, tB, alpha, A′, B′, beta, C, 'O')
end

# gemvi! requires matching eltypes, so promote A and B to match C.
function LinearAlgebra.mul!(C::CuVector{Tc}, tA::AbstractChar, A::DenseCuMatrix{Ta}, B::CuSparseVector{Tb,Cint}, alpha::Number, beta::Number) where {Tc <: BlasFloat, Ta <: Number, Tb <: Number}
    tA, A = unwrap_symmherm(tA, A)
    A′ = Ta === Tc ? A : convert(CuMatrix{Tc}, A)
    B′ = Tb === Tc ? B : sparse_with_eltype(B, Tc)
    gemvi!(tA, alpha, A′, B′, beta, C, 'O')
end

# SpGEMM, which only supports 32-bit indices. gemm! requires matching eltypes, so promote
# A and B to match C.
for SparseMatrixType in (:CuSparseMatrixCSC, :CuSparseMatrixCSR)
    @eval function LinearAlgebra.mul!(C::$SparseMatrixType{Tc}, tA::AbstractChar, tB::AbstractChar, A::$SparseMatrixType{Ta,Cint}, B::$SparseMatrixType{Tb,Cint}, alpha::Number, beta::Number) where {Tc <: BlasFloat, Ta <: Number, Tb <: Number}
        A = vendor_operand(tA, A)
        B = vendor_operand(tB, B)
        A′ = Ta === Tc ? A : sparse_with_eltype(A, Tc)
        B′ = Tb === Tc ? B : sparse_with_eltype(B, Tc)
        gemm!('N', 'N', alpha, A′, B′, beta, C, 'O')
    end
end
function LinearAlgebra.mul!(C::CuSparseMatrixCOO{Tc}, tA::AbstractChar, tB::AbstractChar, A::CuSparseMatrixCOO{Ta,Cint}, B::CuSparseMatrixCOO{Tb,Cint}, alpha::Number, beta::Number) where {Tc <: BlasFloat, Ta <: Number, Tb <: Number}
    C_csr = CuSparseMatrixCSR(C)
    LinearAlgebra.mul!(C_csr, tA, tB, CuSparseMatrixCSR(A), CuSparseMatrixCSR(B), alpha, beta)
    copyto!(C, CuSparseMatrixCOO(C_csr))
    return C
end

for SparseMatrixType in (:CuSparseMatrixCSC, :CuSparseMatrixCSR)
    @eval function LinearAlgebra.:(*)(A::$SparseMatrixType{T,Cint}, B::$SparseMatrixType{T,Cint}) where {T <: BlasFloat}
        gemm('N', 'N', one(T), A, B, 'O')
    end
end
function LinearAlgebra.:(*)(A::CuSparseMatrixCOO{T,Cint}, B::CuSparseMatrixCOO{T,Cint}) where {T <: BlasFloat}
    CuSparseMatrixCOO(CuSparseMatrixCSR(A) * CuSparseMatrixCSR(B))
end

# Julia < 1.13 dispatches on the non-public `generic_matvecmul!` and `generic_matmatmul!`.
# GPUArrays forwards those to `mul!` for the canonical formats; do the same for BSR
# matrices and for dense × sparse vector products.
@static if VERSION < v"1.13.0-rc4"
    for (TA, TB, F) in ((:(CuSparseMatrixBSR{Ta}), :(DenseCuVector{Tb}), :SpMatMulFloat),
                        (:(DenseCuMatrix{Ta}), :(CuSparseVector{Tb,Cint}), :BlasFloat))
        @eval begin
            LinearAlgebra.generic_matvecmul!(C::CuVector{Tc}, tA::AbstractChar, A::$TA, B::$TB, alpha::Number, beta::Number) where {Tc <: $F, Ta <: Number, Tb <: Number} =
                mul!(C, tA, A, B, alpha, beta)
            LinearAlgebra.generic_matvecmul!(C::CuVector{Tc}, tA::AbstractChar, A::$TA, B::$TB, _add::LinearAlgebra.MulAddMul) where {Tc <: $F, Ta <: Number, Tb <: Number} =
                mul!(C, tA, A, B, _add.alpha, _add.beta)
        end
    end
    @eval begin
        LinearAlgebra.generic_matmatmul!(C::CuMatrix{Tc}, tA, tB, A::CuSparseMatrixBSR{Ta}, B::DenseCuMatrix{Tb}, alpha::Number, beta::Number) where {Tc <: SpMatMulFloat, Ta <: Number, Tb <: Number} =
            mul!(C, tA, tB, A, B, alpha, beta)
        LinearAlgebra.generic_matmatmul!(C::CuMatrix{Tc}, tA, tB, A::CuSparseMatrixBSR{Ta}, B::DenseCuMatrix{Tb}, _add::LinearAlgebra.MulAddMul) where {Tc <: SpMatMulFloat, Ta <: Number, Tb <: Number} =
            mul!(C, tA, tB, A, B, _add.alpha, _add.beta)
    end
end


## BSR matrices

# arithmetic converts to CSR, which GPUArrays implements
for op in (:+, :-)
    @eval begin
        Base.$op(A::CuSparseMatrixBSR, B::GPUArrays.GPUSparseMatrix) = $op(CuSparseMatrixCSR(A), B)
        Base.$op(A::GPUArrays.GPUSparseMatrix, B::CuSparseMatrixBSR) = $op(A, CuSparseMatrixCSR(B))
        Base.$op(A::CuSparseMatrixBSR, B::CuSparseMatrixBSR) = $op(CuSparseMatrixCSR(A), CuSparseMatrixCSR(B))
    end
end

LinearAlgebra.istriu(A::CuSparseMatrixBSR, k::Integer=0) = istriu(CuSparseMatrixCSR(A), k)
LinearAlgebra.istril(A::CuSparseMatrixBSR, k::Integer=0) = istril(CuSparseMatrixCSR(A), k)

# Triangular wrappers answer by their structure, so that LinearAlgebra's triangular solves
# reach the methods below (the generic ones would index the parent).
const CuSparseBSROperand = Union{CuSparseMatrixBSR,Transpose{<:Any,<:CuSparseMatrixBSR},
                                 Adjoint{<:Any,<:CuSparseMatrixBSR}}
LinearAlgebra.istriu(::Union{UpperTriangular{<:Any,<:CuSparseBSROperand},
                             UnitUpperTriangular{<:Any,<:CuSparseBSROperand}}) = true
LinearAlgebra.istril(::Union{UpperTriangular{<:Any,<:CuSparseBSROperand},
                             UnitUpperTriangular{<:Any,<:CuSparseBSROperand}}) = false
LinearAlgebra.istril(::Union{LowerTriangular{<:Any,<:CuSparseBSROperand},
                             UnitLowerTriangular{<:Any,<:CuSparseBSROperand}}) = true
LinearAlgebra.istriu(::Union{LowerTriangular{<:Any,<:CuSparseBSROperand},
                             UnitLowerTriangular{<:Any,<:CuSparseBSROperand}}) = false

# `one` and `oneunit`, like SparseArrays does for SparseMatrixCSC
function Base._one(unit::Tv, A::CuSparseMatrixBSR{<:Any,Ti}) where {Tv,Ti}
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("multiplicative identity only defined for square matrices"))
    n = size(A, 1)
    b = A.blockDim
    mb = cld(n, b)

    # a diagonal block for every block row
    rowPtr = CuVector{Ti}(1:mb+1)
    colVal = CuVector{Ti}(1:mb)

    # the blocks themselves are diagonal, so their layout doesn't depend on `A.dir`
    nzVal = CUDACore.zeros(Tv, b*b*mb)
    diag = view(reshape(nzVal, b*b, mb), 1:b+1:b*b, :)
    diag .= unit
    # when `n` isn't a multiple of the block size, the last block extends past the matrix
    pad = mb*b - n
    pad > 0 && (diag[end-pad+1:end, end] .= zero(Tv))

    CuSparseMatrixBSR{Tv,Ti}(rowPtr, colVal, nzVal, size(A), b, A.dir, mb)
end


## triangular solves

for SparseMatrixType in (:CuSparseMatrixBSR,)
    @eval begin
        LinearAlgebra.generic_trimatdiv!(C::DenseCuVector{T}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::DenseCuVector{T}) where {T<:BlasFloat} =
            sv2!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', uploc, isunitc, one(T), A, C === B ? C : copyto!(C, B), 'O')
        LinearAlgebra.generic_trimatdiv!(C::DenseCuMatrix{T}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::DenseCuMatrix{T}) where {T<:BlasFloat} =
            sm2!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', 'N', uploc, isunitc, one(T), A, C === B ? C : copyto!(C, B), 'O')
        function LinearAlgebra.generic_trimatdiv!(C::DenseCuMatrix{T}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::AdjOrTrans{T,<:DenseCuMatrix{T}}) where {T<:BlasFloat}
            transb = LinearAlgebra.wrapper_char(B)
            (transb == 'C') && (T <: Complex) && throw(ErrorException("This operation is not supported by the current CUDA version."))
            C !== parent(B) && copyto!(C, B)
            sm2!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', 'N', uploc, isunitc, one(T), A, C, 'O')
        end
        function LinearAlgebra.generic_trimatdiv!(C::Transpose{T,<:DenseCuMatrix{T}}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::Transpose{T,<:DenseCuMatrix{T}}) where {T<:BlasFloat}
            B === C || throw(ErrorException("This operation is only supported if B and C are identical."))
            sm2!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', 'T', uploc, isunitc, one(T), A, parent(B), 'O')
        end
        function LinearAlgebra.generic_trimatdiv!(C::Adjoint{T,<:DenseCuMatrix{T}}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::Adjoint{T,<:DenseCuMatrix{T}}) where {T<:BlasFloat}
            B === C || throw(ErrorException("This operation is only supported if B and C are identical."))
            sm2!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', 'C', uploc, isunitc, one(T), A, parent(B), 'O')
        end
    end
end # SparseMatrixType loop

for SparseMatrixType in (:CuSparseMatrixCOO, :CuSparseMatrixCSR, :CuSparseMatrixCSC)
    @eval begin
        function LinearAlgebra.generic_trimatdiv!(C::DenseCuVector{T}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::DenseCuVector{T}) where {T<:BlasFloat}
            sv!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', uploc, isunitc, one(T), A, B, C, 'O')
        end
        function LinearAlgebra.generic_trimatdiv!(C::DenseCuMatrix{T}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::DenseCuMatrix{T}) where {T<:BlasFloat}
            sm!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', 'N', uploc, isunitc, one(T), A, B, C, 'O')
        end
        function LinearAlgebra.generic_trimatdiv!(C::DenseCuMatrix{T}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::AdjOrTrans{T,<:DenseCuMatrix{T}}) where {T<:BlasFloat}
            transb = LinearAlgebra.wrapper_char(B)
            (transb == 'C') && (T <: Complex) && throw(ErrorException("This operation is not supported by the current CUDA version."))
            sm!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', transb, uploc, isunitc, one(T), A, parent(B), C, 'O')
        end
        function LinearAlgebra.generic_trimatdiv!(C::Transpose{T,<:DenseCuMatrix{T}}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::Transpose{T,<:DenseCuMatrix{T}}) where {T<:BlasFloat}
            (B !== C) && throw(ErrorException("This operation is only supported if B and C are identical."))
            sm!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', 'T', uploc, isunitc, one(T), A, parent(B), parent(C), 'O')
        end
        function LinearAlgebra.generic_trimatdiv!(C::Adjoint{T,<:DenseCuMatrix{T}}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::Adjoint{T,<:DenseCuMatrix{T}}) where {T<:BlasFloat}
            (B !== C) && throw(ErrorException("This operation is only supported if B and C are identical."))
            sm!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', 'C', uploc, isunitc, one(T), A, parent(B), parent(C), 'O')
        end
    end
end # SparseMatrixType loop
