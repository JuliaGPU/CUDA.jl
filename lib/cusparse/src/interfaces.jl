# interfacing with other packages

using LinearAlgebra
using LinearAlgebra: BlasComplex, BlasFloat, BlasReal, MulAddMul, AdjOrTrans, HermOrSym
using GPUArrays: _spadjoint, _sptranspose

function GPUArrays._spadjoint(A::CuSparseMatrixCSR)
    Aᴴ = CuSparseMatrixCSC(A.rowPtr, A.colVal, conj(A.nzVal), reverse(size(A)))
    CuSparseMatrixCSR(Aᴴ)
end
function GPUArrays._sptranspose(A::CuSparseMatrixCSR)
    Aᵀ = CuSparseMatrixCSC(A.rowPtr, A.colVal, A.nzVal, reverse(size(A)))
    CuSparseMatrixCSR(Aᵀ)
end
function GPUArrays._spadjoint(A::CuSparseMatrixCSC)
    Aᴴ = CuSparseMatrixCSR(A.colPtr, A.rowVal, conj(A.nzVal), reverse(size(A)))
    CuSparseMatrixCSC(Aᴴ)
end
function GPUArrays._sptranspose(A::CuSparseMatrixCSC)
    Aᵀ = CuSparseMatrixCSR(A.colPtr, A.rowVal, A.nzVal, reverse(size(A)))
    CuSparseMatrixCSC(Aᵀ)
end
function GPUArrays._spadjoint(A::CuSparseMatrixCOO)
    # we use sparse instead of CuSparseMatrixCOO because we want to sort the matrix.
    sparse(A.colInd, A.rowInd, conj(A.nzVal), reverse(size(A))..., fmt = :coo)
end
function GPUArrays._sptranspose(A::CuSparseMatrixCOO)
    # we use sparse instead of CuSparseMatrixCOO because we want to sort the matrix.
    sparse(A.colInd, A.rowInd, A.nzVal, reverse(size(A))..., fmt = :coo)
end

function mv_wrapper(transa::SparseChar, alpha::Number, A::CuSparseMatrix, X::DenseCuVector{T},
                    beta::Number, Y::CuVector{T}) where {T}
    mv!(transa, alpha, A, X, beta, Y, 'O')
end

function mm_wrapper(transa::SparseChar, transb::SparseChar, alpha::Number,
                    A::CuSparseMatrix{T}, B::CuMatrix{T}, beta::Number, C::CuMatrix{T}) where {T}
    n_A, m_A = (transa != 'N') ? reverse(size(A)) : size(A)
    n_B, m_B = (transb != 'N') ? reverse(size(B)) : size(B)
    n_C, m_C = size(C)
    m_A == n_B || throw(DimensionMismatch())
    n_A == n_C || throw(DimensionMismatch())
    m_B == m_C || throw(DimensionMismatch())
    isempty(B) && return CUDACore.zeros(eltype(B), size(A, 1), 0)
    mm!(transa, transb, alpha, A, B, beta, C, 'O')
end

# element types accepted by cuSPARSE's generic SpMV/SpMM API (`mv!`, `mm!`, `bmm!`).
# half-precision types are not in `BlasFloat` because CPU BLAS doesn't support them;
# cuSPARSE handles them by accumulating in Float32/ComplexF32 (see `mv!` in generic.jl).
# Routines based on the classical BLAS bindings (`gemvi!`, `gemm!`, `geam`, ...) stay
# restricted to plain `BlasFloat`.
const SpMatMulFloat = Union{Float16, ComplexF16, BlasFloat}

# convert the element type of a CuSparseMatrix/Vector (issue #3128: mixed-eltype matmul)
sparse_with_eltype(A::CuSparseMatrixCSC, ::Type{T}) where {T} =
    CuSparseMatrixCSC(A.colPtr, A.rowVal, convert(CuVector{T}, nonzeros(A)), size(A))
sparse_with_eltype(A::CuSparseMatrixCSR, ::Type{T}) where {T} =
    CuSparseMatrixCSR(A.rowPtr, A.colVal, convert(CuVector{T}, nonzeros(A)), size(A))
sparse_with_eltype(A::CuSparseMatrixCOO, ::Type{T}) where {T} =
    CuSparseMatrixCOO(A.rowInd, A.colInd, convert(CuVector{T}, nonzeros(A)), size(A), nnz(A))
sparse_with_eltype(A::CuSparseVector, ::Type{T}) where {T} =
    CuSparseVector(nonzeroinds(A), convert(CuVector{T}, nonzeros(A)), length(A))

# Mixed-eltype `dense × sparse`: Julia's generic `*(::AbstractMatrix, ::AbstractMatrix)`
# allocates its output as `similar(B, TS)` (on 1.10; via `matprod_dest` on 1.11+); when
# B is sparse, that returns a `CuSparseMatrix`, which then forces a scalar fallback.
# Allocate a dense result and dispatch to `mul!` instead (our relaxed `generic_matmatmul!`
# handles the mixed-eltype case). The loop-generated single-T `*` methods below remain
# more specific and continue to win when `eltype(A) == eltype(B)`.
#
# No override needed for `sparse × dense`: `similar(B_dense, TS)` already returns dense,
# so Julia's default `*` allocates the right container and dispatches via `mul!`.
const DenseCuMatOrAdj = Union{DenseCuMatrix, AdjOrTrans{<:Any, <:DenseCuMatrix},
                              HermOrSym{<:Any, <:DenseCuMatrix}}
const CuSparseMatOrAdj = Union{CuSparseMatrix, AdjOrTrans{<:Any, <:CuSparseMatrix},
                               HermOrSym{<:Any, <:CuSparseMatrix}}
function Base.:(*)(@nospecialize(A::DenseCuMatOrAdj), @nospecialize(B::CuSparseMatOrAdj))
    T = promote_type(eltype(A), eltype(B))
    C = CuMatrix{T}(undef, size(A, 1), size(B, 2))
    return mul!(C, A, B, true, false)
end

LinearAlgebra.dot(x::CuSparseVector{T}, y::DenseCuVector{T}) where {T <: BlasReal} = vv!('N', x, y, 'O')
LinearAlgebra.dot(x::DenseCuVector{T}, y::CuSparseVector{T}) where {T <: BlasReal} = dot(y, x)

LinearAlgebra.dot(x::CuSparseVector{T}, y::DenseCuVector{T}) where {T <: BlasComplex} = vv!('C', x, y, 'O')
LinearAlgebra.dot(x::DenseCuVector{T}, y::CuSparseVector{T}) where {T <: BlasComplex} = conj(dot(y,x))

adjtrans_wrappers = ((identity, identity),
                     (M -> :(Transpose{T, <:$M}), M -> :(_sptranspose(parent($M)))),
                     (M -> :(Adjoint{T, <:$M}), M -> :(_spadjoint(parent($M)))))

op_wrappers = ((identity, T -> 'N', identity),
               (T -> :(Transpose{T, <:$T}), T -> 'T', A -> :(parent($A))),
               (T -> :(Adjoint{T, <:$T}), T -> T <: Real ? 'T' : 'C', A -> :(parent($A))),
               (T -> :(HermOrSym{T, <:$T}), T -> 'N', A -> :(parent($A))))

# legacy methods with final MulAddMul argument
LinearAlgebra.generic_matvecmul!(C::CuVector{Tc}, tA::AbstractChar, A::CuSparseMatrix{Ta}, B::DenseCuVector{Tb}, _add::MulAddMul) where {Tc <: SpMatMulFloat, Ta <: SpMatMulFloat, Tb <: SpMatMulFloat} =
    LinearAlgebra.generic_matvecmul!(C, tA, A, B, _add.alpha, _add.beta)
LinearAlgebra.generic_matvecmul!(C::CuVector{Tc}, tA::AbstractChar, A::CuSparseMatrix{Ta}, B::CuSparseVector{Tb}, _add::MulAddMul) where {Tc <: SpMatMulFloat, Ta <: SpMatMulFloat, Tb <: SpMatMulFloat} =
    LinearAlgebra.generic_matvecmul!(C, tA, A, B, _add.alpha, _add.beta)
LinearAlgebra.generic_matmatmul!(C::CuMatrix{Tc}, tA, tB, A::CuSparseMatrix{Ta}, B::DenseCuMatrix{Tb}, _add::MulAddMul) where {Tc <: SpMatMulFloat, Ta <: SpMatMulFloat, Tb <: SpMatMulFloat} =
    LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, _add.alpha, _add.beta)

# mv! tolerates A.eltype ≠ X.eltype, but requires X.eltype == Y.eltype, so only B may need promotion.
function LinearAlgebra.generic_matvecmul!(C::CuVector{Tc}, tA::AbstractChar, A::CuSparseMatrix{Ta}, B::DenseCuVector{Tb}, alpha::Number, beta::Number) where {Tc <: SpMatMulFloat, Ta <: SpMatMulFloat, Tb <: SpMatMulFloat}
    B′ = Tb === Tc ? B : convert(CuVector{Tc}, B)
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    mv_wrapper(tA, alpha, A, B′, beta, C)
end

function LinearAlgebra.generic_matvecmul!(C::CuVector{Tc}, tA::AbstractChar, A::CuSparseMatrix{Ta}, B::CuSparseVector{Tb}, alpha::Number, beta::Number) where {Tc <: SpMatMulFloat, Ta <: SpMatMulFloat, Tb <: SpMatMulFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    mv_wrapper(tA, alpha, A, CuVector{Tc}(B), beta, C)
end

# mm! requires all three operands to share an eltype, so promote A and B to match C.
function LinearAlgebra.generic_matmatmul!(C::CuMatrix{Tc}, tA, tB, A::CuSparseMatrix{Ta}, B::DenseCuMatrix{Tb}, alpha::Number, beta::Number) where {Tc <: SpMatMulFloat, Ta <: SpMatMulFloat, Tb <: SpMatMulFloat}
    A′ = Ta === Tc ? A : sparse_with_eltype(A, Tc)
    B′ = Tb === Tc ? B : convert(CuMatrix{Tc}, B)
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    mm_wrapper(tA, tB, alpha, A′, B′, beta, C)
end

for (wrapa, transa, unwrapa) in op_wrappers
    TypeA = wrapa(:(CuSparseMatrix{T}))

    @eval function LinearAlgebra.:(*)(A::$TypeA, x::CuSparseVector{T}) where {T <: SpMatMulFloat}
        m, n = size(A)
        length(x) == n || throw(DimensionMismatch())
        y = CuVector{T}(undef, m)
        mul!(y, A, x, true, false)
    end
end

# legacy methods with final MulAddMul argument
LinearAlgebra.generic_matvecmul!(C::CuVector{Tc}, tA::AbstractChar, A::DenseCuMatrix{Ta}, B::CuSparseVector{Tb}, _add::MulAddMul) where {Tc <: BlasFloat, Ta <: BlasFloat, Tb <: BlasFloat} =
    LinearAlgebra.generic_matvecmul!(C, tA, A, B, _add.alpha, _add.beta)

LinearAlgebra.generic_matmatmul!(C::CuMatrix{Tc}, tA, tB, A::DenseCuMatrix{Ta}, B::CuSparseMatrixCSC{Tb}, _add::MulAddMul) where {Tc <: SpMatMulFloat, Ta <: SpMatMulFloat, Tb <: SpMatMulFloat} =
    LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, _add.alpha, _add.beta)
LinearAlgebra.generic_matmatmul!(C::CuMatrix{Tc}, tA, tB, A::DenseCuMatrix{Ta}, B::CuSparseMatrixCSR{Tb}, _add::MulAddMul) where {Tc <: SpMatMulFloat, Ta <: SpMatMulFloat, Tb <: SpMatMulFloat} =
    LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, _add.alpha, _add.beta)
LinearAlgebra.generic_matmatmul!(C::CuMatrix{Tc}, tA, tB, A::DenseCuMatrix{Ta}, B::CuSparseMatrixCOO{Tb}, _add::MulAddMul) where {Tc <: SpMatMulFloat, Ta <: SpMatMulFloat, Tb <: SpMatMulFloat} =
    LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, _add.alpha, _add.beta)

# gemvi! requires matching eltypes, so promote A and B to match C.
function LinearAlgebra.generic_matvecmul!(C::CuVector{Tc}, tA::AbstractChar, A::DenseCuMatrix{Ta}, B::CuSparseVector{Tb}, alpha::Number, beta::Number) where {Tc <: BlasFloat, Ta <: BlasFloat, Tb <: BlasFloat}
    A′ = Ta === Tc ? A : convert(CuMatrix{Tc}, A)
    B′ = Tb === Tc ? B : sparse_with_eltype(B, Tc)
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    gemvi!(tA, alpha, A′, B′, beta, C, 'O')
end

# mm! requires matching eltypes, so promote A and B to match C.
function LinearAlgebra.generic_matmatmul!(C::CuMatrix{Tc}, tA, tB, A::DenseCuMatrix{Ta}, B::CuSparseMatrixCSC{Tb}, alpha::Number, beta::Number) where {Tc <: SpMatMulFloat, Ta <: SpMatMulFloat, Tb <: SpMatMulFloat}
    A′ = Ta === Tc ? A : convert(CuMatrix{Tc}, A)
    B′ = Tb === Tc ? B : sparse_with_eltype(B, Tc)
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    mm!(tA, tB, alpha, A′, B′, beta, C, 'O')
end
function LinearAlgebra.generic_matmatmul!(C::CuMatrix{Tc}, tA, tB, A::DenseCuMatrix{Ta}, B::CuSparseMatrixCSR{Tb}, alpha::Number, beta::Number) where {Tc <: SpMatMulFloat, Ta <: SpMatMulFloat, Tb <: SpMatMulFloat}
    A′ = Ta === Tc ? A : convert(CuMatrix{Tc}, A)
    B′ = Tb === Tc ? B : sparse_with_eltype(B, Tc)
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    mm!(tA, tB, alpha, A′, B′, beta, C, 'O')
end
function LinearAlgebra.generic_matmatmul!(C::CuMatrix{Tc}, tA, tB, A::DenseCuMatrix{Ta}, B::CuSparseMatrixCOO{Tb}, alpha::Number, beta::Number) where {Tc <: SpMatMulFloat, Ta <: SpMatMulFloat, Tb <: SpMatMulFloat}
    A′ = Ta === Tc ? A : convert(CuMatrix{Tc}, A)
    B′ = Tb === Tc ? B : sparse_with_eltype(B, Tc)
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    mm!(tA, tB, alpha, A′, B′, beta, C, 'O')
end

for (wrapa, transa, unwrapa) in op_wrappers
    TypeA = wrapa(:(DenseCuMatrix{T}))

    @eval function Base.:(*)(A::$TypeA, x::CuSparseVector{T}) where {T <: BlasFloat}
        m, n = size(A)
        length(x) == n || throw(DimensionMismatch())
        y = CuVector{T}(undef, m)
        mul!(y, A, x, true, false)
    end

    for (wrapb, transb, unwrapb) in op_wrappers
        for SparseMatrixType in (:(CuSparseMatrixCSC{T}), :(CuSparseMatrixCSR{T}), :(CuSparseMatrixCOO{T}))
            TypeB = wrapb(SparseMatrixType)

            @eval function Base.:(*)(A::$TypeA, B::$TypeB) where {T <: SpMatMulFloat}
                m, n = size(A)
                k, p = size(B)
                n == k || throw(DimensionMismatch())
                C = CuMatrix{T}(undef, m, p)
                mul!(C, A, B, true, false)
            end
        end
    end
end

# legacy methods with final MulAddMul argument
LinearAlgebra.generic_matmatmul!(C::CuSparseMatrixCSC{T}, tA, tB, A::CuSparseMatrixCSC{T}, B::CuSparseMatrixCSC{T}, _add::MulAddMul) where {T <: BlasFloat} =
    LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, _add.alpha, _add.beta)
LinearAlgebra.generic_matmatmul!(C::CuSparseMatrixCSR{T}, tA, tB, A::CuSparseMatrixCSR{T}, B::CuSparseMatrixCSR{T}, _add::MulAddMul) where {T <: BlasFloat} =
    LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, _add.alpha, _add.beta)
LinearAlgebra.generic_matmatmul!(C::CuSparseMatrixCOO{T}, tA, tB, A::CuSparseMatrixCOO{T}, B::CuSparseMatrixCOO{T}, _add::MulAddMul) where {T <: BlasFloat} =
    LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, _add.alpha, _add.beta)

function LinearAlgebra.generic_matmatmul!(C::CuSparseMatrixCSC{T}, tA, tB, A::CuSparseMatrixCSC{T}, B::CuSparseMatrixCSC{T}, alpha::Number, beta::Number) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    gemm!(tA, tB, alpha, A, B, beta, C, 'O')
end
function LinearAlgebra.generic_matmatmul!(C::CuSparseMatrixCSR{T}, tA, tB, A::CuSparseMatrixCSR{T}, B::CuSparseMatrixCSR{T}, alpha::Number, beta::Number) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    gemm!(tA, tB, alpha, A, B, beta, C, 'O')
end
function LinearAlgebra.generic_matmatmul!(C::CuSparseMatrixCOO{T}, tA, tB, A::CuSparseMatrixCOO{T}, B::CuSparseMatrixCOO{T}, alpha::Number, beta::Number) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    A_csr = CuSparseMatrixCSR(A)
    B_csr = CuSparseMatrixCSR(B)
    C_csr = CuSparseMatrixCSR(C)
    LinearAlgebra.generic_matmatmul!(C_csr, tA, tB, A_csr, B_csr, alpha, beta)
    copyto!(C, CuSparseMatrixCOO(C_csr))
    return C
end

for SparseMatrixType in (:CuSparseMatrixCSC, :CuSparseMatrixCSR)
    @eval function LinearAlgebra.:(*)(A::$SparseMatrixType{T}, B::$SparseMatrixType{T}) where {T <: BlasFloat}
        gemm('N', 'N', one(T), A, B, 'O')
    end
end

function LinearAlgebra.:(*)(A::CuSparseMatrixCOO{T}, B::CuSparseMatrixCOO{T}) where {T <: BlasFloat}
    A_csr = CuSparseMatrixCSR(A)
    B_csr = CuSparseMatrixCSR(B)
    CuSparseMatrixCOO(A_csr * B_csr)
end

for (wrapa, unwrapa) in adjtrans_wrappers, (wrapb, unwrapb) in adjtrans_wrappers
    for SparseMatrixType in (:(CuSparseMatrixCSC{T}), :(CuSparseMatrixCSR{T}), :(CuSparseMatrixCOO{T}))
        TypeA = wrapa(SparseMatrixType)
        TypeB = wrapb(SparseMatrixType)
        wrapa == identity && wrapb == identity && continue
        @eval Base.:(*)(A::$TypeA, B::$TypeB) where {T <: BlasFloat} = $(unwrapa(:A)) * $(unwrapb(:B))
    end
end

for op in (:(+), :(-))
    for (wrapa, unwrapa) in adjtrans_wrappers, (wrapb, unwrapb) in adjtrans_wrappers
        for SparseMatrixType in (:(CuSparseMatrixCSC{T}), :(CuSparseMatrixCSR{T}))
            TypeA = wrapa(SparseMatrixType)
            TypeB = wrapb(SparseMatrixType)
            @eval Base.$op(A::$TypeA, B::$TypeB) where {T <: BlasFloat} = geam(one(T), $(unwrapa(:A)), $(op)(one(T)), $(unwrapb(:B)), 'O')
        end
    end

    @eval begin
        Base.$op(A::CuSparseVector{T}, B::CuSparseVector{T}) where {T <: BlasFloat} = axpby(one(T), A, $(op)(one(T)), B, 'O')
        Base.$op(A::Union{CuSparseMatrixCOO{T}, Transpose{T,<:CuSparseMatrixCOO}, Adjoint{T,<:CuSparseMatrixCOO}},
                 B::Union{CuSparseMatrixCOO{T}, Transpose{T,<:CuSparseMatrixCOO}, Adjoint{T,<:CuSparseMatrixCOO}}) where {T <: BlasFloat} =
            CuSparseMatrixCOO($(op)(CuSparseMatrixCSR(A), CuSparseMatrixCSR(B)))
    end

    # Symmetric/Hermitian wrappers: materialize, then defer. (issue #3043)
    for (wrap, _) in adjtrans_wrappers,
        SparseMatrixType in (:CuSparseMatrixCSC, :CuSparseMatrixCSR, :CuSparseMatrixCOO)

        W = wrap(:($SparseMatrixType{T}))
        @eval Base.$op(A::HermOrSym{T,<:$SparseMatrixType}, B::$W) where {T <: BlasFloat} = $op(sparse(A), B)
        @eval Base.$op(A::$W, B::HermOrSym{T,<:$SparseMatrixType}) where {T <: BlasFloat} = $op(A, sparse(B))
    end
end

# triangular
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

## uniform scaling

# these operations materialize the identity matrix and re-use broadcast
# TODO: can we do without this, and just use the broadcast implementation
#       with a singleton argument it knows how to index?

function _sparse_identity(::Type{<:CuSparseMatrixCSR{<:Any,Ti}},
                          I::UniformScaling{Tv}, dims::Dims) where {Tv,Ti}
    len = min(dims[1], dims[2])
    rowPtr = CuVector{Ti}(vcat(1:len, fill(len+1, dims[1]-len+1)))
    colVal = CuVector{Ti}(1:len)
    nzVal = CUDACore.fill(I.λ, len)
    CuSparseMatrixCSR{Tv,Ti}(rowPtr, colVal, nzVal, dims)
end

function _sparse_identity(::Type{<:CuSparseMatrixCSC{<:Any,Ti}},
                          I::UniformScaling{Tv}, dims::Dims) where {Tv,Ti}
    len = min(dims[1], dims[2])
    colPtr = CuVector{Ti}(vcat(1:len, fill(len+1, dims[2]-len+1)))
    rowVal = CuVector{Ti}(1:len)
    nzVal = CUDACore.fill(I.λ, len)
    CuSparseMatrixCSC{Tv,Ti}(colPtr, rowVal, nzVal, dims)
end

function _sparse_identity(::Type{<:CuSparseMatrixCOO{Tv,Ti}},
                        I::UniformScaling, dims::Dims) where {Tv,Ti}
    len = min(dims[1], dims[2])
    rowInd = CuVector{Ti}(1:len)
    colInd = CuVector{Ti}(1:len)
    nzVal = CUDACore.fill(I.λ, len)
    CuSparseMatrixCOO{Tv,Ti}(rowInd, colInd, nzVal, dims)
end

for (wrapa, unwrapa) in adjtrans_wrappers
    for SparseMatrixType in (:(CuSparseMatrixCSC{T}), :(CuSparseMatrixCSR{T}), :(CuSparseMatrixCOO{T}))
        TypeA = wrapa(SparseMatrixType)
        @eval begin
            Base.:(+)(A::$TypeA, J::UniformScaling) where {T} = $(unwrapa(:A)) + _sparse_identity(typeof(A), J, size(A))
            Base.:(+)(J::UniformScaling, A::$TypeA) where {T} = _sparse_identity(typeof(A), J, size(A)) + $(unwrapa(:A))

            Base.:(-)(A::$TypeA, J::UniformScaling) where {T} = $(unwrapa(:A)) - _sparse_identity(typeof(A), J, size(A))
            Base.:(-)(J::UniformScaling, A::$TypeA) where {T} = _sparse_identity(typeof(A), J, size(A)) - $(unwrapa(:A))
        end

        # Broadcasting is not yet supported for COO matrices
        if SparseMatrixType != :(CuSparseMatrixCOO{T})
            @eval begin
                Base.:(*)(A::$TypeA, J::UniformScaling) where {T} = $(unwrapa(:A)) * J.λ
                Base.:(*)(J::UniformScaling, A::$TypeA) where {T} = J.λ * $(unwrapa(:A))
            end
        else
            @eval begin
                Base.:(*)(A::$TypeA, J::UniformScaling) where {T} = $(unwrapa(:A)) * _sparse_identity(typeof(A), J, size(A))
                Base.:(*)(J::UniformScaling, A::$TypeA) where {T} = _sparse_identity(typeof(A), J, size(A)) * $(unwrapa(:A))
            end
        end
    end
end

# TODO: let Broadcast handle this automatically (a la SparseArrays.PromoteToSparse)
for (wrapa, unwrapa) in adjtrans_wrappers, op in (:(+), :(-), :(*))
    for SparseMatrixType in (:(CuSparseMatrixCSC{T}), :(CuSparseMatrixCSR{T}), :(CuSparseMatrixCOO{T}))
        TypeA = wrapa(SparseMatrixType)
        @eval begin
            function Base.$op(lhs::Diagonal, rhs::$TypeA) where {T}
                return $op($SparseMatrixType(lhs), $(unwrapa(:rhs)))
            end
            function Base.$op(lhs::$TypeA, rhs::Diagonal) where {T}
                return $op($(unwrapa(:lhs)), $SparseMatrixType(rhs))
            end
        end
    end
end
