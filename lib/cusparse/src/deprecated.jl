# Deprecated in v6.6: `mv!` multiplies BSR matrices too, with the generic API from cuSPARSE
# 12.6.3 and with bsrmv before. `mv2!` stays exported until the next breaking release.
export mv2!
"""
    mv2!(transa, alpha, A::CuSparseMatrixBSR, X, beta, Y, index)

Deprecated, use [`mv!`](@ref), which supports BSR matrices.
"""
function mv2! end
@deprecate mv2!(transa::SparseChar, alpha::Number, A::CuSparseMatrixBSR, X::CuVector, beta::Number, Y::CuVector, index::SparseChar) mv!(transa, alpha, A, X, beta, Y; index) false

# Deprecated in v6.6: the index base of the sparse operands is the keyword `index`, which
# defaults to one-based ('O') like the constructors of the sparse arrays.
@deprecate gather!(X::CuSparseVector, Y::CuVector, index::SparseChar) gather!(X, Y; index) false
@deprecate scatter!(Y::CuVector, X::CuSparseVector, index::SparseChar) scatter!(Y, X; index) false
@deprecate axpby!(alpha::Number, X::CuSparseVector, beta::Number, Y::CuVector, index::SparseChar) axpby!(alpha, X, beta, Y; index) false
@deprecate rot!(X::CuSparseVector, Y::CuVector, c::Number, s::Number, index::SparseChar) rot!(X, Y, c, s; index) false
@deprecate vv!(transx::SparseChar, X::CuSparseVector, Y::DenseCuVector, index::SparseChar) vv!(transx, X, Y; index) false

@deprecate mv!(transa::SparseChar, alpha::Number, A::CuSparseMatrix, X::DenseCuVector, beta::Number, Y::DenseCuVector, index::SparseChar, algo::cusparseSpMVAlg_t=CUSPARSE_SPMV_ALG_DEFAULT) mv!(transa, alpha, A, X, beta, Y, algo; index) false
@deprecate mm!(transa::SparseChar, transb::SparseChar, alpha::Number, A::Union{CuSparseMatrix,DenseCuMatrix}, B::Union{CuSparseMatrix,DenseCuMatrix,Transpose{<:Any,<:DenseCuMatrix}}, beta::Number, C::Union{DenseCuMatrix,Transpose{<:Any,<:DenseCuMatrix}}, index::SparseChar, algo::cusparseSpMMAlg_t=CUSPARSE_SPMM_ALG_DEFAULT) mm!(transa, transb, alpha, A, B, beta, C, algo; index) false
@deprecate bmm!(transa::SparseChar, transb::SparseChar, alpha::Number, A::CuSparseArrayCSR, B::DenseCuArray, beta::Number, C::DenseCuArray, index::SparseChar, algo::cusparseSpMMAlg_t=CUSPARSE_SPMM_ALG_DEFAULT) bmm!(transa, transb, alpha, A, B, beta, C, algo; index) false
@deprecate gemm!(transa::SparseChar, transb::SparseChar, alpha::Number, A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}, B::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}, beta::Number, C::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}, index::SparseChar, algo::cusparseSpGEMMAlg_t=CUSPARSE_SPGEMM_DEFAULT) gemm!(transa, transb, alpha, A, B, beta, C, algo; index) false
@deprecate gemm(transa::SparseChar, transb::SparseChar, alpha::Number, A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}, B::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}, index::SparseChar, algo::cusparseSpGEMMAlg_t=CUSPARSE_SPGEMM_DEFAULT) gemm(transa, transb, alpha, A, B, algo; index) false
@deprecate gemm(transa::SparseChar, transb::SparseChar, alpha::Number, A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}, B::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}, beta::Number, C::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}, index::SparseChar, algo::cusparseSpGEMMAlg_t=CUSPARSE_SPGEMM_DEFAULT; same_pattern::Bool=false) gemm(transa, transb, alpha, A, B, beta, C, algo; index, same_pattern) false
@deprecate gemv(transa::SparseChar, alpha::Number, A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}, x::CuSparseVector, index::SparseChar, algo::cusparseSpGEMMAlg_t=CUSPARSE_SPGEMM_DEFAULT) gemv(transa, alpha, A, x, algo; index) false
@deprecate sv!(transa::SparseChar, uplo::SparseChar, diag::SparseChar, alpha::Number, A::Union{CuSparseMatrixCSC,CuSparseMatrixCSR,CuSparseMatrixCOO}, X::DenseCuVector, Y::DenseCuVector, index::SparseChar, algo::cusparseSpSVAlg_t=CUSPARSE_SPSV_ALG_DEFAULT) sv!(transa, uplo, diag, alpha, A, X, Y, algo; index) false
@deprecate sm!(transa::SparseChar, transb::SparseChar, uplo::SparseChar, diag::SparseChar, alpha::Number, A::Union{CuSparseMatrixCSC,CuSparseMatrixCSR,CuSparseMatrixCOO}, B::DenseCuMatrix, C::DenseCuMatrix, index::SparseChar, algo::cusparseSpSMAlg_t=CUSPARSE_SPSM_ALG_DEFAULT) sm!(transa, transb, uplo, diag, alpha, A, B, C, algo; index) false
@deprecate sddmm!(transa::SparseChar, transb::SparseChar, alpha::Number, A::DenseCuMatrix, B::DenseCuMatrix, beta::Number, C::Union{CuSparseMatrixCSR,CuSparseMatrixBSR}, index::SparseChar, algo::cusparseSDDMMAlg_t=CUSPARSE_SDDMM_ALG_DEFAULT) sddmm!(transa, transb, alpha, A, B, beta, C, algo; index) false

@deprecate sv2!(transa::SparseChar, uplo::SparseChar, diag::SparseChar, alpha::Number, A::CuSparseMatrixBSR, X::CuVector, index::SparseChar) sv2!(transa, uplo, diag, alpha, A, X; index) false
@deprecate sv2(transa::SparseChar, uplo::SparseChar, diag::SparseChar, alpha::Number, A::CuSparseMatrixBSR, X::CuVector, index::SparseChar) sv2(transa, uplo, diag, alpha, A, X; index) false
@deprecate sv2(transa::SparseChar, uplo::SparseChar, diag::SparseChar, A::CuSparseMatrixBSR, X::CuVector, index::SparseChar) sv2(transa, uplo, diag, A, X; index) false
@deprecate sv2(transa::SparseChar, uplo::SparseChar, A::CuSparseMatrixBSR, X::CuVector, index::SparseChar) sv2(transa, uplo, A, X; index) false
@deprecate sm2!(transa::SparseChar, transxy::SparseChar, uplo::SparseChar, diag::SparseChar, alpha::Number, A::CuSparseMatrixBSR, X::StridedCuMatrix, index::SparseChar) sm2!(transa, transxy, uplo, diag, alpha, A, X; index) false
@deprecate sm2(transa::SparseChar, transxy::SparseChar, uplo::SparseChar, diag::SparseChar, alpha::Number, A::CuSparseMatrixBSR, X::StridedCuMatrix, index::SparseChar) sm2(transa, transxy, uplo, diag, alpha, A, X; index) false
@deprecate sm2(transa::SparseChar, transxy::SparseChar, uplo::SparseChar, diag::SparseChar, A::CuSparseMatrixBSR, X::StridedCuMatrix, index::SparseChar) sm2(transa, transxy, uplo, diag, A, X; index) false
@deprecate sm2(transa::SparseChar, transxy::SparseChar, uplo::SparseChar, A::CuSparseMatrixBSR, X::StridedCuMatrix, index::SparseChar) sm2(transa, transxy, uplo, A, X; index) false
@deprecate gemvi!(transa::SparseChar, alpha::Number, A::CuMatrix, X::CuSparseVector, beta::Number, Y::CuVector, index::SparseChar) gemvi!(transa, alpha, A, X, beta, Y; index) false

@deprecate geam(alpha::Number, A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}, beta::Number, B::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}, index::SparseChar) geam(alpha, A, beta, B; index) false
@deprecate axpby(alpha::Number, x::CuSparseVector, beta::Number, y::CuSparseVector, index::SparseChar) axpby(alpha, x, beta, y; index) false
@deprecate color(A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}, index::SparseChar; percentage::Number=1.0) color(A; index, percentage) false
@deprecate ic02!(A::CuSparseMatrix, index::SparseChar) ic02!(A; index) false
@deprecate ilu02!(A::CuSparseMatrix, index::SparseChar) ilu02!(A; index) false
@deprecate ic02(A, index::SparseChar) ic02(A; index) false
@deprecate ilu02(A, index::SparseChar) ilu02(A; index) false
@deprecate sort_csc(A::CuSparseMatrixCSC, index::SparseChar) sort_csc(A; index) false
@deprecate sort_csr(A::CuSparseMatrixCSR, index::SparseChar) sort_csr(A; index) false
@deprecate prune(A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}, threshold::Number, index::SparseChar) prune(A, threshold; index) false

# Deprecated in v6.6: the tridiagonal solvers have no sparse operand, and ignored `index`.
@deprecate gtsv2!(dl::CuVector, d::CuVector, du::CuVector, B::CuVecOrMat, index::SparseChar; pivoting::Bool=true) gtsv2!(dl, d, du, B; pivoting) false
@deprecate gtsv2(dl::CuVector, d::CuVector, du::CuVector, B::CuVecOrMat, index::SparseChar; pivoting::Bool=true) gtsv2(dl, d, du, B; pivoting) false

# Deprecated in v6.6: `sort_coo` sorts by the `by` keyword, `:row` or `:col`.
function sort_coo(A::CuSparseMatrixCOO, type::SparseChar)
    Base.depwarn("`sort_coo(A, type)` is deprecated, use `sort_coo(A; by=:row)` or `sort_coo(A; by=:col)` instead.", :sort_coo)
    type == 'R' || type == 'C' || throw(ArgumentError("type=$type was used and only type='R' and type='C' are supported."))
    return sort_coo(A; by = type == 'R' ? :row : :col)
end
