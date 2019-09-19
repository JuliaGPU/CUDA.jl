# utility functions for the CUSPARSE wrappers

"""
convert `SparseChar` {`N`,`T`,`C`} to `cusparseOperation_t`
"""
function cusparseop(trans::SparseChar)
    if trans == 'N'
        return CUSPARSE_OPERATION_NON_TRANSPOSE
    end
    if trans == 'T'
        return CUSPARSE_OPERATION_TRANSPOSE
    end
    if trans == 'C'
        return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
    end
    throw(ArgumentError("unknown cusparse operation $trans"))
end

"""
convert `SparseChar` {`G`,`S`,`H`,`T`} to `cusparseMatrixType_t`
"""
function cusparsetype(mattype::SparseChar)
    if mattype == 'G'
        return CUSPARSE_MATRIX_TYPE_GENERAL
    end
    if mattype == 'T'
        return CUSPARSE_MATRIX_TYPE_TRIANGULAR
    end
    if mattype == 'S'
        return CUSPARSE_MATRIX_TYPE_SYMMETRIC
    end
    if mattype == 'H'
        return CUSPARSE_MATRIX_TYPE_HERMITIAN
    end
    throw(ArgumentError("unknown cusparse matrix type $mattype"))
end

"""
convert `SparseChar` {`U`,`L`} to `cusparseFillMode_t`
"""
function cusparsefill(uplo::SparseChar)
    if uplo == 'U'
        return CUSPARSE_FILL_MODE_UPPER
    end
    if uplo == 'L'
        return CUSPARSE_FILL_MODE_LOWER
    end
    throw(ArgumentError("unknown cusparse fill mode $uplo"))
end

"""
convert `SparseChar` {`U`,`N`} to `cusparseDiagType_t`
"""
function cusparsediag(diag::SparseChar)
    if diag == 'U'
        return CUSPARSE_DIAG_TYPE_UNIT
    end
    if diag == 'N'
        return CUSPARSE_DIAG_TYPE_NON_UNIT
    end
    throw(ArgumentError("unknown cusparse diag mode $diag"))
end

"""
convert `SparseChar` {`Z`,`O`} to `cusparseIndexBase_t`
"""
function cusparseindex(index::SparseChar)
    if index == 'Z'
        return CUSPARSE_INDEX_BASE_ZERO
    end
    if index == 'O'
        return CUSPARSE_INDEX_BASE_ONE
    end
    throw(ArgumentError("unknown cusparse index base"))
end

"""
convert `SparseChar` {`R`,`C`} to `cusparseDirection_t`
"""
function cusparsedir(dir::SparseChar)
    if dir == 'R'
        return CUSPARSE_DIRECTION_ROW
    end
    if dir == 'C'
        return CUSPARSE_DIRECTION_COLUMN
    end
    throw(ArgumentError("unknown cusparse direction $dir"))
end

"""
check that the dimensions of matrix `X` and vector `Y` make sense for a multiplication
"""
function chkmvdims(X, n, Y, m)
    if length(X) != n
        throw(DimensionMismatch("X must have length $n, but has length $(length(X))"))
    elseif length(Y) != m
        throw(DimensionMismatch("Y must have length $m, but has length $(length(Y))"))
    end
end

"""
check that the dimensions of matrices `X` and `Y` make sense for a multiplication
"""
function chkmmdims( B, C, k, l, m, n )
    if size(B) != (k,l)
        throw(DimensionMismatch("B has dimensions $(size(B)) but needs ($k,$l)"))
    elseif size(C) != (m,n)
        throw(DimensionMismatch("C has dimensions $(size(C)) but needs ($m,$n)"))
    end
end

"""
form a `cusparseMatDescr` from a `CuSparseMatrix`
"""
function getDescr( A::CuSparseMatrix, index::SparseChar )
    cuind = cusparseindex(index)
    typ   = CUSPARSE_MATRIX_TYPE_GENERAL
    fill  = CUSPARSE_FILL_MODE_LOWER
    if ishermitian(A)
        typ  = CUSPARSE_MATRIX_TYPE_HERMITIAN
        fill = cusparsefill(A.uplo)
    elseif issymmetric(A)
        typ  = CUSPARSE_MATRIX_TYPE_SYMMETRIC
        fill = cusparsefill(A.uplo)
    end
    cudesc = cusparseMatDescr(typ, fill,CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
end

function getDescr( A::Symmetric, index::SparseChar )
    cuind = cusparseindex(index)
    typ  = CUSPARSE_MATRIX_TYPE_SYMMETRIC
    fill = cusparsefill(A.uplo)
    cudesc = cusparseMatDescr(typ, fill,CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
end

function getDescr( A::Hermitian, index::SparseChar )
    cuind = cusparseindex(index)
    typ  = CUSPARSE_MATRIX_TYPE_HERMITIAN
    fill = cusparsefill(A.uplo)
    cudesc = cusparseMatDescr(typ, fill,CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
end
