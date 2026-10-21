# utility functions for the CUSPARSE wrappers

# Legacy ABIs interpret both buffers as int*. The public coordinate parameter alone
# no longer establishes the pointer width.
function require_cint_storage(A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC,CuSparseMatrixCOO})
    ptr_ok = A isa CuSparseMatrixCSR ? eltype(A.rowPtr) === Cint :
             A isa CuSparseMatrixCSC ? eltype(A.colPtr) === Cint : true
    SparseArrays.indtype(A) === Cint && ptr_ok ||
        throw(ArgumentError("this legacy vendor operation requires uniform Int32 coordinate and pointer storage; convert explicitly"))
    return A
end
require_cint_storage(A::CuSparseMatrixBSR) =
    (eltype(A.rowPtr) === Cint && eltype(A.colVal) === Cint ? A :
     throw(ArgumentError("legacy BSR operation requires Int32 indices")))

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
check that the dimensions of matrices `B` and `C` make sense for a multiplication
"""
function chkmmdims(B, C, k, l, m, n)
    if size(B) != (k,l)
        throw(DimensionMismatch("B has dimensions $(size(B)) but needs ($k,$l)"))
    elseif size(C) != (m,n)
        throw(DimensionMismatch("C has dimensions $(size(C)) but needs ($m,$n)"))
    end
end

"""
check that the dimensions of arrays `B` and `C` make sense for a batched matrix-matrix multiplication
"""
function chkbmmdims(B, C, k, l, m, n)
    if size(B)[1:2] != (k,l)
        throw(DimensionMismatch("B has dimensions $(size(B)) but needs ($k,$l)"))
    elseif size(C)[1:2] != (m,n)
        throw(DimensionMismatch("C has dimensions $(size(C)) but needs ($m,$n)"))
    end
end
