## removed in CUDA 11

@checked function cusparseScsr2csc(handle, m, n, nnz, csrSortedVal, csrSortedRowPtr,
                                   csrSortedColInd, cscSortedVal, cscSortedRowInd,
                                   cscSortedColPtr, copyValues, idxBase)
    initialize_api()
    ccall((:cusparseScsr2csc, libcusparse()), cusparseStatus_t,
                   (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cfloat}, CuPtr{Cint},
                    CuPtr{Cint}, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, cusparseAction_t,
                    cusparseIndexBase_t),
                   handle, m, n, nnz, csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                   cscSortedVal, cscSortedRowInd, cscSortedColPtr, copyValues, idxBase)
end

@checked function cusparseDcsr2csc(handle, m, n, nnz, csrSortedVal, csrSortedRowPtr,
                                   csrSortedColInd, cscSortedVal, cscSortedRowInd,
                                   cscSortedColPtr, copyValues, idxBase)
    initialize_api()
    ccall((:cusparseDcsr2csc, libcusparse()), cusparseStatus_t,
                   (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cdouble}, CuPtr{Cint},
                    CuPtr{Cint}, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                    cusparseAction_t, cusparseIndexBase_t),
                   handle, m, n, nnz, csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                   cscSortedVal, cscSortedRowInd, cscSortedColPtr, copyValues, idxBase)
end

@checked function cusparseCcsr2csc(handle, m, n, nnz, csrSortedVal, csrSortedRowPtr,
                                   csrSortedColInd, cscSortedVal, cscSortedRowInd,
                                   cscSortedColPtr, copyValues, idxBase)
    initialize_api()
    ccall((:cusparseCcsr2csc, libcusparse()), cusparseStatus_t,
                   (cusparseHandle_t, Cint, Cint, Cint, CuPtr{cuComplex}, CuPtr{Cint},
                    CuPtr{Cint}, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint},
                    cusparseAction_t, cusparseIndexBase_t),
                   handle, m, n, nnz, csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                   cscSortedVal, cscSortedRowInd, cscSortedColPtr, copyValues, idxBase)
end

@checked function cusparseZcsr2csc(handle, m, n, nnz, csrSortedVal, csrSortedRowPtr,
                                   csrSortedColInd, cscSortedVal, cscSortedRowInd,
                                   cscSortedColPtr, copyValues, idxBase)
    initialize_api()
    ccall((:cusparseZcsr2csc, libcusparse()), cusparseStatus_t,
                   (cusparseHandle_t, Cint, Cint, Cint, CuPtr{cuDoubleComplex},
                    CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuDoubleComplex}, CuPtr{Cint},
                    CuPtr{Cint}, cusparseAction_t, cusparseIndexBase_t),
                   handle, m, n, nnz, csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                   cscSortedVal, cscSortedRowInd, cscSortedColPtr, copyValues, idxBase)
end

@checked function cusparseScsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                                  csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B,
                                  ldb, beta, C, ldc)
    initialize_api()
    ccall((:cusparseScsrmm2, libcusparse()), cusparseStatus_t,
                   (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint,
                    Cint, Cint, Cint, Ref{Cfloat}, cusparseMatDescr_t, CuPtr{Cfloat},
                    CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cfloat}, Cint, Ref{Cfloat},
                    CuPtr{Cfloat}, Cint),
                   handle, transA, transB, m, n, k, nnz, alpha, descrA, csrSortedValA,
                   csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc)
end

@checked function cusparseDcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                                  csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B,
                                  ldb, beta, C, ldc)
    initialize_api()
    ccall((:cusparseDcsrmm2, libcusparse()), cusparseStatus_t,
                   (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint,
                    Cint, Cint, Cint, Ref{Cdouble}, cusparseMatDescr_t, CuPtr{Cdouble},
                    CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cint, Ref{Cdouble},
                    CuPtr{Cdouble}, Cint),
                   handle, transA, transB, m, n, k, nnz, alpha, descrA, csrSortedValA,
                   csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc)
end

@checked function cusparseCcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                                  csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B,
                                  ldb, beta, C, ldc)
    initialize_api()
    ccall((:cusparseCcsrmm2, libcusparse()), cusparseStatus_t,
                   (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint,
                    Cint, Cint, Cint, Ref{cuComplex}, cusparseMatDescr_t, CuPtr{cuComplex},
                    CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuComplex}, Cint, Ref{cuComplex},
                    CuPtr{cuComplex}, Cint),
                   handle, transA, transB, m, n, k, nnz, alpha, descrA, csrSortedValA,
                   csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc)
end

@checked function cusparseZcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                                  csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B,
                                  ldb, beta, C, ldc)
    initialize_api()
    ccall((:cusparseZcsrmm2, libcusparse()), cusparseStatus_t,
                   (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint,
                    Cint, Cint, Cint, Ref{cuDoubleComplex}, cusparseMatDescr_t,
                    CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                    CuPtr{cuDoubleComplex}, Cint, Ref{cuDoubleComplex},
                    CuPtr{cuDoubleComplex}, Cint),
                   handle, transA, transB, m, n, k, nnz, alpha, descrA, csrSortedValA,
                   csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc)
end
