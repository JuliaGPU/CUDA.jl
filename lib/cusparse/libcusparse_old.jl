# wrapers for old functionality

## removed in CUDA 11

@checked function cusparseScsr2csc(handle, m, n, nnz, csrSortedVal, csrSortedRowPtr,
                                   csrSortedColInd, cscSortedVal, cscSortedRowInd,
                                   cscSortedColPtr, copyValues, idxBase)
    initialize_api()
    @runtime_ccall((:cusparseScsr2csc, libcusparse()), cusparseStatus_t,
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
    @runtime_ccall((:cusparseDcsr2csc, libcusparse()), cusparseStatus_t,
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
    @runtime_ccall((:cusparseCcsr2csc, libcusparse()), cusparseStatus_t,
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
    @runtime_ccall((:cusparseZcsr2csc, libcusparse()), cusparseStatus_t,
                   (cusparseHandle_t, Cint, Cint, Cint, CuPtr{cuDoubleComplex},
                    CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuDoubleComplex}, CuPtr{Cint},
                    CuPtr{Cint}, cusparseAction_t, cusparseIndexBase_t),
                   handle, m, n, nnz, csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                   cscSortedVal, cscSortedRowInd, cscSortedColPtr, copyValues, idxBase)
end

