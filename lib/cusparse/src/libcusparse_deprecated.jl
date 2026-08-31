# wrappers for legacy CUSPARSE functionality, removed from CUDA 11+,
# but still required to support CUDA 10.x toolkits.

for (fname, elty) in ((:cusparseSdense2csr, :Float32),
                      (:cusparseDdense2csr, :Float64),
                      (:cusparseCdense2csr, :ComplexF32),
                      (:cusparseZdense2csr, :ComplexF64))
    @eval @checked function $fname(handle, m, n, descrA, A, lda, nnzPerRow,
                                   csrSortedValA, csrSortedRowPtrA, csrSortedColIndA)
        initialize_context()
        @gcsafe_ccall libcusparse.$fname(handle::cusparseHandle_t, m::Cint, n::Cint,
                                         descrA::cusparseMatDescr_t, A::CuPtr{$elty},
                                         lda::Cint, nnzPerRow::CuPtr{Cint},
                                         csrSortedValA::CuPtr{$elty},
                                         csrSortedRowPtrA::CuPtr{Cint},
                                         csrSortedColIndA::CuPtr{Cint})::cusparseStatus_t
    end
end

for (fname, elty) in ((:cusparseSdense2csc, :Float32),
                      (:cusparseDdense2csc, :Float64),
                      (:cusparseCdense2csc, :ComplexF32),
                      (:cusparseZdense2csc, :ComplexF64))
    @eval @checked function $fname(handle, m, n, descrA, A, lda, nnzPerCol,
                                   cscSortedValA, cscSortedRowIndA, cscSortedColPtrA)
        initialize_context()
        @gcsafe_ccall libcusparse.$fname(handle::cusparseHandle_t, m::Cint, n::Cint,
                                         descrA::cusparseMatDescr_t, A::CuPtr{$elty},
                                         lda::Cint, nnzPerCol::CuPtr{Cint},
                                         cscSortedValA::CuPtr{$elty},
                                         cscSortedRowIndA::CuPtr{Cint},
                                         cscSortedColPtrA::CuPtr{Cint})::cusparseStatus_t
    end
end

for (fname, elty) in ((:cusparseScsr2dense, :Float32),
                      (:cusparseDcsr2dense, :Float64),
                      (:cusparseCcsr2dense, :ComplexF32),
                      (:cusparseZcsr2dense, :ComplexF64))
    @eval @checked function $fname(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                                   csrSortedColIndA, A, lda)
        initialize_context()
        @gcsafe_ccall libcusparse.$fname(handle::cusparseHandle_t, m::Cint, n::Cint,
                                         descrA::cusparseMatDescr_t,
                                         csrSortedValA::CuPtr{$elty},
                                         csrSortedRowPtrA::CuPtr{Cint},
                                         csrSortedColIndA::CuPtr{Cint}, A::CuPtr{$elty},
                                         lda::Cint)::cusparseStatus_t
    end
end

for (fname, elty) in ((:cusparseScsc2dense, :Float32),
                      (:cusparseDcsc2dense, :Float64),
                      (:cusparseCcsc2dense, :ComplexF32),
                      (:cusparseZcsc2dense, :ComplexF64))
    @eval @checked function $fname(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA,
                                   cscSortedColPtrA, A, lda)
        initialize_context()
        @gcsafe_ccall libcusparse.$fname(handle::cusparseHandle_t, m::Cint, n::Cint,
                                         descrA::cusparseMatDescr_t,
                                         cscSortedValA::CuPtr{$elty},
                                         cscSortedRowIndA::CuPtr{Cint},
                                         cscSortedColPtrA::CuPtr{Cint}, A::CuPtr{$elty},
                                         lda::Cint)::cusparseStatus_t
    end
end
