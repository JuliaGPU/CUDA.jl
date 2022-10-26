# Float16 functionality is only enabled when using C++ (defining __cplusplus breaks things)

@checked function cusparseHpruneDense2csr_bufferSizeExt(handle, m, n, A, lda, threshold,
                                                        descrC, csrSortedValC,
                                                        csrSortedRowPtrC, csrSortedColIndC,
                                                        pBufferSizeInBytes)
    initialize_context()
    @ccall libcusparse.cusparseHpruneDense2csr_bufferSizeExt(handle::cusparseHandle_t,
                                                             m::Cint, n::Cint,
                                                             A::Ptr{Float16}, lda::Cint,
                                                             threshold::Ptr{Float16},
                                                             descrC::cusparseMatDescr_t,
                                                             csrSortedValC::Ptr{Float16},
                                                             csrSortedRowPtrC::Ptr{Cint},
                                                             csrSortedColIndC::Ptr{Cint},
                                                             pBufferSizeInBytes::Ptr{Csize_t})::cusparseStatus_t
end

@checked function cusparseHpruneDense2csrNnz(handle, m, n, A, lda, threshold, descrC,
                                             csrRowPtrC, nnzTotalDevHostPtr, pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseHpruneDense2csrNnz(handle::cusparseHandle_t, m::Cint,
                                                  n::Cint, A::Ptr{Float16}, lda::Cint,
                                                  threshold::Ptr{Float16},
                                                  descrC::cusparseMatDescr_t,
                                                  csrRowPtrC::Ptr{Cint},
                                                  nnzTotalDevHostPtr::Ptr{Cint},
                                                  pBuffer::Ptr{Cvoid})::cusparseStatus_t
end

@checked function cusparseHpruneDense2csr(handle, m, n, A, lda, threshold, descrC,
                                          csrSortedValC, csrSortedRowPtrC, csrSortedColIndC,
                                          pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseHpruneDense2csr(handle::cusparseHandle_t, m::Cint, n::Cint,
                                               A::Ptr{Float16}, lda::Cint,
                                               threshold::Ptr{Float16},
                                               descrC::cusparseMatDescr_t,
                                               csrSortedValC::Ptr{Float16},
                                               csrSortedRowPtrC::Ptr{Cint},
                                               csrSortedColIndC::Ptr{Cint},
                                               pBuffer::Ptr{Cvoid})::cusparseStatus_t
end

@checked function cusparseHpruneCsr2csr_bufferSizeExt(handle, m, n, nnzA, descrA,
                                                      csrSortedValA, csrSortedRowPtrA,
                                                      csrSortedColIndA, threshold, descrC,
                                                      csrSortedValC, csrSortedRowPtrC,
                                                      csrSortedColIndC, pBufferSizeInBytes)
    initialize_context()
    @ccall libcusparse.cusparseHpruneCsr2csr_bufferSizeExt(handle::cusparseHandle_t,
                                                           m::Cint, n::Cint, nnzA::Cint,
                                                           descrA::cusparseMatDescr_t,
                                                           csrSortedValA::Ptr{Float16},
                                                           csrSortedRowPtrA::Ptr{Cint},
                                                           csrSortedColIndA::Ptr{Cint},
                                                           threshold::Ptr{Float16},
                                                           descrC::cusparseMatDescr_t,
                                                           csrSortedValC::Ptr{Float16},
                                                           csrSortedRowPtrC::Ptr{Cint},
                                                           csrSortedColIndC::Ptr{Cint},
                                                           pBufferSizeInBytes::Ptr{Csize_t})::cusparseStatus_t
end

@checked function cusparseHpruneCsr2csrNnz(handle, m, n, nnzA, descrA, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA, threshold,
                                           descrC, csrSortedRowPtrC, nnzTotalDevHostPtr,
                                           pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseHpruneCsr2csrNnz(handle::cusparseHandle_t, m::Cint, n::Cint,
                                                nnzA::Cint, descrA::cusparseMatDescr_t,
                                                csrSortedValA::Ptr{Float16},
                                                csrSortedRowPtrA::Ptr{Cint},
                                                csrSortedColIndA::Ptr{Cint},
                                                threshold::Ptr{Float16},
                                                descrC::cusparseMatDescr_t,
                                                csrSortedRowPtrC::Ptr{Cint},
                                                nnzTotalDevHostPtr::Ptr{Cint},
                                                pBuffer::Ptr{Cvoid})::cusparseStatus_t
end

@checked function cusparseHpruneCsr2csr(handle, m, n, nnzA, descrA, csrSortedValA,
                                        csrSortedRowPtrA, csrSortedColIndA, threshold,
                                        descrC, csrSortedValC, csrSortedRowPtrC,
                                        csrSortedColIndC, pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseHpruneCsr2csr(handle::cusparseHandle_t, m::Cint, n::Cint,
                                             nnzA::Cint, descrA::cusparseMatDescr_t,
                                             csrSortedValA::Ptr{Float16},
                                             csrSortedRowPtrA::Ptr{Cint},
                                             csrSortedColIndA::Ptr{Cint},
                                             threshold::Ptr{Float16},
                                             descrC::cusparseMatDescr_t,
                                             csrSortedValC::Ptr{Float16},
                                             csrSortedRowPtrC::Ptr{Cint},
                                             csrSortedColIndC::Ptr{Cint},
                                             pBuffer::Ptr{Cvoid})::cusparseStatus_t
end

@checked function cusparseHpruneDense2csrByPercentage_bufferSizeExt(handle, m, n, A, lda,
                                                                    percentage, descrC,
                                                                    csrSortedValC,
                                                                    csrSortedRowPtrC,
                                                                    csrSortedColIndC, info,
                                                                    pBufferSizeInBytes)
    initialize_context()
    @ccall libcusparse.cusparseHpruneDense2csrByPercentage_bufferSizeExt(handle::cusparseHandle_t,
                                                                         m::Cint, n::Cint,
                                                                         A::Ptr{Float16},
                                                                         lda::Cint,
                                                                         percentage::Cfloat,
                                                                         descrC::cusparseMatDescr_t,
                                                                         csrSortedValC::Ptr{Float16},
                                                                         csrSortedRowPtrC::Ptr{Cint},
                                                                         csrSortedColIndC::Ptr{Cint},
                                                                         info::pruneInfo_t,
                                                                         pBufferSizeInBytes::Ptr{Csize_t})::cusparseStatus_t
end

@checked function cusparseHpruneDense2csrNnzByPercentage(handle, m, n, A, lda, percentage,
                                                         descrC, csrRowPtrC,
                                                         nnzTotalDevHostPtr, info, pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseHpruneDense2csrNnzByPercentage(handle::cusparseHandle_t,
                                                              m::Cint, n::Cint,
                                                              A::Ptr{Float16}, lda::Cint,
                                                              percentage::Cfloat,
                                                              descrC::cusparseMatDescr_t,
                                                              csrRowPtrC::Ptr{Cint},
                                                              nnzTotalDevHostPtr::Ptr{Cint},
                                                              info::pruneInfo_t,
                                                              pBuffer::Ptr{Cvoid})::cusparseStatus_t
end

@checked function cusparseHpruneDense2csrByPercentage(handle, m, n, A, lda, percentage,
                                                      descrC, csrSortedValC,
                                                      csrSortedRowPtrC, csrSortedColIndC,
                                                      info, pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseHpruneDense2csrByPercentage(handle::cusparseHandle_t,
                                                           m::Cint, n::Cint,
                                                           A::Ptr{Float16}, lda::Cint,
                                                           percentage::Cfloat,
                                                           descrC::cusparseMatDescr_t,
                                                           csrSortedValC::Ptr{Float16},
                                                           csrSortedRowPtrC::Ptr{Cint},
                                                           csrSortedColIndC::Ptr{Cint},
                                                           info::pruneInfo_t,
                                                           pBuffer::Ptr{Cvoid})::cusparseStatus_t
end

@checked function cusparseHpruneCsr2csrByPercentage_bufferSizeExt(handle, m, n, nnzA,
                                                                  descrA, csrSortedValA,
                                                                  csrSortedRowPtrA,
                                                                  csrSortedColIndA,
                                                                  percentage, descrC,
                                                                  csrSortedValC,
                                                                  csrSortedRowPtrC,
                                                                  csrSortedColIndC, info,
                                                                  pBufferSizeInBytes)
    initialize_context()
    @ccall libcusparse.cusparseHpruneCsr2csrByPercentage_bufferSizeExt(handle::cusparseHandle_t,
                                                                       m::Cint, n::Cint,
                                                                       nnzA::Cint,
                                                                       descrA::cusparseMatDescr_t,
                                                                       csrSortedValA::Ptr{Float16},
                                                                       csrSortedRowPtrA::Ptr{Cint},
                                                                       csrSortedColIndA::Ptr{Cint},
                                                                       percentage::Cfloat,
                                                                       descrC::cusparseMatDescr_t,
                                                                       csrSortedValC::Ptr{Float16},
                                                                       csrSortedRowPtrC::Ptr{Cint},
                                                                       csrSortedColIndC::Ptr{Cint},
                                                                       info::pruneInfo_t,
                                                                       pBufferSizeInBytes::Ptr{Csize_t})::cusparseStatus_t
end

@checked function cusparseHpruneCsr2csrNnzByPercentage(handle, m, n, nnzA, descrA,
                                                       csrSortedValA, csrSortedRowPtrA,
                                                       csrSortedColIndA, percentage, descrC,
                                                       csrSortedRowPtrC, nnzTotalDevHostPtr,
                                                       info, pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseHpruneCsr2csrNnzByPercentage(handle::cusparseHandle_t,
                                                            m::Cint, n::Cint, nnzA::Cint,
                                                            descrA::cusparseMatDescr_t,
                                                            csrSortedValA::Ptr{Float16},
                                                            csrSortedRowPtrA::Ptr{Cint},
                                                            csrSortedColIndA::Ptr{Cint},
                                                            percentage::Cfloat,
                                                            descrC::cusparseMatDescr_t,
                                                            csrSortedRowPtrC::Ptr{Cint},
                                                            nnzTotalDevHostPtr::Ptr{Cint},
                                                            info::pruneInfo_t,
                                                            pBuffer::Ptr{Cvoid})::cusparseStatus_t
end

@checked function cusparseHpruneCsr2csrByPercentage(handle, m, n, nnzA, descrA,
                                                    csrSortedValA, csrSortedRowPtrA,
                                                    csrSortedColIndA, percentage, descrC,
                                                    csrSortedValC, csrSortedRowPtrC,
                                                    csrSortedColIndC, info, pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseHpruneCsr2csrByPercentage(handle::cusparseHandle_t, m::Cint,
                                                         n::Cint, nnzA::Cint,
                                                         descrA::cusparseMatDescr_t,
                                                         csrSortedValA::Ptr{Float16},
                                                         csrSortedRowPtrA::Ptr{Cint},
                                                         csrSortedColIndA::Ptr{Cint},
                                                         percentage::Cfloat,
                                                         descrC::cusparseMatDescr_t,
                                                         csrSortedValC::Ptr{Float16},
                                                         csrSortedRowPtrC::Ptr{Cint},
                                                         csrSortedColIndC::Ptr{Cint},
                                                         info::pruneInfo_t,
                                                         pBuffer::Ptr{Cvoid})::cusparseStatus_t
end
