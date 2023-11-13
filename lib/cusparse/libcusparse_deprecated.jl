# removed in CUDA 12

mutable struct csrsv2Info end

const csrsv2Info_t = Ptr{csrsv2Info}

@checked function cusparseCreateCsrsv2Info(info)
    initialize_context()
    @ccall libcusparse.cusparseCreateCsrsv2Info(info::Ptr{csrsv2Info_t})::cusparseStatus_t
end

@checked function cusparseDestroyCsrsv2Info(info)
    initialize_context()
    @ccall libcusparse.cusparseDestroyCsrsv2Info(info::csrsv2Info_t)::cusparseStatus_t
end

mutable struct csrsm2Info end

const csrsm2Info_t = Ptr{csrsm2Info}

@checked function cusparseCreateCsrsm2Info(info)
    initialize_context()
    @ccall libcusparse.cusparseCreateCsrsm2Info(info::Ptr{csrsm2Info_t})::cusparseStatus_t
end

@checked function cusparseDestroyCsrsm2Info(info)
    initialize_context()
    @ccall libcusparse.cusparseDestroyCsrsm2Info(info::csrsm2Info_t)::cusparseStatus_t
end

@checked function cusparseXcsrsv2_zeroPivot(handle, info, position)
    initialize_context()
    @ccall libcusparse.cusparseXcsrsv2_zeroPivot(handle::cusparseHandle_t,
                                                 info::csrsv2Info_t,
                                                 position::Ptr{Cint})::cusparseStatus_t
end

@checked function cusparseScsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA,
                                             csrSortedRowPtrA, csrSortedColIndA, info,
                                             pBufferSizeInBytes)
    initialize_context()
    @ccall libcusparse.cusparseScsrsv2_bufferSize(handle::cusparseHandle_t,
                                                  transA::cusparseOperation_t, m::Cint,
                                                  nnz::Cint, descrA::cusparseMatDescr_t,
                                                  csrSortedValA::CuPtr{Cfloat},
                                                  csrSortedRowPtrA::CuPtr{Cint},
                                                  csrSortedColIndA::CuPtr{Cint},
                                                  info::csrsv2Info_t,
                                                  pBufferSizeInBytes::Ptr{Cint})::cusparseStatus_t
end

@checked function cusparseDcsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA,
                                             csrSortedRowPtrA, csrSortedColIndA, info,
                                             pBufferSizeInBytes)
    initialize_context()
    @ccall libcusparse.cusparseDcsrsv2_bufferSize(handle::cusparseHandle_t,
                                                  transA::cusparseOperation_t, m::Cint,
                                                  nnz::Cint, descrA::cusparseMatDescr_t,
                                                  csrSortedValA::CuPtr{Cdouble},
                                                  csrSortedRowPtrA::CuPtr{Cint},
                                                  csrSortedColIndA::CuPtr{Cint},
                                                  info::csrsv2Info_t,
                                                  pBufferSizeInBytes::Ptr{Cint})::cusparseStatus_t
end

@checked function cusparseCcsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA,
                                             csrSortedRowPtrA, csrSortedColIndA, info,
                                             pBufferSizeInBytes)
    initialize_context()
    @ccall libcusparse.cusparseCcsrsv2_bufferSize(handle::cusparseHandle_t,
                                                  transA::cusparseOperation_t, m::Cint,
                                                  nnz::Cint, descrA::cusparseMatDescr_t,
                                                  csrSortedValA::CuPtr{cuComplex},
                                                  csrSortedRowPtrA::CuPtr{Cint},
                                                  csrSortedColIndA::CuPtr{Cint},
                                                  info::csrsv2Info_t,
                                                  pBufferSizeInBytes::Ptr{Cint})::cusparseStatus_t
end

@checked function cusparseZcsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA,
                                             csrSortedRowPtrA, csrSortedColIndA, info,
                                             pBufferSizeInBytes)
    initialize_context()
    @ccall libcusparse.cusparseZcsrsv2_bufferSize(handle::cusparseHandle_t,
                                                  transA::cusparseOperation_t, m::Cint,
                                                  nnz::Cint, descrA::cusparseMatDescr_t,
                                                  csrSortedValA::CuPtr{cuDoubleComplex},
                                                  csrSortedRowPtrA::CuPtr{Cint},
                                                  csrSortedColIndA::CuPtr{Cint},
                                                  info::csrsv2Info_t,
                                                  pBufferSizeInBytes::Ptr{Cint})::cusparseStatus_t
end

@checked function cusparseScsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA, info, policy,
                                           pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseScsrsv2_analysis(handle::cusparseHandle_t,
                                                transA::cusparseOperation_t, m::Cint,
                                                nnz::Cint, descrA::cusparseMatDescr_t,
                                                csrSortedValA::CuPtr{Cfloat},
                                                csrSortedRowPtrA::CuPtr{Cint},
                                                csrSortedColIndA::CuPtr{Cint},
                                                info::csrsv2Info_t,
                                                policy::cusparseSolvePolicy_t,
                                                pBuffer::CuPtr{Cvoid})::cusparseStatus_t
end

@checked function cusparseDcsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA, info, policy,
                                           pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseDcsrsv2_analysis(handle::cusparseHandle_t,
                                                transA::cusparseOperation_t, m::Cint,
                                                nnz::Cint, descrA::cusparseMatDescr_t,
                                                csrSortedValA::CuPtr{Cdouble},
                                                csrSortedRowPtrA::CuPtr{Cint},
                                                csrSortedColIndA::CuPtr{Cint},
                                                info::csrsv2Info_t,
                                                policy::cusparseSolvePolicy_t,
                                                pBuffer::CuPtr{Cvoid})::cusparseStatus_t
end

@checked function cusparseCcsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA, info, policy,
                                           pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseCcsrsv2_analysis(handle::cusparseHandle_t,
                                                transA::cusparseOperation_t, m::Cint,
                                                nnz::Cint, descrA::cusparseMatDescr_t,
                                                csrSortedValA::CuPtr{cuComplex},
                                                csrSortedRowPtrA::CuPtr{Cint},
                                                csrSortedColIndA::CuPtr{Cint},
                                                info::csrsv2Info_t,
                                                policy::cusparseSolvePolicy_t,
                                                pBuffer::CuPtr{Cvoid})::cusparseStatus_t
end

@checked function cusparseZcsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA, info, policy,
                                           pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseZcsrsv2_analysis(handle::cusparseHandle_t,
                                                transA::cusparseOperation_t, m::Cint,
                                                nnz::Cint, descrA::cusparseMatDescr_t,
                                                csrSortedValA::CuPtr{cuDoubleComplex},
                                                csrSortedRowPtrA::CuPtr{Cint},
                                                csrSortedColIndA::CuPtr{Cint},
                                                info::csrsv2Info_t,
                                                policy::cusparseSolvePolicy_t,
                                                pBuffer::CuPtr{Cvoid})::cusparseStatus_t
end

@checked function cusparseScsrsv2_solve(handle, transA, m, nnz, alpha, descrA,
                                        csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                                        info, f, x, policy, pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseScsrsv2_solve(handle::cusparseHandle_t,
                                             transA::cusparseOperation_t, m::Cint,
                                             nnz::Cint, alpha::Ref{Cfloat},
                                             descrA::cusparseMatDescr_t,
                                             csrSortedValA::CuPtr{Cfloat},
                                             csrSortedRowPtrA::CuPtr{Cint},
                                             csrSortedColIndA::CuPtr{Cint},
                                             info::csrsv2Info_t, f::CuPtr{Cfloat},
                                             x::CuPtr{Cfloat},
                                             policy::cusparseSolvePolicy_t,
                                             pBuffer::CuPtr{Cvoid})::cusparseStatus_t
end

@checked function cusparseDcsrsv2_solve(handle, transA, m, nnz, alpha, descrA,
                                        csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                                        info, f, x, policy, pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseDcsrsv2_solve(handle::cusparseHandle_t,
                                             transA::cusparseOperation_t, m::Cint,
                                             nnz::Cint, alpha::Ref{Cdouble},
                                             descrA::cusparseMatDescr_t,
                                             csrSortedValA::CuPtr{Cdouble},
                                             csrSortedRowPtrA::CuPtr{Cint},
                                             csrSortedColIndA::CuPtr{Cint},
                                             info::csrsv2Info_t, f::CuPtr{Cdouble},
                                             x::CuPtr{Cdouble},
                                             policy::cusparseSolvePolicy_t,
                                             pBuffer::CuPtr{Cvoid})::cusparseStatus_t
end

@checked function cusparseCcsrsv2_solve(handle, transA, m, nnz, alpha, descrA,
                                        csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                                        info, f, x, policy, pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseCcsrsv2_solve(handle::cusparseHandle_t,
                                             transA::cusparseOperation_t, m::Cint,
                                             nnz::Cint, alpha::Ref{cuComplex},
                                             descrA::cusparseMatDescr_t,
                                             csrSortedValA::CuPtr{cuComplex},
                                             csrSortedRowPtrA::CuPtr{Cint},
                                             csrSortedColIndA::CuPtr{Cint},
                                             info::csrsv2Info_t, f::CuPtr{cuComplex},
                                             x::CuPtr{cuComplex},
                                             policy::cusparseSolvePolicy_t,
                                             pBuffer::CuPtr{Cvoid})::cusparseStatus_t
end

@checked function cusparseZcsrsv2_solve(handle, transA, m, nnz, alpha, descrA,
                                        csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                                        info, f, x, policy, pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseZcsrsv2_solve(handle::cusparseHandle_t,
                                             transA::cusparseOperation_t, m::Cint,
                                             nnz::Cint, alpha::Ref{cuDoubleComplex},
                                             descrA::cusparseMatDescr_t,
                                             csrSortedValA::CuPtr{cuDoubleComplex},
                                             csrSortedRowPtrA::CuPtr{Cint},
                                             csrSortedColIndA::CuPtr{Cint},
                                             info::csrsv2Info_t, f::CuPtr{cuDoubleComplex},
                                             x::CuPtr{cuDoubleComplex},
                                             policy::cusparseSolvePolicy_t,
                                             pBuffer::CuPtr{Cvoid})::cusparseStatus_t
end

@checked function cusparseXcsrsm2_zeroPivot(handle, info, position)
    initialize_context()
    @ccall libcusparse.cusparseXcsrsm2_zeroPivot(handle::cusparseHandle_t,
                                                 info::csrsm2Info_t,
                                                 position::Ptr{Cint})::cusparseStatus_t
end

@checked function cusparseScsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz,
                                                alpha, descrA, csrSortedValA,
                                                csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                                                info, policy, pBufferSize)
    initialize_context()
    @ccall libcusparse.cusparseScsrsm2_bufferSizeExt(handle::cusparseHandle_t, algo::Cint,
                                                     transA::cusparseOperation_t,
                                                     transB::cusparseOperation_t, m::Cint,
                                                     nrhs::Cint, nnz::Cint,
                                                     alpha::Ref{Cfloat},
                                                     descrA::cusparseMatDescr_t,
                                                     csrSortedValA::CuPtr{Cfloat},
                                                     csrSortedRowPtrA::CuPtr{Cint},
                                                     csrSortedColIndA::CuPtr{Cint},
                                                     B::CuPtr{Cfloat}, ldb::Cint,
                                                     info::csrsm2Info_t,
                                                     policy::cusparseSolvePolicy_t,
                                                     pBufferSize::Ptr{Csize_t})::cusparseStatus_t
end

@checked function cusparseDcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz,
                                                alpha, descrA, csrSortedValA,
                                                csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                                                info, policy, pBufferSize)
    initialize_context()
    @ccall libcusparse.cusparseDcsrsm2_bufferSizeExt(handle::cusparseHandle_t, algo::Cint,
                                                     transA::cusparseOperation_t,
                                                     transB::cusparseOperation_t, m::Cint,
                                                     nrhs::Cint, nnz::Cint,
                                                     alpha::Ref{Cdouble},
                                                     descrA::cusparseMatDescr_t,
                                                     csrSortedValA::CuPtr{Cdouble},
                                                     csrSortedRowPtrA::CuPtr{Cint},
                                                     csrSortedColIndA::CuPtr{Cint},
                                                     B::CuPtr{Cdouble}, ldb::Cint,
                                                     info::csrsm2Info_t,
                                                     policy::cusparseSolvePolicy_t,
                                                     pBufferSize::Ptr{Csize_t})::cusparseStatus_t
end

@checked function cusparseCcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz,
                                                alpha, descrA, csrSortedValA,
                                                csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                                                info, policy, pBufferSize)
    initialize_context()
    @ccall libcusparse.cusparseCcsrsm2_bufferSizeExt(handle::cusparseHandle_t, algo::Cint,
                                                     transA::cusparseOperation_t,
                                                     transB::cusparseOperation_t, m::Cint,
                                                     nrhs::Cint, nnz::Cint,
                                                     alpha::Ref{cuComplex},
                                                     descrA::cusparseMatDescr_t,
                                                     csrSortedValA::CuPtr{cuComplex},
                                                     csrSortedRowPtrA::CuPtr{Cint},
                                                     csrSortedColIndA::CuPtr{Cint},
                                                     B::CuPtr{cuComplex}, ldb::Cint,
                                                     info::csrsm2Info_t,
                                                     policy::cusparseSolvePolicy_t,
                                                     pBufferSize::Ptr{Csize_t})::cusparseStatus_t
end

@checked function cusparseZcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz,
                                                alpha, descrA, csrSortedValA,
                                                csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                                                info, policy, pBufferSize)
    initialize_context()
    @ccall libcusparse.cusparseZcsrsm2_bufferSizeExt(handle::cusparseHandle_t, algo::Cint,
                                                     transA::cusparseOperation_t,
                                                     transB::cusparseOperation_t, m::Cint,
                                                     nrhs::Cint, nnz::Cint,
                                                     alpha::Ref{cuDoubleComplex},
                                                     descrA::cusparseMatDescr_t,
                                                     csrSortedValA::CuPtr{cuDoubleComplex},
                                                     csrSortedRowPtrA::CuPtr{Cint},
                                                     csrSortedColIndA::CuPtr{Cint},
                                                     B::CuPtr{cuDoubleComplex}, ldb::Cint,
                                                     info::csrsm2Info_t,
                                                     policy::cusparseSolvePolicy_t,
                                                     pBufferSize::Ptr{Csize_t})::cusparseStatus_t
end

@checked function cusparseScsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz,
                                           alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                                           csrSortedColIndA, B, ldb, info, policy, pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseScsrsm2_analysis(handle::cusparseHandle_t, algo::Cint,
                                                transA::cusparseOperation_t,
                                                transB::cusparseOperation_t, m::Cint,
                                                nrhs::Cint, nnz::Cint, alpha::Ref{Cfloat},
                                                descrA::cusparseMatDescr_t,
                                                csrSortedValA::CuPtr{Cfloat},
                                                csrSortedRowPtrA::CuPtr{Cint},
                                                csrSortedColIndA::CuPtr{Cint},
                                                B::CuPtr{Cfloat}, ldb::Cint,
                                                info::csrsm2Info_t,
                                                policy::cusparseSolvePolicy_t,
                                                pBuffer::CuPtr{Cvoid})::cusparseStatus_t
end

@checked function cusparseDcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz,
                                           alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                                           csrSortedColIndA, B, ldb, info, policy, pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseDcsrsm2_analysis(handle::cusparseHandle_t, algo::Cint,
                                                transA::cusparseOperation_t,
                                                transB::cusparseOperation_t, m::Cint,
                                                nrhs::Cint, nnz::Cint, alpha::Ref{Cdouble},
                                                descrA::cusparseMatDescr_t,
                                                csrSortedValA::CuPtr{Cdouble},
                                                csrSortedRowPtrA::CuPtr{Cint},
                                                csrSortedColIndA::CuPtr{Cint},
                                                B::CuPtr{Cdouble}, ldb::Cint,
                                                info::csrsm2Info_t,
                                                policy::cusparseSolvePolicy_t,
                                                pBuffer::CuPtr{Cvoid})::cusparseStatus_t
end

@checked function cusparseCcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz,
                                           alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                                           csrSortedColIndA, B, ldb, info, policy, pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseCcsrsm2_analysis(handle::cusparseHandle_t, algo::Cint,
                                                transA::cusparseOperation_t,
                                                transB::cusparseOperation_t, m::Cint,
                                                nrhs::Cint, nnz::Cint,
                                                alpha::Ref{cuComplex},
                                                descrA::cusparseMatDescr_t,
                                                csrSortedValA::CuPtr{cuComplex},
                                                csrSortedRowPtrA::CuPtr{Cint},
                                                csrSortedColIndA::CuPtr{Cint},
                                                B::CuPtr{cuComplex}, ldb::Cint,
                                                info::csrsm2Info_t,
                                                policy::cusparseSolvePolicy_t,
                                                pBuffer::CuPtr{Cvoid})::cusparseStatus_t
end

@checked function cusparseZcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz,
                                           alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                                           csrSortedColIndA, B, ldb, info, policy, pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseZcsrsm2_analysis(handle::cusparseHandle_t, algo::Cint,
                                                transA::cusparseOperation_t,
                                                transB::cusparseOperation_t, m::Cint,
                                                nrhs::Cint, nnz::Cint,
                                                alpha::Ref{cuDoubleComplex},
                                                descrA::cusparseMatDescr_t,
                                                csrSortedValA::CuPtr{cuDoubleComplex},
                                                csrSortedRowPtrA::CuPtr{Cint},
                                                csrSortedColIndA::CuPtr{Cint},
                                                B::CuPtr{cuDoubleComplex}, ldb::Cint,
                                                info::csrsm2Info_t,
                                                policy::cusparseSolvePolicy_t,
                                                pBuffer::CuPtr{Cvoid})::cusparseStatus_t
end

@checked function cusparseScsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha,
                                        descrA, csrSortedValA, csrSortedRowPtrA,
                                        csrSortedColIndA, B, ldb, info, policy, pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseScsrsm2_solve(handle::cusparseHandle_t, algo::Cint,
                                             transA::cusparseOperation_t,
                                             transB::cusparseOperation_t, m::Cint,
                                             nrhs::Cint, nnz::Cint, alpha::Ref{Cfloat},
                                             descrA::cusparseMatDescr_t,
                                             csrSortedValA::CuPtr{Cfloat},
                                             csrSortedRowPtrA::CuPtr{Cint},
                                             csrSortedColIndA::CuPtr{Cint},
                                             B::CuPtr{Cfloat}, ldb::Cint,
                                             info::csrsm2Info_t,
                                             policy::cusparseSolvePolicy_t,
                                             pBuffer::CuPtr{Cvoid})::cusparseStatus_t
end

@checked function cusparseDcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha,
                                        descrA, csrSortedValA, csrSortedRowPtrA,
                                        csrSortedColIndA, B, ldb, info, policy, pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseDcsrsm2_solve(handle::cusparseHandle_t, algo::Cint,
                                             transA::cusparseOperation_t,
                                             transB::cusparseOperation_t, m::Cint,
                                             nrhs::Cint, nnz::Cint, alpha::Ref{Cdouble},
                                             descrA::cusparseMatDescr_t,
                                             csrSortedValA::CuPtr{Cdouble},
                                             csrSortedRowPtrA::CuPtr{Cint},
                                             csrSortedColIndA::CuPtr{Cint},
                                             B::CuPtr{Cdouble}, ldb::Cint,
                                             info::csrsm2Info_t,
                                             policy::cusparseSolvePolicy_t,
                                             pBuffer::CuPtr{Cvoid})::cusparseStatus_t
end

@checked function cusparseCcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha,
                                        descrA, csrSortedValA, csrSortedRowPtrA,
                                        csrSortedColIndA, B, ldb, info, policy, pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseCcsrsm2_solve(handle::cusparseHandle_t, algo::Cint,
                                             transA::cusparseOperation_t,
                                             transB::cusparseOperation_t, m::Cint,
                                             nrhs::Cint, nnz::Cint, alpha::Ref{cuComplex},
                                             descrA::cusparseMatDescr_t,
                                             csrSortedValA::CuPtr{cuComplex},
                                             csrSortedRowPtrA::CuPtr{Cint},
                                             csrSortedColIndA::CuPtr{Cint},
                                             B::CuPtr{cuComplex}, ldb::Cint,
                                             info::csrsm2Info_t,
                                             policy::cusparseSolvePolicy_t,
                                             pBuffer::CuPtr{Cvoid})::cusparseStatus_t
end

@checked function cusparseZcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha,
                                        descrA, csrSortedValA, csrSortedRowPtrA,
                                        csrSortedColIndA, B, ldb, info, policy, pBuffer)
    initialize_context()
    @ccall libcusparse.cusparseZcsrsm2_solve(handle::cusparseHandle_t, algo::Cint,
                                             transA::cusparseOperation_t,
                                             transB::cusparseOperation_t, m::Cint,
                                             nrhs::Cint, nnz::Cint,
                                             alpha::Ref{cuDoubleComplex},
                                             descrA::cusparseMatDescr_t,
                                             csrSortedValA::CuPtr{cuDoubleComplex},
                                             csrSortedRowPtrA::CuPtr{Cint},
                                             csrSortedColIndA::CuPtr{Cint},
                                             B::CuPtr{cuDoubleComplex}, ldb::Cint,
                                             info::csrsm2Info_t,
                                             policy::cusparseSolvePolicy_t,
                                             pBuffer::CuPtr{Cvoid})::cusparseStatus_t
end
