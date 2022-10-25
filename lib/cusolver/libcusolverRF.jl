using CEnum

@cenum cusolverRfResetValuesFastMode_t::UInt32 begin
    CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF = 0
    CUSOLVERRF_RESET_VALUES_FAST_MODE_ON = 1
end

@cenum cusolverRfMatrixFormat_t::UInt32 begin
    CUSOLVERRF_MATRIX_FORMAT_CSR = 0
    CUSOLVERRF_MATRIX_FORMAT_CSC = 1
end

@cenum cusolverRfUnitDiagonal_t::UInt32 begin
    CUSOLVERRF_UNIT_DIAGONAL_STORED_L = 0
    CUSOLVERRF_UNIT_DIAGONAL_STORED_U = 1
    CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L = 2
    CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U = 3
end

@cenum cusolverRfFactorization_t::UInt32 begin
    CUSOLVERRF_FACTORIZATION_ALG0 = 0
    CUSOLVERRF_FACTORIZATION_ALG1 = 1
    CUSOLVERRF_FACTORIZATION_ALG2 = 2
end

@cenum cusolverRfTriangularSolve_t::UInt32 begin
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG1 = 1
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG2 = 2
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG3 = 3
end

@cenum cusolverRfNumericBoostReport_t::UInt32 begin
    CUSOLVERRF_NUMERIC_BOOST_NOT_USED = 0
    CUSOLVERRF_NUMERIC_BOOST_USED = 1
end

mutable struct cusolverRfCommon end

const cusolverRfHandle_t = Ptr{cusolverRfCommon}

@checked function cusolverRfCreate(handle)
    initialize_context()
    ccall((:cusolverRfCreate, libcusolver), cusolverStatus_t, (Ptr{cusolverRfHandle_t},),
          handle)
end

@checked function cusolverRfDestroy(handle)
    initialize_context()
    ccall((:cusolverRfDestroy, libcusolver), cusolverStatus_t, (cusolverRfHandle_t,),
          handle)
end

@checked function cusolverRfGetMatrixFormat(handle, format, diag)
    initialize_context()
    ccall((:cusolverRfGetMatrixFormat, libcusolver), cusolverStatus_t,
          (cusolverRfHandle_t, Ptr{cusolverRfMatrixFormat_t},
           Ptr{cusolverRfUnitDiagonal_t}), handle, format, diag)
end

@checked function cusolverRfSetMatrixFormat(handle, format, diag)
    initialize_context()
    ccall((:cusolverRfSetMatrixFormat, libcusolver), cusolverStatus_t,
          (cusolverRfHandle_t, cusolverRfMatrixFormat_t, cusolverRfUnitDiagonal_t), handle,
          format, diag)
end

@checked function cusolverRfSetNumericProperties(handle, zero, boost)
    initialize_context()
    ccall((:cusolverRfSetNumericProperties, libcusolver), cusolverStatus_t,
          (cusolverRfHandle_t, Cdouble, Cdouble), handle, zero, boost)
end

@checked function cusolverRfGetNumericProperties(handle, zero, boost)
    initialize_context()
    ccall((:cusolverRfGetNumericProperties, libcusolver), cusolverStatus_t,
          (cusolverRfHandle_t, Ptr{Cdouble}, Ptr{Cdouble}), handle, zero, boost)
end

@checked function cusolverRfGetNumericBoostReport(handle, report)
    initialize_context()
    ccall((:cusolverRfGetNumericBoostReport, libcusolver), cusolverStatus_t,
          (cusolverRfHandle_t, Ptr{cusolverRfNumericBoostReport_t}), handle, report)
end

@checked function cusolverRfSetAlgs(handle, factAlg, solveAlg)
    initialize_context()
    ccall((:cusolverRfSetAlgs, libcusolver), cusolverStatus_t,
          (cusolverRfHandle_t, cusolverRfFactorization_t, cusolverRfTriangularSolve_t),
          handle, factAlg, solveAlg)
end

@checked function cusolverRfGetAlgs(handle, factAlg, solveAlg)
    initialize_context()
    ccall((:cusolverRfGetAlgs, libcusolver), cusolverStatus_t,
          (cusolverRfHandle_t, Ptr{cusolverRfFactorization_t},
           Ptr{cusolverRfTriangularSolve_t}), handle, factAlg, solveAlg)
end

@checked function cusolverRfGetResetValuesFastMode(handle, fastMode)
    initialize_context()
    ccall((:cusolverRfGetResetValuesFastMode, libcusolver), cusolverStatus_t,
          (cusolverRfHandle_t, Ptr{cusolverRfResetValuesFastMode_t}), handle, fastMode)
end

@checked function cusolverRfSetResetValuesFastMode(handle, fastMode)
    initialize_context()
    ccall((:cusolverRfSetResetValuesFastMode, libcusolver), cusolverStatus_t,
          (cusolverRfHandle_t, cusolverRfResetValuesFastMode_t), handle, fastMode)
end

@checked function cusolverRfSetupHost(n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA, nnzL,
                                      h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU,
                                      h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q,
                                      handle)
    initialize_context()
    ccall((:cusolverRfSetupHost, libcusolver), cusolverStatus_t,
          (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cint, Ptr{Cint}, Ptr{Cint},
           Ptr{Cdouble}, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint},
           cusolverRfHandle_t), n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA, nnzL,
          h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU,
          h_csrValU, h_P, h_Q, handle)
end

@checked function cusolverRfSetupDevice(n, nnzA, csrRowPtrA, csrColIndA, csrValA, nnzL,
                                        csrRowPtrL, csrColIndL, csrValL, nnzU, csrRowPtrU,
                                        csrColIndU, csrValU, P, Q, handle)
    initialize_context()
    ccall((:cusolverRfSetupDevice, libcusolver), cusolverStatus_t,
          (Cint, Cint, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cint},
           CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble},
           CuPtr{Cint}, CuPtr{Cint}, cusolverRfHandle_t), n, nnzA, csrRowPtrA, csrColIndA,
          csrValA, nnzL, csrRowPtrL, csrColIndL, csrValL, nnzU, csrRowPtrU, csrColIndU,
          csrValU, P, Q, handle)
end

@checked function cusolverRfResetValues(n, nnzA, csrRowPtrA, csrColIndA, csrValA, P, Q,
                                        handle)
    initialize_context()
    ccall((:cusolverRfResetValues, libcusolver), cusolverStatus_t,
          (Cint, Cint, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
           cusolverRfHandle_t), n, nnzA, csrRowPtrA, csrColIndA, csrValA, P, Q, handle)
end

@checked function cusolverRfAnalyze(handle)
    initialize_context()
    ccall((:cusolverRfAnalyze, libcusolver), cusolverStatus_t, (cusolverRfHandle_t,),
          handle)
end

@checked function cusolverRfRefactor(handle)
    initialize_context()
    ccall((:cusolverRfRefactor, libcusolver), cusolverStatus_t, (cusolverRfHandle_t,),
          handle)
end

@checked function cusolverRfAccessBundledFactorsDevice(handle, nnzM, Mp, Mi, Mx)
    initialize_context()
    ccall((:cusolverRfAccessBundledFactorsDevice, libcusolver), cusolverStatus_t,
          (cusolverRfHandle_t, Ptr{Cint}, CuPtr{Ptr{Cint}}, CuPtr{Ptr{Cint}},
           CuPtr{Ptr{Cdouble}}), handle, nnzM, Mp, Mi, Mx)
end

@checked function cusolverRfExtractBundledFactorsHost(handle, h_nnzM, h_Mp, h_Mi, h_Mx)
    initialize_context()
    ccall((:cusolverRfExtractBundledFactorsHost, libcusolver), cusolverStatus_t,
          (cusolverRfHandle_t, Ptr{Cint}, Ptr{Ptr{Cint}}, Ptr{Ptr{Cint}},
           Ptr{Ptr{Cdouble}}), handle, h_nnzM, h_Mp, h_Mi, h_Mx)
end

@checked function cusolverRfExtractSplitFactorsHost(handle, h_nnzL, h_csrRowPtrL,
                                                    h_csrColIndL, h_csrValL, h_nnzU,
                                                    h_csrRowPtrU, h_csrColIndU, h_csrValU)
    initialize_context()
    ccall((:cusolverRfExtractSplitFactorsHost, libcusolver), cusolverStatus_t,
          (cusolverRfHandle_t, Ptr{Cint}, Ptr{Ptr{Cint}}, Ptr{Ptr{Cint}}, Ptr{Ptr{Cdouble}},
           Ptr{Cint}, Ptr{Ptr{Cint}}, Ptr{Ptr{Cint}}, Ptr{Ptr{Cdouble}}), handle, h_nnzL,
          h_csrRowPtrL, h_csrColIndL, h_csrValL, h_nnzU, h_csrRowPtrU, h_csrColIndU,
          h_csrValU)
end

@checked function cusolverRfSolve(handle, P, Q, nrhs, Temp, ldt, XF, ldxf)
    initialize_context()
    ccall((:cusolverRfSolve, libcusolver), cusolverStatus_t,
          (cusolverRfHandle_t, CuPtr{Cint}, CuPtr{Cint}, Cint, CuPtr{Cdouble}, Cint,
           CuPtr{Cdouble}, Cint), handle, P, Q, nrhs, Temp, ldt, XF, ldxf)
end

@checked function cusolverRfBatchSetupHost(batchSize, n, nnzA, h_csrRowPtrA, h_csrColIndA,
                                           h_csrValA_array, nnzL, h_csrRowPtrL,
                                           h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU,
                                           h_csrColIndU, h_csrValU, h_P, h_Q, handle)
    initialize_context()
    ccall((:cusolverRfBatchSetupHost, libcusolver), cusolverStatus_t,
          (Cint, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint},
           Ptr{Cint}, Ptr{Cdouble}, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint},
           Ptr{Cint}, cusolverRfHandle_t), batchSize, n, nnzA, h_csrRowPtrA, h_csrColIndA,
          h_csrValA_array, nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU,
          h_csrColIndU, h_csrValU, h_P, h_Q, handle)
end

@checked function cusolverRfBatchResetValues(batchSize, n, nnzA, csrRowPtrA, csrColIndA,
                                             csrValA_array, P, Q, handle)
    initialize_context()
    ccall((:cusolverRfBatchResetValues, libcusolver), cusolverStatus_t,
          (Cint, Cint, Cint, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Ptr{Cdouble}}, CuPtr{Cint},
           CuPtr{Cint}, cusolverRfHandle_t), batchSize, n, nnzA, csrRowPtrA, csrColIndA,
          csrValA_array, P, Q, handle)
end

@checked function cusolverRfBatchAnalyze(handle)
    initialize_context()
    ccall((:cusolverRfBatchAnalyze, libcusolver), cusolverStatus_t, (cusolverRfHandle_t,),
          handle)
end

@checked function cusolverRfBatchRefactor(handle)
    initialize_context()
    ccall((:cusolverRfBatchRefactor, libcusolver), cusolverStatus_t, (cusolverRfHandle_t,),
          handle)
end

@checked function cusolverRfBatchSolve(handle, P, Q, nrhs, Temp, ldt, XF_array, ldxf)
    initialize_context()
    ccall((:cusolverRfBatchSolve, libcusolver), cusolverStatus_t,
          (cusolverRfHandle_t, CuPtr{Cint}, CuPtr{Cint}, Cint, CuPtr{Cdouble}, Cint,
           CuPtr{Ptr{Cdouble}}, Cint), handle, P, Q, nrhs, Temp, ldt, XF_array, ldxf)
end

@checked function cusolverRfBatchZeroPivot(handle, position)
    initialize_context()
    ccall((:cusolverRfBatchZeroPivot, libcusolver), cusolverStatus_t,
          (cusolverRfHandle_t, CuPtr{Cint}), handle, position)
end
