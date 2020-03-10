# Julia wrapper for header: cusolverDn.h
# Automatically generated using Clang.jl


@checked function cusolverGetProperty(type, value)
    @runtime_ccall((:cusolverGetProperty, libcusolver()), cusolverStatus_t,
                   (libraryPropertyType, Ptr{Cint}),
                   type, value)
end

@checked function cusolverGetVersion(version)
    @runtime_ccall((:cusolverGetVersion, libcusolver()), cusolverStatus_t,
                   (Ptr{Cint},),
                   version)
end

@checked function cusolverDnCreate(handle)
    initialize_api()
    @runtime_ccall((:cusolverDnCreate, libcusolver()), cusolverStatus_t,
                   (Ptr{cusolverDnHandle_t},),
                   handle)
end

@checked function cusolverDnDestroy(handle)
    initialize_api()
    @runtime_ccall((:cusolverDnDestroy, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t,),
                   handle)
end

@checked function cusolverDnSetStream(handle, streamId)
    initialize_api()
    @runtime_ccall((:cusolverDnSetStream, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, CUstream),
                   handle, streamId)
end

@checked function cusolverDnGetStream(handle, streamId)
    initialize_api()
    @runtime_ccall((:cusolverDnGetStream, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Ptr{CUstream}),
                   handle, streamId)
end

@checked function cusolverDnIRSParamsCreate(params_ptr)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSParamsCreate, libcusolver()), cusolverStatus_t,
                   (Ptr{cusolverDnIRSParams_t},),
                   params_ptr)
end

@checked function cusolverDnIRSParamsDestroy(params)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSParamsDestroy, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t,),
                   params)
end

@checked function cusolverDnIRSParamsSetTol(params, data_type, val)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSParamsSetTol, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cudaDataType, Cdouble),
                   params, data_type, val)
end

@checked function cusolverDnIRSParamsSetTolInner(params, data_type, val)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSParamsSetTolInner, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cudaDataType, Cdouble),
                   params, data_type, val)
end

@checked function cusolverDnIRSParamsSetSolverPrecisions(params, solver_main_precision,
                                                         solver_lowest_precision)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSParamsSetSolverPrecisions, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cudaDataType, cudaDataType),
                   params, solver_main_precision, solver_lowest_precision)
end

@checked function cusolverDnIRSParamsSetRefinementSolver(params, refinement_solver)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSParamsSetRefinementSolver, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cusolverIRSRefinement_t),
                   params, refinement_solver)
end

@checked function cusolverDnIRSParamsSetMaxIters(params, maxiters)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSParamsSetMaxIters, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cusolver_int_t),
                   params, maxiters)
end

@checked function cusolverDnIRSParamsSetMaxItersInner(params, maxiters_inner)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSParamsSetMaxItersInner, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cusolver_int_t),
                   params, maxiters_inner)
end

@checked function cusolverDnIRSParamsGetNiters(params, niters)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSParamsGetNiters, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, Ptr{cusolver_int_t}),
                   params, niters)
end

@checked function cusolverDnIRSParamsGetOuterNiters(params, outer_niters)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSParamsGetOuterNiters, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, Ptr{cusolver_int_t}),
                   params, outer_niters)
end

@checked function cusolverDnIRSParamsGetMaxIters(params, maxiters)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSParamsGetMaxIters, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, Ptr{cusolver_int_t}),
                   params, maxiters)
end

@checked function cusolverDnIRSParamsSetSolverMainPrecision(params, solver_main_precision)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSParamsSetSolverMainPrecision, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cudaDataType),
                   params, solver_main_precision)
end

@checked function cusolverDnIRSParamsSetSolverLowestPrecision(params,
                                                              solver_lowest_precision)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSParamsSetSolverLowestPrecision, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cudaDataType),
                   params, solver_lowest_precision)
end

@checked function cusolverDnIRSInfosDestroy(params, infos)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSInfosDestroy, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cusolverDnIRSInfos_t),
                   params, infos)
end

@checked function cusolverDnIRSInfosCreate(params, infos_ptr)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSInfosCreate, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, Ptr{cusolverDnIRSInfos_t}),
                   params, infos_ptr)
end

@checked function cusolverDnIRSInfosGetNiters(params, infos, niters)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSInfosGetNiters, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cusolverDnIRSInfos_t, Ptr{cusolver_int_t}),
                   params, infos, niters)
end

@checked function cusolverDnIRSInfosGetOuterNiters(params, infos, outer_niters)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSInfosGetOuterNiters, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cusolverDnIRSInfos_t, Ptr{cusolver_int_t}),
                   params, infos, outer_niters)
end

@checked function cusolverDnIRSInfosGetMaxIters(params, infos, maxiters)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSInfosGetMaxIters, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cusolverDnIRSInfos_t, Ptr{cusolver_int_t}),
                   params, infos, maxiters)
end

@checked function cusolverDnIRSInfosRequestResidual(params, infos)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSInfosRequestResidual, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cusolverDnIRSInfos_t),
                   params, infos)
end

@checked function cusolverDnIRSInfosGetResidualHistory(params, infos, residual_history)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSInfosGetResidualHistory, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cusolverDnIRSInfos_t, Ptr{Ptr{Cvoid}}),
                   params, infos, residual_history)
end

@checked function cusolverDnZZgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_api()
    @runtime_ccall((:cusolverDnZZgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cusolver_int_t},
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{Cvoid}, Csize_t, Ptr{cusolver_int_t},
                    CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnZCgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_api()
    @runtime_ccall((:cusolverDnZCgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cusolver_int_t},
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{Cvoid}, Csize_t, Ptr{cusolver_int_t},
                    CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnZKgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_api()
    @runtime_ccall((:cusolverDnZKgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cusolver_int_t},
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{Cvoid}, Csize_t, Ptr{cusolver_int_t},
                    CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnCCgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_api()
    @runtime_ccall((:cusolverDnCCgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Csize_t, Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnCKgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_api()
    @runtime_ccall((:cusolverDnCKgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Csize_t, Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnDDgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_api()
    @runtime_ccall((:cusolverDnDDgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cdouble},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnDSgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_api()
    @runtime_ccall((:cusolverDnDSgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cdouble},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnDHgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_api()
    @runtime_ccall((:cusolverDnDHgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cdouble},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnSSgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_api()
    @runtime_ccall((:cusolverDnSSgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cfloat},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnSHgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_api()
    @runtime_ccall((:cusolverDnSHgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cfloat},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnZZgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_api()
    @runtime_ccall((:cusolverDnZZgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cusolver_int_t},
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnZCgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_api()
    @runtime_ccall((:cusolverDnZCgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cusolver_int_t},
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnZKgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_api()
    @runtime_ccall((:cusolverDnZKgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cusolver_int_t},
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnCCgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_api()
    @runtime_ccall((:cusolverDnCCgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnCKgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_api()
    @runtime_ccall((:cusolverDnCKgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnDDgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_api()
    @runtime_ccall((:cusolverDnDDgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cdouble},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnDSgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_api()
    @runtime_ccall((:cusolverDnDSgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cdouble},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnDHgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_api()
    @runtime_ccall((:cusolverDnDHgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cdouble},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnSSgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_api()
    @runtime_ccall((:cusolverDnSSgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cfloat},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnSHgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_api()
    @runtime_ccall((:cusolverDnSHgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cfloat},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnIRSXgesv(handle, gesv_irs_params, gesv_irs_infos,
                                     inout_data_type, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                     dX, lddx, dWorkspace, lwork_bytes, niters, d_info)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSXgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnIRSParams_t, cusolverDnIRSInfos_t,
                    cudaDataType, cusolver_int_t, cusolver_int_t, CuPtr{Cvoid},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cvoid}, cusolver_int_t,
                    CuPtr{Cvoid}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, gesv_irs_params, gesv_irs_infos, inout_data_type, n, nrhs, dA,
                   ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, niters,
                   d_info)
end

@checked function cusolverDnIRSXgesv_bufferSize(handle, params, n, nrhs, lwork_bytes)
    initialize_api()
    @runtime_ccall((:cusolverDnIRSXgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnIRSParams_t, cusolver_int_t,
                    cusolver_int_t, Ptr{Csize_t}),
                   handle, params, n, nrhs, lwork_bytes)
end

@checked function cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSpotrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    Ptr{Cint}),
                   handle, uplo, n, A, lda, Lwork)
end

@checked function cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDpotrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    Ptr{Cint}),
                   handle, uplo, n, A, lda, Lwork)
end

@checked function cusolverDnCpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnCpotrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    Ptr{Cint}),
                   handle, uplo, n, A, lda, Lwork)
end

@checked function cusolverDnZpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZpotrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, Ptr{Cint}),
                   handle, uplo, n, A, lda, Lwork)
end

@checked function cusolverDnSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnSpotrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
end

@checked function cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnDpotrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
end

@checked function cusolverDnCpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnCpotrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
end

@checked function cusolverDnZpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnZpotrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
end

@checked function cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnSpotrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
end

@checked function cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnDpotrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cdouble},
                    Cint, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
end

@checked function cusolverDnCpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnCpotrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{cuComplex},
                    Cint, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
end

@checked function cusolverDnZpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnZpotrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
end

@checked function cusolverDnSpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnSpotrfBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Ptr{Cfloat}}, Cint,
                    CuPtr{Cint}, Cint),
                   handle, uplo, n, Aarray, lda, infoArray, batchSize)
end

@checked function cusolverDnDpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnDpotrfBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Ptr{Cdouble}}, Cint,
                    CuPtr{Cint}, Cint),
                   handle, uplo, n, Aarray, lda, infoArray, batchSize)
end

@checked function cusolverDnCpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnCpotrfBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Ptr{cuComplex}},
                    Cint, CuPtr{Cint}, Cint),
                   handle, uplo, n, Aarray, lda, infoArray, batchSize)
end

@checked function cusolverDnZpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnZpotrfBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint,
                    CuPtr{Ptr{cuDoubleComplex}}, Cint, CuPtr{Cint}, Cint),
                   handle, uplo, n, Aarray, lda, infoArray, batchSize)
end

@checked function cusolverDnSpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info,
                                          batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnSpotrsBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Ptr{Cfloat}},
                    Cint, CuPtr{Ptr{Cfloat}}, Cint, CuPtr{Cint}, Cint),
                   handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
end

@checked function cusolverDnDpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info,
                                          batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnDpotrsBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Ptr{Cdouble}},
                    Cint, CuPtr{Ptr{Cdouble}}, Cint, CuPtr{Cint}, Cint),
                   handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
end

@checked function cusolverDnCpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info,
                                          batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnCpotrsBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                    CuPtr{Ptr{cuComplex}}, Cint, CuPtr{Ptr{cuComplex}}, Cint, CuPtr{Cint},
                    Cint),
                   handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
end

@checked function cusolverDnZpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info,
                                          batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnZpotrsBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                    CuPtr{Ptr{cuDoubleComplex}}, Cint, CuPtr{Ptr{cuDoubleComplex}}, Cint,
                    CuPtr{Cint}, Cint),
                   handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
end

@checked function cusolverDnSpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSpotri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    Ptr{Cint}),
                   handle, uplo, n, A, lda, lwork)
end

@checked function cusolverDnDpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDpotri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    Ptr{Cint}),
                   handle, uplo, n, A, lda, lwork)
end

@checked function cusolverDnCpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnCpotri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    Ptr{Cint}),
                   handle, uplo, n, A, lda, lwork)
end

@checked function cusolverDnZpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZpotri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, Ptr{Cint}),
                   handle, uplo, n, A, lda, lwork)
end

@checked function cusolverDnSpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnSpotri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnDpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnDpotri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnCpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnCpotri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnZpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnZpotri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnStrtri_bufferSize(handle, uplo, diag, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnStrtri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                    CuPtr{Cfloat}, Cint, Ptr{Cint}),
                   handle, uplo, diag, n, A, lda, lwork)
end

@checked function cusolverDnDtrtri_bufferSize(handle, uplo, diag, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDtrtri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                    CuPtr{Cdouble}, Cint, Ptr{Cint}),
                   handle, uplo, diag, n, A, lda, lwork)
end

@checked function cusolverDnCtrtri_bufferSize(handle, uplo, diag, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnCtrtri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                    CuPtr{cuComplex}, Cint, Ptr{Cint}),
                   handle, uplo, diag, n, A, lda, lwork)
end

@checked function cusolverDnZtrtri_bufferSize(handle, uplo, diag, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZtrtri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                    CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                   handle, uplo, diag, n, A, lda, lwork)
end

@checked function cusolverDnStrtri(handle, uplo, diag, n, A, lda, work, lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnStrtri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, uplo, diag, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnDtrtri(handle, uplo, diag, n, A, lda, work, lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnDtrtri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, uplo, diag, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnCtrtri(handle, uplo, diag, n, A, lda, work, lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnCtrtri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, diag, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnZtrtri(handle, uplo, diag, n, A, lda, work, lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnZtrtri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, diag, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnSlauum_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSlauum_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    Ptr{Cint}),
                   handle, uplo, n, A, lda, lwork)
end

@checked function cusolverDnDlauum_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDlauum_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    Ptr{Cint}),
                   handle, uplo, n, A, lda, lwork)
end

@checked function cusolverDnClauum_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnClauum_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    Ptr{Cint}),
                   handle, uplo, n, A, lda, lwork)
end

@checked function cusolverDnZlauum_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZlauum_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, Ptr{Cint}),
                   handle, uplo, n, A, lda, lwork)
end

@checked function cusolverDnSlauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnSlauum, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnDlauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnDlauum, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnClauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnClauum, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnZlauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnZlauum, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSgetrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, Ptr{Cint}),
                   handle, m, n, A, lda, Lwork)
end

@checked function cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDgetrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, Ptr{Cint}),
                   handle, m, n, A, lda, Lwork)
end

@checked function cusolverDnCgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnCgetrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint, Ptr{Cint}),
                   handle, m, n, A, lda, Lwork)
end

@checked function cusolverDnZgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZgetrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                    Ptr{Cint}),
                   handle, m, n, A, lda, Lwork)
end

@checked function cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnSgetrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                    CuPtr{Cint}, CuPtr{Cint}),
                   handle, m, n, A, lda, Workspace, devIpiv, devInfo)
end

@checked function cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnDgetrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                    CuPtr{Cint}, CuPtr{Cint}),
                   handle, m, n, A, lda, Workspace, devIpiv, devInfo)
end

@checked function cusolverDnCgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnCgetrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}),
                   handle, m, n, A, lda, Workspace, devIpiv, devInfo)
end

@checked function cusolverDnZgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnZgetrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}),
                   handle, m, n, A, lda, Workspace, devIpiv, devInfo)
end

@checked function cusolverDnSlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    initialize_api()
    @runtime_ccall((:cusolverDnSlaswp, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, CuPtr{Cfloat}, Cint, Cint, Cint,
                    CuPtr{Cint}, Cint),
                   handle, n, A, lda, k1, k2, devIpiv, incx)
end

@checked function cusolverDnDlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    initialize_api()
    @runtime_ccall((:cusolverDnDlaswp, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, CuPtr{Cdouble}, Cint, Cint, Cint,
                    CuPtr{Cint}, Cint),
                   handle, n, A, lda, k1, k2, devIpiv, incx)
end

@checked function cusolverDnClaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    initialize_api()
    @runtime_ccall((:cusolverDnClaswp, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, CuPtr{cuComplex}, Cint, Cint, Cint,
                    CuPtr{Cint}, Cint),
                   handle, n, A, lda, k1, k2, devIpiv, incx)
end

@checked function cusolverDnZlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    initialize_api()
    @runtime_ccall((:cusolverDnZlaswp, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, CuPtr{cuDoubleComplex}, Cint, Cint, Cint,
                    CuPtr{Cint}, Cint),
                   handle, n, A, lda, k1, k2, devIpiv, incx)
end

@checked function cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnSgetrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasOperation_t, Cint, Cint, CuPtr{Cfloat},
                    Cint, CuPtr{Cint}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
end

@checked function cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnDgetrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasOperation_t, Cint, Cint, CuPtr{Cdouble},
                    Cint, CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
end

@checked function cusolverDnCgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnCgetrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasOperation_t, Cint, Cint, CuPtr{cuComplex},
                    Cint, CuPtr{Cint}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
end

@checked function cusolverDnZgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnZgetrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasOperation_t, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cint}),
                   handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
end

@checked function cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSgeqrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, Ptr{Cint}),
                   handle, m, n, A, lda, lwork)
end

@checked function cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDgeqrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, Ptr{Cint}),
                   handle, m, n, A, lda, lwork)
end

@checked function cusolverDnCgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnCgeqrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint, Ptr{Cint}),
                   handle, m, n, A, lda, lwork)
end

@checked function cusolverDnZgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZgeqrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                    Ptr{Cint}),
                   handle, m, n, A, lda, lwork)
end

@checked function cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnSgeqrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                    CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
end

@checked function cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnDgeqrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                    CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
end

@checked function cusolverDnCgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnCgeqrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
end

@checked function cusolverDnZgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnZgeqrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
end

@checked function cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSorgqr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Ptr{Cint}),
                   handle, m, n, k, A, lda, tau, lwork)
end

@checked function cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDorgqr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Ptr{Cint}),
                   handle, m, n, k, A, lda, tau, lwork)
end

@checked function cusolverDnCungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnCungqr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Ptr{Cint}),
                   handle, m, n, k, A, lda, tau, lwork)
end

@checked function cusolverDnZungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZungqr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, Ptr{Cint}),
                   handle, m, n, k, A, lda, tau, lwork)
end

@checked function cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnSorgqr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, m, n, k, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnDorgqr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, m, n, k, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnCungqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnCungqr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, m, n, k, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnZungqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnZungqr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, m, n, k, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C,
                                              ldc, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSormqr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                    Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                    Ptr{Cint}),
                   handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
end

@checked function cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C,
                                              ldc, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDormqr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                    Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                    Ptr{Cint}),
                   handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
end

@checked function cusolverDnCunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C,
                                              ldc, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnCunmqr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                    Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, CuPtr{cuComplex}, Cint,
                    Ptr{Cint}),
                   handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
end

@checked function cusolverDnZunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C,
                                              ldc, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZunmqr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                    Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex},
                    CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                   handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
end

@checked function cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work,
                                   lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnSormqr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                    Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
end

@checked function cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work,
                                   lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnDormqr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                    Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
end

@checked function cusolverDnCunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work,
                                   lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnCunmqr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                    Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
end

@checked function cusolverDnZunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work,
                                   lwork, devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnZunmqr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                    Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex},
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
end

@checked function cusolverDnSsytrf_bufferSize(handle, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSsytrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, CuPtr{Cfloat}, Cint, Ptr{Cint}),
                   handle, n, A, lda, lwork)
end

@checked function cusolverDnDsytrf_bufferSize(handle, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDsytrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, CuPtr{Cdouble}, Cint, Ptr{Cint}),
                   handle, n, A, lda, lwork)
end

@checked function cusolverDnCsytrf_bufferSize(handle, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnCsytrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, CuPtr{cuComplex}, Cint, Ptr{Cint}),
                   handle, n, A, lda, lwork)
end

@checked function cusolverDnZsytrf_bufferSize(handle, n, A, lda, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZsytrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                   handle, n, A, lda, lwork)
end

@checked function cusolverDnSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnSsytrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cint}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

@checked function cusolverDnDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnDsytrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

@checked function cusolverDnCsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnCsytrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

@checked function cusolverDnZsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnZsytrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

@checked function cusolverDnSsytrs_bufferSize(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb,
                                              lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSsytrs_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cint}, CuPtr{Cfloat}, Cint, Ptr{Cint}),
                   handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
end

@checked function cusolverDnDsytrs_bufferSize(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb,
                                              lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDsytrs_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cdouble},
                    Cint, CuPtr{Cint}, CuPtr{Cdouble}, Cint, Ptr{Cint}),
                   handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
end

@checked function cusolverDnCsytrs_bufferSize(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb,
                                              lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnCsytrs_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{cuComplex},
                    Cint, CuPtr{Cint}, CuPtr{cuComplex}, Cint, Ptr{Cint}),
                   handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
end

@checked function cusolverDnZsytrs_bufferSize(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb,
                                              lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZsytrs_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                    Cint, Ptr{Cint}),
                   handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
end

@checked function cusolverDnSsytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work,
                                   lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnSsytrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cint}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
end

@checked function cusolverDnDsytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work,
                                   lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnDsytrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cdouble},
                    Cint, CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cint}),
                   handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
end

@checked function cusolverDnCsytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work,
                                   lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnCsytrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{cuComplex},
                    Cint, CuPtr{Cint}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}),
                   handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
end

@checked function cusolverDnZsytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work,
                                   lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnZsytrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
end

@checked function cusolverDnSsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSsytri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cint}, Ptr{Cint}),
                   handle, uplo, n, A, lda, ipiv, lwork)
end

@checked function cusolverDnDsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDsytri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cint}, Ptr{Cint}),
                   handle, uplo, n, A, lda, ipiv, lwork)
end

@checked function cusolverDnCsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnCsytri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}, Ptr{Cint}),
                   handle, uplo, n, A, lda, ipiv, lwork)
end

@checked function cusolverDnZsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZsytri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cint}, Ptr{Cint}),
                   handle, uplo, n, A, lda, ipiv, lwork)
end

@checked function cusolverDnSsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnSsytri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cint}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

@checked function cusolverDnDsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnDsytri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

@checked function cusolverDnCsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnCsytri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

@checked function cusolverDnZsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnZsytri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

@checked function cusolverDnSgebrd_bufferSize(handle, m, n, Lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSgebrd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                   handle, m, n, Lwork)
end

@checked function cusolverDnDgebrd_bufferSize(handle, m, n, Lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDgebrd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                   handle, m, n, Lwork)
end

@checked function cusolverDnCgebrd_bufferSize(handle, m, n, Lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnCgebrd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                   handle, m, n, Lwork)
end

@checked function cusolverDnZgebrd_bufferSize(handle, m, n, Lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZgebrd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                   handle, m, n, Lwork)
end

@checked function cusolverDnSgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork,
                                   devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnSgebrd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                    CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                    CuPtr{Cint}),
                   handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
end

@checked function cusolverDnDgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork,
                                   devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnDgebrd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                    CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                    CuPtr{Cint}),
                   handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
end

@checked function cusolverDnCgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork,
                                   devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnCgebrd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint, CuPtr{Cfloat},
                    CuPtr{Cfloat}, CuPtr{cuComplex}, CuPtr{cuComplex}, CuPtr{cuComplex},
                    Cint, CuPtr{Cint}),
                   handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
end

@checked function cusolverDnZgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork,
                                   devInfo)
    initialize_api()
    @runtime_ccall((:cusolverDnZgebrd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
end

@checked function cusolverDnSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSorgbr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint, CuPtr{Cfloat},
                    Cint, CuPtr{Cfloat}, Ptr{Cint}),
                   handle, side, m, n, k, A, lda, tau, lwork)
end

@checked function cusolverDnDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDorgbr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Ptr{Cint}),
                   handle, side, m, n, k, A, lda, tau, lwork)
end

@checked function cusolverDnCungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnCungbr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Ptr{Cint}),
                   handle, side, m, n, k, A, lda, tau, lwork)
end

@checked function cusolverDnZungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZungbr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Ptr{Cint}),
                   handle, side, m, n, k, A, lda, tau, lwork)
end

@checked function cusolverDnSorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnSorgbr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint, CuPtr{Cfloat},
                    Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, side, m, n, k, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnDorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnDorgbr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, side, m, n, k, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnCungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnCungbr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}),
                   handle, side, m, n, k, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnZungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnZungbr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex},
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, side, m, n, k, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnSsytrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSsytrd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, Ptr{Cint}),
                   handle, uplo, n, A, lda, d, e, tau, lwork)
end

@checked function cusolverDnDsytrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDsytrd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, Ptr{Cint}),
                   handle, uplo, n, A, lda, d, e, tau, lwork)
end

@checked function cusolverDnChetrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnChetrd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{cuComplex}, Ptr{Cint}),
                   handle, uplo, n, A, lda, d, e, tau, lwork)
end

@checked function cusolverDnZhetrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZhetrd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Ptr{Cint}),
                   handle, uplo, n, A, lda, d, e, tau, lwork)
end

@checked function cusolverDnSsytrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnSsytrd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                    CuPtr{Cint}),
                   handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
end

@checked function cusolverDnDsytrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnDsytrd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                    CuPtr{Cint}),
                   handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
end

@checked function cusolverDnChetrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnChetrd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{cuComplex}, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}),
                   handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
end

@checked function cusolverDnZhetrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnZhetrd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
end

@checked function cusolverDnSorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSorgtr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Ptr{Cint}),
                   handle, uplo, n, A, lda, tau, lwork)
end

@checked function cusolverDnDorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDorgtr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Ptr{Cint}),
                   handle, uplo, n, A, lda, tau, lwork)
end

@checked function cusolverDnCungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnCungtr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Ptr{Cint}),
                   handle, uplo, n, A, lda, tau, lwork)
end

@checked function cusolverDnZungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZungtr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Ptr{Cint}),
                   handle, uplo, n, A, lda, tau, lwork)
end

@checked function cusolverDnSorgtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnSorgtr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnDorgtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnDorgtr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnCungtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnCungtr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnZungtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnZungtr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnSormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau,
                                              C, ldc, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSormtr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                    cublasOperation_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                    CuPtr{Cfloat}, Cint, Ptr{Cint}),
                   handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
end

@checked function cusolverDnDormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau,
                                              C, ldc, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDormtr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                    cublasOperation_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                    CuPtr{Cdouble}, Cint, Ptr{Cint}),
                   handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
end

@checked function cusolverDnCunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau,
                                              C, ldc, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnCunmtr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                    cublasOperation_t, Cint, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, Ptr{Cint}),
                   handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
end

@checked function cusolverDnZunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau,
                                              C, ldc, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZunmtr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                    cublasOperation_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                   handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
end

@checked function cusolverDnSormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                   work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnSormtr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                    cublasOperation_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
end

@checked function cusolverDnDormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                   work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnDormtr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                    cublasOperation_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
end

@checked function cusolverDnCunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                   work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnCunmtr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                    cublasOperation_t, Cint, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}),
                   handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
end

@checked function cusolverDnZunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                   work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnZunmtr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                    cublasOperation_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
end

@checked function cusolverDnSgesvd_bufferSize(handle, m, n, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSgesvd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                   handle, m, n, lwork)
end

@checked function cusolverDnDgesvd_bufferSize(handle, m, n, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDgesvd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                   handle, m, n, lwork)
end

@checked function cusolverDnCgesvd_bufferSize(handle, m, n, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnCgesvd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                   handle, m, n, lwork)
end

@checked function cusolverDnZgesvd_bufferSize(handle, m, n, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZgesvd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                   handle, m, n, lwork)
end

@checked function cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,
                                   work, lwork, rwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnSgesvd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, UInt8, UInt8, Cint, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                    Cint, CuPtr{Cfloat}, CuPtr{Cint}),
                   handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork,
                   rwork, info)
end

@checked function cusolverDnDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,
                                   work, lwork, rwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnDgesvd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, UInt8, UInt8, Cint, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cint}),
                   handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork,
                   rwork, info)
end

@checked function cusolverDnCgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,
                                   work, lwork, rwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnCgesvd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, UInt8, UInt8, Cint, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{Cint}),
                   handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork,
                   rwork, info)
end

@checked function cusolverDnZgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,
                                   work, lwork, rwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnZgesvd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, UInt8, UInt8, Cint, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{Cdouble}, CuPtr{Cint}),
                   handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork,
                   rwork, info)
end

@checked function cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSsyevd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Ptr{Cint}),
                   handle, jobz, uplo, n, A, lda, W, lwork)
end

@checked function cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDsyevd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Ptr{Cint}),
                   handle, jobz, uplo, n, A, lda, W, lwork)
end

@checked function cusolverDnCheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnCheevd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, Ptr{Cint}),
                   handle, jobz, uplo, n, A, lda, W, lwork)
end

@checked function cusolverDnZheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZheevd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}),
                   handle, jobz, uplo, n, A, lda, W, lwork)
end

@checked function cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnSsyevd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info)
end

@checked function cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnDsyevd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info)
end

@checked function cusolverDnCheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnCheevd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info)
end

@checked function cusolverDnZheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnZheevd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cint}),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info)
end

@checked function cusolverDnSsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl,
                                               vu, il, iu, meig, W, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSsyevdx_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                    cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, Cfloat, Cfloat, Cint,
                    Cint, Ptr{Cint}, CuPtr{Cfloat}, Ptr{Cint}),
                   handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
end

@checked function cusolverDnDsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl,
                                               vu, il, iu, meig, W, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDsyevdx_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                    cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, Cdouble, Cdouble, Cint,
                    Cint, Ptr{Cint}, CuPtr{Cdouble}, Ptr{Cint}),
                   handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
end

@checked function cusolverDnCheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl,
                                               vu, il, iu, meig, W, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnCheevdx_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                    cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, Cfloat, Cfloat, Cint,
                    Cint, Ptr{Cint}, CuPtr{Cfloat}, Ptr{Cint}),
                   handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
end

@checked function cusolverDnZheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl,
                                               vu, il, iu, meig, W, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZheevdx_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                    cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint, Cdouble, Cdouble,
                    Cint, Cint, Ptr{Cint}, CuPtr{Cdouble}, Ptr{Cint}),
                   handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
end

@checked function cusolverDnSsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                    meig, W, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnSsyevdx, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                    cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, Cfloat, Cfloat, Cint,
                    Cint, Ptr{Cint}, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work,
                   lwork, info)
end

@checked function cusolverDnDsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                    meig, W, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnDsyevdx, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                    cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, Cdouble, Cdouble, Cint,
                    Cint, Ptr{Cint}, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work,
                   lwork, info)
end

@checked function cusolverDnCheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                    meig, W, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnCheevdx, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                    cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, Cfloat, Cfloat, Cint,
                    Cint, Ptr{Cint}, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work,
                   lwork, info)
end

@checked function cusolverDnZheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                    meig, W, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnZheevdx, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                    cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint, Cdouble, Cdouble,
                    Cint, Cint, Ptr{Cint}, CuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{Cint}),
                   handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work,
                   lwork, info)
end

@checked function cusolverDnSsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda,
                                               B, ldb, vl, vu, il, iu, meig, W, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSsygvdx_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Cint, Cfloat, Cfloat, Cint, Cint, Ptr{Cint},
                    CuPtr{Cfloat}, Ptr{Cint}),
                   handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu,
                   meig, W, lwork)
end

@checked function cusolverDnDsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda,
                                               B, ldb, vl, vu, il, iu, meig, W, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDsygvdx_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, Cdouble, Cdouble, Cint, Cint, Ptr{Cint},
                    CuPtr{Cdouble}, Ptr{Cint}),
                   handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu,
                   meig, W, lwork)
end

@checked function cusolverDnChegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda,
                                               B, ldb, vl, vu, il, iu, meig, W, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnChegvdx_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, Cfloat, Cfloat, Cint, Cint, Ptr{Cint},
                    CuPtr{Cfloat}, Ptr{Cint}),
                   handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu,
                   meig, W, lwork)
end

@checked function cusolverDnZhegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda,
                                               B, ldb, vl, vu, il, iu, meig, W, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZhegvdx_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Cint, Cdouble, Cdouble, Cint, Cint,
                    Ptr{Cint}, CuPtr{Cdouble}, Ptr{Cint}),
                   handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu,
                   meig, W, lwork)
end

@checked function cusolverDnSsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb,
                                    vl, vu, il, iu, meig, W, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnSsygvdx, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Cint, Cfloat, Cfloat, Cint, Cint, Ptr{Cint},
                    CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu,
                   meig, W, work, lwork, info)
end

@checked function cusolverDnDsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb,
                                    vl, vu, il, iu, meig, W, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnDsygvdx, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, Cdouble, Cdouble, Cint, Cint, Ptr{Cint},
                    CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu,
                   meig, W, work, lwork, info)
end

@checked function cusolverDnChegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb,
                                    vl, vu, il, iu, meig, W, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnChegvdx, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, Cfloat, Cfloat, Cint, Cint, Ptr{Cint},
                    CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu,
                   meig, W, work, lwork, info)
end

@checked function cusolverDnZhegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb,
                                    vl, vu, il, iu, meig, W, work, lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnZhegvdx, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Cint, Cdouble, Cdouble, Cint, Cint,
                    Ptr{Cint}, CuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu,
                   meig, W, work, lwork, info)
end

@checked function cusolverDnSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnSsygvd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Ptr{Cint}),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
end

@checked function cusolverDnDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnDsygvd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Ptr{Cint}),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
end

@checked function cusolverDnChegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnChegvd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cfloat}, Ptr{Cint}),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
end

@checked function cusolverDnZhegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork)
    initialize_api()
    @runtime_ccall((:cusolverDnZhegvd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
end

@checked function cusolverDnSsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnSsygvd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
end

@checked function cusolverDnDsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnDsygvd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
end

@checked function cusolverDnChegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnChegvd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
end

@checked function cusolverDnZhegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info)
    initialize_api()
    @runtime_ccall((:cusolverDnZhegvd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cint}),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
end

@checked function cusolverDnCreateSyevjInfo(info)
    initialize_api()
    @runtime_ccall((:cusolverDnCreateSyevjInfo, libcusolver()), cusolverStatus_t,
                   (Ptr{syevjInfo_t},),
                   info)
end

@checked function cusolverDnDestroySyevjInfo(info)
    initialize_api()
    @runtime_ccall((:cusolverDnDestroySyevjInfo, libcusolver()), cusolverStatus_t,
                   (syevjInfo_t,),
                   info)
end

@checked function cusolverDnXsyevjSetTolerance(info, tolerance)
    initialize_api()
    @runtime_ccall((:cusolverDnXsyevjSetTolerance, libcusolver()), cusolverStatus_t,
                   (syevjInfo_t, Cdouble),
                   info, tolerance)
end

@checked function cusolverDnXsyevjSetMaxSweeps(info, max_sweeps)
    initialize_api()
    @runtime_ccall((:cusolverDnXsyevjSetMaxSweeps, libcusolver()), cusolverStatus_t,
                   (syevjInfo_t, Cint),
                   info, max_sweeps)
end

@checked function cusolverDnXsyevjSetSortEig(info, sort_eig)
    initialize_api()
    @runtime_ccall((:cusolverDnXsyevjSetSortEig, libcusolver()), cusolverStatus_t,
                   (syevjInfo_t, Cint),
                   info, sort_eig)
end

@checked function cusolverDnXsyevjGetResidual(handle, info, residual)
    initialize_api()
    @runtime_ccall((:cusolverDnXsyevjGetResidual, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, syevjInfo_t, Ptr{Cdouble}),
                   handle, info, residual)
end

@checked function cusolverDnXsyevjGetSweeps(handle, info, executed_sweeps)
    initialize_api()
    @runtime_ccall((:cusolverDnXsyevjGetSweeps, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, syevjInfo_t, Ptr{Cint}),
                   handle, info, executed_sweeps)
end

@checked function cusolverDnSsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W,
                                                     lwork, params, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnSsyevjBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t, Cint),
                   handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
end

@checked function cusolverDnDsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W,
                                                     lwork, params, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnDsyevjBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t, Cint),
                   handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
end

@checked function cusolverDnCheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W,
                                                     lwork, params, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnCheevjBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t, Cint),
                   handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
end

@checked function cusolverDnZheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W,
                                                     lwork, params, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnZheevjBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t,
                    Cint),
                   handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
end

@checked function cusolverDnSsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork,
                                          info, params, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnSsyevjBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint},
                    syevjInfo_t, Cint),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
end

@checked function cusolverDnDsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork,
                                          info, params, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnDsyevjBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                    CuPtr{Cint}, syevjInfo_t, Cint),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
end

@checked function cusolverDnCheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork,
                                          info, params, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnCheevjBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}, syevjInfo_t, Cint),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
end

@checked function cusolverDnZheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork,
                                          info, params, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnZheevjBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cint}, syevjInfo_t, Cint),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
end

@checked function cusolverDnSsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                              params)
    initialize_api()
    @runtime_ccall((:cusolverDnSsyevj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t),
                   handle, jobz, uplo, n, A, lda, W, lwork, params)
end

@checked function cusolverDnDsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                              params)
    initialize_api()
    @runtime_ccall((:cusolverDnDsyevj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t),
                   handle, jobz, uplo, n, A, lda, W, lwork, params)
end

@checked function cusolverDnCheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                              params)
    initialize_api()
    @runtime_ccall((:cusolverDnCheevj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t),
                   handle, jobz, uplo, n, A, lda, W, lwork, params)
end

@checked function cusolverDnZheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                              params)
    initialize_api()
    @runtime_ccall((:cusolverDnZheevj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t),
                   handle, jobz, uplo, n, A, lda, W, lwork, params)
end

@checked function cusolverDnSsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                   params)
    initialize_api()
    @runtime_ccall((:cusolverDnSsyevj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint},
                    syevjInfo_t),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
end

@checked function cusolverDnDsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                   params)
    initialize_api()
    @runtime_ccall((:cusolverDnDsyevj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                    CuPtr{Cint}, syevjInfo_t),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
end

@checked function cusolverDnCheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                   params)
    initialize_api()
    @runtime_ccall((:cusolverDnCheevj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}, syevjInfo_t),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
end

@checked function cusolverDnZheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                   params)
    initialize_api()
    @runtime_ccall((:cusolverDnZheevj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cint}, syevjInfo_t),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
end

@checked function cusolverDnSsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork, params)
    initialize_api()
    @runtime_ccall((:cusolverDnSsygvj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
end

@checked function cusolverDnDsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork, params)
    initialize_api()
    @runtime_ccall((:cusolverDnDsygvj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
end

@checked function cusolverDnChegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork, params)
    initialize_api()
    @runtime_ccall((:cusolverDnChegvj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
end

@checked function cusolverDnZhegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork, params)
    initialize_api()
    @runtime_ccall((:cusolverDnZhegvj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
end

@checked function cusolverDnSsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info, params)
    initialize_api()
    @runtime_ccall((:cusolverDnSsygvj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}, syevjInfo_t),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info,
                   params)
end

@checked function cusolverDnDsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info, params)
    initialize_api()
    @runtime_ccall((:cusolverDnDsygvj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}, syevjInfo_t),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info,
                   params)
end

@checked function cusolverDnChegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info, params)
    initialize_api()
    @runtime_ccall((:cusolverDnChegvj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{Cint}, syevjInfo_t),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info,
                   params)
end

@checked function cusolverDnZhegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info, params)
    initialize_api()
    @runtime_ccall((:cusolverDnZhegvj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cint}, syevjInfo_t),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info,
                   params)
end

@checked function cusolverDnCreateGesvdjInfo(info)
    initialize_api()
    @runtime_ccall((:cusolverDnCreateGesvdjInfo, libcusolver()), cusolverStatus_t,
                   (Ptr{gesvdjInfo_t},),
                   info)
end

@checked function cusolverDnDestroyGesvdjInfo(info)
    initialize_api()
    @runtime_ccall((:cusolverDnDestroyGesvdjInfo, libcusolver()), cusolverStatus_t,
                   (gesvdjInfo_t,),
                   info)
end

@checked function cusolverDnXgesvdjSetTolerance(info, tolerance)
    initialize_api()
    @runtime_ccall((:cusolverDnXgesvdjSetTolerance, libcusolver()), cusolverStatus_t,
                   (gesvdjInfo_t, Cdouble),
                   info, tolerance)
end

@checked function cusolverDnXgesvdjSetMaxSweeps(info, max_sweeps)
    initialize_api()
    @runtime_ccall((:cusolverDnXgesvdjSetMaxSweeps, libcusolver()), cusolverStatus_t,
                   (gesvdjInfo_t, Cint),
                   info, max_sweeps)
end

@checked function cusolverDnXgesvdjSetSortEig(info, sort_svd)
    initialize_api()
    @runtime_ccall((:cusolverDnXgesvdjSetSortEig, libcusolver()), cusolverStatus_t,
                   (gesvdjInfo_t, Cint),
                   info, sort_svd)
end

@checked function cusolverDnXgesvdjGetResidual(handle, info, residual)
    initialize_api()
    @runtime_ccall((:cusolverDnXgesvdjGetResidual, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, gesvdjInfo_t, Ptr{Cdouble}),
                   handle, info, residual)
end

@checked function cusolverDnXgesvdjGetSweeps(handle, info, executed_sweeps)
    initialize_api()
    @runtime_ccall((:cusolverDnXgesvdjGetSweeps, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, gesvdjInfo_t, Ptr{Cint}),
                   handle, info, executed_sweeps)
end

@checked function cusolverDnSgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U,
                                                      ldu, V, ldv, lwork, params, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnSgesvdjBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{Cfloat},
                    Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                    Ptr{Cint}, gesvdjInfo_t, Cint),
                   handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
end

@checked function cusolverDnDgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U,
                                                      ldu, V, ldv, lwork, params, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnDgesvdjBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{Cdouble},
                    Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                    Ptr{Cint}, gesvdjInfo_t, Cint),
                   handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
end

@checked function cusolverDnCgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U,
                                                      ldu, V, ldv, lwork, params, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnCgesvdjBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{cuComplex},
                    Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                    Ptr{Cint}, gesvdjInfo_t, Cint),
                   handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
end

@checked function cusolverDnZgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U,
                                                      ldu, V, ldv, lwork, params, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnZgesvdjBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}, gesvdjInfo_t, Cint),
                   handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
end

@checked function cusolverDnSgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                           work, lwork, info, params, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnSgesvdjBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{Cfloat},
                    Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cint}, gesvdjInfo_t, Cint),
                   handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                   params, batchSize)
end

@checked function cusolverDnDgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                           work, lwork, info, params, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnDgesvdjBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{Cdouble},
                    Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cint}, gesvdjInfo_t, Cint),
                   handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                   params, batchSize)
end

@checked function cusolverDnCgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                           work, lwork, info, params, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnCgesvdjBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{cuComplex},
                    Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cint}, gesvdjInfo_t, Cint),
                   handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                   params, batchSize)
end

@checked function cusolverDnZgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                           work, lwork, info, params, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnZgesvdjBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{Cint}, gesvdjInfo_t, Cint),
                   handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                   params, batchSize)
end

@checked function cusolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu,
                                               V, ldv, lwork, params)
    initialize_api()
    @runtime_ccall((:cusolverDnSgesvdj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                    Cint, Ptr{Cint}, gesvdjInfo_t),
                   handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
end

@checked function cusolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu,
                                               V, ldv, lwork, params)
    initialize_api()
    @runtime_ccall((:cusolverDnDgesvdj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, Ptr{Cint}, gesvdjInfo_t),
                   handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
end

@checked function cusolverDnCgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu,
                                               V, ldv, lwork, params)
    initialize_api()
    @runtime_ccall((:cusolverDnCgesvdj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, Ptr{Cint}, gesvdjInfo_t),
                   handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
end

@checked function cusolverDnZgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu,
                                               V, ldv, lwork, params)
    initialize_api()
    @runtime_ccall((:cusolverDnZgesvdj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}, gesvdjInfo_t),
                   handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
end

@checked function cusolverDnSgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                    work, lwork, info, params)
    initialize_api()
    @runtime_ccall((:cusolverDnSgesvdj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                    Cint, CuPtr{Cfloat}, Cint, CuPtr{Cint}, gesvdjInfo_t),
                   handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                   params)
end

@checked function cusolverDnDgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                    work, lwork, info, params)
    initialize_api()
    @runtime_ccall((:cusolverDnDgesvdj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cint}, gesvdjInfo_t),
                   handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                   params)
end

@checked function cusolverDnCgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                    work, lwork, info, params)
    initialize_api()
    @runtime_ccall((:cusolverDnCgesvdj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint, CuPtr{Cint},
                    gesvdjInfo_t),
                   handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                   params)
end

@checked function cusolverDnZgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                    work, lwork, info, params)
    initialize_api()
    @runtime_ccall((:cusolverDnZgesvdj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{Cint}, gesvdjInfo_t),
                   handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                   params)
end

@checked function cusolverDnSgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A,
                                                             lda, strideA, d_S, strideS,
                                                             d_U, ldu, strideU, d_V, ldv,
                                                             strideV, lwork, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnSgesvdaStridedBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{Cfloat}, Cint, Clonglong, CuPtr{Cfloat}, Clonglong,
                    CuPtr{Cfloat}, Cint, Clonglong, CuPtr{Cfloat}, Cint, Clonglong,
                    Ptr{Cint}, Cint),
                   handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu,
                   strideU, d_V, ldv, strideV, lwork, batchSize)
end

@checked function cusolverDnDgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A,
                                                             lda, strideA, d_S, strideS,
                                                             d_U, ldu, strideU, d_V, ldv,
                                                             strideV, lwork, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnDgesvdaStridedBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{Cdouble}, Cint, Clonglong, CuPtr{Cdouble}, Clonglong,
                    CuPtr{Cdouble}, Cint, Clonglong, CuPtr{Cdouble}, Cint, Clonglong,
                    Ptr{Cint}, Cint),
                   handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu,
                   strideU, d_V, ldv, strideV, lwork, batchSize)
end

@checked function cusolverDnCgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A,
                                                             lda, strideA, d_S, strideS,
                                                             d_U, ldu, strideU, d_V, ldv,
                                                             strideV, lwork, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnCgesvdaStridedBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{cuComplex}, Cint, Clonglong, CuPtr{Cfloat}, Clonglong,
                    CuPtr{cuComplex}, Cint, Clonglong, CuPtr{cuComplex}, Cint, Clonglong,
                    Ptr{Cint}, Cint),
                   handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu,
                   strideU, d_V, ldv, strideV, lwork, batchSize)
end

@checked function cusolverDnZgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A,
                                                             lda, strideA, d_S, strideS,
                                                             d_U, ldu, strideU, d_V, ldv,
                                                             strideV, lwork, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnZgesvdaStridedBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, Clonglong, CuPtr{Cdouble}, Clonglong,
                    CuPtr{cuDoubleComplex}, Cint, Clonglong, CuPtr{cuDoubleComplex}, Cint,
                    Clonglong, Ptr{Cint}, Cint),
                   handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu,
                   strideU, d_V, ldv, strideV, lwork, batchSize)
end

@checked function cusolverDnSgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda,
                                                  strideA, d_S, strideS, d_U, ldu, strideU,
                                                  d_V, ldv, strideV, d_work, lwork, d_info,
                                                  h_R_nrmF, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnSgesvdaStridedBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{Cfloat}, Cint, Clonglong, CuPtr{Cfloat}, Clonglong,
                    CuPtr{Cfloat}, Cint, Clonglong, CuPtr{Cfloat}, Cint, Clonglong,
                    CuPtr{Cfloat}, Cint, CuPtr{Cint}, Ptr{Cdouble}, Cint),
                   handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu,
                   strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)
end

@checked function cusolverDnDgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda,
                                                  strideA, d_S, strideS, d_U, ldu, strideU,
                                                  d_V, ldv, strideV, d_work, lwork, d_info,
                                                  h_R_nrmF, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnDgesvdaStridedBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{Cdouble}, Cint, Clonglong, CuPtr{Cdouble}, Clonglong,
                    CuPtr{Cdouble}, Cint, Clonglong, CuPtr{Cdouble}, Cint, Clonglong,
                    CuPtr{Cdouble}, Cint, CuPtr{Cint}, Ptr{Cdouble}, Cint),
                   handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu,
                   strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)
end

@checked function cusolverDnCgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda,
                                                  strideA, d_S, strideS, d_U, ldu, strideU,
                                                  d_V, ldv, strideV, d_work, lwork, d_info,
                                                  h_R_nrmF, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnCgesvdaStridedBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{cuComplex}, Cint, Clonglong, CuPtr{Cfloat}, Clonglong,
                    CuPtr{cuComplex}, Cint, Clonglong, CuPtr{cuComplex}, Cint, Clonglong,
                    CuPtr{cuComplex}, Cint, CuPtr{Cint}, Ptr{Cdouble}, Cint),
                   handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu,
                   strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)
end

@checked function cusolverDnZgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda,
                                                  strideA, d_S, strideS, d_U, ldu, strideU,
                                                  d_V, ldv, strideV, d_work, lwork, d_info,
                                                  h_R_nrmF, batchSize)
    initialize_api()
    @runtime_ccall((:cusolverDnZgesvdaStridedBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, Clonglong, CuPtr{Cdouble}, Clonglong,
                    CuPtr{cuDoubleComplex}, Cint, Clonglong, CuPtr{cuDoubleComplex}, Cint,
                    Clonglong, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}, Ptr{Cdouble},
                    Cint),
                   handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu,
                   strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)
end
# Julia wrapper for header: cusolverSp.h
# Automatically generated using Clang.jl


@checked function cusolverSpCreate(handle)
    initialize_api()
    @runtime_ccall((:cusolverSpCreate, libcusolver()), cusolverStatus_t,
                   (Ptr{cusolverSpHandle_t},),
                   handle)
end

@checked function cusolverSpDestroy(handle)
    initialize_api()
    @runtime_ccall((:cusolverSpDestroy, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t,),
                   handle)
end

@checked function cusolverSpSetStream(handle, streamId)
    initialize_api()
    @runtime_ccall((:cusolverSpSetStream, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, CUstream),
                   handle, streamId)
end

@checked function cusolverSpGetStream(handle, streamId)
    initialize_api()
    @runtime_ccall((:cusolverSpGetStream, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Ptr{CUstream}),
                   handle, streamId)
end

@checked function cusolverSpXcsrissymHost(handle, m, nnzA, descrA, csrRowPtrA, csrEndPtrA,
                                          csrColIndA, issym)
    initialize_api()
    @runtime_ccall((:cusolverSpXcsrissymHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                   handle, m, nnzA, descrA, csrRowPtrA, csrEndPtrA, csrColIndA, issym)
end

@checked function cusolverSpScsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpScsrlsvluHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cint}),
                   handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   reorder, x, singularity)
end

@checked function cusolverSpDcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpDcsrlsvluHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cdouble, Cint, Ptr{Cdouble},
                    Ptr{Cint}),
                   handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   reorder, x, singularity)
end

@checked function cusolverSpCcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpCcsrlsvluHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                    Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cfloat, Cint, Ptr{cuComplex},
                    Ptr{Cint}),
                   handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   reorder, x, singularity)
end

@checked function cusolverSpZcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpZcsrlsvluHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                    Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex},
                    Cdouble, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}),
                   handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   reorder, x, singularity)
end

@checked function cusolverSpScsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd,
                                      b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpScsrlsvqr, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                    CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cfloat}, Cfloat, Cint, CuPtr{Cfloat},
                    Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpDcsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd,
                                      b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpDcsrlsvqr, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                    CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cdouble, Cint,
                    CuPtr{Cdouble}, Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpCcsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd,
                                      b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpCcsrlsvqr, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                    CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuComplex}, Cfloat, Cint,
                    CuPtr{cuComplex}, Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpZcsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd,
                                      b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpZcsrlsvqr, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                    CuPtr{cuDoubleComplex}, Cdouble, Cint, CuPtr{cuDoubleComplex},
                    Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpScsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpScsrlsvqrHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cint}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   reorder, x, singularity)
end

@checked function cusolverSpDcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpDcsrlsvqrHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cdouble, Cint, Ptr{Cdouble},
                    Ptr{Cint}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   reorder, x, singularity)
end

@checked function cusolverSpCcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpCcsrlsvqrHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                    Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cfloat, Cint, Ptr{cuComplex},
                    Ptr{Cint}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   reorder, x, singularity)
end

@checked function cusolverSpZcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpZcsrlsvqrHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                    Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex},
                    Cdouble, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   reorder, x, singularity)
end

@checked function cusolverSpScsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                            csrColInd, b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpScsrlsvcholHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpDcsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                            csrColInd, b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpDcsrlsvcholHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cdouble, Cint, Ptr{Cdouble},
                    Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpCcsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                            csrColInd, b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpCcsrlsvcholHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                    Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cfloat, Cint, Ptr{cuComplex},
                    Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpZcsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                            csrColInd, b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpZcsrlsvcholHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                    Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex},
                    Cdouble, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpScsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                        csrColInd, b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpScsrlsvchol, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                    CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cfloat}, Cfloat, Cint, CuPtr{Cfloat},
                    Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpDcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                        csrColInd, b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpDcsrlsvchol, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                    CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cdouble, Cint,
                    CuPtr{Cdouble}, Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpCcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                        csrColInd, b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpCcsrlsvchol, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                    CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuComplex}, Cfloat, Cint,
                    CuPtr{cuComplex}, Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpZcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                        csrColInd, b, tol, reorder, x, singularity)
    initialize_api()
    @runtime_ccall((:cusolverSpZcsrlsvchol, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                    CuPtr{cuDoubleComplex}, Cdouble, Cint, CuPtr{cuDoubleComplex},
                    Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpScsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, b, tol, rankA, x, p, min_norm)
    initialize_api()
    @runtime_ccall((:cusolverSpScsrlsqvqrHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cfloat, Ptr{Cint}, Ptr{Cfloat},
                    Ptr{Cint}, Ptr{Cfloat}),
                   handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   rankA, x, p, min_norm)
end

@checked function cusolverSpDcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, b, tol, rankA, x, p, min_norm)
    initialize_api()
    @runtime_ccall((:cusolverSpDcsrlsqvqrHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cdouble, Ptr{Cint},
                    Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}),
                   handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   rankA, x, p, min_norm)
end

@checked function cusolverSpCcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, b, tol, rankA, x, p, min_norm)
    initialize_api()
    @runtime_ccall((:cusolverSpCcsrlsqvqrHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cfloat,
                    Ptr{Cint}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cfloat}),
                   handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   rankA, x, p, min_norm)
end

@checked function cusolverSpZcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, b, tol, rankA, x, p, min_norm)
    initialize_api()
    @runtime_ccall((:cusolverSpZcsrlsqvqrHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex},
                    Cdouble, Ptr{Cint}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cdouble}),
                   handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   rankA, x, p, min_norm)
end

@checked function cusolverSpScsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, mu0, x0, maxite, tol, mu, x)
    initialize_api()
    @runtime_ccall((:cusolverSpScsreigvsiHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                    Ptr{Cint}, Ptr{Cint}, Cfloat, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat},
                    Ptr{Cfloat}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0,
                   maxite, tol, mu, x)
end

@checked function cusolverSpDcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, mu0, x0, maxite, tol, mu, x)
    initialize_api()
    @runtime_ccall((:cusolverSpDcsreigvsiHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                    Ptr{Cint}, Ptr{Cint}, Cdouble, Ptr{Cdouble}, Cint, Cdouble,
                    Ptr{Cdouble}, Ptr{Cdouble}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0,
                   maxite, tol, mu, x)
end

@checked function cusolverSpCcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, mu0, x0, maxite, tol, mu, x)
    initialize_api()
    @runtime_ccall((:cusolverSpCcsreigvsiHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                    Ptr{Cint}, Ptr{Cint}, cuComplex, Ptr{cuComplex}, Cint, Cfloat,
                    Ptr{cuComplex}, Ptr{cuComplex}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0,
                   maxite, tol, mu, x)
end

@checked function cusolverSpZcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, mu0, x0, maxite, tol, mu, x)
    initialize_api()
    @runtime_ccall((:cusolverSpZcsreigvsiHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                    Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, cuDoubleComplex,
                    Ptr{cuDoubleComplex}, Cint, Cdouble, Ptr{cuDoubleComplex},
                    Ptr{cuDoubleComplex}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0,
                   maxite, tol, mu, x)
end

@checked function cusolverSpScsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                       csrColIndA, mu0, x0, maxite, eps, mu, x)
    initialize_api()
    @runtime_ccall((:cusolverSpScsreigvsi, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                    CuPtr{Cint}, CuPtr{Cint}, Cfloat, CuPtr{Cfloat}, Cint, Cfloat,
                    CuPtr{Cfloat}, CuPtr{Cfloat}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0,
                   maxite, eps, mu, x)
end

@checked function cusolverSpDcsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                       csrColIndA, mu0, x0, maxite, eps, mu, x)
    initialize_api()
    @runtime_ccall((:cusolverSpDcsreigvsi, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                    CuPtr{Cint}, CuPtr{Cint}, Cdouble, CuPtr{Cdouble}, Cint, Cdouble,
                    CuPtr{Cdouble}, CuPtr{Cdouble}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0,
                   maxite, eps, mu, x)
end

@checked function cusolverSpCcsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                       csrColIndA, mu0, x0, maxite, eps, mu, x)
    initialize_api()
    @runtime_ccall((:cusolverSpCcsreigvsi, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                    CuPtr{Cint}, CuPtr{Cint}, cuComplex, CuPtr{cuComplex}, Cint, Cfloat,
                    CuPtr{cuComplex}, CuPtr{cuComplex}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0,
                   maxite, eps, mu, x)
end

@checked function cusolverSpZcsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                       csrColIndA, mu0, x0, maxite, eps, mu, x)
    initialize_api()
    @runtime_ccall((:cusolverSpZcsreigvsi, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, cuDoubleComplex,
                    CuPtr{cuDoubleComplex}, Cint, Cdouble, CuPtr{cuDoubleComplex},
                    CuPtr{cuDoubleComplex}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0,
                   maxite, eps, mu, x)
end

@checked function cusolverSpScsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                         csrColIndA, left_bottom_corner,
                                         right_upper_corner, num_eigs)
    initialize_api()
    @runtime_ccall((:cusolverSpScsreigsHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                    Ptr{Cint}, Ptr{Cint}, cuComplex, cuComplex, Ptr{Cint}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                   left_bottom_corner, right_upper_corner, num_eigs)
end

@checked function cusolverSpDcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                         csrColIndA, left_bottom_corner,
                                         right_upper_corner, num_eigs)
    initialize_api()
    @runtime_ccall((:cusolverSpDcsreigsHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                    Ptr{Cint}, Ptr{Cint}, cuDoubleComplex, cuDoubleComplex, Ptr{Cint}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                   left_bottom_corner, right_upper_corner, num_eigs)
end

@checked function cusolverSpCcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                         csrColIndA, left_bottom_corner,
                                         right_upper_corner, num_eigs)
    initialize_api()
    @runtime_ccall((:cusolverSpCcsreigsHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                    Ptr{Cint}, Ptr{Cint}, cuComplex, cuComplex, Ptr{Cint}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                   left_bottom_corner, right_upper_corner, num_eigs)
end

@checked function cusolverSpZcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                         csrColIndA, left_bottom_corner,
                                         right_upper_corner, num_eigs)
    initialize_api()
    @runtime_ccall((:cusolverSpZcsreigsHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                    Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, cuDoubleComplex,
                    cuDoubleComplex, Ptr{Cint}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                   left_bottom_corner, right_upper_corner, num_eigs)
end

@checked function cusolverSpXcsrsymrcmHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA,
                                           p)
    initialize_api()
    @runtime_ccall((:cusolverSpXcsrsymrcmHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}),
                   handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
end

@checked function cusolverSpXcsrsymmdqHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA,
                                           p)
    initialize_api()
    @runtime_ccall((:cusolverSpXcsrsymmdqHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}),
                   handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
end

@checked function cusolverSpXcsrsymamdHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA,
                                           p)
    initialize_api()
    @runtime_ccall((:cusolverSpXcsrsymamdHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}),
                   handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
end

@checked function cusolverSpXcsrmetisndHost(handle, n, nnzA, descrA, csrRowPtrA,
                                            csrColIndA, options, p)
    initialize_api()
    @runtime_ccall((:cusolverSpXcsrmetisndHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                    Ptr{Cint}, Ptr{Int64}, Ptr{Cint}),
                   handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, options, p)
end

@checked function cusolverSpScsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA,
                                        csrColIndA, P, numnz)
    initialize_api()
    @runtime_ccall((:cusolverSpScsrzfdHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                   handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
end

@checked function cusolverSpDcsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA,
                                        csrColIndA, P, numnz)
    initialize_api()
    @runtime_ccall((:cusolverSpDcsrzfdHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                    CuPtr{Cint}, CuPtr{Cint}, Ptr{Cint}, Ptr{Cint}),
                   handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
end

@checked function cusolverSpCcsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA,
                                        csrColIndA, P, numnz)
    initialize_api()
    @runtime_ccall((:cusolverSpCcsrzfdHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                   handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
end

@checked function cusolverSpZcsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA,
                                        csrColIndA, P, numnz)
    initialize_api()
    @runtime_ccall((:cusolverSpZcsrzfdHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                    Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                   handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
end

@checked function cusolverSpXcsrperm_bufferSizeHost(handle, m, n, nnzA, descrA, csrRowPtrA,
                                                    csrColIndA, p, q, bufferSizeInBytes)
    initialize_api()
    @runtime_ccall((:cusolverSpXcsrperm_bufferSizeHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Csize_t}),
                   handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q,
                   bufferSizeInBytes)
end

@checked function cusolverSpXcsrpermHost(handle, m, n, nnzA, descrA, csrRowPtrA,
                                         csrColIndA, p, q, map, pBuffer)
    initialize_api()
    @runtime_ccall((:cusolverSpXcsrpermHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}),
                   handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q, map, pBuffer)
end

@checked function cusolverSpCreateCsrqrInfo(info)
    initialize_api()
    @runtime_ccall((:cusolverSpCreateCsrqrInfo, libcusolver()), cusolverStatus_t,
                   (Ptr{csrqrInfo_t},),
                   info)
end

@checked function cusolverSpDestroyCsrqrInfo(info)
    initialize_api()
    @runtime_ccall((:cusolverSpDestroyCsrqrInfo, libcusolver()), cusolverStatus_t,
                   (csrqrInfo_t,),
                   info)
end

@checked function cusolverSpXcsrqrAnalysisBatched(handle, m, n, nnzA, descrA, csrRowPtrA,
                                                  csrColIndA, info)
    initialize_api()
    @runtime_ccall((:cusolverSpXcsrqrAnalysisBatched, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cint},
                    CuPtr{Cint}, csrqrInfo_t),
                   handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
end

@checked function cusolverSpScsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal,
                                                    csrRowPtr, csrColInd, batchSize, info,
                                                    internalDataInBytes, workspaceInBytes)
    initialize_api()
    @runtime_ccall((:cusolverSpScsrqrBufferInfoBatched, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, csrqrInfo_t,
                    Ptr{Csize_t}, Ptr{Csize_t}),
                   handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize,
                   info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpDcsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal,
                                                    csrRowPtr, csrColInd, batchSize, info,
                                                    internalDataInBytes, workspaceInBytes)
    initialize_api()
    @runtime_ccall((:cusolverSpDcsrqrBufferInfoBatched, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, csrqrInfo_t,
                    Ptr{Csize_t}, Ptr{Csize_t}),
                   handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize,
                   info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpCcsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal,
                                                    csrRowPtr, csrColInd, batchSize, info,
                                                    internalDataInBytes, workspaceInBytes)
    initialize_api()
    @runtime_ccall((:cusolverSpCcsrqrBufferInfoBatched, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, csrqrInfo_t,
                    Ptr{Csize_t}, Ptr{Csize_t}),
                   handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize,
                   info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpZcsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal,
                                                    csrRowPtr, csrColInd, batchSize, info,
                                                    internalDataInBytes, workspaceInBytes)
    initialize_api()
    @runtime_ccall((:cusolverSpZcsrqrBufferInfoBatched, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, csrqrInfo_t,
                    Ptr{Csize_t}, Ptr{Csize_t}),
                   handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize,
                   info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpScsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                            csrColIndA, b, x, batchSize, info, pBuffer)
    initialize_api()
    @runtime_ccall((:cusolverSpScsrqrsvBatched, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cfloat}, CuPtr{Cfloat},
                    Cint, csrqrInfo_t, CuPtr{Cvoid}),
                   handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x,
                   batchSize, info, pBuffer)
end

@checked function cusolverSpDcsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                            csrColIndA, b, x, batchSize, info, pBuffer)
    initialize_api()
    @runtime_ccall((:cusolverSpDcsrqrsvBatched, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble},
                    CuPtr{Cdouble}, Cint, csrqrInfo_t, CuPtr{Cvoid}),
                   handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x,
                   batchSize, info, pBuffer)
end

@checked function cusolverSpCcsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                            csrColIndA, b, x, batchSize, info, pBuffer)
    initialize_api()
    @runtime_ccall((:cusolverSpCcsrqrsvBatched, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuComplex},
                    CuPtr{cuComplex}, Cint, csrqrInfo_t, CuPtr{Cvoid}),
                   handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x,
                   batchSize, info, pBuffer)
end

@checked function cusolverSpZcsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                            csrColIndA, b, x, batchSize, info, pBuffer)
    initialize_api()
    @runtime_ccall((:cusolverSpZcsrqrsvBatched, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                    CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, csrqrInfo_t,
                    CuPtr{Cvoid}),
                   handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x,
                   batchSize, info, pBuffer)
end
