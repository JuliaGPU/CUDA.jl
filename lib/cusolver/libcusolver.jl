# Julia wrapper for header: cusolverDn.h
# Automatically generated using Clang.jl

@checked function cusolverGetProperty(type, value)
    ccall((:cusolverGetProperty, libcusolver()), cusolverStatus_t,
                   (libraryPropertyType, Ptr{Cint}),
                   type, value)
end

@checked function cusolverGetVersion(version)
    ccall((:cusolverGetVersion, libcusolver()), cusolverStatus_t,
                   (Ptr{Cint},),
                   version)
end

@checked function cusolverDnCreate(handle)
    initialize_context()
    ccall((:cusolverDnCreate, libcusolver()), cusolverStatus_t,
                   (Ptr{cusolverDnHandle_t},),
                   handle)
end

@checked function cusolverDnDestroy(handle)
    initialize_context()
    ccall((:cusolverDnDestroy, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t,),
                   handle)
end

@checked function cusolverDnSetStream(handle, streamId)
    initialize_context()
    ccall((:cusolverDnSetStream, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, CUstream),
                   handle, streamId)
end

@checked function cusolverDnGetStream(handle, streamId)
    initialize_context()
    ccall((:cusolverDnGetStream, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Ptr{CUstream}),
                   handle, streamId)
end

@checked function cusolverDnIRSParamsCreate(params_ptr)
    initialize_context()
    ccall((:cusolverDnIRSParamsCreate, libcusolver()), cusolverStatus_t,
                   (Ptr{cusolverDnIRSParams_t},),
                   params_ptr)
end

@checked function cusolverDnIRSParamsDestroy(params)
    initialize_context()
    ccall((:cusolverDnIRSParamsDestroy, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t,),
                   params)
end

@checked function cusolverDnIRSParamsSetRefinementSolver(params, refinement_solver)
    initialize_context()
    ccall((:cusolverDnIRSParamsSetRefinementSolver, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cusolverIRSRefinement_t),
                   params, refinement_solver)
end

@checked function cusolverDnIRSParamsSetSolverMainPrecision(params, solver_main_precision)
    initialize_context()
    ccall((:cusolverDnIRSParamsSetSolverMainPrecision, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cusolverPrecType_t),
                   params, solver_main_precision)
end

@checked function cusolverDnIRSParamsSetSolverLowestPrecision(params,
                                                              solver_lowest_precision)
    initialize_context()
    ccall((:cusolverDnIRSParamsSetSolverLowestPrecision, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cusolverPrecType_t),
                   params, solver_lowest_precision)
end

@checked function cusolverDnIRSParamsSetSolverPrecisions(params, solver_main_precision,
                                                         solver_lowest_precision)
    initialize_context()
    ccall((:cusolverDnIRSParamsSetSolverPrecisions, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cusolverPrecType_t, cusolverPrecType_t),
                   params, solver_main_precision, solver_lowest_precision)
end

@checked function cusolverDnIRSParamsSetTol(params, val)
    initialize_context()
    ccall((:cusolverDnIRSParamsSetTol, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, Cdouble),
                   params, val)
end

@checked function cusolverDnIRSParamsSetTolInner(params, val)
    initialize_context()
    ccall((:cusolverDnIRSParamsSetTolInner, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, Cdouble),
                   params, val)
end

@checked function cusolverDnIRSParamsSetMaxIters(params, maxiters)
    initialize_context()
    ccall((:cusolverDnIRSParamsSetMaxIters, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cusolver_int_t),
                   params, maxiters)
end

@checked function cusolverDnIRSParamsSetMaxItersInner(params, maxiters_inner)
    initialize_context()
    ccall((:cusolverDnIRSParamsSetMaxItersInner, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, cusolver_int_t),
                   params, maxiters_inner)
end

@checked function cusolverDnIRSParamsGetMaxIters(params, maxiters)
    initialize_context()
    ccall((:cusolverDnIRSParamsGetMaxIters, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t, Ptr{cusolver_int_t}),
                   params, maxiters)
end

@checked function cusolverDnIRSParamsEnableFallback(params)
    initialize_context()
    ccall((:cusolverDnIRSParamsEnableFallback, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t,),
                   params)
end

@checked function cusolverDnIRSParamsDisableFallback(params)
    initialize_context()
    ccall((:cusolverDnIRSParamsDisableFallback, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSParams_t,),
                   params)
end

@checked function cusolverDnIRSInfosDestroy(infos)
    initialize_context()
    ccall((:cusolverDnIRSInfosDestroy, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSInfos_t,),
                   infos)
end

@checked function cusolverDnIRSInfosCreate(infos_ptr)
    initialize_context()
    ccall((:cusolverDnIRSInfosCreate, libcusolver()), cusolverStatus_t,
                   (Ptr{cusolverDnIRSInfos_t},),
                   infos_ptr)
end

@checked function cusolverDnIRSInfosGetNiters(infos, niters)
    initialize_context()
    ccall((:cusolverDnIRSInfosGetNiters, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSInfos_t, Ptr{cusolver_int_t}),
                   infos, niters)
end

@checked function cusolverDnIRSInfosGetOuterNiters(infos, outer_niters)
    initialize_context()
    ccall((:cusolverDnIRSInfosGetOuterNiters, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSInfos_t, Ptr{cusolver_int_t}),
                   infos, outer_niters)
end

@checked function cusolverDnIRSInfosRequestResidual(infos)
    initialize_context()
    ccall((:cusolverDnIRSInfosRequestResidual, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSInfos_t,),
                   infos)
end

@checked function cusolverDnIRSInfosGetResidualHistory(infos, residual_history)
    initialize_context()
    ccall((:cusolverDnIRSInfosGetResidualHistory, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSInfos_t, Ptr{Ptr{Cvoid}}),
                   infos, residual_history)
end

@checked function cusolverDnIRSInfosGetMaxIters(infos, maxiters)
    initialize_context()
    ccall((:cusolverDnIRSInfosGetMaxIters, libcusolver()), cusolverStatus_t,
                   (cusolverDnIRSInfos_t, Ptr{cusolver_int_t}),
                   infos, maxiters)
end

@checked function cusolverDnZZgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnZZgesv, libcusolver()), cusolverStatus_t,
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
    initialize_context()
    ccall((:cusolverDnZCgesv, libcusolver()), cusolverStatus_t,
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
    initialize_context()
    ccall((:cusolverDnZKgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cusolver_int_t},
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{Cvoid}, Csize_t, Ptr{cusolver_int_t},
                    CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnZEgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnZEgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cusolver_int_t},
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{Cvoid}, Csize_t, Ptr{cusolver_int_t},
                    CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnZYgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnZYgesv, libcusolver()), cusolverStatus_t,
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
    initialize_context()
    ccall((:cusolverDnCCgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Csize_t, Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnCEgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnCEgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Csize_t, Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnCKgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnCKgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Csize_t, Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnCYgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnCYgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Csize_t, Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnDDgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnDDgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cdouble},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnDSgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnDSgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cdouble},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnDHgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnDHgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cdouble},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnDBgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnDBgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cdouble},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnDXgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnDXgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cdouble},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnSSgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnSSgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cfloat},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnSHgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnSHgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cfloat},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnSBgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnSBgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cfloat},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnSXgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnSXgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cfloat},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnZZgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnZZgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cusolver_int_t},
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnZCgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnZCgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cusolver_int_t},
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnZKgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnZKgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cusolver_int_t},
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnZEgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnZEgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cusolver_int_t},
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnZYgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnZYgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cusolver_int_t},
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnCCgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnCCgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnCKgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnCKgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnCEgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnCEgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnCYgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnCYgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{cuComplex},
                    cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnDDgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnDDgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cdouble},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnDSgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnDSgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cdouble},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnDHgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnDHgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cdouble},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnDBgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnDBgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cdouble},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnDXgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnDXgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cdouble},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnSSgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnSSgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cfloat},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnSHgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnSHgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cfloat},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnSBgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnSBgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cfloat},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnSXgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnSXgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, CuPtr{Cfloat},
                    cusolver_int_t, CuPtr{cusolver_int_t}, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnZZgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnZZgels, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Csize_t, Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnZCgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnZCgels, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Csize_t, Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnZKgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnZKgels, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Csize_t, Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnZEgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnZEgels, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Csize_t, Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnZYgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnZYgels, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Csize_t, Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnCCgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnCCgels, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuComplex}, cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t,
                    CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnCKgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnCKgels, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuComplex}, cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t,
                    CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnCEgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnCEgels, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuComplex}, cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t,
                    CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnCYgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnCYgels, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuComplex}, cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t,
                    CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnDDgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnDDgels, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnDSgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnDSgels, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnDHgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnDHgels, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnDBgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnDBgels, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnDXgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnDXgels, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnSSgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnSSgels, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnSHgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnSHgels, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnSBgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnSBgels, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnSXgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    ccall((:cusolverDnSXgels, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Csize_t,
                    Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes, iter, d_info)
end

@checked function cusolverDnZZgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnZZgels_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Ptr{Csize_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnZCgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnZCgels_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Ptr{Csize_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnZKgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnZKgels_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Ptr{Csize_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnZEgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnZEgels_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Ptr{Csize_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnZYgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnZYgels_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{cuDoubleComplex},
                    cusolver_int_t, CuPtr{cuDoubleComplex}, cusolver_int_t, CuPtr{Cvoid},
                    Ptr{Csize_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnCCgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnCCgels_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuComplex}, cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t,
                    CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnCKgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnCKgels_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuComplex}, cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t,
                    CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnCEgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnCEgels_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuComplex}, cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t,
                    CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnCYgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnCYgels_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{cuComplex}, cusolver_int_t, CuPtr{cuComplex}, cusolver_int_t,
                    CuPtr{cuComplex}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnDDgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnDDgels_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnDSgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnDSgels_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnDHgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnDHgels_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnDBgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnDBgels_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnDXgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnDXgels_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cdouble}, cusolver_int_t,
                    CuPtr{Cdouble}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnSSgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnSSgels_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnSHgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnSHgels_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnSBgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnSBgels_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnSXgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnSXgels_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cfloat}, cusolver_int_t,
                    CuPtr{Cfloat}, cusolver_int_t, CuPtr{Cvoid}, Ptr{Csize_t}),
                   handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                   lwork_bytes)
end

@checked function cusolverDnIRSXgesv(handle, gesv_irs_params, gesv_irs_infos, n, nrhs, dA,
                                     ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes,
                                     niters, d_info)
    initialize_context()
    ccall((:cusolverDnIRSXgesv, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnIRSParams_t, cusolverDnIRSInfos_t,
                    cusolver_int_t, cusolver_int_t, CuPtr{Cvoid}, cusolver_int_t,
                    CuPtr{Cvoid}, cusolver_int_t, CuPtr{Cvoid}, cusolver_int_t,
                    CuPtr{Cvoid}, Csize_t, Ptr{cusolver_int_t}, CuPtr{cusolver_int_t}),
                   handle, gesv_irs_params, gesv_irs_infos, n, nrhs, dA, ldda, dB, lddb,
                   dX, lddx, dWorkspace, lwork_bytes, niters, d_info)
end

@checked function cusolverDnIRSXgesv_bufferSize(handle, params, n, nrhs, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnIRSXgesv_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnIRSParams_t, cusolver_int_t,
                    cusolver_int_t, Ptr{Csize_t}),
                   handle, params, n, nrhs, lwork_bytes)
end

@checked function cusolverDnIRSXgels(handle, gels_irs_params, gels_irs_infos, m, n, nrhs,
                                     dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes,
                                     niters, d_info)
    initialize_context()
    ccall((:cusolverDnIRSXgels, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnIRSParams_t, cusolverDnIRSInfos_t,
                    cusolver_int_t, cusolver_int_t, cusolver_int_t, CuPtr{Cvoid},
                    cusolver_int_t, CuPtr{Cvoid}, cusolver_int_t, CuPtr{Cvoid},
                    cusolver_int_t, CuPtr{Cvoid}, Csize_t, Ptr{cusolver_int_t},
                    CuPtr{cusolver_int_t}),
                   handle, gels_irs_params, gels_irs_infos, m, n, nrhs, dA, ldda, dB, lddb,
                   dX, lddx, dWorkspace, lwork_bytes, niters, d_info)
end

@checked function cusolverDnIRSXgels_bufferSize(handle, params, m, n, nrhs, lwork_bytes)
    initialize_context()
    ccall((:cusolverDnIRSXgels_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnIRSParams_t, cusolver_int_t,
                    cusolver_int_t, cusolver_int_t, Ptr{Csize_t}),
                   handle, params, m, n, nrhs, lwork_bytes)
end

@checked function cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    initialize_context()
    ccall((:cusolverDnSpotrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    Ptr{Cint}),
                   handle, uplo, n, A, lda, Lwork)
end

@checked function cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    initialize_context()
    ccall((:cusolverDnDpotrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    Ptr{Cint}),
                   handle, uplo, n, A, lda, Lwork)
end

@checked function cusolverDnCpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    initialize_context()
    ccall((:cusolverDnCpotrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    Ptr{Cint}),
                   handle, uplo, n, A, lda, Lwork)
end

@checked function cusolverDnZpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    initialize_context()
    ccall((:cusolverDnZpotrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, Ptr{Cint}),
                   handle, uplo, n, A, lda, Lwork)
end

@checked function cusolverDnSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnSpotrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
end

@checked function cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnDpotrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
end

@checked function cusolverDnCpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnCpotrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
end

@checked function cusolverDnZpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnZpotrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
end

@checked function cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    initialize_context()
    ccall((:cusolverDnSpotrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
end

@checked function cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    initialize_context()
    ccall((:cusolverDnDpotrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cdouble},
                    Cint, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
end

@checked function cusolverDnCpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    initialize_context()
    ccall((:cusolverDnCpotrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{cuComplex},
                    Cint, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
end

@checked function cusolverDnZpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    initialize_context()
    ccall((:cusolverDnZpotrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
end

@checked function cusolverDnSpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    initialize_context()
    ccall((:cusolverDnSpotrfBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Ptr{Cfloat}}, Cint,
                    CuPtr{Cint}, Cint),
                   handle, uplo, n, Aarray, lda, infoArray, batchSize)
end

@checked function cusolverDnDpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    initialize_context()
    ccall((:cusolverDnDpotrfBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Ptr{Cdouble}}, Cint,
                    CuPtr{Cint}, Cint),
                   handle, uplo, n, Aarray, lda, infoArray, batchSize)
end

@checked function cusolverDnCpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    initialize_context()
    ccall((:cusolverDnCpotrfBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Ptr{cuComplex}},
                    Cint, CuPtr{Cint}, Cint),
                   handle, uplo, n, Aarray, lda, infoArray, batchSize)
end

@checked function cusolverDnZpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    initialize_context()
    ccall((:cusolverDnZpotrfBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint,
                    CuPtr{Ptr{cuDoubleComplex}}, Cint, CuPtr{Cint}, Cint),
                   handle, uplo, n, Aarray, lda, infoArray, batchSize)
end

@checked function cusolverDnSpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info,
                                          batchSize)
    initialize_context()
    ccall((:cusolverDnSpotrsBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Ptr{Cfloat}},
                    Cint, CuPtr{Ptr{Cfloat}}, Cint, CuPtr{Cint}, Cint),
                   handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
end

@checked function cusolverDnDpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info,
                                          batchSize)
    initialize_context()
    ccall((:cusolverDnDpotrsBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Ptr{Cdouble}},
                    Cint, CuPtr{Ptr{Cdouble}}, Cint, CuPtr{Cint}, Cint),
                   handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
end

@checked function cusolverDnCpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info,
                                          batchSize)
    initialize_context()
    ccall((:cusolverDnCpotrsBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                    CuPtr{Ptr{cuComplex}}, Cint, CuPtr{Ptr{cuComplex}}, Cint, CuPtr{Cint},
                    Cint),
                   handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
end

@checked function cusolverDnZpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info,
                                          batchSize)
    initialize_context()
    ccall((:cusolverDnZpotrsBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                    CuPtr{Ptr{cuDoubleComplex}}, Cint, CuPtr{Ptr{cuDoubleComplex}}, Cint,
                    CuPtr{Cint}, Cint),
                   handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
end

@checked function cusolverDnSpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnSpotri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    Ptr{Cint}),
                   handle, uplo, n, A, lda, lwork)
end

@checked function cusolverDnDpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnDpotri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    Ptr{Cint}),
                   handle, uplo, n, A, lda, lwork)
end

@checked function cusolverDnCpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnCpotri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    Ptr{Cint}),
                   handle, uplo, n, A, lda, lwork)
end

@checked function cusolverDnZpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnZpotri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, Ptr{Cint}),
                   handle, uplo, n, A, lda, lwork)
end

@checked function cusolverDnSpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnSpotri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnDpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnDpotri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnCpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnCpotri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnZpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnZpotri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnStrtri_bufferSize(handle, uplo, diag, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnStrtri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                    CuPtr{Cfloat}, Cint, Ptr{Cint}),
                   handle, uplo, diag, n, A, lda, lwork)
end

@checked function cusolverDnDtrtri_bufferSize(handle, uplo, diag, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnDtrtri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                    CuPtr{Cdouble}, Cint, Ptr{Cint}),
                   handle, uplo, diag, n, A, lda, lwork)
end

@checked function cusolverDnCtrtri_bufferSize(handle, uplo, diag, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnCtrtri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                    CuPtr{cuComplex}, Cint, Ptr{Cint}),
                   handle, uplo, diag, n, A, lda, lwork)
end

@checked function cusolverDnZtrtri_bufferSize(handle, uplo, diag, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnZtrtri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                    CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                   handle, uplo, diag, n, A, lda, lwork)
end

@checked function cusolverDnStrtri(handle, uplo, diag, n, A, lda, work, lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnStrtri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, uplo, diag, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnDtrtri(handle, uplo, diag, n, A, lda, work, lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnDtrtri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, uplo, diag, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnCtrtri(handle, uplo, diag, n, A, lda, work, lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnCtrtri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, diag, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnZtrtri(handle, uplo, diag, n, A, lda, work, lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnZtrtri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, diag, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnSlauum_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnSlauum_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    Ptr{Cint}),
                   handle, uplo, n, A, lda, lwork)
end

@checked function cusolverDnDlauum_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnDlauum_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    Ptr{Cint}),
                   handle, uplo, n, A, lda, lwork)
end

@checked function cusolverDnClauum_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnClauum_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    Ptr{Cint}),
                   handle, uplo, n, A, lda, lwork)
end

@checked function cusolverDnZlauum_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnZlauum_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, Ptr{Cint}),
                   handle, uplo, n, A, lda, lwork)
end

@checked function cusolverDnSlauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnSlauum, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnDlauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnDlauum, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnClauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnClauum, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnZlauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnZlauum, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, work, lwork, devInfo)
end

@checked function cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    initialize_context()
    ccall((:cusolverDnSgetrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, Ptr{Cint}),
                   handle, m, n, A, lda, Lwork)
end

@checked function cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    initialize_context()
    ccall((:cusolverDnDgetrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, Ptr{Cint}),
                   handle, m, n, A, lda, Lwork)
end

@checked function cusolverDnCgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    initialize_context()
    ccall((:cusolverDnCgetrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint, Ptr{Cint}),
                   handle, m, n, A, lda, Lwork)
end

@checked function cusolverDnZgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    initialize_context()
    ccall((:cusolverDnZgetrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                    Ptr{Cint}),
                   handle, m, n, A, lda, Lwork)
end

@checked function cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    initialize_context()
    ccall((:cusolverDnSgetrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                    CuPtr{Cint}, CuPtr{Cint}),
                   handle, m, n, A, lda, Workspace, devIpiv, devInfo)
end

@checked function cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    initialize_context()
    ccall((:cusolverDnDgetrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                    CuPtr{Cint}, CuPtr{Cint}),
                   handle, m, n, A, lda, Workspace, devIpiv, devInfo)
end

@checked function cusolverDnCgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    initialize_context()
    ccall((:cusolverDnCgetrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}),
                   handle, m, n, A, lda, Workspace, devIpiv, devInfo)
end

@checked function cusolverDnZgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    initialize_context()
    ccall((:cusolverDnZgetrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}),
                   handle, m, n, A, lda, Workspace, devIpiv, devInfo)
end

@checked function cusolverDnSlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    initialize_context()
    ccall((:cusolverDnSlaswp, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, CuPtr{Cfloat}, Cint, Cint, Cint,
                    CuPtr{Cint}, Cint),
                   handle, n, A, lda, k1, k2, devIpiv, incx)
end

@checked function cusolverDnDlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    initialize_context()
    ccall((:cusolverDnDlaswp, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, CuPtr{Cdouble}, Cint, Cint, Cint,
                    CuPtr{Cint}, Cint),
                   handle, n, A, lda, k1, k2, devIpiv, incx)
end

@checked function cusolverDnClaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    initialize_context()
    ccall((:cusolverDnClaswp, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, CuPtr{cuComplex}, Cint, Cint, Cint,
                    CuPtr{Cint}, Cint),
                   handle, n, A, lda, k1, k2, devIpiv, incx)
end

@checked function cusolverDnZlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    initialize_context()
    ccall((:cusolverDnZlaswp, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, CuPtr{cuDoubleComplex}, Cint, Cint, Cint,
                    CuPtr{Cint}, Cint),
                   handle, n, A, lda, k1, k2, devIpiv, incx)
end

@checked function cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    initialize_context()
    ccall((:cusolverDnSgetrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasOperation_t, Cint, Cint, CuPtr{Cfloat},
                    Cint, CuPtr{Cint}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
end

@checked function cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    initialize_context()
    ccall((:cusolverDnDgetrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasOperation_t, Cint, Cint, CuPtr{Cdouble},
                    Cint, CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
end

@checked function cusolverDnCgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    initialize_context()
    ccall((:cusolverDnCgetrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasOperation_t, Cint, Cint, CuPtr{cuComplex},
                    Cint, CuPtr{Cint}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
end

@checked function cusolverDnZgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    initialize_context()
    ccall((:cusolverDnZgetrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasOperation_t, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cint}),
                   handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
end

@checked function cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnSgeqrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, Ptr{Cint}),
                   handle, m, n, A, lda, lwork)
end

@checked function cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnDgeqrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, Ptr{Cint}),
                   handle, m, n, A, lda, lwork)
end

@checked function cusolverDnCgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnCgeqrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint, Ptr{Cint}),
                   handle, m, n, A, lda, lwork)
end

@checked function cusolverDnZgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnZgeqrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                    Ptr{Cint}),
                   handle, m, n, A, lda, lwork)
end

@checked function cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnSgeqrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                    CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
end

@checked function cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnDgeqrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                    CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
end

@checked function cusolverDnCgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnCgeqrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
end

@checked function cusolverDnZgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnZgeqrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
end

@checked function cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    initialize_context()
    ccall((:cusolverDnSorgqr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Ptr{Cint}),
                   handle, m, n, k, A, lda, tau, lwork)
end

@checked function cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    initialize_context()
    ccall((:cusolverDnDorgqr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Ptr{Cint}),
                   handle, m, n, k, A, lda, tau, lwork)
end

@checked function cusolverDnCungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    initialize_context()
    ccall((:cusolverDnCungqr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Ptr{Cint}),
                   handle, m, n, k, A, lda, tau, lwork)
end

@checked function cusolverDnZungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    initialize_context()
    ccall((:cusolverDnZungqr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, Ptr{Cint}),
                   handle, m, n, k, A, lda, tau, lwork)
end

@checked function cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnSorgqr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, m, n, k, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnDorgqr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, m, n, k, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnCungqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnCungqr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, m, n, k, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnZungqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnZungqr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, m, n, k, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C,
                                              ldc, lwork)
    initialize_context()
    ccall((:cusolverDnSormqr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                    Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                    Ptr{Cint}),
                   handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
end

@checked function cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C,
                                              ldc, lwork)
    initialize_context()
    ccall((:cusolverDnDormqr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                    Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                    Ptr{Cint}),
                   handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
end

@checked function cusolverDnCunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C,
                                              ldc, lwork)
    initialize_context()
    ccall((:cusolverDnCunmqr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                    Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, CuPtr{cuComplex}, Cint,
                    Ptr{Cint}),
                   handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
end

@checked function cusolverDnZunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C,
                                              ldc, lwork)
    initialize_context()
    ccall((:cusolverDnZunmqr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                    Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex},
                    CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                   handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
end

@checked function cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work,
                                   lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnSormqr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                    Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
end

@checked function cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work,
                                   lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnDormqr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                    Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
end

@checked function cusolverDnCunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work,
                                   lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnCunmqr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                    Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
end

@checked function cusolverDnZunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work,
                                   lwork, devInfo)
    initialize_context()
    ccall((:cusolverDnZunmqr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                    Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex},
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
end

@checked function cusolverDnSsytrf_bufferSize(handle, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnSsytrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, CuPtr{Cfloat}, Cint, Ptr{Cint}),
                   handle, n, A, lda, lwork)
end

@checked function cusolverDnDsytrf_bufferSize(handle, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnDsytrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, CuPtr{Cdouble}, Cint, Ptr{Cint}),
                   handle, n, A, lda, lwork)
end

@checked function cusolverDnCsytrf_bufferSize(handle, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnCsytrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, CuPtr{cuComplex}, Cint, Ptr{Cint}),
                   handle, n, A, lda, lwork)
end

@checked function cusolverDnZsytrf_bufferSize(handle, n, A, lda, lwork)
    initialize_context()
    ccall((:cusolverDnZsytrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                   handle, n, A, lda, lwork)
end

@checked function cusolverDnSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnSsytrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cint}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

@checked function cusolverDnDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnDsytrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

@checked function cusolverDnCsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnCsytrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

@checked function cusolverDnZsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnZsytrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

@checked function cusolverDnSsytrs_bufferSize(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb,
                                              lwork)
    initialize_context()
    ccall((:cusolverDnSsytrs_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cint}, CuPtr{Cfloat}, Cint, Ptr{Cint}),
                   handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
end

@checked function cusolverDnDsytrs_bufferSize(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb,
                                              lwork)
    initialize_context()
    ccall((:cusolverDnDsytrs_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cdouble},
                    Cint, CuPtr{Cint}, CuPtr{Cdouble}, Cint, Ptr{Cint}),
                   handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
end

@checked function cusolverDnCsytrs_bufferSize(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb,
                                              lwork)
    initialize_context()
    ccall((:cusolverDnCsytrs_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{cuComplex},
                    Cint, CuPtr{Cint}, CuPtr{cuComplex}, Cint, Ptr{Cint}),
                   handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
end

@checked function cusolverDnZsytrs_bufferSize(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb,
                                              lwork)
    initialize_context()
    ccall((:cusolverDnZsytrs_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                    Cint, Ptr{Cint}),
                   handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
end

@checked function cusolverDnSsytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work,
                                   lwork, info)
    initialize_context()
    ccall((:cusolverDnSsytrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cint}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
end

@checked function cusolverDnDsytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work,
                                   lwork, info)
    initialize_context()
    ccall((:cusolverDnDsytrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cdouble},
                    Cint, CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cint}),
                   handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
end

@checked function cusolverDnCsytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work,
                                   lwork, info)
    initialize_context()
    ccall((:cusolverDnCsytrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{cuComplex},
                    Cint, CuPtr{Cint}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}),
                   handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
end

@checked function cusolverDnZsytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work,
                                   lwork, info)
    initialize_context()
    ccall((:cusolverDnZsytrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
end

@checked function cusolverDnSsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    initialize_context()
    ccall((:cusolverDnSsytri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cint}, Ptr{Cint}),
                   handle, uplo, n, A, lda, ipiv, lwork)
end

@checked function cusolverDnDsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    initialize_context()
    ccall((:cusolverDnDsytri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cint}, Ptr{Cint}),
                   handle, uplo, n, A, lda, ipiv, lwork)
end

@checked function cusolverDnCsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    initialize_context()
    ccall((:cusolverDnCsytri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}, Ptr{Cint}),
                   handle, uplo, n, A, lda, ipiv, lwork)
end

@checked function cusolverDnZsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    initialize_context()
    ccall((:cusolverDnZsytri_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cint}, Ptr{Cint}),
                   handle, uplo, n, A, lda, ipiv, lwork)
end

@checked function cusolverDnSsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnSsytri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cint}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

@checked function cusolverDnDsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnDsytri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

@checked function cusolverDnCsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnCsytri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

@checked function cusolverDnZsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnZsytri, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

@checked function cusolverDnSgebrd_bufferSize(handle, m, n, Lwork)
    initialize_context()
    ccall((:cusolverDnSgebrd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                   handle, m, n, Lwork)
end

@checked function cusolverDnDgebrd_bufferSize(handle, m, n, Lwork)
    initialize_context()
    ccall((:cusolverDnDgebrd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                   handle, m, n, Lwork)
end

@checked function cusolverDnCgebrd_bufferSize(handle, m, n, Lwork)
    initialize_context()
    ccall((:cusolverDnCgebrd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                   handle, m, n, Lwork)
end

@checked function cusolverDnZgebrd_bufferSize(handle, m, n, Lwork)
    initialize_context()
    ccall((:cusolverDnZgebrd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                   handle, m, n, Lwork)
end

@checked function cusolverDnSgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork,
                                   devInfo)
    initialize_context()
    ccall((:cusolverDnSgebrd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                    CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                    CuPtr{Cint}),
                   handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
end

@checked function cusolverDnDgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork,
                                   devInfo)
    initialize_context()
    ccall((:cusolverDnDgebrd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                    CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                    CuPtr{Cint}),
                   handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
end

@checked function cusolverDnCgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork,
                                   devInfo)
    initialize_context()
    ccall((:cusolverDnCgebrd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint, CuPtr{Cfloat},
                    CuPtr{Cfloat}, CuPtr{cuComplex}, CuPtr{cuComplex}, CuPtr{cuComplex},
                    Cint, CuPtr{Cint}),
                   handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
end

@checked function cusolverDnZgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork,
                                   devInfo)
    initialize_context()
    ccall((:cusolverDnZgebrd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
end

@checked function cusolverDnSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    initialize_context()
    ccall((:cusolverDnSorgbr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint, CuPtr{Cfloat},
                    Cint, CuPtr{Cfloat}, Ptr{Cint}),
                   handle, side, m, n, k, A, lda, tau, lwork)
end

@checked function cusolverDnDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    initialize_context()
    ccall((:cusolverDnDorgbr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Ptr{Cint}),
                   handle, side, m, n, k, A, lda, tau, lwork)
end

@checked function cusolverDnCungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    initialize_context()
    ccall((:cusolverDnCungbr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Ptr{Cint}),
                   handle, side, m, n, k, A, lda, tau, lwork)
end

@checked function cusolverDnZungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    initialize_context()
    ccall((:cusolverDnZungbr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Ptr{Cint}),
                   handle, side, m, n, k, A, lda, tau, lwork)
end

@checked function cusolverDnSorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnSorgbr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint, CuPtr{Cfloat},
                    Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, side, m, n, k, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnDorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnDorgbr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, side, m, n, k, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnCungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnCungbr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}),
                   handle, side, m, n, k, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnZungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnZungbr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex},
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, side, m, n, k, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnSsytrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    initialize_context()
    ccall((:cusolverDnSsytrd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, Ptr{Cint}),
                   handle, uplo, n, A, lda, d, e, tau, lwork)
end

@checked function cusolverDnDsytrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    initialize_context()
    ccall((:cusolverDnDsytrd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, Ptr{Cint}),
                   handle, uplo, n, A, lda, d, e, tau, lwork)
end

@checked function cusolverDnChetrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    initialize_context()
    ccall((:cusolverDnChetrd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{cuComplex}, Ptr{Cint}),
                   handle, uplo, n, A, lda, d, e, tau, lwork)
end

@checked function cusolverDnZhetrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    initialize_context()
    ccall((:cusolverDnZhetrd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Ptr{Cint}),
                   handle, uplo, n, A, lda, d, e, tau, lwork)
end

@checked function cusolverDnSsytrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnSsytrd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                    CuPtr{Cint}),
                   handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
end

@checked function cusolverDnDsytrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnDsytrd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                    CuPtr{Cint}),
                   handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
end

@checked function cusolverDnChetrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnChetrd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{cuComplex}, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}),
                   handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
end

@checked function cusolverDnZhetrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnZhetrd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
end

@checked function cusolverDnSorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    initialize_context()
    ccall((:cusolverDnSorgtr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Ptr{Cint}),
                   handle, uplo, n, A, lda, tau, lwork)
end

@checked function cusolverDnDorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    initialize_context()
    ccall((:cusolverDnDorgtr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Ptr{Cint}),
                   handle, uplo, n, A, lda, tau, lwork)
end

@checked function cusolverDnCungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    initialize_context()
    ccall((:cusolverDnCungtr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Ptr{Cint}),
                   handle, uplo, n, A, lda, tau, lwork)
end

@checked function cusolverDnZungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    initialize_context()
    ccall((:cusolverDnZungtr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Ptr{Cint}),
                   handle, uplo, n, A, lda, tau, lwork)
end

@checked function cusolverDnSorgtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnSorgtr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnDorgtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnDorgtr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnCungtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnCungtr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnZungtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnZungtr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, uplo, n, A, lda, tau, work, lwork, info)
end

@checked function cusolverDnSormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau,
                                              C, ldc, lwork)
    initialize_context()
    ccall((:cusolverDnSormtr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                    cublasOperation_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                    CuPtr{Cfloat}, Cint, Ptr{Cint}),
                   handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
end

@checked function cusolverDnDormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau,
                                              C, ldc, lwork)
    initialize_context()
    ccall((:cusolverDnDormtr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                    cublasOperation_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                    CuPtr{Cdouble}, Cint, Ptr{Cint}),
                   handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
end

@checked function cusolverDnCunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau,
                                              C, ldc, lwork)
    initialize_context()
    ccall((:cusolverDnCunmtr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                    cublasOperation_t, Cint, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, Ptr{Cint}),
                   handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
end

@checked function cusolverDnZunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau,
                                              C, ldc, lwork)
    initialize_context()
    ccall((:cusolverDnZunmtr_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                    cublasOperation_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                   handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
end

@checked function cusolverDnSormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                   work, lwork, info)
    initialize_context()
    ccall((:cusolverDnSormtr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                    cublasOperation_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
end

@checked function cusolverDnDormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                   work, lwork, info)
    initialize_context()
    ccall((:cusolverDnDormtr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                    cublasOperation_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
end

@checked function cusolverDnCunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                   work, lwork, info)
    initialize_context()
    ccall((:cusolverDnCunmtr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                    cublasOperation_t, Cint, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}),
                   handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
end

@checked function cusolverDnZunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                   work, lwork, info)
    initialize_context()
    ccall((:cusolverDnZunmtr, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                    cublasOperation_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
end

@checked function cusolverDnSgesvd_bufferSize(handle, m, n, lwork)
    initialize_context()
    ccall((:cusolverDnSgesvd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                   handle, m, n, lwork)
end

@checked function cusolverDnDgesvd_bufferSize(handle, m, n, lwork)
    initialize_context()
    ccall((:cusolverDnDgesvd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                   handle, m, n, lwork)
end

@checked function cusolverDnCgesvd_bufferSize(handle, m, n, lwork)
    initialize_context()
    ccall((:cusolverDnCgesvd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                   handle, m, n, lwork)
end

@checked function cusolverDnZgesvd_bufferSize(handle, m, n, lwork)
    initialize_context()
    ccall((:cusolverDnZgesvd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                   handle, m, n, lwork)
end

@checked function cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,
                                   work, lwork, rwork, info)
    initialize_context()
    ccall((:cusolverDnSgesvd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, UInt8, UInt8, Cint, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                    Cint, CuPtr{Cfloat}, CuPtr{Cint}),
                   handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork,
                   rwork, info)
end

@checked function cusolverDnDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,
                                   work, lwork, rwork, info)
    initialize_context()
    ccall((:cusolverDnDgesvd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, UInt8, UInt8, Cint, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cint}),
                   handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork,
                   rwork, info)
end

@checked function cusolverDnCgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,
                                   work, lwork, rwork, info)
    initialize_context()
    ccall((:cusolverDnCgesvd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, UInt8, UInt8, Cint, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{Cint}),
                   handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork,
                   rwork, info)
end

@checked function cusolverDnZgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,
                                   work, lwork, rwork, info)
    initialize_context()
    ccall((:cusolverDnZgesvd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, UInt8, UInt8, Cint, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{Cdouble}, CuPtr{Cint}),
                   handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork,
                   rwork, info)
end

@checked function cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    initialize_context()
    ccall((:cusolverDnSsyevd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Ptr{Cint}),
                   handle, jobz, uplo, n, A, lda, W, lwork)
end

@checked function cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    initialize_context()
    ccall((:cusolverDnDsyevd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Ptr{Cint}),
                   handle, jobz, uplo, n, A, lda, W, lwork)
end

@checked function cusolverDnCheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    initialize_context()
    ccall((:cusolverDnCheevd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, Ptr{Cint}),
                   handle, jobz, uplo, n, A, lda, W, lwork)
end

@checked function cusolverDnZheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    initialize_context()
    ccall((:cusolverDnZheevd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}),
                   handle, jobz, uplo, n, A, lda, W, lwork)
end

@checked function cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnSsyevd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info)
end

@checked function cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnDsyevd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info)
end

@checked function cusolverDnCheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnCheevd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info)
end

@checked function cusolverDnZheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnZheevd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cint}),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info)
end

@checked function cusolverDnSsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl,
                                               vu, il, iu, meig, W, lwork)
    initialize_context()
    ccall((:cusolverDnSsyevdx_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                    cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, Cfloat, Cfloat, Cint,
                    Cint, Ptr{Cint}, CuPtr{Cfloat}, Ptr{Cint}),
                   handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
end

@checked function cusolverDnDsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl,
                                               vu, il, iu, meig, W, lwork)
    initialize_context()
    ccall((:cusolverDnDsyevdx_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                    cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, Cdouble, Cdouble, Cint,
                    Cint, Ptr{Cint}, CuPtr{Cdouble}, Ptr{Cint}),
                   handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
end

@checked function cusolverDnCheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl,
                                               vu, il, iu, meig, W, lwork)
    initialize_context()
    ccall((:cusolverDnCheevdx_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                    cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, Cfloat, Cfloat, Cint,
                    Cint, Ptr{Cint}, CuPtr{Cfloat}, Ptr{Cint}),
                   handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
end

@checked function cusolverDnZheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl,
                                               vu, il, iu, meig, W, lwork)
    initialize_context()
    ccall((:cusolverDnZheevdx_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                    cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint, Cdouble, Cdouble,
                    Cint, Cint, Ptr{Cint}, CuPtr{Cdouble}, Ptr{Cint}),
                   handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
end

@checked function cusolverDnSsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                    meig, W, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnSsyevdx, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                    cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, Cfloat, Cfloat, Cint,
                    Cint, Ptr{Cint}, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work,
                   lwork, info)
end

@checked function cusolverDnDsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                    meig, W, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnDsyevdx, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                    cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, Cdouble, Cdouble, Cint,
                    Cint, Ptr{Cint}, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work,
                   lwork, info)
end

@checked function cusolverDnCheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                    meig, W, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnCheevdx, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                    cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, Cfloat, Cfloat, Cint,
                    Cint, Ptr{Cint}, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work,
                   lwork, info)
end

@checked function cusolverDnZheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                    meig, W, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnZheevdx, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                    cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint, Cdouble, Cdouble,
                    Cint, Cint, Ptr{Cint}, CuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{Cint}),
                   handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work,
                   lwork, info)
end

@checked function cusolverDnSsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda,
                                               B, ldb, vl, vu, il, iu, meig, W, lwork)
    initialize_context()
    ccall((:cusolverDnSsygvdx_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Cint, Cfloat, Cfloat, Cint, Cint, Ptr{Cint},
                    CuPtr{Cfloat}, Ptr{Cint}),
                   handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu,
                   meig, W, lwork)
end

@checked function cusolverDnDsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda,
                                               B, ldb, vl, vu, il, iu, meig, W, lwork)
    initialize_context()
    ccall((:cusolverDnDsygvdx_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, Cdouble, Cdouble, Cint, Cint, Ptr{Cint},
                    CuPtr{Cdouble}, Ptr{Cint}),
                   handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu,
                   meig, W, lwork)
end

@checked function cusolverDnChegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda,
                                               B, ldb, vl, vu, il, iu, meig, W, lwork)
    initialize_context()
    ccall((:cusolverDnChegvdx_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, Cfloat, Cfloat, Cint, Cint, Ptr{Cint},
                    CuPtr{Cfloat}, Ptr{Cint}),
                   handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu,
                   meig, W, lwork)
end

@checked function cusolverDnZhegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda,
                                               B, ldb, vl, vu, il, iu, meig, W, lwork)
    initialize_context()
    ccall((:cusolverDnZhegvdx_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Cint, Cdouble, Cdouble, Cint, Cint,
                    Ptr{Cint}, CuPtr{Cdouble}, Ptr{Cint}),
                   handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu,
                   meig, W, lwork)
end

@checked function cusolverDnSsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb,
                                    vl, vu, il, iu, meig, W, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnSsygvdx, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Cint, Cfloat, Cfloat, Cint, Cint, Ptr{Cint},
                    CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu,
                   meig, W, work, lwork, info)
end

@checked function cusolverDnDsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb,
                                    vl, vu, il, iu, meig, W, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnDsygvdx, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, Cdouble, Cdouble, Cint, Cint, Ptr{Cint},
                    CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu,
                   meig, W, work, lwork, info)
end

@checked function cusolverDnChegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb,
                                    vl, vu, il, iu, meig, W, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnChegvdx, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, Cfloat, Cfloat, Cint, Cint, Ptr{Cint},
                    CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu,
                   meig, W, work, lwork, info)
end

@checked function cusolverDnZhegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb,
                                    vl, vu, il, iu, meig, W, work, lwork, info)
    initialize_context()
    ccall((:cusolverDnZhegvdx, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Cint, Cdouble, Cdouble, Cint, Cint,
                    Ptr{Cint}, CuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                   handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu,
                   meig, W, work, lwork, info)
end

@checked function cusolverDnSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork)
    initialize_context()
    ccall((:cusolverDnSsygvd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Ptr{Cint}),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
end

@checked function cusolverDnDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork)
    initialize_context()
    ccall((:cusolverDnDsygvd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Ptr{Cint}),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
end

@checked function cusolverDnChegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork)
    initialize_context()
    ccall((:cusolverDnChegvd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cfloat}, Ptr{Cint}),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
end

@checked function cusolverDnZhegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork)
    initialize_context()
    ccall((:cusolverDnZhegvd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
end

@checked function cusolverDnSsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info)
    initialize_context()
    ccall((:cusolverDnSsygvd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
end

@checked function cusolverDnDsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info)
    initialize_context()
    ccall((:cusolverDnDsygvd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
end

@checked function cusolverDnChegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info)
    initialize_context()
    ccall((:cusolverDnChegvd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
end

@checked function cusolverDnZhegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info)
    initialize_context()
    ccall((:cusolverDnZhegvd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cint}),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
end

@checked function cusolverDnCreateSyevjInfo(info)
    initialize_context()
    ccall((:cusolverDnCreateSyevjInfo, libcusolver()), cusolverStatus_t,
                   (Ptr{syevjInfo_t},),
                   info)
end

@checked function cusolverDnDestroySyevjInfo(info)
    initialize_context()
    ccall((:cusolverDnDestroySyevjInfo, libcusolver()), cusolverStatus_t,
                   (syevjInfo_t,),
                   info)
end

@checked function cusolverDnXsyevjSetTolerance(info, tolerance)
    initialize_context()
    ccall((:cusolverDnXsyevjSetTolerance, libcusolver()), cusolverStatus_t,
                   (syevjInfo_t, Cdouble),
                   info, tolerance)
end

@checked function cusolverDnXsyevjSetMaxSweeps(info, max_sweeps)
    initialize_context()
    ccall((:cusolverDnXsyevjSetMaxSweeps, libcusolver()), cusolverStatus_t,
                   (syevjInfo_t, Cint),
                   info, max_sweeps)
end

@checked function cusolverDnXsyevjSetSortEig(info, sort_eig)
    initialize_context()
    ccall((:cusolverDnXsyevjSetSortEig, libcusolver()), cusolverStatus_t,
                   (syevjInfo_t, Cint),
                   info, sort_eig)
end

@checked function cusolverDnXsyevjGetResidual(handle, info, residual)
    initialize_context()
    ccall((:cusolverDnXsyevjGetResidual, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, syevjInfo_t, Ptr{Cdouble}),
                   handle, info, residual)
end

@checked function cusolverDnXsyevjGetSweeps(handle, info, executed_sweeps)
    initialize_context()
    ccall((:cusolverDnXsyevjGetSweeps, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, syevjInfo_t, Ptr{Cint}),
                   handle, info, executed_sweeps)
end

@checked function cusolverDnSsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W,
                                                     lwork, params, batchSize)
    initialize_context()
    ccall((:cusolverDnSsyevjBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t, Cint),
                   handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
end

@checked function cusolverDnDsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W,
                                                     lwork, params, batchSize)
    initialize_context()
    ccall((:cusolverDnDsyevjBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t, Cint),
                   handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
end

@checked function cusolverDnCheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W,
                                                     lwork, params, batchSize)
    initialize_context()
    ccall((:cusolverDnCheevjBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t, Cint),
                   handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
end

@checked function cusolverDnZheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W,
                                                     lwork, params, batchSize)
    initialize_context()
    ccall((:cusolverDnZheevjBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t,
                    Cint),
                   handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
end

@checked function cusolverDnSsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork,
                                          info, params, batchSize)
    initialize_context()
    ccall((:cusolverDnSsyevjBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint},
                    syevjInfo_t, Cint),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
end

@checked function cusolverDnDsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork,
                                          info, params, batchSize)
    initialize_context()
    ccall((:cusolverDnDsyevjBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                    CuPtr{Cint}, syevjInfo_t, Cint),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
end

@checked function cusolverDnCheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork,
                                          info, params, batchSize)
    initialize_context()
    ccall((:cusolverDnCheevjBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}, syevjInfo_t, Cint),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
end

@checked function cusolverDnZheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork,
                                          info, params, batchSize)
    initialize_context()
    ccall((:cusolverDnZheevjBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cint}, syevjInfo_t, Cint),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
end

@checked function cusolverDnSsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                              params)
    initialize_context()
    ccall((:cusolverDnSsyevj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t),
                   handle, jobz, uplo, n, A, lda, W, lwork, params)
end

@checked function cusolverDnDsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                              params)
    initialize_context()
    ccall((:cusolverDnDsyevj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t),
                   handle, jobz, uplo, n, A, lda, W, lwork, params)
end

@checked function cusolverDnCheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                              params)
    initialize_context()
    ccall((:cusolverDnCheevj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t),
                   handle, jobz, uplo, n, A, lda, W, lwork, params)
end

@checked function cusolverDnZheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                              params)
    initialize_context()
    ccall((:cusolverDnZheevj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t),
                   handle, jobz, uplo, n, A, lda, W, lwork, params)
end

@checked function cusolverDnSsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                   params)
    initialize_context()
    ccall((:cusolverDnSsyevj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint},
                    syevjInfo_t),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
end

@checked function cusolverDnDsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                   params)
    initialize_context()
    ccall((:cusolverDnDsyevj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                    CuPtr{Cint}, syevjInfo_t),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
end

@checked function cusolverDnCheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                   params)
    initialize_context()
    ccall((:cusolverDnCheevj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                    CuPtr{Cint}, syevjInfo_t),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
end

@checked function cusolverDnZheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                   params)
    initialize_context()
    ccall((:cusolverDnZheevj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cint}, syevjInfo_t),
                   handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
end

@checked function cusolverDnSsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork, params)
    initialize_context()
    ccall((:cusolverDnSsygvj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
end

@checked function cusolverDnDsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork, params)
    initialize_context()
    ccall((:cusolverDnDsygvj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
end

@checked function cusolverDnChegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork, params)
    initialize_context()
    ccall((:cusolverDnChegvj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
end

@checked function cusolverDnZhegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork, params)
    initialize_context()
    ccall((:cusolverDnZhegvj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
end

@checked function cusolverDnSsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info, params)
    initialize_context()
    ccall((:cusolverDnSsygvj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}, syevjInfo_t),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info,
                   params)
end

@checked function cusolverDnDsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info, params)
    initialize_context()
    ccall((:cusolverDnDsygvj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}, syevjInfo_t),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info,
                   params)
end

@checked function cusolverDnChegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info, params)
    initialize_context()
    ccall((:cusolverDnChegvj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{Cint}, syevjInfo_t),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info,
                   params)
end

@checked function cusolverDnZhegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info, params)
    initialize_context()
    ccall((:cusolverDnZhegvj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                    cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{Cint}, syevjInfo_t),
                   handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info,
                   params)
end

@checked function cusolverDnCreateGesvdjInfo(info)
    initialize_context()
    ccall((:cusolverDnCreateGesvdjInfo, libcusolver()), cusolverStatus_t,
                   (Ptr{gesvdjInfo_t},),
                   info)
end

@checked function cusolverDnDestroyGesvdjInfo(info)
    initialize_context()
    ccall((:cusolverDnDestroyGesvdjInfo, libcusolver()), cusolverStatus_t,
                   (gesvdjInfo_t,),
                   info)
end

@checked function cusolverDnXgesvdjSetTolerance(info, tolerance)
    initialize_context()
    ccall((:cusolverDnXgesvdjSetTolerance, libcusolver()), cusolverStatus_t,
                   (gesvdjInfo_t, Cdouble),
                   info, tolerance)
end

@checked function cusolverDnXgesvdjSetMaxSweeps(info, max_sweeps)
    initialize_context()
    ccall((:cusolverDnXgesvdjSetMaxSweeps, libcusolver()), cusolverStatus_t,
                   (gesvdjInfo_t, Cint),
                   info, max_sweeps)
end

@checked function cusolverDnXgesvdjSetSortEig(info, sort_svd)
    initialize_context()
    ccall((:cusolverDnXgesvdjSetSortEig, libcusolver()), cusolverStatus_t,
                   (gesvdjInfo_t, Cint),
                   info, sort_svd)
end

@checked function cusolverDnXgesvdjGetResidual(handle, info, residual)
    initialize_context()
    ccall((:cusolverDnXgesvdjGetResidual, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, gesvdjInfo_t, Ptr{Cdouble}),
                   handle, info, residual)
end

@checked function cusolverDnXgesvdjGetSweeps(handle, info, executed_sweeps)
    initialize_context()
    ccall((:cusolverDnXgesvdjGetSweeps, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, gesvdjInfo_t, Ptr{Cint}),
                   handle, info, executed_sweeps)
end

@checked function cusolverDnSgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U,
                                                      ldu, V, ldv, lwork, params, batchSize)
    initialize_context()
    ccall((:cusolverDnSgesvdjBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{Cfloat},
                    Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                    Ptr{Cint}, gesvdjInfo_t, Cint),
                   handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
end

@checked function cusolverDnDgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U,
                                                      ldu, V, ldv, lwork, params, batchSize)
    initialize_context()
    ccall((:cusolverDnDgesvdjBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{Cdouble},
                    Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                    Ptr{Cint}, gesvdjInfo_t, Cint),
                   handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
end

@checked function cusolverDnCgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U,
                                                      ldu, V, ldv, lwork, params, batchSize)
    initialize_context()
    ccall((:cusolverDnCgesvdjBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{cuComplex},
                    Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                    Ptr{Cint}, gesvdjInfo_t, Cint),
                   handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
end

@checked function cusolverDnZgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U,
                                                      ldu, V, ldv, lwork, params, batchSize)
    initialize_context()
    ccall((:cusolverDnZgesvdjBatched_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}, gesvdjInfo_t, Cint),
                   handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
end

@checked function cusolverDnSgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                           work, lwork, info, params, batchSize)
    initialize_context()
    ccall((:cusolverDnSgesvdjBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{Cfloat},
                    Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cint}, gesvdjInfo_t, Cint),
                   handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                   params, batchSize)
end

@checked function cusolverDnDgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                           work, lwork, info, params, batchSize)
    initialize_context()
    ccall((:cusolverDnDgesvdjBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{Cdouble},
                    Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cint}, gesvdjInfo_t, Cint),
                   handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                   params, batchSize)
end

@checked function cusolverDnCgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                           work, lwork, info, params, batchSize)
    initialize_context()
    ccall((:cusolverDnCgesvdjBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{cuComplex},
                    Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cint}, gesvdjInfo_t, Cint),
                   handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                   params, batchSize)
end

@checked function cusolverDnZgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                           work, lwork, info, params, batchSize)
    initialize_context()
    ccall((:cusolverDnZgesvdjBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint,
                    CuPtr{Cint}, gesvdjInfo_t, Cint),
                   handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                   params, batchSize)
end

@checked function cusolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu,
                                               V, ldv, lwork, params)
    initialize_context()
    ccall((:cusolverDnSgesvdj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                    Cint, Ptr{Cint}, gesvdjInfo_t),
                   handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
end

@checked function cusolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu,
                                               V, ldv, lwork, params)
    initialize_context()
    ccall((:cusolverDnDgesvdj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, Ptr{Cint}, gesvdjInfo_t),
                   handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
end

@checked function cusolverDnCgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu,
                                               V, ldv, lwork, params)
    initialize_context()
    ccall((:cusolverDnCgesvdj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, Ptr{Cint}, gesvdjInfo_t),
                   handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
end

@checked function cusolverDnZgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu,
                                               V, ldv, lwork, params)
    initialize_context()
    ccall((:cusolverDnZgesvdj_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                    Cint, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}, gesvdjInfo_t),
                   handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
end

@checked function cusolverDnSgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                    work, lwork, info, params)
    initialize_context()
    ccall((:cusolverDnSgesvdj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                    Cint, CuPtr{Cfloat}, Cint, CuPtr{Cint}, gesvdjInfo_t),
                   handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                   params)
end

@checked function cusolverDnDgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                    work, lwork, info, params)
    initialize_context()
    ccall((:cusolverDnDgesvdj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                    CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cint}, gesvdjInfo_t),
                   handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                   params)
end

@checked function cusolverDnCgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                    work, lwork, info, params)
    initialize_context()
    ccall((:cusolverDnCgesvdj, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                    CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint, CuPtr{Cint},
                    gesvdjInfo_t),
                   handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                   params)
end

@checked function cusolverDnZgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                    work, lwork, info, params)
    initialize_context()
    ccall((:cusolverDnZgesvdj, libcusolver()), cusolverStatus_t,
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
    initialize_context()
    ccall((:cusolverDnSgesvdaStridedBatched_bufferSize, libcusolver()), cusolverStatus_t,
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
    initialize_context()
    ccall((:cusolverDnDgesvdaStridedBatched_bufferSize, libcusolver()), cusolverStatus_t,
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
    initialize_context()
    ccall((:cusolverDnCgesvdaStridedBatched_bufferSize, libcusolver()), cusolverStatus_t,
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
    initialize_context()
    ccall((:cusolverDnZgesvdaStridedBatched_bufferSize, libcusolver()), cusolverStatus_t,
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
    initialize_context()
    ccall((:cusolverDnSgesvdaStridedBatched, libcusolver()), cusolverStatus_t,
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
    initialize_context()
    ccall((:cusolverDnDgesvdaStridedBatched, libcusolver()), cusolverStatus_t,
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
    initialize_context()
    ccall((:cusolverDnCgesvdaStridedBatched, libcusolver()), cusolverStatus_t,
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
    initialize_context()
    ccall((:cusolverDnZgesvdaStridedBatched, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                    CuPtr{cuDoubleComplex}, Cint, Clonglong, CuPtr{Cdouble}, Clonglong,
                    CuPtr{cuDoubleComplex}, Cint, Clonglong, CuPtr{cuDoubleComplex}, Cint,
                    Clonglong, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}, Ptr{Cdouble},
                    Cint),
                   handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu,
                   strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)
end

@checked function cusolverDnCreateParams(params)
    initialize_context()
    ccall((:cusolverDnCreateParams, libcusolver()), cusolverStatus_t,
                   (Ptr{cusolverDnParams_t},),
                   params)
end

@checked function cusolverDnDestroyParams(params)
    initialize_context()
    ccall((:cusolverDnDestroyParams, libcusolver()), cusolverStatus_t,
                   (cusolverDnParams_t,),
                   params)
end

@checked function cusolverDnSetAdvOptions(params, _function, algo)
    initialize_context()
    ccall((:cusolverDnSetAdvOptions, libcusolver()), cusolverStatus_t,
                   (cusolverDnParams_t, cusolverDnFunction_t, cusolverAlgMode_t),
                   params, _function, algo)
end

@checked function cusolverDnPotrf_bufferSize(handle, params, uplo, n, dataTypeA, A, lda,
                                             computeType, workspaceInBytes)
    initialize_context()
    ccall((:cusolverDnPotrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, Int64,
                    cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, Ptr{Csize_t}),
                   handle, params, uplo, n, dataTypeA, A, lda, computeType,
                   workspaceInBytes)
end

@checked function cusolverDnPotrf(handle, params, uplo, n, dataTypeA, A, lda, computeType,
                                  pBuffer, workspaceInBytes, info)
    initialize_context()
    ccall((:cusolverDnPotrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, Int64,
                    cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, Csize_t,
                    CuPtr{Cint}),
                   handle, params, uplo, n, dataTypeA, A, lda, computeType, pBuffer,
                   workspaceInBytes, info)
end

@checked function cusolverDnPotrs(handle, params, uplo, n, nrhs, dataTypeA, A, lda,
                                  dataTypeB, B, ldb, info)
    initialize_context()
    ccall((:cusolverDnPotrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, Int64,
                    Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid},
                    Int64, CuPtr{Cint}),
                   handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb,
                   info)
end

@checked function cusolverDnGeqrf_bufferSize(handle, params, m, n, dataTypeA, A, lda,
                                             dataTypeTau, tau, computeType, workspaceInBytes)
    initialize_context()
    ccall((:cusolverDnGeqrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnParams_t, Int64, Int64, cudaDataType,
                    CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, cudaDataType,
                    Ptr{Csize_t}),
                   handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType,
                   workspaceInBytes)
end

@checked function cusolverDnGeqrf(handle, params, m, n, dataTypeA, A, lda, dataTypeTau,
                                  tau, computeType, pBuffer, workspaceInBytes, info)
    initialize_context()
    ccall((:cusolverDnGeqrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnParams_t, Int64, Int64, cudaDataType,
                    CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, cudaDataType,
                    CuPtr{Cvoid}, Csize_t, CuPtr{Cint}),
                   handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType,
                   pBuffer, workspaceInBytes, info)
end

@checked function cusolverDnGetrf_bufferSize(handle, params, m, n, dataTypeA, A, lda,
                                             computeType, workspaceInBytes)
    initialize_context()
    ccall((:cusolverDnGetrf_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnParams_t, Int64, Int64, cudaDataType,
                    CuPtr{Cvoid}, Int64, cudaDataType, Ptr{Csize_t}),
                   handle, params, m, n, dataTypeA, A, lda, computeType, workspaceInBytes)
end

@checked function cusolverDnGetrf(handle, params, m, n, dataTypeA, A, lda, ipiv,
                                  computeType, pBuffer, workspaceInBytes, info)
    initialize_context()
    ccall((:cusolverDnGetrf, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnParams_t, Int64, Int64, cudaDataType,
                    CuPtr{Cvoid}, Int64, CuPtr{Int64}, cudaDataType, CuPtr{Cvoid}, Csize_t,
                    CuPtr{Cint}),
                   handle, params, m, n, dataTypeA, A, lda, ipiv, computeType, pBuffer,
                   workspaceInBytes, info)
end

@checked function cusolverDnGetrs(handle, params, trans, n, nrhs, dataTypeA, A, lda, ipiv,
                                  dataTypeB, B, ldb, info)
    initialize_context()
    ccall((:cusolverDnGetrs, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnParams_t, cublasOperation_t, Int64,
                    Int64, cudaDataType, CuPtr{Cvoid}, Int64, CuPtr{Int64}, cudaDataType,
                    CuPtr{Cvoid}, Int64, CuPtr{Cint}),
                   handle, params, trans, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B,
                   ldb, info)
end

@checked function cusolverDnSyevd_bufferSize(handle, params, jobz, uplo, n, dataTypeA, A,
                                             lda, dataTypeW, W, computeType,
                                             workspaceInBytes)
    initialize_context()
    ccall((:cusolverDnSyevd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t,
                    cublasFillMode_t, Int64, cudaDataType, CuPtr{Cvoid}, Int64,
                    cudaDataType, CuPtr{Cvoid}, cudaDataType, Ptr{Csize_t}),
                   handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W,
                   computeType, workspaceInBytes)
end

@checked function cusolverDnSyevd(handle, params, jobz, uplo, n, dataTypeA, A, lda,
                                  dataTypeW, W, computeType, pBuffer, workspaceInBytes, info)
    initialize_context()
    ccall((:cusolverDnSyevd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t,
                    cublasFillMode_t, Int64, cudaDataType, CuPtr{Cvoid}, Int64,
                    cudaDataType, CuPtr{Cvoid}, cudaDataType, CuPtr{Cvoid}, Csize_t,
                    CuPtr{Cint}),
                   handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W,
                   computeType, pBuffer, workspaceInBytes, info)
end

@checked function cusolverDnSyevdx_bufferSize(handle, params, jobz, range, uplo, n,
                                              dataTypeA, A, lda, vl, vu, il, iu, h_meig,
                                              dataTypeW, W, computeType, workspaceInBytes)
    initialize_context()
    ccall((:cusolverDnSyevdx_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t,
                    cusolverEigRange_t, cublasFillMode_t, Int64, cudaDataType,
                    CuPtr{Cvoid}, Int64, Ptr{Cvoid}, Ptr{Cvoid}, Int64, Int64, Ptr{Int64},
                    cudaDataType, CuPtr{Cvoid}, cudaDataType, Ptr{Csize_t}),
                   handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu,
                   h_meig, dataTypeW, W, computeType, workspaceInBytes)
end

@checked function cusolverDnSyevdx(handle, params, jobz, range, uplo, n, dataTypeA, A, lda,
                                   vl, vu, il, iu, meig64, dataTypeW, W, computeType,
                                   pBuffer, workspaceInBytes, info)
    initialize_context()
    ccall((:cusolverDnSyevdx, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t,
                    cusolverEigRange_t, cublasFillMode_t, Int64, cudaDataType,
                    CuPtr{Cvoid}, Int64, Ptr{Cvoid}, Ptr{Cvoid}, Int64, Int64, Ptr{Int64},
                    cudaDataType, CuPtr{Cvoid}, cudaDataType, CuPtr{Cvoid}, Csize_t,
                    CuPtr{Cint}),
                   handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu,
                   meig64, dataTypeW, W, computeType, pBuffer, workspaceInBytes, info)
end

@checked function cusolverDnGesvd_bufferSize(handle, params, jobu, jobvt, m, n, dataTypeA,
                                             A, lda, dataTypeS, S, dataTypeU, U, ldu,
                                             dataTypeVT, VT, ldvt, computeType,
                                             workspaceInBytes)
    initialize_context()
    ccall((:cusolverDnGesvd_bufferSize, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnParams_t, UInt8, UInt8, Int64, Int64,
                    cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid},
                    cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, Int64,
                    cudaDataType, Ptr{Csize_t}),
                   handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S,
                   dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, workspaceInBytes)
end

@checked function cusolverDnGesvd(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda,
                                  dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt,
                                  computeType, pBuffer, workspaceInBytes, info)
    initialize_context()
    ccall((:cusolverDnGesvd, libcusolver()), cusolverStatus_t,
                   (cusolverDnHandle_t, cusolverDnParams_t, UInt8, UInt8, Int64, Int64,
                    cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid},
                    cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, Int64,
                    cudaDataType, CuPtr{Cvoid}, Csize_t, CuPtr{Cint}),
                   handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S,
                   dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, pBuffer,
                   workspaceInBytes, info)
end
# Julia wrapper for header: cusolverSp.h
# Automatically generated using Clang.jl

@checked function cusolverSpCreate(handle)
    initialize_context()
    ccall((:cusolverSpCreate, libcusolver()), cusolverStatus_t,
                   (Ptr{cusolverSpHandle_t},),
                   handle)
end

@checked function cusolverSpDestroy(handle)
    initialize_context()
    ccall((:cusolverSpDestroy, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t,),
                   handle)
end

@checked function cusolverSpSetStream(handle, streamId)
    initialize_context()
    ccall((:cusolverSpSetStream, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, CUstream),
                   handle, streamId)
end

@checked function cusolverSpGetStream(handle, streamId)
    initialize_context()
    ccall((:cusolverSpGetStream, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Ptr{CUstream}),
                   handle, streamId)
end

@checked function cusolverSpXcsrissymHost(handle, m, nnzA, descrA, csrRowPtrA, csrEndPtrA,
                                          csrColIndA, issym)
    initialize_context()
    ccall((:cusolverSpXcsrissymHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                   handle, m, nnzA, descrA, csrRowPtrA, csrEndPtrA, csrColIndA, issym)
end

@checked function cusolverSpScsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpScsrlsvluHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cint}),
                   handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   reorder, x, singularity)
end

@checked function cusolverSpDcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpDcsrlsvluHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cdouble, Cint, Ptr{Cdouble},
                    Ptr{Cint}),
                   handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   reorder, x, singularity)
end

@checked function cusolverSpCcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpCcsrlsvluHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                    Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cfloat, Cint, Ptr{cuComplex},
                    Ptr{Cint}),
                   handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   reorder, x, singularity)
end

@checked function cusolverSpZcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpZcsrlsvluHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                    Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex},
                    Cdouble, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}),
                   handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   reorder, x, singularity)
end

@checked function cusolverSpScsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd,
                                      b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpScsrlsvqr, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                    CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cfloat}, Cfloat, Cint, CuPtr{Cfloat},
                    Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpDcsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd,
                                      b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpDcsrlsvqr, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                    CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cdouble, Cint,
                    CuPtr{Cdouble}, Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpCcsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd,
                                      b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpCcsrlsvqr, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                    CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuComplex}, Cfloat, Cint,
                    CuPtr{cuComplex}, Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpZcsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd,
                                      b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpZcsrlsvqr, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                    CuPtr{cuDoubleComplex}, Cdouble, Cint, CuPtr{cuDoubleComplex},
                    Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpScsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpScsrlsvqrHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cint}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   reorder, x, singularity)
end

@checked function cusolverSpDcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpDcsrlsvqrHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cdouble, Cint, Ptr{Cdouble},
                    Ptr{Cint}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   reorder, x, singularity)
end

@checked function cusolverSpCcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpCcsrlsvqrHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                    Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cfloat, Cint, Ptr{cuComplex},
                    Ptr{Cint}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   reorder, x, singularity)
end

@checked function cusolverSpZcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpZcsrlsvqrHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                    Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex},
                    Cdouble, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   reorder, x, singularity)
end

@checked function cusolverSpScsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                            csrColInd, b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpScsrlsvcholHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpDcsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                            csrColInd, b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpDcsrlsvcholHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cdouble, Cint, Ptr{Cdouble},
                    Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpCcsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                            csrColInd, b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpCcsrlsvcholHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                    Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cfloat, Cint, Ptr{cuComplex},
                    Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpZcsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                            csrColInd, b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpZcsrlsvcholHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                    Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex},
                    Cdouble, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpScsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                        csrColInd, b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpScsrlsvchol, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                    CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cfloat}, Cfloat, Cint, CuPtr{Cfloat},
                    Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpDcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                        csrColInd, b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpDcsrlsvchol, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                    CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cdouble, Cint,
                    CuPtr{Cdouble}, Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpCcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                        csrColInd, b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpCcsrlsvchol, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                    CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuComplex}, Cfloat, Cint,
                    CuPtr{cuComplex}, Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpZcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                        csrColInd, b, tol, reorder, x, singularity)
    initialize_context()
    ccall((:cusolverSpZcsrlsvchol, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                    CuPtr{cuDoubleComplex}, Cdouble, Cint, CuPtr{cuDoubleComplex},
                    Ptr{Cint}),
                   handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder,
                   x, singularity)
end

@checked function cusolverSpScsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, b, tol, rankA, x, p, min_norm)
    initialize_context()
    ccall((:cusolverSpScsrlsqvqrHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cfloat, Ptr{Cint}, Ptr{Cfloat},
                    Ptr{Cint}, Ptr{Cfloat}),
                   handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   rankA, x, p, min_norm)
end

@checked function cusolverSpDcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, b, tol, rankA, x, p, min_norm)
    initialize_context()
    ccall((:cusolverSpDcsrlsqvqrHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cdouble, Ptr{Cint},
                    Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}),
                   handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   rankA, x, p, min_norm)
end

@checked function cusolverSpCcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, b, tol, rankA, x, p, min_norm)
    initialize_context()
    ccall((:cusolverSpCcsrlsqvqrHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cfloat,
                    Ptr{Cint}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cfloat}),
                   handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   rankA, x, p, min_norm)
end

@checked function cusolverSpZcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, b, tol, rankA, x, p, min_norm)
    initialize_context()
    ccall((:cusolverSpZcsrlsqvqrHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex},
                    Cdouble, Ptr{Cint}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cdouble}),
                   handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                   rankA, x, p, min_norm)
end

@checked function cusolverSpScsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, mu0, x0, maxite, tol, mu, x)
    initialize_context()
    ccall((:cusolverSpScsreigvsiHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                    Ptr{Cint}, Ptr{Cint}, Cfloat, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat},
                    Ptr{Cfloat}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0,
                   maxite, tol, mu, x)
end

@checked function cusolverSpDcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, mu0, x0, maxite, tol, mu, x)
    initialize_context()
    ccall((:cusolverSpDcsreigvsiHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                    Ptr{Cint}, Ptr{Cint}, Cdouble, Ptr{Cdouble}, Cint, Cdouble,
                    Ptr{Cdouble}, Ptr{Cdouble}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0,
                   maxite, tol, mu, x)
end

@checked function cusolverSpCcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, mu0, x0, maxite, tol, mu, x)
    initialize_context()
    ccall((:cusolverSpCcsreigvsiHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                    Ptr{Cint}, Ptr{Cint}, cuComplex, Ptr{cuComplex}, Cint, Cfloat,
                    Ptr{cuComplex}, Ptr{cuComplex}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0,
                   maxite, tol, mu, x)
end

@checked function cusolverSpZcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, mu0, x0, maxite, tol, mu, x)
    initialize_context()
    ccall((:cusolverSpZcsreigvsiHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                    Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, cuDoubleComplex,
                    Ptr{cuDoubleComplex}, Cint, Cdouble, Ptr{cuDoubleComplex},
                    Ptr{cuDoubleComplex}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0,
                   maxite, tol, mu, x)
end

@checked function cusolverSpScsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                       csrColIndA, mu0, x0, maxite, eps, mu, x)
    initialize_context()
    ccall((:cusolverSpScsreigvsi, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                    CuPtr{Cint}, CuPtr{Cint}, Cfloat, CuPtr{Cfloat}, Cint, Cfloat,
                    CuPtr{Cfloat}, CuPtr{Cfloat}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0,
                   maxite, eps, mu, x)
end

@checked function cusolverSpDcsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                       csrColIndA, mu0, x0, maxite, eps, mu, x)
    initialize_context()
    ccall((:cusolverSpDcsreigvsi, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                    CuPtr{Cint}, CuPtr{Cint}, Cdouble, CuPtr{Cdouble}, Cint, Cdouble,
                    CuPtr{Cdouble}, CuPtr{Cdouble}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0,
                   maxite, eps, mu, x)
end

@checked function cusolverSpCcsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                       csrColIndA, mu0, x0, maxite, eps, mu, x)
    initialize_context()
    ccall((:cusolverSpCcsreigvsi, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                    CuPtr{Cint}, CuPtr{Cint}, cuComplex, CuPtr{cuComplex}, Cint, Cfloat,
                    CuPtr{cuComplex}, CuPtr{cuComplex}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0,
                   maxite, eps, mu, x)
end

@checked function cusolverSpZcsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                       csrColIndA, mu0, x0, maxite, eps, mu, x)
    initialize_context()
    ccall((:cusolverSpZcsreigvsi, libcusolver()), cusolverStatus_t,
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
    initialize_context()
    ccall((:cusolverSpScsreigsHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                    Ptr{Cint}, Ptr{Cint}, cuComplex, cuComplex, Ptr{Cint}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                   left_bottom_corner, right_upper_corner, num_eigs)
end

@checked function cusolverSpDcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                         csrColIndA, left_bottom_corner,
                                         right_upper_corner, num_eigs)
    initialize_context()
    ccall((:cusolverSpDcsreigsHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                    Ptr{Cint}, Ptr{Cint}, cuDoubleComplex, cuDoubleComplex, Ptr{Cint}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                   left_bottom_corner, right_upper_corner, num_eigs)
end

@checked function cusolverSpCcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                         csrColIndA, left_bottom_corner,
                                         right_upper_corner, num_eigs)
    initialize_context()
    ccall((:cusolverSpCcsreigsHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                    Ptr{Cint}, Ptr{Cint}, cuComplex, cuComplex, Ptr{Cint}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                   left_bottom_corner, right_upper_corner, num_eigs)
end

@checked function cusolverSpZcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                         csrColIndA, left_bottom_corner,
                                         right_upper_corner, num_eigs)
    initialize_context()
    ccall((:cusolverSpZcsreigsHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                    Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, cuDoubleComplex,
                    cuDoubleComplex, Ptr{Cint}),
                   handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                   left_bottom_corner, right_upper_corner, num_eigs)
end

@checked function cusolverSpXcsrsymrcmHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA,
                                           p)
    initialize_context()
    ccall((:cusolverSpXcsrsymrcmHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}),
                   handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
end

@checked function cusolverSpXcsrsymmdqHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA,
                                           p)
    initialize_context()
    ccall((:cusolverSpXcsrsymmdqHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}),
                   handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
end

@checked function cusolverSpXcsrsymamdHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA,
                                           p)
    initialize_context()
    ccall((:cusolverSpXcsrsymamdHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}),
                   handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
end

@checked function cusolverSpXcsrmetisndHost(handle, n, nnzA, descrA, csrRowPtrA,
                                            csrColIndA, options, p)
    initialize_context()
    ccall((:cusolverSpXcsrmetisndHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                    Ptr{Cint}, Ptr{Int64}, Ptr{Cint}),
                   handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, options, p)
end

@checked function cusolverSpScsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA,
                                        csrColIndA, P, numnz)
    initialize_context()
    ccall((:cusolverSpScsrzfdHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                   handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
end

@checked function cusolverSpDcsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA,
                                        csrColIndA, P, numnz)
    initialize_context()
    ccall((:cusolverSpDcsrzfdHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                   handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
end

@checked function cusolverSpCcsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA,
                                        csrColIndA, P, numnz)
    initialize_context()
    ccall((:cusolverSpCcsrzfdHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                   handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
end

@checked function cusolverSpZcsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA,
                                        csrColIndA, P, numnz)
    initialize_context()
    ccall((:cusolverSpZcsrzfdHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                    Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                   handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
end

@checked function cusolverSpXcsrperm_bufferSizeHost(handle, m, n, nnzA, descrA, csrRowPtrA,
                                                    csrColIndA, p, q, bufferSizeInBytes)
    initialize_context()
    ccall((:cusolverSpXcsrperm_bufferSizeHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Csize_t}),
                   handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q,
                   bufferSizeInBytes)
end

@checked function cusolverSpXcsrpermHost(handle, m, n, nnzA, descrA, csrRowPtrA,
                                         csrColIndA, p, q, map, pBuffer)
    initialize_context()
    ccall((:cusolverSpXcsrpermHost, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}),
                   handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q, map, pBuffer)
end

@checked function cusolverSpCreateCsrqrInfo(info)
    initialize_context()
    ccall((:cusolverSpCreateCsrqrInfo, libcusolver()), cusolverStatus_t,
                   (Ptr{csrqrInfo_t},),
                   info)
end

@checked function cusolverSpDestroyCsrqrInfo(info)
    initialize_context()
    ccall((:cusolverSpDestroyCsrqrInfo, libcusolver()), cusolverStatus_t,
                   (csrqrInfo_t,),
                   info)
end

@checked function cusolverSpXcsrqrAnalysisBatched(handle, m, n, nnzA, descrA, csrRowPtrA,
                                                  csrColIndA, info)
    initialize_context()
    ccall((:cusolverSpXcsrqrAnalysisBatched, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cint},
                    CuPtr{Cint}, csrqrInfo_t),
                   handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
end

@checked function cusolverSpScsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal,
                                                    csrRowPtr, csrColInd, batchSize, info,
                                                    internalDataInBytes, workspaceInBytes)
    initialize_context()
    ccall((:cusolverSpScsrqrBufferInfoBatched, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, csrqrInfo_t,
                    Ptr{Csize_t}, Ptr{Csize_t}),
                   handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize,
                   info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpDcsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal,
                                                    csrRowPtr, csrColInd, batchSize, info,
                                                    internalDataInBytes, workspaceInBytes)
    initialize_context()
    ccall((:cusolverSpDcsrqrBufferInfoBatched, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, csrqrInfo_t,
                    Ptr{Csize_t}, Ptr{Csize_t}),
                   handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize,
                   info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpCcsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal,
                                                    csrRowPtr, csrColInd, batchSize, info,
                                                    internalDataInBytes, workspaceInBytes)
    initialize_context()
    ccall((:cusolverSpCcsrqrBufferInfoBatched, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, csrqrInfo_t,
                    Ptr{Csize_t}, Ptr{Csize_t}),
                   handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize,
                   info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpZcsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal,
                                                    csrRowPtr, csrColInd, batchSize, info,
                                                    internalDataInBytes, workspaceInBytes)
    initialize_context()
    ccall((:cusolverSpZcsrqrBufferInfoBatched, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, csrqrInfo_t,
                    Ptr{Csize_t}, Ptr{Csize_t}),
                   handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize,
                   info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpScsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                            csrColIndA, b, x, batchSize, info, pBuffer)
    initialize_context()
    ccall((:cusolverSpScsrqrsvBatched, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cfloat}, CuPtr{Cfloat},
                    Cint, csrqrInfo_t, CuPtr{Cvoid}),
                   handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x,
                   batchSize, info, pBuffer)
end

@checked function cusolverSpDcsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                            csrColIndA, b, x, batchSize, info, pBuffer)
    initialize_context()
    ccall((:cusolverSpDcsrqrsvBatched, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble},
                    CuPtr{Cdouble}, Cint, csrqrInfo_t, CuPtr{Cvoid}),
                   handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x,
                   batchSize, info, pBuffer)
end

@checked function cusolverSpCcsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                            csrColIndA, b, x, batchSize, info, pBuffer)
    initialize_context()
    ccall((:cusolverSpCcsrqrsvBatched, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuComplex},
                    CuPtr{cuComplex}, Cint, csrqrInfo_t, CuPtr{Cvoid}),
                   handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x,
                   batchSize, info, pBuffer)
end

@checked function cusolverSpZcsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                            csrColIndA, b, x, batchSize, info, pBuffer)
    initialize_context()
    ccall((:cusolverSpZcsrqrsvBatched, libcusolver()), cusolverStatus_t,
                   (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                    CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                    CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, csrqrInfo_t,
                    CuPtr{Cvoid}),
                   handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x,
                   batchSize, info, pBuffer)
end

## Added in CUDA 11.1

@checked function cusolverDnXpotrf_bufferSize(handle, params, uplo, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    initialize_context()
    ccall((:cusolverDnXpotrf_bufferSize, libcusolver()), cusolverStatus_t, (cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, Ptr{Csize_t}, Ptr{Csize_t}), handle, params, uplo, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
end

@checked function cusolverDnXgetrs(handle, params, trans, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, info)
    initialize_context()
    ccall((:cusolverDnXgetrs, libcusolver()), cusolverStatus_t, (cusolverDnHandle_t, cusolverDnParams_t, cublasOperation_t, Int64, Int64, cudaDataType, CuPtr{Cvoid}, Int64, CuPtr{Int64}, cudaDataType, CuPtr{Cvoid}, Int64, CuPtr{Cint}), handle, params, trans, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, info)
end

@checked function cusolverDnXgesvdp(handle, params, jobz, econ, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeV, V, ldv, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_info, h_err_sigma)
    initialize_context()
    ccall((:cusolverDnXgesvdp, libcusolver()), cusolverStatus_t, (cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, Cint, Int64, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid}, Csize_t, CuPtr{Cint}, Ptr{Cdouble}), handle, params, jobz, econ, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeV, V, ldv, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_info, h_err_sigma)
end

@checked function cusolverDnXgesvd(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    initialize_context()
    ccall((:cusolverDnXgesvd, libcusolver()), cusolverStatus_t, (cusolverDnHandle_t, cusolverDnParams_t, UInt8, UInt8, Int64, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid}, Csize_t, CuPtr{Cint}), handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
end

@checked function cusolverDnXsyevdx_bufferSize(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, h_meig, dataTypeW, W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    initialize_context()
    ccall((:cusolverDnXsyevdx_bufferSize, libcusolver()), cusolverStatus_t, (cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, Int64, cudaDataType, CuPtr{Cvoid}, Int64, CuPtr{Cvoid}, CuPtr{Cvoid}, Int64, Int64, CuPtr{Int64}, cudaDataType, CuPtr{Cvoid}, cudaDataType, Ptr{Csize_t}, Ptr{Csize_t}), handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, h_meig, dataTypeW, W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
end

@checked function cusolverDnXpotrs(handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb, info)
    initialize_context()
    ccall((:cusolverDnXpotrs, libcusolver()), cusolverStatus_t, (cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, Int64, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, Int64, CuPtr{Cint}), handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb, info)
end

@checked function cusolverDnXgetrf_bufferSize(handle, params, m, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    initialize_context()
    ccall((:cusolverDnXgetrf_bufferSize, libcusolver()), cusolverStatus_t, (cusolverDnHandle_t, cusolverDnParams_t, Int64, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, Ptr{Csize_t}, Ptr{Csize_t}), handle, params, m, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
end

@checked function cusolverDnXgesvdp_bufferSize(handle, params, jobz, econ, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeV, V, ldv, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    initialize_context()
    ccall((:cusolverDnXgesvdp_bufferSize, libcusolver()), cusolverStatus_t, (cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, Cint, Int64, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, Ptr{Csize_t}, Ptr{Csize_t}), handle, params, jobz, econ, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeV, V, ldv, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
end

@checked function cusolverDnXgesvd_bufferSize(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    initialize_context()
    ccall((:cusolverDnXgesvd_bufferSize, libcusolver()), cusolverStatus_t, (cusolverDnHandle_t, cusolverDnParams_t, UInt8, UInt8, Int64, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, Ptr{Csize_t}, Ptr{Csize_t}), handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
end

@checked function cusolverDnXgeqrf_bufferSize(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    initialize_context()
    ccall((:cusolverDnXgeqrf_bufferSize, libcusolver()), cusolverStatus_t, (cusolverDnHandle_t, cusolverDnParams_t, Int64, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, cudaDataType, Ptr{Csize_t}, Ptr{Csize_t}), handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
end

@checked function cusolverDnXsyevdx(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, meig64, dataTypeW, W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    initialize_context()
    ccall((:cusolverDnXsyevdx, libcusolver()), cusolverStatus_t, (cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, Int64, cudaDataType, CuPtr{Cvoid}, Int64, CuPtr{Cvoid}, CuPtr{Cvoid}, Int64, Int64, CuPtr{Int64}, cudaDataType, CuPtr{Cvoid}, cudaDataType, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid}, Csize_t, CuPtr{Cint}), handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, meig64, dataTypeW, W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
end

@checked function cusolverDnXgetrf(handle, params, m, n, dataTypeA, A, lda, ipiv, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    initialize_context()
    ccall((:cusolverDnXgetrf, libcusolver()), cusolverStatus_t, (cusolverDnHandle_t, cusolverDnParams_t, Int64, Int64, cudaDataType, CuPtr{Cvoid}, Int64, CuPtr{Int64}, cudaDataType, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid}, Csize_t, CuPtr{Cint}), handle, params, m, n, dataTypeA, A, lda, ipiv, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
end

@checked function cusolverDnXsyevd_bufferSize(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    initialize_context()
    ccall((:cusolverDnXsyevd_bufferSize, libcusolver()), cusolverStatus_t, (cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, cudaDataType, Ptr{Csize_t}, Ptr{Csize_t}), handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
end

@checked function cusolverDnXpotrf(handle, params, uplo, n, dataTypeA, A, lda, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    initialize_context()
    ccall((:cusolverDnXpotrf, libcusolver()), cusolverStatus_t, (cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid}, Csize_t, CuPtr{Cint}), handle, params, uplo, n, dataTypeA, A, lda, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
end

@checked function cusolverDnXgeqrf(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    initialize_context()
    ccall((:cusolverDnXgeqrf, libcusolver()), cusolverStatus_t, (cusolverDnHandle_t, cusolverDnParams_t, Int64, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, cudaDataType, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid}, Csize_t, CuPtr{Cint}), handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
end

@checked function cusolverDnXsyevd(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    initialize_context()
    ccall((:cusolverDnXsyevd, libcusolver()), cusolverStatus_t, (cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, cudaDataType, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid}, Csize_t, CuPtr{Cint}), handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
end

## Added in CUDA 11.2

@checked function cusolverDnXgesvdr(handle, params, jobu, jobv, m, n, k, p, niters, dataTypeA, A, lda, dataTypeSrand, Srand, dataTypeUrand, Urand, ldUrand, dataTypeVrand, Vrand, ldVrand, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_info)
    initialize_context()
    ccall((:cusolverDnXgesvdr, libcusolver), cusolverStatus_t, (cusolverDnHandle_t, cusolverDnParams_t, UInt8, UInt8, Int64, Int64, Int64, Int64, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, Csize_t, Ptr{Cvoid}, Csize_t, CuPtr{Cint}), handle, params, jobu, jobv, m, n, k, p, niters, dataTypeA, A, lda, dataTypeSrand, Srand, dataTypeUrand, Urand, ldUrand, dataTypeVrand, Vrand, ldVrand, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_info)
end

@checked function cusolverDnXgesvdr_bufferSize(handle, params, jobu, jobv, m, n, k, p, niters, dataTypeA, A, lda, dataTypeSrand, Srand, dataTypeUrand, Urand, ldUrand, dataTypeVrand, Vrand, ldVrand, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    initialize_context()
    ccall((:cusolverDnXgesvdr_bufferSize, libcusolver), cusolverStatus_t, (cusolverDnHandle_t, cusolverDnParams_t, UInt8, UInt8, Int64, Int64, Int64, Int64, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Cvoid}, Int64, cudaDataType, CuPtr{Csize_t}, Ptr{Csize_t}), handle, params, jobu, jobv, m, n, k, p, niters, dataTypeA, A, lda, dataTypeSrand, Srand, dataTypeUrand, Urand, ldUrand, dataTypeVrand, Vrand, ldVrand, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
end

## from cusolverMg.h

@checked function cusolverMgCreate(handle)
    initialize_context()
    ccall((:cusolverMgCreate, libcusolvermg()), cusolverStatus_t,
                   (Ptr{cusolverMgHandle_t},),
                   handle)
end

@checked function cusolverMgDestroy(handle)
    initialize_context()
    ccall((:cusolverMgDestroy, libcusolvermg()), cusolverStatus_t,
                   (cusolverMgHandle_t,),
                   handle)
end

@checked function cusolverMgDeviceSelect(handle, nbDevices, deviceId)
    initialize_context()
    ccall((:cusolverMgDeviceSelect, libcusolvermg()), cusolverStatus_t,
                   (cusolverMgHandle_t, Cint, Ptr{Cint}),
                   handle, nbDevices, deviceId)
end

@checked function cusolverMgCreateDeviceGrid(grid, numRowDevices, numColDevices, deviceId,
                                             mapping)
    initialize_context()
    ccall((:cusolverMgCreateDeviceGrid, libcusolvermg()), cusolverStatus_t,
                   (Ptr{cudaLibMgGrid_t}, Int32, Int32, Ptr{Int32},
                    cusolverMgGridMapping_t),
                   grid, numRowDevices, numColDevices, deviceId, mapping)
end

@checked function cusolverMgDestroyGrid(grid)
    initialize_context()
    ccall((:cusolverMgDestroyGrid, libcusolvermg()), cusolverStatus_t,
                   (cudaLibMgGrid_t,),
                   grid)
end

@checked function cusolverMgCreateMatrixDesc(desc, numRows, numCols, rowBlockSize,
                                             colBlockSize, dataType, grid)
    initialize_context()
    ccall((:cusolverMgCreateMatrixDesc, libcusolvermg()), cusolverStatus_t,
                   (Ptr{cudaLibMgMatrixDesc_t}, Int64, Int64, Int64, Int64, cudaDataType,
                    cudaLibMgGrid_t),
                   desc, numRows, numCols, rowBlockSize, colBlockSize, dataType, grid)
end

@checked function cusolverMgDestroyMatrixDesc(desc)
    initialize_context()
    ccall((:cusolverMgDestroyMatrixDesc, libcusolvermg()), cusolverStatus_t,
                   (cudaLibMgMatrixDesc_t,),
                   desc)
end

@checked function cusolverMgSyevd_bufferSize(handle, jobz, uplo, N, array_d_A, IA, JA,
                                             descrA, W, dataTypeW, computeType, lwork)
    initialize_context()
    ccall((:cusolverMgSyevd_bufferSize, libcusolvermg()), cusolverStatus_t,
                   (cusolverMgHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    Ptr{CuPtr{Cvoid}}, Cint, Cint, cudaLibMgMatrixDesc_t, Ptr{Cvoid},
                    cudaDataType, cudaDataType, Ptr{Int64}),
                   handle, jobz, uplo, N, array_d_A, IA, JA, descrA, W, dataTypeW,
                   computeType, lwork)
end

@checked function cusolverMgSyevd(handle, jobz, uplo, N, array_d_A, IA, JA, descrA, W,
                                  dataTypeW, computeType, array_d_work, lwork, info)
    initialize_context()
    ccall((:cusolverMgSyevd, libcusolvermg()), cusolverStatus_t,
                   (cusolverMgHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                    Ptr{CuPtr{Cvoid}}, Cint, Cint, cudaLibMgMatrixDesc_t, Ptr{Cvoid},
                    cudaDataType, cudaDataType, Ptr{CuPtr{Cvoid}}, Int64, Ptr{Cint}),
                   handle, jobz, uplo, N, array_d_A, IA, JA, descrA, W, dataTypeW,
                   computeType, array_d_work, lwork, info)
end

@checked function cusolverMgGetrf_bufferSize(handle, M, N, array_d_A, IA, JA, descrA,
                                             array_d_IPIV, computeType, lwork)
    initialize_context()
    ccall((:cusolverMgGetrf_bufferSize, libcusolvermg()), cusolverStatus_t,
                   (cusolverMgHandle_t, Cint, Cint, Ptr{CuPtr{Cvoid}}, Cint, Cint,
                    cudaLibMgMatrixDesc_t, Ptr{CuPtr{Cint}}, cudaDataType, Ptr{Int64}),
                   handle, M, N, array_d_A, IA, JA, descrA, array_d_IPIV, computeType,
                   lwork)
end

@checked function cusolverMgGetrf(handle, M, N, array_d_A, IA, JA, descrA, array_d_IPIV,
                                  computeType, array_d_work, lwork, info)
    initialize_context()
    ccall((:cusolverMgGetrf, libcusolvermg()), cusolverStatus_t,
                   (cusolverMgHandle_t, Cint, Cint, Ptr{CuPtr{Cvoid}}, Cint, Cint,
                    cudaLibMgMatrixDesc_t, Ptr{CuPtr{Cint}}, cudaDataType, Ptr{CuPtr{Cvoid}},
                    Int64, Ptr{Cint}),
                   handle, M, N, array_d_A, IA, JA, descrA, array_d_IPIV, computeType,
                   array_d_work, lwork, info)
end

@checked function cusolverMgGetrs_bufferSize(handle, TRANS, N, NRHS, array_d_A, IA, JA,
                                             descrA, array_d_IPIV, array_d_B, IB, JB,
                                             descrB, computeType, lwork)
    initialize_context()
    ccall((:cusolverMgGetrs_bufferSize, libcusolvermg()), cusolverStatus_t,
                   (cusolverMgHandle_t, cublasOperation_t, Cint, Cint, Ptr{CuPtr{Cvoid}},
                    Cint, Cint, cudaLibMgMatrixDesc_t, Ptr{CuPtr{Cint}}, Ptr{CuPtr{Cvoid}},
                    Cint, Cint, cudaLibMgMatrixDesc_t, cudaDataType, Ptr{Int64}),
                   handle, TRANS, N, NRHS, array_d_A, IA, JA, descrA, array_d_IPIV,
                   array_d_B, IB, JB, descrB, computeType, lwork)
end

@checked function cusolverMgGetrs(handle, TRANS, N, NRHS, array_d_A, IA, JA, descrA,
                                  array_d_IPIV, array_d_B, IB, JB, descrB, computeType,
                                  array_d_work, lwork, info)
    initialize_context()
    ccall((:cusolverMgGetrs, libcusolvermg()), cusolverStatus_t,
                   (cusolverMgHandle_t, cublasOperation_t, Cint, Cint, Ptr{CuPtr{Cvoid}},
                    Cint, Cint, cudaLibMgMatrixDesc_t, Ptr{CuPtr{Cint}}, Ptr{CuPtr{Cvoid}},
                    Cint, Cint, cudaLibMgMatrixDesc_t, cudaDataType, Ptr{CuPtr{Cvoid}},
                    Int64, Ptr{Cint}),
                   handle, TRANS, N, NRHS, array_d_A, IA, JA, descrA, array_d_IPIV,
                   array_d_B, IB, JB, descrB, computeType, array_d_work, lwork, info)
end

@checked function cusolverMgPotrf_bufferSize(handle, uplo, N, array_d_A, IA, JA, descrA,
                                             computeType, lwork)
    initialize_context()
    ccall((:cusolverMgPotrf_bufferSize, libcusolvermg()), cusolverStatus_t,
                   (cusolverMgHandle_t, cublasFillMode_t, Cint, Ptr{CuPtr{Cvoid}}, Cint,
                    Cint, cudaLibMgMatrixDesc_t, cudaDataType, Ptr{Int64}),
                   handle, uplo, N, array_d_A, IA, JA, descrA, computeType, lwork)
end

@checked function cusolverMgPotrf(handle, uplo, N, array_d_A, IA, JA, descrA, computeType,
                                  array_d_work, lwork, h_info)
    initialize_context()
    ccall((:cusolverMgPotrf, libcusolvermg()), cusolverStatus_t,
                   (cusolverMgHandle_t, cublasFillMode_t, Cint, Ptr{CuPtr{Cvoid}}, Cint,
                    Cint, cudaLibMgMatrixDesc_t, cudaDataType, Ptr{CuPtr{Cvoid}}, Int64,
                    Ptr{Cint}),
                   handle, uplo, N, array_d_A, IA, JA, descrA, computeType, array_d_work,
                   lwork, h_info)
end

@checked function cusolverMgPotrs_bufferSize(handle, uplo, n, nrhs, array_d_A, IA, JA,
                                             descrA, array_d_B, IB, JB, descrB,
                                             computeType, lwork)
    initialize_context()
    ccall((:cusolverMgPotrs_bufferSize, libcusolvermg()), cusolverStatus_t,
                   (cusolverMgHandle_t, cublasFillMode_t, Cint, Cint, Ptr{CuPtr{Cvoid}},
                    Cint, Cint, cudaLibMgMatrixDesc_t, Ptr{CuPtr{Cvoid}}, Cint, Cint,
                    cudaLibMgMatrixDesc_t, cudaDataType, Ptr{Int64}),
                   handle, uplo, n, nrhs, array_d_A, IA, JA, descrA, array_d_B, IB, JB,
                   descrB, computeType, lwork)
end

@checked function cusolverMgPotrs(handle, uplo, n, nrhs, array_d_A, IA, JA, descrA,
                                  array_d_B, IB, JB, descrB, computeType, array_d_work,
                                  lwork, h_info)
    initialize_context()
    ccall((:cusolverMgPotrs, libcusolvermg()), cusolverStatus_t,
                   (cusolverMgHandle_t, cublasFillMode_t, Cint, Cint, Ptr{CuPtr{Cvoid}},
                    Cint, Cint, cudaLibMgMatrixDesc_t, Ptr{CuPtr{Cvoid}}, Cint, Cint,
                    cudaLibMgMatrixDesc_t, cudaDataType, Ptr{CuPtr{Cvoid}}, Int64, Ptr{Cint}),
                   handle, uplo, n, nrhs, array_d_A, IA, JA, descrA, array_d_B, IB, JB,
                   descrB, computeType, array_d_work, lwork, h_info)
end

@checked function cusolverMgPotri_bufferSize(handle, uplo, N, array_d_A, IA, JA, descrA,
                                             computeType, lwork)
    initialize_context()
    ccall((:cusolverMgPotri_bufferSize, libcusolvermg()), cusolverStatus_t,
                   (cusolverMgHandle_t, cublasFillMode_t, Cint, Ptr{CuPtr{Cvoid}}, Cint,
                    Cint, cudaLibMgMatrixDesc_t, cudaDataType, Ptr{Int64}),
                   handle, uplo, N, array_d_A, IA, JA, descrA, computeType, lwork)
end

@checked function cusolverMgPotri(handle, uplo, N, array_d_A, IA, JA, descrA, computeType,
                                  array_d_work, lwork, h_info)
    initialize_context()
    ccall((:cusolverMgPotri, libcusolvermg()), cusolverStatus_t,
                   (cusolverMgHandle_t, cublasFillMode_t, Cint, Ptr{CuPtr{Cvoid}}, Cint,
                    Cint, cudaLibMgMatrixDesc_t, cudaDataType, Ptr{CuPtr{Cvoid}}, Int64,
                    Ptr{Cint}),
                   handle, uplo, N, array_d_A, IA, JA, descrA, computeType, array_d_work,
                   lwork, h_info)
end
