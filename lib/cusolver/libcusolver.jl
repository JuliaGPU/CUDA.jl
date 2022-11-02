using CEnum

# CUSOLVER uses CUDA runtime objects, which are compatible with our driver usage
const cudaStream_t = CUstream

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == CUSOLVER_STATUS_ALLOC_FAILED
        throw(OutOfGPUMemoryError())
    else
        throw(CUSOLVERError(res))
    end
end

macro check(ex, errs...)
    check = :(isequal(err, CUSOLVER_STATUS_ALLOC_FAILED))
    for err in errs
        check = :($check || isequal(err, $(esc(err))))
    end

    quote
        res = @retry_reclaim err -> $check $(esc(ex))
        if res != CUSOLVER_STATUS_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end

mutable struct cusolverDnContext end

const cusolverDnHandle_t = Ptr{cusolverDnContext}

mutable struct syevjInfo end

const syevjInfo_t = Ptr{syevjInfo}

mutable struct gesvdjInfo end

const gesvdjInfo_t = Ptr{gesvdjInfo}

mutable struct cusolverDnIRSParams end

const cusolverDnIRSParams_t = Ptr{cusolverDnIRSParams}

mutable struct cusolverDnIRSInfos end

const cusolverDnIRSInfos_t = Ptr{cusolverDnIRSInfos}

mutable struct cusolverDnParams end

const cusolverDnParams_t = Ptr{cusolverDnParams}

@cenum cusolverDnFunction_t::UInt32 begin
    CUSOLVERDN_GETRF = 0
    CUSOLVERDN_POTRF = 1
end

const cusolver_int_t = Cint

@cenum cusolverStatus_t::UInt32 begin
    CUSOLVER_STATUS_SUCCESS = 0
    CUSOLVER_STATUS_NOT_INITIALIZED = 1
    CUSOLVER_STATUS_ALLOC_FAILED = 2
    CUSOLVER_STATUS_INVALID_VALUE = 3
    CUSOLVER_STATUS_ARCH_MISMATCH = 4
    CUSOLVER_STATUS_MAPPING_ERROR = 5
    CUSOLVER_STATUS_EXECUTION_FAILED = 6
    CUSOLVER_STATUS_INTERNAL_ERROR = 7
    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8
    CUSOLVER_STATUS_NOT_SUPPORTED = 9
    CUSOLVER_STATUS_ZERO_PIVOT = 10
    CUSOLVER_STATUS_INVALID_LICENSE = 11
    CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED = 12
    CUSOLVER_STATUS_IRS_PARAMS_INVALID = 13
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC = 14
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE = 15
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER = 16
    CUSOLVER_STATUS_IRS_INTERNAL_ERROR = 20
    CUSOLVER_STATUS_IRS_NOT_SUPPORTED = 21
    CUSOLVER_STATUS_IRS_OUT_OF_RANGE = 22
    CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES = 23
    CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED = 25
    CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED = 26
    CUSOLVER_STATUS_IRS_MATRIX_SINGULAR = 30
    CUSOLVER_STATUS_INVALID_WORKSPACE = 31
end

@cenum cusolverEigType_t::UInt32 begin
    CUSOLVER_EIG_TYPE_1 = 1
    CUSOLVER_EIG_TYPE_2 = 2
    CUSOLVER_EIG_TYPE_3 = 3
end

@cenum cusolverEigMode_t::UInt32 begin
    CUSOLVER_EIG_MODE_NOVECTOR = 0
    CUSOLVER_EIG_MODE_VECTOR = 1
end

@cenum cusolverEigRange_t::UInt32 begin
    CUSOLVER_EIG_RANGE_ALL = 1001
    CUSOLVER_EIG_RANGE_I = 1002
    CUSOLVER_EIG_RANGE_V = 1003
end

@cenum cusolverNorm_t::UInt32 begin
    CUSOLVER_INF_NORM = 104
    CUSOLVER_MAX_NORM = 105
    CUSOLVER_ONE_NORM = 106
    CUSOLVER_FRO_NORM = 107
end

@cenum cusolverIRSRefinement_t::UInt32 begin
    CUSOLVER_IRS_REFINE_NOT_SET = 1100
    CUSOLVER_IRS_REFINE_NONE = 1101
    CUSOLVER_IRS_REFINE_CLASSICAL = 1102
    CUSOLVER_IRS_REFINE_CLASSICAL_GMRES = 1103
    CUSOLVER_IRS_REFINE_GMRES = 1104
    CUSOLVER_IRS_REFINE_GMRES_GMRES = 1105
    CUSOLVER_IRS_REFINE_GMRES_NOPCOND = 1106
    CUSOLVER_PREC_DD = 1150
    CUSOLVER_PREC_SS = 1151
    CUSOLVER_PREC_SHT = 1152
end

@cenum cusolverPrecType_t::UInt32 begin
    CUSOLVER_R_8I = 1201
    CUSOLVER_R_8U = 1202
    CUSOLVER_R_64F = 1203
    CUSOLVER_R_32F = 1204
    CUSOLVER_R_16F = 1205
    CUSOLVER_R_16BF = 1206
    CUSOLVER_R_TF32 = 1207
    CUSOLVER_R_AP = 1208
    CUSOLVER_C_8I = 1211
    CUSOLVER_C_8U = 1212
    CUSOLVER_C_64F = 1213
    CUSOLVER_C_32F = 1214
    CUSOLVER_C_16F = 1215
    CUSOLVER_C_16BF = 1216
    CUSOLVER_C_TF32 = 1217
    CUSOLVER_C_AP = 1218
end

@cenum cusolverAlgMode_t::UInt32 begin
    CUSOLVER_ALG_0 = 0
    CUSOLVER_ALG_1 = 1
    CUSOLVER_ALG_2 = 2
end

@cenum cusolverStorevMode_t::UInt32 begin
    CUBLAS_STOREV_COLUMNWISE = 0
    CUBLAS_STOREV_ROWWISE = 1
end

@cenum cusolverDirectMode_t::UInt32 begin
    CUBLAS_DIRECT_FORWARD = 0
    CUBLAS_DIRECT_BACKWARD = 1
end

@checked function cusolverGetProperty(type, value)
    @ccall libcusolver.cusolverGetProperty(type::libraryPropertyType,
                                           value::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverGetVersion(version)
    @ccall libcusolver.cusolverGetVersion(version::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCreate(handle)
    initialize_context()
    @ccall libcusolver.cusolverDnCreate(handle::Ptr{cusolverDnHandle_t})::cusolverStatus_t
end

@checked function cusolverDnDestroy(handle)
    initialize_context()
    @ccall libcusolver.cusolverDnDestroy(handle::cusolverDnHandle_t)::cusolverStatus_t
end

@checked function cusolverDnSetStream(handle, streamId)
    initialize_context()
    @ccall libcusolver.cusolverDnSetStream(handle::cusolverDnHandle_t,
                                           streamId::cudaStream_t)::cusolverStatus_t
end

@checked function cusolverDnGetStream(handle, streamId)
    initialize_context()
    @ccall libcusolver.cusolverDnGetStream(handle::cusolverDnHandle_t,
                                           streamId::Ptr{cudaStream_t})::cusolverStatus_t
end

@checked function cusolverDnIRSParamsCreate(params_ptr)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSParamsCreate(params_ptr::Ptr{cusolverDnIRSParams_t})::cusolverStatus_t
end

@checked function cusolverDnIRSParamsDestroy(params)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSParamsDestroy(params::cusolverDnIRSParams_t)::cusolverStatus_t
end

@checked function cusolverDnIRSParamsSetRefinementSolver(params, refinement_solver)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSParamsSetRefinementSolver(params::cusolverDnIRSParams_t,
                                                              refinement_solver::cusolverIRSRefinement_t)::cusolverStatus_t
end

@checked function cusolverDnIRSParamsSetSolverMainPrecision(params, solver_main_precision)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSParamsSetSolverMainPrecision(params::cusolverDnIRSParams_t,
                                                                 solver_main_precision::cusolverPrecType_t)::cusolverStatus_t
end

@checked function cusolverDnIRSParamsSetSolverLowestPrecision(params,
                                                              solver_lowest_precision)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSParamsSetSolverLowestPrecision(params::cusolverDnIRSParams_t,
                                                                   solver_lowest_precision::cusolverPrecType_t)::cusolverStatus_t
end

@checked function cusolverDnIRSParamsSetSolverPrecisions(params, solver_main_precision,
                                                         solver_lowest_precision)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSParamsSetSolverPrecisions(params::cusolverDnIRSParams_t,
                                                              solver_main_precision::cusolverPrecType_t,
                                                              solver_lowest_precision::cusolverPrecType_t)::cusolverStatus_t
end

@checked function cusolverDnIRSParamsSetTol(params, val)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSParamsSetTol(params::cusolverDnIRSParams_t,
                                                 val::Cdouble)::cusolverStatus_t
end

@checked function cusolverDnIRSParamsSetTolInner(params, val)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSParamsSetTolInner(params::cusolverDnIRSParams_t,
                                                      val::Cdouble)::cusolverStatus_t
end

@checked function cusolverDnIRSParamsSetMaxIters(params, maxiters)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSParamsSetMaxIters(params::cusolverDnIRSParams_t,
                                                      maxiters::cusolver_int_t)::cusolverStatus_t
end

@checked function cusolverDnIRSParamsSetMaxItersInner(params, maxiters_inner)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSParamsSetMaxItersInner(params::cusolverDnIRSParams_t,
                                                           maxiters_inner::cusolver_int_t)::cusolverStatus_t
end

@checked function cusolverDnIRSParamsGetMaxIters(params, maxiters)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSParamsGetMaxIters(params::cusolverDnIRSParams_t,
                                                      maxiters::Ptr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnIRSParamsEnableFallback(params)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSParamsEnableFallback(params::cusolverDnIRSParams_t)::cusolverStatus_t
end

@checked function cusolverDnIRSParamsDisableFallback(params)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSParamsDisableFallback(params::cusolverDnIRSParams_t)::cusolverStatus_t
end

@checked function cusolverDnIRSInfosDestroy(infos)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSInfosDestroy(infos::cusolverDnIRSInfos_t)::cusolverStatus_t
end

@checked function cusolverDnIRSInfosCreate(infos_ptr)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSInfosCreate(infos_ptr::Ptr{cusolverDnIRSInfos_t})::cusolverStatus_t
end

@checked function cusolverDnIRSInfosGetNiters(infos, niters)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSInfosGetNiters(infos::cusolverDnIRSInfos_t,
                                                   niters::Ptr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnIRSInfosGetOuterNiters(infos, outer_niters)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSInfosGetOuterNiters(infos::cusolverDnIRSInfos_t,
                                                        outer_niters::Ptr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnIRSInfosRequestResidual(infos)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSInfosRequestResidual(infos::cusolverDnIRSInfos_t)::cusolverStatus_t
end

@checked function cusolverDnIRSInfosGetResidualHistory(infos, residual_history)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSInfosGetResidualHistory(infos::cusolverDnIRSInfos_t,
                                                            residual_history::Ptr{Ptr{Cvoid}})::cusolverStatus_t
end

@checked function cusolverDnIRSInfosGetMaxIters(infos, maxiters)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSInfosGetMaxIters(infos::cusolverDnIRSInfos_t,
                                                     maxiters::Ptr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnZZgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnZZgesv(handle::cusolverDnHandle_t, n::cusolver_int_t,
                                        nrhs::cusolver_int_t, dA::CuPtr{cuDoubleComplex},
                                        ldda::cusolver_int_t, dipiv::CuPtr{cusolver_int_t},
                                        dB::CuPtr{cuDoubleComplex}, lddb::cusolver_int_t,
                                        dX::CuPtr{cuDoubleComplex}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnZCgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnZCgesv(handle::cusolverDnHandle_t, n::cusolver_int_t,
                                        nrhs::cusolver_int_t, dA::CuPtr{cuDoubleComplex},
                                        ldda::cusolver_int_t, dipiv::CuPtr{cusolver_int_t},
                                        dB::CuPtr{cuDoubleComplex}, lddb::cusolver_int_t,
                                        dX::CuPtr{cuDoubleComplex}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnZKgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnZKgesv(handle::cusolverDnHandle_t, n::cusolver_int_t,
                                        nrhs::cusolver_int_t, dA::CuPtr{cuDoubleComplex},
                                        ldda::cusolver_int_t, dipiv::CuPtr{cusolver_int_t},
                                        dB::CuPtr{cuDoubleComplex}, lddb::cusolver_int_t,
                                        dX::CuPtr{cuDoubleComplex}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnZEgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnZEgesv(handle::cusolverDnHandle_t, n::cusolver_int_t,
                                        nrhs::cusolver_int_t, dA::CuPtr{cuDoubleComplex},
                                        ldda::cusolver_int_t, dipiv::CuPtr{cusolver_int_t},
                                        dB::CuPtr{cuDoubleComplex}, lddb::cusolver_int_t,
                                        dX::CuPtr{cuDoubleComplex}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnZYgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnZYgesv(handle::cusolverDnHandle_t, n::cusolver_int_t,
                                        nrhs::cusolver_int_t, dA::CuPtr{cuDoubleComplex},
                                        ldda::cusolver_int_t, dipiv::CuPtr{cusolver_int_t},
                                        dB::CuPtr{cuDoubleComplex}, lddb::cusolver_int_t,
                                        dX::CuPtr{cuDoubleComplex}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnCCgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnCCgesv(handle::cusolverDnHandle_t, n::cusolver_int_t,
                                        nrhs::cusolver_int_t, dA::CuPtr{cuComplex},
                                        ldda::cusolver_int_t, dipiv::CuPtr{cusolver_int_t},
                                        dB::CuPtr{cuComplex}, lddb::cusolver_int_t,
                                        dX::CuPtr{cuComplex}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnCEgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnCEgesv(handle::cusolverDnHandle_t, n::cusolver_int_t,
                                        nrhs::cusolver_int_t, dA::CuPtr{cuComplex},
                                        ldda::cusolver_int_t, dipiv::CuPtr{cusolver_int_t},
                                        dB::CuPtr{cuComplex}, lddb::cusolver_int_t,
                                        dX::CuPtr{cuComplex}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnCKgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnCKgesv(handle::cusolverDnHandle_t, n::cusolver_int_t,
                                        nrhs::cusolver_int_t, dA::CuPtr{cuComplex},
                                        ldda::cusolver_int_t, dipiv::CuPtr{cusolver_int_t},
                                        dB::CuPtr{cuComplex}, lddb::cusolver_int_t,
                                        dX::CuPtr{cuComplex}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnCYgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnCYgesv(handle::cusolverDnHandle_t, n::cusolver_int_t,
                                        nrhs::cusolver_int_t, dA::CuPtr{cuComplex},
                                        ldda::cusolver_int_t, dipiv::CuPtr{cusolver_int_t},
                                        dB::CuPtr{cuComplex}, lddb::cusolver_int_t,
                                        dX::CuPtr{cuComplex}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnDDgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnDDgesv(handle::cusolverDnHandle_t, n::cusolver_int_t,
                                        nrhs::cusolver_int_t, dA::CuPtr{Cdouble},
                                        ldda::cusolver_int_t, dipiv::CuPtr{cusolver_int_t},
                                        dB::CuPtr{Cdouble}, lddb::cusolver_int_t,
                                        dX::CuPtr{Cdouble}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnDSgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnDSgesv(handle::cusolverDnHandle_t, n::cusolver_int_t,
                                        nrhs::cusolver_int_t, dA::CuPtr{Cdouble},
                                        ldda::cusolver_int_t, dipiv::CuPtr{cusolver_int_t},
                                        dB::CuPtr{Cdouble}, lddb::cusolver_int_t,
                                        dX::CuPtr{Cdouble}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnDHgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnDHgesv(handle::cusolverDnHandle_t, n::cusolver_int_t,
                                        nrhs::cusolver_int_t, dA::CuPtr{Cdouble},
                                        ldda::cusolver_int_t, dipiv::CuPtr{cusolver_int_t},
                                        dB::CuPtr{Cdouble}, lddb::cusolver_int_t,
                                        dX::CuPtr{Cdouble}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnDBgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnDBgesv(handle::cusolverDnHandle_t, n::cusolver_int_t,
                                        nrhs::cusolver_int_t, dA::CuPtr{Cdouble},
                                        ldda::cusolver_int_t, dipiv::CuPtr{cusolver_int_t},
                                        dB::CuPtr{Cdouble}, lddb::cusolver_int_t,
                                        dX::CuPtr{Cdouble}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnDXgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnDXgesv(handle::cusolverDnHandle_t, n::cusolver_int_t,
                                        nrhs::cusolver_int_t, dA::CuPtr{Cdouble},
                                        ldda::cusolver_int_t, dipiv::CuPtr{cusolver_int_t},
                                        dB::CuPtr{Cdouble}, lddb::cusolver_int_t,
                                        dX::CuPtr{Cdouble}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnSSgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnSSgesv(handle::cusolverDnHandle_t, n::cusolver_int_t,
                                        nrhs::cusolver_int_t, dA::CuPtr{Cfloat},
                                        ldda::cusolver_int_t, dipiv::CuPtr{cusolver_int_t},
                                        dB::CuPtr{Cfloat}, lddb::cusolver_int_t,
                                        dX::CuPtr{Cfloat}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnSHgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnSHgesv(handle::cusolverDnHandle_t, n::cusolver_int_t,
                                        nrhs::cusolver_int_t, dA::CuPtr{Cfloat},
                                        ldda::cusolver_int_t, dipiv::CuPtr{cusolver_int_t},
                                        dB::CuPtr{Cfloat}, lddb::cusolver_int_t,
                                        dX::CuPtr{Cfloat}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnSBgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnSBgesv(handle::cusolverDnHandle_t, n::cusolver_int_t,
                                        nrhs::cusolver_int_t, dA::CuPtr{Cfloat},
                                        ldda::cusolver_int_t, dipiv::CuPtr{cusolver_int_t},
                                        dB::CuPtr{Cfloat}, lddb::cusolver_int_t,
                                        dX::CuPtr{Cfloat}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnSXgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnSXgesv(handle::cusolverDnHandle_t, n::cusolver_int_t,
                                        nrhs::cusolver_int_t, dA::CuPtr{Cfloat},
                                        ldda::cusolver_int_t, dipiv::CuPtr{cusolver_int_t},
                                        dB::CuPtr{Cfloat}, lddb::cusolver_int_t,
                                        dX::CuPtr{Cfloat}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnZZgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnZZgesv_bufferSize(handle::cusolverDnHandle_t,
                                                   n::cusolver_int_t, nrhs::cusolver_int_t,
                                                   dA::CuPtr{cuDoubleComplex},
                                                   ldda::cusolver_int_t,
                                                   dipiv::CuPtr{cusolver_int_t},
                                                   dB::CuPtr{cuDoubleComplex},
                                                   lddb::cusolver_int_t,
                                                   dX::CuPtr{cuDoubleComplex},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnZCgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnZCgesv_bufferSize(handle::cusolverDnHandle_t,
                                                   n::cusolver_int_t, nrhs::cusolver_int_t,
                                                   dA::CuPtr{cuDoubleComplex},
                                                   ldda::cusolver_int_t,
                                                   dipiv::CuPtr{cusolver_int_t},
                                                   dB::CuPtr{cuDoubleComplex},
                                                   lddb::cusolver_int_t,
                                                   dX::CuPtr{cuDoubleComplex},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnZKgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnZKgesv_bufferSize(handle::cusolverDnHandle_t,
                                                   n::cusolver_int_t, nrhs::cusolver_int_t,
                                                   dA::CuPtr{cuDoubleComplex},
                                                   ldda::cusolver_int_t,
                                                   dipiv::CuPtr{cusolver_int_t},
                                                   dB::CuPtr{cuDoubleComplex},
                                                   lddb::cusolver_int_t,
                                                   dX::CuPtr{cuDoubleComplex},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnZEgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnZEgesv_bufferSize(handle::cusolverDnHandle_t,
                                                   n::cusolver_int_t, nrhs::cusolver_int_t,
                                                   dA::CuPtr{cuDoubleComplex},
                                                   ldda::cusolver_int_t,
                                                   dipiv::CuPtr{cusolver_int_t},
                                                   dB::CuPtr{cuDoubleComplex},
                                                   lddb::cusolver_int_t,
                                                   dX::CuPtr{cuDoubleComplex},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnZYgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnZYgesv_bufferSize(handle::cusolverDnHandle_t,
                                                   n::cusolver_int_t, nrhs::cusolver_int_t,
                                                   dA::CuPtr{cuDoubleComplex},
                                                   ldda::cusolver_int_t,
                                                   dipiv::CuPtr{cusolver_int_t},
                                                   dB::CuPtr{cuDoubleComplex},
                                                   lddb::cusolver_int_t,
                                                   dX::CuPtr{cuDoubleComplex},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnCCgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnCCgesv_bufferSize(handle::cusolverDnHandle_t,
                                                   n::cusolver_int_t, nrhs::cusolver_int_t,
                                                   dA::CuPtr{cuComplex},
                                                   ldda::cusolver_int_t,
                                                   dipiv::CuPtr{cusolver_int_t},
                                                   dB::CuPtr{cuComplex},
                                                   lddb::cusolver_int_t,
                                                   dX::CuPtr{cuComplex},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnCKgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnCKgesv_bufferSize(handle::cusolverDnHandle_t,
                                                   n::cusolver_int_t, nrhs::cusolver_int_t,
                                                   dA::CuPtr{cuComplex},
                                                   ldda::cusolver_int_t,
                                                   dipiv::CuPtr{cusolver_int_t},
                                                   dB::CuPtr{cuComplex},
                                                   lddb::cusolver_int_t,
                                                   dX::CuPtr{cuComplex},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnCEgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnCEgesv_bufferSize(handle::cusolverDnHandle_t,
                                                   n::cusolver_int_t, nrhs::cusolver_int_t,
                                                   dA::CuPtr{cuComplex},
                                                   ldda::cusolver_int_t,
                                                   dipiv::CuPtr{cusolver_int_t},
                                                   dB::CuPtr{cuComplex},
                                                   lddb::cusolver_int_t,
                                                   dX::CuPtr{cuComplex},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnCYgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnCYgesv_bufferSize(handle::cusolverDnHandle_t,
                                                   n::cusolver_int_t, nrhs::cusolver_int_t,
                                                   dA::CuPtr{cuComplex},
                                                   ldda::cusolver_int_t,
                                                   dipiv::CuPtr{cusolver_int_t},
                                                   dB::CuPtr{cuComplex},
                                                   lddb::cusolver_int_t,
                                                   dX::CuPtr{cuComplex},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnDDgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnDDgesv_bufferSize(handle::cusolverDnHandle_t,
                                                   n::cusolver_int_t, nrhs::cusolver_int_t,
                                                   dA::CuPtr{Cdouble}, ldda::cusolver_int_t,
                                                   dipiv::CuPtr{cusolver_int_t},
                                                   dB::CuPtr{Cdouble}, lddb::cusolver_int_t,
                                                   dX::CuPtr{Cdouble}, lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnDSgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnDSgesv_bufferSize(handle::cusolverDnHandle_t,
                                                   n::cusolver_int_t, nrhs::cusolver_int_t,
                                                   dA::CuPtr{Cdouble}, ldda::cusolver_int_t,
                                                   dipiv::CuPtr{cusolver_int_t},
                                                   dB::CuPtr{Cdouble}, lddb::cusolver_int_t,
                                                   dX::CuPtr{Cdouble}, lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnDHgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnDHgesv_bufferSize(handle::cusolverDnHandle_t,
                                                   n::cusolver_int_t, nrhs::cusolver_int_t,
                                                   dA::CuPtr{Cdouble}, ldda::cusolver_int_t,
                                                   dipiv::CuPtr{cusolver_int_t},
                                                   dB::CuPtr{Cdouble}, lddb::cusolver_int_t,
                                                   dX::CuPtr{Cdouble}, lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnDBgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnDBgesv_bufferSize(handle::cusolverDnHandle_t,
                                                   n::cusolver_int_t, nrhs::cusolver_int_t,
                                                   dA::CuPtr{Cdouble}, ldda::cusolver_int_t,
                                                   dipiv::CuPtr{cusolver_int_t},
                                                   dB::CuPtr{Cdouble}, lddb::cusolver_int_t,
                                                   dX::CuPtr{Cdouble}, lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnDXgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnDXgesv_bufferSize(handle::cusolverDnHandle_t,
                                                   n::cusolver_int_t, nrhs::cusolver_int_t,
                                                   dA::CuPtr{Cdouble}, ldda::cusolver_int_t,
                                                   dipiv::CuPtr{cusolver_int_t},
                                                   dB::CuPtr{Cdouble}, lddb::cusolver_int_t,
                                                   dX::CuPtr{Cdouble}, lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnSSgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnSSgesv_bufferSize(handle::cusolverDnHandle_t,
                                                   n::cusolver_int_t, nrhs::cusolver_int_t,
                                                   dA::CuPtr{Cfloat}, ldda::cusolver_int_t,
                                                   dipiv::CuPtr{cusolver_int_t},
                                                   dB::CuPtr{Cfloat}, lddb::cusolver_int_t,
                                                   dX::CuPtr{Cfloat}, lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnSHgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnSHgesv_bufferSize(handle::cusolverDnHandle_t,
                                                   n::cusolver_int_t, nrhs::cusolver_int_t,
                                                   dA::CuPtr{Cfloat}, ldda::cusolver_int_t,
                                                   dipiv::CuPtr{cusolver_int_t},
                                                   dB::CuPtr{Cfloat}, lddb::cusolver_int_t,
                                                   dX::CuPtr{Cfloat}, lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnSBgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnSBgesv_bufferSize(handle::cusolverDnHandle_t,
                                                   n::cusolver_int_t, nrhs::cusolver_int_t,
                                                   dA::CuPtr{Cfloat}, ldda::cusolver_int_t,
                                                   dipiv::CuPtr{cusolver_int_t},
                                                   dB::CuPtr{Cfloat}, lddb::cusolver_int_t,
                                                   dX::CuPtr{Cfloat}, lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnSXgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                                              dX, lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnSXgesv_bufferSize(handle::cusolverDnHandle_t,
                                                   n::cusolver_int_t, nrhs::cusolver_int_t,
                                                   dA::CuPtr{Cfloat}, ldda::cusolver_int_t,
                                                   dipiv::CuPtr{cusolver_int_t},
                                                   dB::CuPtr{Cfloat}, lddb::cusolver_int_t,
                                                   dX::CuPtr{Cfloat}, lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnZZgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnZZgels(handle::cusolverDnHandle_t, m::cusolver_int_t,
                                        n::cusolver_int_t, nrhs::cusolver_int_t,
                                        dA::CuPtr{cuDoubleComplex}, ldda::cusolver_int_t,
                                        dB::CuPtr{cuDoubleComplex}, lddb::cusolver_int_t,
                                        dX::CuPtr{cuDoubleComplex}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnZCgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnZCgels(handle::cusolverDnHandle_t, m::cusolver_int_t,
                                        n::cusolver_int_t, nrhs::cusolver_int_t,
                                        dA::CuPtr{cuDoubleComplex}, ldda::cusolver_int_t,
                                        dB::CuPtr{cuDoubleComplex}, lddb::cusolver_int_t,
                                        dX::CuPtr{cuDoubleComplex}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnZKgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnZKgels(handle::cusolverDnHandle_t, m::cusolver_int_t,
                                        n::cusolver_int_t, nrhs::cusolver_int_t,
                                        dA::CuPtr{cuDoubleComplex}, ldda::cusolver_int_t,
                                        dB::CuPtr{cuDoubleComplex}, lddb::cusolver_int_t,
                                        dX::CuPtr{cuDoubleComplex}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnZEgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnZEgels(handle::cusolverDnHandle_t, m::cusolver_int_t,
                                        n::cusolver_int_t, nrhs::cusolver_int_t,
                                        dA::CuPtr{cuDoubleComplex}, ldda::cusolver_int_t,
                                        dB::CuPtr{cuDoubleComplex}, lddb::cusolver_int_t,
                                        dX::CuPtr{cuDoubleComplex}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnZYgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnZYgels(handle::cusolverDnHandle_t, m::cusolver_int_t,
                                        n::cusolver_int_t, nrhs::cusolver_int_t,
                                        dA::CuPtr{cuDoubleComplex}, ldda::cusolver_int_t,
                                        dB::CuPtr{cuDoubleComplex}, lddb::cusolver_int_t,
                                        dX::CuPtr{cuDoubleComplex}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnCCgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnCCgels(handle::cusolverDnHandle_t, m::cusolver_int_t,
                                        n::cusolver_int_t, nrhs::cusolver_int_t,
                                        dA::CuPtr{cuComplex}, ldda::cusolver_int_t,
                                        dB::CuPtr{cuComplex}, lddb::cusolver_int_t,
                                        dX::CuPtr{cuComplex}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnCKgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnCKgels(handle::cusolverDnHandle_t, m::cusolver_int_t,
                                        n::cusolver_int_t, nrhs::cusolver_int_t,
                                        dA::CuPtr{cuComplex}, ldda::cusolver_int_t,
                                        dB::CuPtr{cuComplex}, lddb::cusolver_int_t,
                                        dX::CuPtr{cuComplex}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnCEgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnCEgels(handle::cusolverDnHandle_t, m::cusolver_int_t,
                                        n::cusolver_int_t, nrhs::cusolver_int_t,
                                        dA::CuPtr{cuComplex}, ldda::cusolver_int_t,
                                        dB::CuPtr{cuComplex}, lddb::cusolver_int_t,
                                        dX::CuPtr{cuComplex}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnCYgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnCYgels(handle::cusolverDnHandle_t, m::cusolver_int_t,
                                        n::cusolver_int_t, nrhs::cusolver_int_t,
                                        dA::CuPtr{cuComplex}, ldda::cusolver_int_t,
                                        dB::CuPtr{cuComplex}, lddb::cusolver_int_t,
                                        dX::CuPtr{cuComplex}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnDDgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnDDgels(handle::cusolverDnHandle_t, m::cusolver_int_t,
                                        n::cusolver_int_t, nrhs::cusolver_int_t,
                                        dA::CuPtr{Cdouble}, ldda::cusolver_int_t,
                                        dB::CuPtr{Cdouble}, lddb::cusolver_int_t,
                                        dX::CuPtr{Cdouble}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnDSgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnDSgels(handle::cusolverDnHandle_t, m::cusolver_int_t,
                                        n::cusolver_int_t, nrhs::cusolver_int_t,
                                        dA::CuPtr{Cdouble}, ldda::cusolver_int_t,
                                        dB::CuPtr{Cdouble}, lddb::cusolver_int_t,
                                        dX::CuPtr{Cdouble}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnDHgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnDHgels(handle::cusolverDnHandle_t, m::cusolver_int_t,
                                        n::cusolver_int_t, nrhs::cusolver_int_t,
                                        dA::CuPtr{Cdouble}, ldda::cusolver_int_t,
                                        dB::CuPtr{Cdouble}, lddb::cusolver_int_t,
                                        dX::CuPtr{Cdouble}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnDBgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnDBgels(handle::cusolverDnHandle_t, m::cusolver_int_t,
                                        n::cusolver_int_t, nrhs::cusolver_int_t,
                                        dA::CuPtr{Cdouble}, ldda::cusolver_int_t,
                                        dB::CuPtr{Cdouble}, lddb::cusolver_int_t,
                                        dX::CuPtr{Cdouble}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnDXgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnDXgels(handle::cusolverDnHandle_t, m::cusolver_int_t,
                                        n::cusolver_int_t, nrhs::cusolver_int_t,
                                        dA::CuPtr{Cdouble}, ldda::cusolver_int_t,
                                        dB::CuPtr{Cdouble}, lddb::cusolver_int_t,
                                        dX::CuPtr{Cdouble}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnSSgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnSSgels(handle::cusolverDnHandle_t, m::cusolver_int_t,
                                        n::cusolver_int_t, nrhs::cusolver_int_t,
                                        dA::CuPtr{Cfloat}, ldda::cusolver_int_t,
                                        dB::CuPtr{Cfloat}, lddb::cusolver_int_t,
                                        dX::CuPtr{Cfloat}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnSHgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnSHgels(handle::cusolverDnHandle_t, m::cusolver_int_t,
                                        n::cusolver_int_t, nrhs::cusolver_int_t,
                                        dA::CuPtr{Cfloat}, ldda::cusolver_int_t,
                                        dB::CuPtr{Cfloat}, lddb::cusolver_int_t,
                                        dX::CuPtr{Cfloat}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnSBgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnSBgels(handle::cusolverDnHandle_t, m::cusolver_int_t,
                                        n::cusolver_int_t, nrhs::cusolver_int_t,
                                        dA::CuPtr{Cfloat}, ldda::cusolver_int_t,
                                        dB::CuPtr{Cfloat}, lddb::cusolver_int_t,
                                        dX::CuPtr{Cfloat}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnSXgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx,
                                   dWorkspace, lwork_bytes, iter, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnSXgels(handle::cusolverDnHandle_t, m::cusolver_int_t,
                                        n::cusolver_int_t, nrhs::cusolver_int_t,
                                        dA::CuPtr{Cfloat}, ldda::cusolver_int_t,
                                        dB::CuPtr{Cfloat}, lddb::cusolver_int_t,
                                        dX::CuPtr{Cfloat}, lddx::cusolver_int_t,
                                        dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                        iter::Ptr{cusolver_int_t},
                                        d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnZZgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnZZgels_bufferSize(handle::cusolverDnHandle_t,
                                                   m::cusolver_int_t, n::cusolver_int_t,
                                                   nrhs::cusolver_int_t,
                                                   dA::CuPtr{cuDoubleComplex},
                                                   ldda::cusolver_int_t,
                                                   dB::CuPtr{cuDoubleComplex},
                                                   lddb::cusolver_int_t,
                                                   dX::CuPtr{cuDoubleComplex},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnZCgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnZCgels_bufferSize(handle::cusolverDnHandle_t,
                                                   m::cusolver_int_t, n::cusolver_int_t,
                                                   nrhs::cusolver_int_t,
                                                   dA::CuPtr{cuDoubleComplex},
                                                   ldda::cusolver_int_t,
                                                   dB::CuPtr{cuDoubleComplex},
                                                   lddb::cusolver_int_t,
                                                   dX::CuPtr{cuDoubleComplex},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnZKgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnZKgels_bufferSize(handle::cusolverDnHandle_t,
                                                   m::cusolver_int_t, n::cusolver_int_t,
                                                   nrhs::cusolver_int_t,
                                                   dA::CuPtr{cuDoubleComplex},
                                                   ldda::cusolver_int_t,
                                                   dB::CuPtr{cuDoubleComplex},
                                                   lddb::cusolver_int_t,
                                                   dX::CuPtr{cuDoubleComplex},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnZEgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnZEgels_bufferSize(handle::cusolverDnHandle_t,
                                                   m::cusolver_int_t, n::cusolver_int_t,
                                                   nrhs::cusolver_int_t,
                                                   dA::CuPtr{cuDoubleComplex},
                                                   ldda::cusolver_int_t,
                                                   dB::CuPtr{cuDoubleComplex},
                                                   lddb::cusolver_int_t,
                                                   dX::CuPtr{cuDoubleComplex},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnZYgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnZYgels_bufferSize(handle::cusolverDnHandle_t,
                                                   m::cusolver_int_t, n::cusolver_int_t,
                                                   nrhs::cusolver_int_t,
                                                   dA::CuPtr{cuDoubleComplex},
                                                   ldda::cusolver_int_t,
                                                   dB::CuPtr{cuDoubleComplex},
                                                   lddb::cusolver_int_t,
                                                   dX::CuPtr{cuDoubleComplex},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnCCgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnCCgels_bufferSize(handle::cusolverDnHandle_t,
                                                   m::cusolver_int_t, n::cusolver_int_t,
                                                   nrhs::cusolver_int_t,
                                                   dA::CuPtr{cuComplex},
                                                   ldda::cusolver_int_t,
                                                   dB::CuPtr{cuComplex},
                                                   lddb::cusolver_int_t,
                                                   dX::CuPtr{cuComplex},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnCKgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnCKgels_bufferSize(handle::cusolverDnHandle_t,
                                                   m::cusolver_int_t, n::cusolver_int_t,
                                                   nrhs::cusolver_int_t,
                                                   dA::CuPtr{cuComplex},
                                                   ldda::cusolver_int_t,
                                                   dB::CuPtr{cuComplex},
                                                   lddb::cusolver_int_t,
                                                   dX::CuPtr{cuComplex},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnCEgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnCEgels_bufferSize(handle::cusolverDnHandle_t,
                                                   m::cusolver_int_t, n::cusolver_int_t,
                                                   nrhs::cusolver_int_t,
                                                   dA::CuPtr{cuComplex},
                                                   ldda::cusolver_int_t,
                                                   dB::CuPtr{cuComplex},
                                                   lddb::cusolver_int_t,
                                                   dX::CuPtr{cuComplex},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnCYgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnCYgels_bufferSize(handle::cusolverDnHandle_t,
                                                   m::cusolver_int_t, n::cusolver_int_t,
                                                   nrhs::cusolver_int_t,
                                                   dA::CuPtr{cuComplex},
                                                   ldda::cusolver_int_t,
                                                   dB::CuPtr{cuComplex},
                                                   lddb::cusolver_int_t,
                                                   dX::CuPtr{cuComplex},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnDDgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnDDgels_bufferSize(handle::cusolverDnHandle_t,
                                                   m::cusolver_int_t, n::cusolver_int_t,
                                                   nrhs::cusolver_int_t, dA::CuPtr{Cdouble},
                                                   ldda::cusolver_int_t, dB::CuPtr{Cdouble},
                                                   lddb::cusolver_int_t, dX::CuPtr{Cdouble},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnDSgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnDSgels_bufferSize(handle::cusolverDnHandle_t,
                                                   m::cusolver_int_t, n::cusolver_int_t,
                                                   nrhs::cusolver_int_t, dA::CuPtr{Cdouble},
                                                   ldda::cusolver_int_t, dB::CuPtr{Cdouble},
                                                   lddb::cusolver_int_t, dX::CuPtr{Cdouble},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnDHgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnDHgels_bufferSize(handle::cusolverDnHandle_t,
                                                   m::cusolver_int_t, n::cusolver_int_t,
                                                   nrhs::cusolver_int_t, dA::CuPtr{Cdouble},
                                                   ldda::cusolver_int_t, dB::CuPtr{Cdouble},
                                                   lddb::cusolver_int_t, dX::CuPtr{Cdouble},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnDBgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnDBgels_bufferSize(handle::cusolverDnHandle_t,
                                                   m::cusolver_int_t, n::cusolver_int_t,
                                                   nrhs::cusolver_int_t, dA::CuPtr{Cdouble},
                                                   ldda::cusolver_int_t, dB::CuPtr{Cdouble},
                                                   lddb::cusolver_int_t, dX::CuPtr{Cdouble},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnDXgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnDXgels_bufferSize(handle::cusolverDnHandle_t,
                                                   m::cusolver_int_t, n::cusolver_int_t,
                                                   nrhs::cusolver_int_t, dA::CuPtr{Cdouble},
                                                   ldda::cusolver_int_t, dB::CuPtr{Cdouble},
                                                   lddb::cusolver_int_t, dX::CuPtr{Cdouble},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnSSgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnSSgels_bufferSize(handle::cusolverDnHandle_t,
                                                   m::cusolver_int_t, n::cusolver_int_t,
                                                   nrhs::cusolver_int_t, dA::CuPtr{Cfloat},
                                                   ldda::cusolver_int_t, dB::CuPtr{Cfloat},
                                                   lddb::cusolver_int_t, dX::CuPtr{Cfloat},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnSHgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnSHgels_bufferSize(handle::cusolverDnHandle_t,
                                                   m::cusolver_int_t, n::cusolver_int_t,
                                                   nrhs::cusolver_int_t, dA::CuPtr{Cfloat},
                                                   ldda::cusolver_int_t, dB::CuPtr{Cfloat},
                                                   lddb::cusolver_int_t, dX::CuPtr{Cfloat},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnSBgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnSBgels_bufferSize(handle::cusolverDnHandle_t,
                                                   m::cusolver_int_t, n::cusolver_int_t,
                                                   nrhs::cusolver_int_t, dA::CuPtr{Cfloat},
                                                   ldda::cusolver_int_t, dB::CuPtr{Cfloat},
                                                   lddb::cusolver_int_t, dX::CuPtr{Cfloat},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnSXgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                                              lddx, dWorkspace, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnSXgels_bufferSize(handle::cusolverDnHandle_t,
                                                   m::cusolver_int_t, n::cusolver_int_t,
                                                   nrhs::cusolver_int_t, dA::CuPtr{Cfloat},
                                                   ldda::cusolver_int_t, dB::CuPtr{Cfloat},
                                                   lddb::cusolver_int_t, dX::CuPtr{Cfloat},
                                                   lddx::cusolver_int_t,
                                                   dWorkspace::CuPtr{Cvoid},
                                                   lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnIRSXgesv(handle, gesv_irs_params, gesv_irs_infos, n, nrhs, dA,
                                     ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes,
                                     niters, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSXgesv(handle::cusolverDnHandle_t,
                                          gesv_irs_params::cusolverDnIRSParams_t,
                                          gesv_irs_infos::cusolverDnIRSInfos_t,
                                          n::cusolver_int_t, nrhs::cusolver_int_t,
                                          dA::CuPtr{Cvoid}, ldda::cusolver_int_t,
                                          dB::CuPtr{Cvoid}, lddb::cusolver_int_t,
                                          dX::CuPtr{Cvoid}, lddx::cusolver_int_t,
                                          dWorkspace::CuPtr{Cvoid}, lwork_bytes::Csize_t,
                                          niters::Ptr{cusolver_int_t},
                                          d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnIRSXgesv_bufferSize(handle, params, n, nrhs, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSXgesv_bufferSize(handle::cusolverDnHandle_t,
                                                     params::cusolverDnIRSParams_t,
                                                     n::cusolver_int_t,
                                                     nrhs::cusolver_int_t,
                                                     lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnIRSXgels(handle, gels_irs_params, gels_irs_infos, m, n, nrhs,
                                     dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes,
                                     niters, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSXgels(handle::cusolverDnHandle_t,
                                          gels_irs_params::cusolverDnIRSParams_t,
                                          gels_irs_infos::cusolverDnIRSInfos_t,
                                          m::cusolver_int_t, n::cusolver_int_t,
                                          nrhs::cusolver_int_t, dA::CuPtr{Cvoid},
                                          ldda::cusolver_int_t, dB::CuPtr{Cvoid},
                                          lddb::cusolver_int_t, dX::CuPtr{Cvoid},
                                          lddx::cusolver_int_t, dWorkspace::CuPtr{Cvoid},
                                          lwork_bytes::Csize_t, niters::Ptr{cusolver_int_t},
                                          d_info::CuPtr{cusolver_int_t})::cusolverStatus_t
end

@checked function cusolverDnIRSXgels_bufferSize(handle, params, m, n, nrhs, lwork_bytes)
    initialize_context()
    @ccall libcusolver.cusolverDnIRSXgels_bufferSize(handle::cusolverDnHandle_t,
                                                     params::cusolverDnIRSParams_t,
                                                     m::cusolver_int_t, n::cusolver_int_t,
                                                     nrhs::cusolver_int_t,
                                                     lwork_bytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnSpotrf_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cfloat}, lda::Cint,
                                                   Lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnDpotrf_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cdouble}, lda::Cint,
                                                   Lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnCpotrf_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuComplex}, lda::Cint,
                                                   Lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnZpotrf_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuDoubleComplex}, lda::Cint,
                                                   Lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnSpotrf(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{Cfloat}, lda::Cint,
                                        Workspace::CuPtr{Cfloat}, Lwork::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnDpotrf(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{Cdouble}, lda::Cint,
                                        Workspace::CuPtr{Cdouble}, Lwork::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnCpotrf(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{cuComplex}, lda::Cint,
                                        Workspace::CuPtr{cuComplex}, Lwork::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnZpotrf(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{cuDoubleComplex}, lda::Cint,
                                        Workspace::CuPtr{cuDoubleComplex}, Lwork::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnSpotrs(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, nrhs::Cint, A::CuPtr{Cfloat}, lda::Cint,
                                        B::CuPtr{Cfloat}, ldb::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnDpotrs(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, nrhs::Cint, A::CuPtr{Cdouble}, lda::Cint,
                                        B::CuPtr{Cdouble}, ldb::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnCpotrs(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, nrhs::Cint, A::CuPtr{cuComplex}, lda::Cint,
                                        B::CuPtr{cuComplex}, ldb::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnZpotrs(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, nrhs::Cint, A::CuPtr{cuDoubleComplex},
                                        lda::Cint, B::CuPtr{cuDoubleComplex}, ldb::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSpotrfBatched(handle, uplo, n, Aarray, lda, infoArray,
                                          batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnSpotrfBatched(handle::cusolverDnHandle_t,
                                               uplo::cublasFillMode_t, n::Cint,
                                               Aarray::CuPtr{Ptr{Cfloat}}, lda::Cint,
                                               infoArray::CuPtr{Cint},
                                               batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnDpotrfBatched(handle, uplo, n, Aarray, lda, infoArray,
                                          batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnDpotrfBatched(handle::cusolverDnHandle_t,
                                               uplo::cublasFillMode_t, n::Cint,
                                               Aarray::CuPtr{Ptr{Cdouble}}, lda::Cint,
                                               infoArray::CuPtr{Cint},
                                               batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnCpotrfBatched(handle, uplo, n, Aarray, lda, infoArray,
                                          batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnCpotrfBatched(handle::cusolverDnHandle_t,
                                               uplo::cublasFillMode_t, n::Cint,
                                               Aarray::CuPtr{Ptr{cuComplex}}, lda::Cint,
                                               infoArray::CuPtr{Cint},
                                               batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnZpotrfBatched(handle, uplo, n, Aarray, lda, infoArray,
                                          batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnZpotrfBatched(handle::cusolverDnHandle_t,
                                               uplo::cublasFillMode_t, n::Cint,
                                               Aarray::CuPtr{Ptr{cuDoubleComplex}},
                                               lda::Cint, infoArray::CuPtr{Cint},
                                               batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnSpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info,
                                          batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnSpotrsBatched(handle::cusolverDnHandle_t,
                                               uplo::cublasFillMode_t, n::Cint, nrhs::Cint,
                                               A::CuPtr{Ptr{Cfloat}}, lda::Cint,
                                               B::CuPtr{Ptr{Cfloat}}, ldb::Cint,
                                               d_info::CuPtr{Cint},
                                               batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnDpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info,
                                          batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnDpotrsBatched(handle::cusolverDnHandle_t,
                                               uplo::cublasFillMode_t, n::Cint, nrhs::Cint,
                                               A::CuPtr{Ptr{Cdouble}}, lda::Cint,
                                               B::CuPtr{Ptr{Cdouble}}, ldb::Cint,
                                               d_info::CuPtr{Cint},
                                               batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnCpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info,
                                          batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnCpotrsBatched(handle::cusolverDnHandle_t,
                                               uplo::cublasFillMode_t, n::Cint, nrhs::Cint,
                                               A::CuPtr{Ptr{cuComplex}}, lda::Cint,
                                               B::CuPtr{Ptr{cuComplex}}, ldb::Cint,
                                               d_info::CuPtr{Cint},
                                               batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnZpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info,
                                          batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnZpotrsBatched(handle::cusolverDnHandle_t,
                                               uplo::cublasFillMode_t, n::Cint, nrhs::Cint,
                                               A::CuPtr{Ptr{cuDoubleComplex}}, lda::Cint,
                                               B::CuPtr{Ptr{cuDoubleComplex}}, ldb::Cint,
                                               d_info::CuPtr{Cint},
                                               batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnSpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnSpotri_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cfloat}, lda::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnDpotri_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cdouble}, lda::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnCpotri_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuComplex}, lda::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnZpotri_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuDoubleComplex}, lda::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnSpotri(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{Cfloat}, lda::Cint,
                                        work::CuPtr{Cfloat}, lwork::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnDpotri(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{Cdouble}, lda::Cint,
                                        work::CuPtr{Cdouble}, lwork::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnCpotri(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{cuComplex}, lda::Cint,
                                        work::CuPtr{cuComplex}, lwork::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnZpotri(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{cuDoubleComplex}, lda::Cint,
                                        work::CuPtr{cuDoubleComplex}, lwork::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnXtrtri_bufferSize(handle, uplo, diag, n, dataTypeA, A, lda,
                                              workspaceInBytesOnDevice,
                                              workspaceInBytesOnHost)
    initialize_context()
    @ccall libcusolver.cusolverDnXtrtri_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t,
                                                   diag::cublasDiagType_t, n::Int64,
                                                   dataTypeA::cudaDataType, A::Ptr{Cvoid},
                                                   lda::Int64,
                                                   workspaceInBytesOnDevice::Ptr{Csize_t},
                                                   workspaceInBytesOnHost::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnXtrtri(handle, uplo, diag, n, dataTypeA, A, lda, bufferOnDevice,
                                   workspaceInBytesOnDevice, bufferOnHost,
                                   workspaceInBytesOnHost, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnXtrtri(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        diag::cublasDiagType_t, n::Int64,
                                        dataTypeA::cudaDataType, A::Ptr{Cvoid}, lda::Int64,
                                        bufferOnDevice::Ptr{Cvoid},
                                        workspaceInBytesOnDevice::Csize_t,
                                        bufferOnHost::Ptr{Cvoid},
                                        workspaceInBytesOnHost::Csize_t,
                                        devInfo::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSlauum_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnSlauum_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cfloat}, lda::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDlauum_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnDlauum_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cdouble}, lda::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnClauum_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnClauum_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuComplex}, lda::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZlauum_bufferSize(handle, uplo, n, A, lda, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnZlauum_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuDoubleComplex}, lda::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSlauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnSlauum(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{Cfloat}, lda::Cint,
                                        work::CuPtr{Cfloat}, lwork::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDlauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnDlauum(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{Cdouble}, lda::Cint,
                                        work::CuPtr{Cdouble}, lwork::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnClauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnClauum(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{cuComplex}, lda::Cint,
                                        work::CuPtr{cuComplex}, lwork::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZlauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnZlauum(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{cuDoubleComplex}, lda::Cint,
                                        work::CuPtr{cuDoubleComplex}, lwork::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnSgetrf_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint, A::CuPtr{Cfloat}, lda::Cint,
                                                   Lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnDgetrf_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint, A::CuPtr{Cdouble}, lda::Cint,
                                                   Lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnCgetrf_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint, A::CuPtr{cuComplex}, lda::Cint,
                                                   Lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnZgetrf_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint, A::CuPtr{cuDoubleComplex},
                                                   lda::Cint,
                                                   Lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnSgetrf(handle::cusolverDnHandle_t, m::Cint, n::Cint,
                                        A::CuPtr{Cfloat}, lda::Cint,
                                        Workspace::CuPtr{Cfloat}, devIpiv::CuPtr{Cint},
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnDgetrf(handle::cusolverDnHandle_t, m::Cint, n::Cint,
                                        A::CuPtr{Cdouble}, lda::Cint,
                                        Workspace::CuPtr{Cdouble}, devIpiv::CuPtr{Cint},
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnCgetrf(handle::cusolverDnHandle_t, m::Cint, n::Cint,
                                        A::CuPtr{cuComplex}, lda::Cint,
                                        Workspace::CuPtr{cuComplex}, devIpiv::CuPtr{Cint},
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnZgetrf(handle::cusolverDnHandle_t, m::Cint, n::Cint,
                                        A::CuPtr{cuDoubleComplex}, lda::Cint,
                                        Workspace::CuPtr{cuDoubleComplex},
                                        devIpiv::CuPtr{Cint},
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    initialize_context()
    @ccall libcusolver.cusolverDnSlaswp(handle::cusolverDnHandle_t, n::Cint,
                                        A::CuPtr{Cfloat}, lda::Cint, k1::Cint, k2::Cint,
                                        devIpiv::CuPtr{Cint}, incx::Cint)::cusolverStatus_t
end

@checked function cusolverDnDlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    initialize_context()
    @ccall libcusolver.cusolverDnDlaswp(handle::cusolverDnHandle_t, n::Cint,
                                        A::CuPtr{Cdouble}, lda::Cint, k1::Cint, k2::Cint,
                                        devIpiv::CuPtr{Cint}, incx::Cint)::cusolverStatus_t
end

@checked function cusolverDnClaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    initialize_context()
    @ccall libcusolver.cusolverDnClaswp(handle::cusolverDnHandle_t, n::Cint,
                                        A::CuPtr{cuComplex}, lda::Cint, k1::Cint, k2::Cint,
                                        devIpiv::CuPtr{Cint}, incx::Cint)::cusolverStatus_t
end

@checked function cusolverDnZlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    initialize_context()
    @ccall libcusolver.cusolverDnZlaswp(handle::cusolverDnHandle_t, n::Cint,
                                        A::CuPtr{cuDoubleComplex}, lda::Cint, k1::Cint,
                                        k2::Cint, devIpiv::CuPtr{Cint},
                                        incx::Cint)::cusolverStatus_t
end

@checked function cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnSgetrs(handle::cusolverDnHandle_t,
                                        trans::cublasOperation_t, n::Cint, nrhs::Cint,
                                        A::CuPtr{Cfloat}, lda::Cint, devIpiv::CuPtr{Cint},
                                        B::CuPtr{Cfloat}, ldb::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnDgetrs(handle::cusolverDnHandle_t,
                                        trans::cublasOperation_t, n::Cint, nrhs::Cint,
                                        A::CuPtr{Cdouble}, lda::Cint, devIpiv::CuPtr{Cint},
                                        B::CuPtr{Cdouble}, ldb::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnCgetrs(handle::cusolverDnHandle_t,
                                        trans::cublasOperation_t, n::Cint, nrhs::Cint,
                                        A::CuPtr{cuComplex}, lda::Cint,
                                        devIpiv::CuPtr{Cint}, B::CuPtr{cuComplex},
                                        ldb::Cint, devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnZgetrs(handle::cusolverDnHandle_t,
                                        trans::cublasOperation_t, n::Cint, nrhs::Cint,
                                        A::CuPtr{cuDoubleComplex}, lda::Cint,
                                        devIpiv::CuPtr{Cint}, B::CuPtr{cuDoubleComplex},
                                        ldb::Cint, devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnSgeqrf_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint, A::CuPtr{Cfloat}, lda::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnDgeqrf_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint, A::CuPtr{Cdouble}, lda::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnCgeqrf_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint, A::CuPtr{cuComplex}, lda::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnZgeqrf_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint, A::CuPtr{cuDoubleComplex},
                                                   lda::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnSgeqrf(handle::cusolverDnHandle_t, m::Cint, n::Cint,
                                        A::CuPtr{Cfloat}, lda::Cint, TAU::CuPtr{Cfloat},
                                        Workspace::CuPtr{Cfloat}, Lwork::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnDgeqrf(handle::cusolverDnHandle_t, m::Cint, n::Cint,
                                        A::CuPtr{Cdouble}, lda::Cint, TAU::CuPtr{Cdouble},
                                        Workspace::CuPtr{Cdouble}, Lwork::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnCgeqrf(handle::cusolverDnHandle_t, m::Cint, n::Cint,
                                        A::CuPtr{cuComplex}, lda::Cint,
                                        TAU::CuPtr{cuComplex}, Workspace::CuPtr{cuComplex},
                                        Lwork::Cint, devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnZgeqrf(handle::cusolverDnHandle_t, m::Cint, n::Cint,
                                        A::CuPtr{cuDoubleComplex}, lda::Cint,
                                        TAU::CuPtr{cuDoubleComplex},
                                        Workspace::CuPtr{cuDoubleComplex}, Lwork::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnSorgqr_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint, k::Cint, A::CuPtr{Cfloat},
                                                   lda::Cint, tau::CuPtr{Cfloat},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnDorgqr_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint, k::Cint, A::CuPtr{Cdouble},
                                                   lda::Cint, tau::CuPtr{Cdouble},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnCungqr_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint, k::Cint, A::CuPtr{cuComplex},
                                                   lda::Cint, tau::CuPtr{cuComplex},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnZungqr_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint, k::Cint,
                                                   A::CuPtr{cuDoubleComplex}, lda::Cint,
                                                   tau::CuPtr{cuDoubleComplex},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnSorgqr(handle::cusolverDnHandle_t, m::Cint, n::Cint,
                                        k::Cint, A::CuPtr{Cfloat}, lda::Cint,
                                        tau::CuPtr{Cfloat}, work::CuPtr{Cfloat},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnDorgqr(handle::cusolverDnHandle_t, m::Cint, n::Cint,
                                        k::Cint, A::CuPtr{Cdouble}, lda::Cint,
                                        tau::CuPtr{Cdouble}, work::CuPtr{Cdouble},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCungqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnCungqr(handle::cusolverDnHandle_t, m::Cint, n::Cint,
                                        k::Cint, A::CuPtr{cuComplex}, lda::Cint,
                                        tau::CuPtr{cuComplex}, work::CuPtr{cuComplex},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZungqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnZungqr(handle::cusolverDnHandle_t, m::Cint, n::Cint,
                                        k::Cint, A::CuPtr{cuDoubleComplex}, lda::Cint,
                                        tau::CuPtr{cuDoubleComplex},
                                        work::CuPtr{cuDoubleComplex}, lwork::Cint,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C,
                                              ldc, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnSormqr_bufferSize(handle::cusolverDnHandle_t,
                                                   side::cublasSideMode_t,
                                                   trans::cublasOperation_t, m::Cint,
                                                   n::Cint, k::Cint, A::CuPtr{Cfloat},
                                                   lda::Cint, tau::CuPtr{Cfloat},
                                                   C::CuPtr{Cfloat}, ldc::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C,
                                              ldc, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnDormqr_bufferSize(handle::cusolverDnHandle_t,
                                                   side::cublasSideMode_t,
                                                   trans::cublasOperation_t, m::Cint,
                                                   n::Cint, k::Cint, A::CuPtr{Cdouble},
                                                   lda::Cint, tau::CuPtr{Cdouble},
                                                   C::CuPtr{Cdouble}, ldc::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C,
                                              ldc, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnCunmqr_bufferSize(handle::cusolverDnHandle_t,
                                                   side::cublasSideMode_t,
                                                   trans::cublasOperation_t, m::Cint,
                                                   n::Cint, k::Cint, A::CuPtr{cuComplex},
                                                   lda::Cint, tau::CuPtr{cuComplex},
                                                   C::CuPtr{cuComplex}, ldc::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C,
                                              ldc, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnZunmqr_bufferSize(handle::cusolverDnHandle_t,
                                                   side::cublasSideMode_t,
                                                   trans::cublasOperation_t, m::Cint,
                                                   n::Cint, k::Cint,
                                                   A::CuPtr{cuDoubleComplex}, lda::Cint,
                                                   tau::CuPtr{cuDoubleComplex},
                                                   C::CuPtr{cuDoubleComplex}, ldc::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work,
                                   lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnSormqr(handle::cusolverDnHandle_t, side::cublasSideMode_t,
                                        trans::cublasOperation_t, m::Cint, n::Cint, k::Cint,
                                        A::CuPtr{Cfloat}, lda::Cint, tau::CuPtr{Cfloat},
                                        C::CuPtr{Cfloat}, ldc::Cint, work::CuPtr{Cfloat},
                                        lwork::Cint, devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work,
                                   lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnDormqr(handle::cusolverDnHandle_t, side::cublasSideMode_t,
                                        trans::cublasOperation_t, m::Cint, n::Cint, k::Cint,
                                        A::CuPtr{Cdouble}, lda::Cint, tau::CuPtr{Cdouble},
                                        C::CuPtr{Cdouble}, ldc::Cint, work::CuPtr{Cdouble},
                                        lwork::Cint, devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work,
                                   lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnCunmqr(handle::cusolverDnHandle_t, side::cublasSideMode_t,
                                        trans::cublasOperation_t, m::Cint, n::Cint, k::Cint,
                                        A::CuPtr{cuComplex}, lda::Cint,
                                        tau::CuPtr{cuComplex}, C::CuPtr{cuComplex},
                                        ldc::Cint, work::CuPtr{cuComplex}, lwork::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work,
                                   lwork, devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnZunmqr(handle::cusolverDnHandle_t, side::cublasSideMode_t,
                                        trans::cublasOperation_t, m::Cint, n::Cint, k::Cint,
                                        A::CuPtr{cuDoubleComplex}, lda::Cint,
                                        tau::CuPtr{cuDoubleComplex},
                                        C::CuPtr{cuDoubleComplex}, ldc::Cint,
                                        work::CuPtr{cuDoubleComplex}, lwork::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSsytrf_bufferSize(handle, n, A, lda, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnSsytrf_bufferSize(handle::cusolverDnHandle_t, n::Cint,
                                                   A::CuPtr{Cfloat}, lda::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDsytrf_bufferSize(handle, n, A, lda, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnDsytrf_bufferSize(handle::cusolverDnHandle_t, n::Cint,
                                                   A::CuPtr{Cdouble}, lda::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCsytrf_bufferSize(handle, n, A, lda, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnCsytrf_bufferSize(handle::cusolverDnHandle_t, n::Cint,
                                                   A::CuPtr{cuComplex}, lda::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZsytrf_bufferSize(handle, n, A, lda, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnZsytrf_bufferSize(handle::cusolverDnHandle_t, n::Cint,
                                                   A::CuPtr{cuDoubleComplex}, lda::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnSsytrf(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{Cfloat}, lda::Cint,
                                        ipiv::CuPtr{Cint}, work::CuPtr{Cfloat}, lwork::Cint,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnDsytrf(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{Cdouble}, lda::Cint,
                                        ipiv::CuPtr{Cint}, work::CuPtr{Cdouble},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnCsytrf(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{cuComplex}, lda::Cint,
                                        ipiv::CuPtr{Cint}, work::CuPtr{cuComplex},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnZsytrf(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{cuDoubleComplex}, lda::Cint,
                                        ipiv::CuPtr{Cint}, work::CuPtr{cuDoubleComplex},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnXsytrs_bufferSize(handle, uplo, n, nrhs, dataTypeA, A, lda,
                                              ipiv, dataTypeB, B, ldb,
                                              workspaceInBytesOnDevice,
                                              workspaceInBytesOnHost)
    initialize_context()
    @ccall libcusolver.cusolverDnXsytrs_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Int64,
                                                   nrhs::Int64, dataTypeA::cudaDataType,
                                                   A::Ptr{Cvoid}, lda::Int64,
                                                   ipiv::Ptr{Int64},
                                                   dataTypeB::cudaDataType, B::Ptr{Cvoid},
                                                   ldb::Int64,
                                                   workspaceInBytesOnDevice::Ptr{Csize_t},
                                                   workspaceInBytesOnHost::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnXsytrs(handle, uplo, n, nrhs, dataTypeA, A, lda, ipiv,
                                   dataTypeB, B, ldb, bufferOnDevice,
                                   workspaceInBytesOnDevice, bufferOnHost,
                                   workspaceInBytesOnHost, info)
    initialize_context()
    @ccall libcusolver.cusolverDnXsytrs(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Int64, nrhs::Int64, dataTypeA::cudaDataType,
                                        A::Ptr{Cvoid}, lda::Int64, ipiv::Ptr{Int64},
                                        dataTypeB::cudaDataType, B::Ptr{Cvoid}, ldb::Int64,
                                        bufferOnDevice::Ptr{Cvoid},
                                        workspaceInBytesOnDevice::Csize_t,
                                        bufferOnHost::Ptr{Cvoid},
                                        workspaceInBytesOnHost::Csize_t,
                                        info::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnSsytri_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cfloat}, lda::Cint,
                                                   ipiv::CuPtr{Cint},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnDsytri_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cdouble}, lda::Cint,
                                                   ipiv::CuPtr{Cint},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnCsytri_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuComplex}, lda::Cint,
                                                   ipiv::CuPtr{Cint},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnZsytri_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuDoubleComplex}, lda::Cint,
                                                   ipiv::CuPtr{Cint},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnSsytri(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{Cfloat}, lda::Cint,
                                        ipiv::CuPtr{Cint}, work::CuPtr{Cfloat}, lwork::Cint,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnDsytri(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{Cdouble}, lda::Cint,
                                        ipiv::CuPtr{Cint}, work::CuPtr{Cdouble},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnCsytri(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{cuComplex}, lda::Cint,
                                        ipiv::CuPtr{Cint}, work::CuPtr{cuComplex},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnZsytri(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{cuDoubleComplex}, lda::Cint,
                                        ipiv::CuPtr{Cint}, work::CuPtr{cuDoubleComplex},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSgebrd_bufferSize(handle, m, n, Lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnSgebrd_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint,
                                                   Lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDgebrd_bufferSize(handle, m, n, Lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnDgebrd_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint,
                                                   Lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCgebrd_bufferSize(handle, m, n, Lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnCgebrd_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint,
                                                   Lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZgebrd_bufferSize(handle, m, n, Lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnZgebrd_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint,
                                                   Lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork,
                                   devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnSgebrd(handle::cusolverDnHandle_t, m::Cint, n::Cint,
                                        A::CuPtr{Cfloat}, lda::Cint, D::CuPtr{Cfloat},
                                        E::CuPtr{Cfloat}, TAUQ::CuPtr{Cfloat},
                                        TAUP::CuPtr{Cfloat}, Work::CuPtr{Cfloat},
                                        Lwork::Cint, devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork,
                                   devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnDgebrd(handle::cusolverDnHandle_t, m::Cint, n::Cint,
                                        A::CuPtr{Cdouble}, lda::Cint, D::CuPtr{Cdouble},
                                        E::CuPtr{Cdouble}, TAUQ::CuPtr{Cdouble},
                                        TAUP::CuPtr{Cdouble}, Work::CuPtr{Cdouble},
                                        Lwork::Cint, devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork,
                                   devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnCgebrd(handle::cusolverDnHandle_t, m::Cint, n::Cint,
                                        A::CuPtr{cuComplex}, lda::Cint, D::CuPtr{Cfloat},
                                        E::CuPtr{Cfloat}, TAUQ::CuPtr{cuComplex},
                                        TAUP::CuPtr{cuComplex}, Work::CuPtr{cuComplex},
                                        Lwork::Cint, devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork,
                                   devInfo)
    initialize_context()
    @ccall libcusolver.cusolverDnZgebrd(handle::cusolverDnHandle_t, m::Cint, n::Cint,
                                        A::CuPtr{cuDoubleComplex}, lda::Cint,
                                        D::CuPtr{Cdouble}, E::CuPtr{Cdouble},
                                        TAUQ::CuPtr{cuDoubleComplex},
                                        TAUP::CuPtr{cuDoubleComplex},
                                        Work::CuPtr{cuDoubleComplex}, Lwork::Cint,
                                        devInfo::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnSorgbr_bufferSize(handle::cusolverDnHandle_t,
                                                   side::cublasSideMode_t, m::Cint, n::Cint,
                                                   k::Cint, A::CuPtr{Cfloat}, lda::Cint,
                                                   tau::CuPtr{Cfloat},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnDorgbr_bufferSize(handle::cusolverDnHandle_t,
                                                   side::cublasSideMode_t, m::Cint, n::Cint,
                                                   k::Cint, A::CuPtr{Cdouble}, lda::Cint,
                                                   tau::CuPtr{Cdouble},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnCungbr_bufferSize(handle::cusolverDnHandle_t,
                                                   side::cublasSideMode_t, m::Cint, n::Cint,
                                                   k::Cint, A::CuPtr{cuComplex}, lda::Cint,
                                                   tau::CuPtr{cuComplex},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnZungbr_bufferSize(handle::cusolverDnHandle_t,
                                                   side::cublasSideMode_t, m::Cint, n::Cint,
                                                   k::Cint, A::CuPtr{cuDoubleComplex},
                                                   lda::Cint, tau::CuPtr{cuDoubleComplex},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnSorgbr(handle::cusolverDnHandle_t, side::cublasSideMode_t,
                                        m::Cint, n::Cint, k::Cint, A::CuPtr{Cfloat},
                                        lda::Cint, tau::CuPtr{Cfloat}, work::CuPtr{Cfloat},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnDorgbr(handle::cusolverDnHandle_t, side::cublasSideMode_t,
                                        m::Cint, n::Cint, k::Cint, A::CuPtr{Cdouble},
                                        lda::Cint, tau::CuPtr{Cdouble},
                                        work::CuPtr{Cdouble}, lwork::Cint,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnCungbr(handle::cusolverDnHandle_t, side::cublasSideMode_t,
                                        m::Cint, n::Cint, k::Cint, A::CuPtr{cuComplex},
                                        lda::Cint, tau::CuPtr{cuComplex},
                                        work::CuPtr{cuComplex}, lwork::Cint,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnZungbr(handle::cusolverDnHandle_t, side::cublasSideMode_t,
                                        m::Cint, n::Cint, k::Cint,
                                        A::CuPtr{cuDoubleComplex}, lda::Cint,
                                        tau::CuPtr{cuDoubleComplex},
                                        work::CuPtr{cuDoubleComplex}, lwork::Cint,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSsytrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnSsytrd_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cfloat}, lda::Cint,
                                                   d::CuPtr{Cfloat}, e::CuPtr{Cfloat},
                                                   tau::CuPtr{Cfloat},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDsytrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnDsytrd_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cdouble}, lda::Cint,
                                                   d::CuPtr{Cdouble}, e::CuPtr{Cdouble},
                                                   tau::CuPtr{Cdouble},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnChetrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnChetrd_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuComplex}, lda::Cint,
                                                   d::CuPtr{Cfloat}, e::CuPtr{Cfloat},
                                                   tau::CuPtr{cuComplex},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZhetrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnZhetrd_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuDoubleComplex}, lda::Cint,
                                                   d::CuPtr{Cdouble}, e::CuPtr{Cdouble},
                                                   tau::CuPtr{cuDoubleComplex},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSsytrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnSsytrd(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{Cfloat}, lda::Cint,
                                        d::CuPtr{Cfloat}, e::CuPtr{Cfloat},
                                        tau::CuPtr{Cfloat}, work::CuPtr{Cfloat},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDsytrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnDsytrd(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{Cdouble}, lda::Cint,
                                        d::CuPtr{Cdouble}, e::CuPtr{Cdouble},
                                        tau::CuPtr{Cdouble}, work::CuPtr{Cdouble},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnChetrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnChetrd(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{cuComplex}, lda::Cint,
                                        d::CuPtr{Cfloat}, e::CuPtr{Cfloat},
                                        tau::CuPtr{cuComplex}, work::CuPtr{cuComplex},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZhetrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnZhetrd(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{cuDoubleComplex}, lda::Cint,
                                        d::CuPtr{Cdouble}, e::CuPtr{Cdouble},
                                        tau::CuPtr{cuDoubleComplex},
                                        work::CuPtr{cuDoubleComplex}, lwork::Cint,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnSorgtr_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cfloat}, lda::Cint,
                                                   tau::CuPtr{Cfloat},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnDorgtr_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cdouble}, lda::Cint,
                                                   tau::CuPtr{Cdouble},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnCungtr_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuComplex}, lda::Cint,
                                                   tau::CuPtr{cuComplex},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnZungtr_bufferSize(handle::cusolverDnHandle_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuDoubleComplex}, lda::Cint,
                                                   tau::CuPtr{cuDoubleComplex},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSorgtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnSorgtr(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{Cfloat}, lda::Cint,
                                        tau::CuPtr{Cfloat}, work::CuPtr{Cfloat},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDorgtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnDorgtr(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{Cdouble}, lda::Cint,
                                        tau::CuPtr{Cdouble}, work::CuPtr{Cdouble},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCungtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnCungtr(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{cuComplex}, lda::Cint,
                                        tau::CuPtr{cuComplex}, work::CuPtr{cuComplex},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZungtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnZungtr(handle::cusolverDnHandle_t, uplo::cublasFillMode_t,
                                        n::Cint, A::CuPtr{cuDoubleComplex}, lda::Cint,
                                        tau::CuPtr{cuDoubleComplex},
                                        work::CuPtr{cuDoubleComplex}, lwork::Cint,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau,
                                              C, ldc, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnSormtr_bufferSize(handle::cusolverDnHandle_t,
                                                   side::cublasSideMode_t,
                                                   uplo::cublasFillMode_t,
                                                   trans::cublasOperation_t, m::Cint,
                                                   n::Cint, A::CuPtr{Cfloat}, lda::Cint,
                                                   tau::CuPtr{Cfloat}, C::CuPtr{Cfloat},
                                                   ldc::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau,
                                              C, ldc, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnDormtr_bufferSize(handle::cusolverDnHandle_t,
                                                   side::cublasSideMode_t,
                                                   uplo::cublasFillMode_t,
                                                   trans::cublasOperation_t, m::Cint,
                                                   n::Cint, A::CuPtr{Cdouble}, lda::Cint,
                                                   tau::CuPtr{Cdouble}, C::CuPtr{Cdouble},
                                                   ldc::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau,
                                              C, ldc, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnCunmtr_bufferSize(handle::cusolverDnHandle_t,
                                                   side::cublasSideMode_t,
                                                   uplo::cublasFillMode_t,
                                                   trans::cublasOperation_t, m::Cint,
                                                   n::Cint, A::CuPtr{cuComplex}, lda::Cint,
                                                   tau::CuPtr{cuComplex},
                                                   C::CuPtr{cuComplex}, ldc::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau,
                                              C, ldc, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnZunmtr_bufferSize(handle::cusolverDnHandle_t,
                                                   side::cublasSideMode_t,
                                                   uplo::cublasFillMode_t,
                                                   trans::cublasOperation_t, m::Cint,
                                                   n::Cint, A::CuPtr{cuDoubleComplex},
                                                   lda::Cint, tau::CuPtr{cuDoubleComplex},
                                                   C::CuPtr{cuDoubleComplex}, ldc::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                   work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnSormtr(handle::cusolverDnHandle_t, side::cublasSideMode_t,
                                        uplo::cublasFillMode_t, trans::cublasOperation_t,
                                        m::Cint, n::Cint, A::CuPtr{Cfloat}, lda::Cint,
                                        tau::CuPtr{Cfloat}, C::CuPtr{Cfloat}, ldc::Cint,
                                        work::CuPtr{Cfloat}, lwork::Cint,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                   work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnDormtr(handle::cusolverDnHandle_t, side::cublasSideMode_t,
                                        uplo::cublasFillMode_t, trans::cublasOperation_t,
                                        m::Cint, n::Cint, A::CuPtr{Cdouble}, lda::Cint,
                                        tau::CuPtr{Cdouble}, C::CuPtr{Cdouble}, ldc::Cint,
                                        work::CuPtr{Cdouble}, lwork::Cint,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                   work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnCunmtr(handle::cusolverDnHandle_t, side::cublasSideMode_t,
                                        uplo::cublasFillMode_t, trans::cublasOperation_t,
                                        m::Cint, n::Cint, A::CuPtr{cuComplex}, lda::Cint,
                                        tau::CuPtr{cuComplex}, C::CuPtr{cuComplex},
                                        ldc::Cint, work::CuPtr{cuComplex}, lwork::Cint,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                   work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnZunmtr(handle::cusolverDnHandle_t, side::cublasSideMode_t,
                                        uplo::cublasFillMode_t, trans::cublasOperation_t,
                                        m::Cint, n::Cint, A::CuPtr{cuDoubleComplex},
                                        lda::Cint, tau::CuPtr{cuDoubleComplex},
                                        C::CuPtr{cuDoubleComplex}, ldc::Cint,
                                        work::CuPtr{cuDoubleComplex}, lwork::Cint,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSgesvd_bufferSize(handle, m, n, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnSgesvd_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDgesvd_bufferSize(handle, m, n, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnDgesvd_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCgesvd_bufferSize(handle, m, n, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnCgesvd_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZgesvd_bufferSize(handle, m, n, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnZgesvd_bufferSize(handle::cusolverDnHandle_t, m::Cint,
                                                   n::Cint,
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,
                                   work, lwork, rwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnSgesvd(handle::cusolverDnHandle_t, jobu::Int8, jobvt::Int8,
                                        m::Cint, n::Cint, A::CuPtr{Cfloat}, lda::Cint,
                                        S::CuPtr{Cfloat}, U::CuPtr{Cfloat}, ldu::Cint,
                                        VT::CuPtr{Cfloat}, ldvt::Cint, work::CuPtr{Cfloat},
                                        lwork::Cint, rwork::CuPtr{Cfloat},
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,
                                   work, lwork, rwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnDgesvd(handle::cusolverDnHandle_t, jobu::Int8, jobvt::Int8,
                                        m::Cint, n::Cint, A::CuPtr{Cdouble}, lda::Cint,
                                        S::CuPtr{Cdouble}, U::CuPtr{Cdouble}, ldu::Cint,
                                        VT::CuPtr{Cdouble}, ldvt::Cint,
                                        work::CuPtr{Cdouble}, lwork::Cint,
                                        rwork::CuPtr{Cdouble},
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,
                                   work, lwork, rwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnCgesvd(handle::cusolverDnHandle_t, jobu::Int8, jobvt::Int8,
                                        m::Cint, n::Cint, A::CuPtr{cuComplex}, lda::Cint,
                                        S::CuPtr{Cfloat}, U::CuPtr{cuComplex}, ldu::Cint,
                                        VT::CuPtr{cuComplex}, ldvt::Cint,
                                        work::CuPtr{cuComplex}, lwork::Cint,
                                        rwork::CuPtr{Cfloat},
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,
                                   work, lwork, rwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnZgesvd(handle::cusolverDnHandle_t, jobu::Int8, jobvt::Int8,
                                        m::Cint, n::Cint, A::CuPtr{cuDoubleComplex},
                                        lda::Cint, S::CuPtr{Cdouble},
                                        U::CuPtr{cuDoubleComplex}, ldu::Cint,
                                        VT::CuPtr{cuDoubleComplex}, ldvt::Cint,
                                        work::CuPtr{cuDoubleComplex}, lwork::Cint,
                                        rwork::CuPtr{Cdouble},
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnSsyevd_bufferSize(handle::cusolverDnHandle_t,
                                                   jobz::cusolverEigMode_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cfloat}, lda::Cint,
                                                   W::CuPtr{Cfloat},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnDsyevd_bufferSize(handle::cusolverDnHandle_t,
                                                   jobz::cusolverEigMode_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cdouble}, lda::Cint,
                                                   W::CuPtr{Cdouble},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnCheevd_bufferSize(handle::cusolverDnHandle_t,
                                                   jobz::cusolverEigMode_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuComplex}, lda::Cint,
                                                   W::CuPtr{Cfloat},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnZheevd_bufferSize(handle::cusolverDnHandle_t,
                                                   jobz::cusolverEigMode_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuDoubleComplex}, lda::Cint,
                                                   W::CuPtr{Cdouble},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnSsyevd(handle::cusolverDnHandle_t, jobz::cusolverEigMode_t,
                                        uplo::cublasFillMode_t, n::Cint, A::CuPtr{Cfloat},
                                        lda::Cint, W::CuPtr{Cfloat}, work::CuPtr{Cfloat},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnDsyevd(handle::cusolverDnHandle_t, jobz::cusolverEigMode_t,
                                        uplo::cublasFillMode_t, n::Cint, A::CuPtr{Cdouble},
                                        lda::Cint, W::CuPtr{Cdouble}, work::CuPtr{Cdouble},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnCheevd(handle::cusolverDnHandle_t, jobz::cusolverEigMode_t,
                                        uplo::cublasFillMode_t, n::Cint,
                                        A::CuPtr{cuComplex}, lda::Cint, W::CuPtr{Cfloat},
                                        work::CuPtr{cuComplex}, lwork::Cint,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnZheevd(handle::cusolverDnHandle_t, jobz::cusolverEigMode_t,
                                        uplo::cublasFillMode_t, n::Cint,
                                        A::CuPtr{cuDoubleComplex}, lda::Cint,
                                        W::CuPtr{Cdouble}, work::CuPtr{cuDoubleComplex},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu,
                                               il, iu, meig, W, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnSsyevdx_bufferSize(handle::cusolverDnHandle_t,
                                                    jobz::cusolverEigMode_t,
                                                    range::cusolverEigRange_t,
                                                    uplo::cublasFillMode_t, n::Cint,
                                                    A::CuPtr{Cfloat}, lda::Cint, vl::Cfloat,
                                                    vu::Cfloat, il::Cint, iu::Cint,
                                                    meig::Ptr{Cint}, W::CuPtr{Cfloat},
                                                    lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu,
                                               il, iu, meig, W, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnDsyevdx_bufferSize(handle::cusolverDnHandle_t,
                                                    jobz::cusolverEigMode_t,
                                                    range::cusolverEigRange_t,
                                                    uplo::cublasFillMode_t, n::Cint,
                                                    A::CuPtr{Cdouble}, lda::Cint,
                                                    vl::Cdouble, vu::Cdouble, il::Cint,
                                                    iu::Cint, meig::Ptr{Cint},
                                                    W::CuPtr{Cdouble},
                                                    lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu,
                                               il, iu, meig, W, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnCheevdx_bufferSize(handle::cusolverDnHandle_t,
                                                    jobz::cusolverEigMode_t,
                                                    range::cusolverEigRange_t,
                                                    uplo::cublasFillMode_t, n::Cint,
                                                    A::CuPtr{cuComplex}, lda::Cint,
                                                    vl::Cfloat, vu::Cfloat, il::Cint,
                                                    iu::Cint, meig::Ptr{Cint},
                                                    W::CuPtr{Cfloat},
                                                    lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu,
                                               il, iu, meig, W, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnZheevdx_bufferSize(handle::cusolverDnHandle_t,
                                                    jobz::cusolverEigMode_t,
                                                    range::cusolverEigRange_t,
                                                    uplo::cublasFillMode_t, n::Cint,
                                                    A::CuPtr{cuDoubleComplex}, lda::Cint,
                                                    vl::Cdouble, vu::Cdouble, il::Cint,
                                                    iu::Cint, meig::Ptr{Cint},
                                                    W::CuPtr{Cdouble},
                                                    lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                    meig, W, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnSsyevdx(handle::cusolverDnHandle_t,
                                         jobz::cusolverEigMode_t, range::cusolverEigRange_t,
                                         uplo::cublasFillMode_t, n::Cint, A::CuPtr{Cfloat},
                                         lda::Cint, vl::Cfloat, vu::Cfloat, il::Cint,
                                         iu::Cint, meig::Ptr{Cint}, W::CuPtr{Cfloat},
                                         work::CuPtr{Cfloat}, lwork::Cint,
                                         info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                    meig, W, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnDsyevdx(handle::cusolverDnHandle_t,
                                         jobz::cusolverEigMode_t, range::cusolverEigRange_t,
                                         uplo::cublasFillMode_t, n::Cint, A::CuPtr{Cdouble},
                                         lda::Cint, vl::Cdouble, vu::Cdouble, il::Cint,
                                         iu::Cint, meig::Ptr{Cint}, W::CuPtr{Cdouble},
                                         work::CuPtr{Cdouble}, lwork::Cint,
                                         info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                    meig, W, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnCheevdx(handle::cusolverDnHandle_t,
                                         jobz::cusolverEigMode_t, range::cusolverEigRange_t,
                                         uplo::cublasFillMode_t, n::Cint,
                                         A::CuPtr{cuComplex}, lda::Cint, vl::Cfloat,
                                         vu::Cfloat, il::Cint, iu::Cint, meig::Ptr{Cint},
                                         W::CuPtr{Cfloat}, work::CuPtr{cuComplex},
                                         lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                    meig, W, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnZheevdx(handle::cusolverDnHandle_t,
                                         jobz::cusolverEigMode_t, range::cusolverEigRange_t,
                                         uplo::cublasFillMode_t, n::Cint,
                                         A::CuPtr{cuDoubleComplex}, lda::Cint, vl::Cdouble,
                                         vu::Cdouble, il::Cint, iu::Cint, meig::Ptr{Cint},
                                         W::CuPtr{Cdouble}, work::CuPtr{cuDoubleComplex},
                                         lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda,
                                               B, ldb, vl, vu, il, iu, meig, W, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnSsygvdx_bufferSize(handle::cusolverDnHandle_t,
                                                    itype::cusolverEigType_t,
                                                    jobz::cusolverEigMode_t,
                                                    range::cusolverEigRange_t,
                                                    uplo::cublasFillMode_t, n::Cint,
                                                    A::CuPtr{Cfloat}, lda::Cint,
                                                    B::CuPtr{Cfloat}, ldb::Cint, vl::Cfloat,
                                                    vu::Cfloat, il::Cint, iu::Cint,
                                                    meig::Ptr{Cint}, W::CuPtr{Cfloat},
                                                    lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda,
                                               B, ldb, vl, vu, il, iu, meig, W, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnDsygvdx_bufferSize(handle::cusolverDnHandle_t,
                                                    itype::cusolverEigType_t,
                                                    jobz::cusolverEigMode_t,
                                                    range::cusolverEigRange_t,
                                                    uplo::cublasFillMode_t, n::Cint,
                                                    A::CuPtr{Cdouble}, lda::Cint,
                                                    B::CuPtr{Cdouble}, ldb::Cint,
                                                    vl::Cdouble, vu::Cdouble, il::Cint,
                                                    iu::Cint, meig::Ptr{Cint},
                                                    W::CuPtr{Cdouble},
                                                    lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnChegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda,
                                               B, ldb, vl, vu, il, iu, meig, W, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnChegvdx_bufferSize(handle::cusolverDnHandle_t,
                                                    itype::cusolverEigType_t,
                                                    jobz::cusolverEigMode_t,
                                                    range::cusolverEigRange_t,
                                                    uplo::cublasFillMode_t, n::Cint,
                                                    A::CuPtr{cuComplex}, lda::Cint,
                                                    B::CuPtr{cuComplex}, ldb::Cint,
                                                    vl::Cfloat, vu::Cfloat, il::Cint,
                                                    iu::Cint, meig::Ptr{Cint},
                                                    W::CuPtr{Cfloat},
                                                    lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZhegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda,
                                               B, ldb, vl, vu, il, iu, meig, W, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnZhegvdx_bufferSize(handle::cusolverDnHandle_t,
                                                    itype::cusolverEigType_t,
                                                    jobz::cusolverEigMode_t,
                                                    range::cusolverEigRange_t,
                                                    uplo::cublasFillMode_t, n::Cint,
                                                    A::CuPtr{cuDoubleComplex}, lda::Cint,
                                                    B::CuPtr{cuDoubleComplex}, ldb::Cint,
                                                    vl::Cdouble, vu::Cdouble, il::Cint,
                                                    iu::Cint, meig::Ptr{Cint},
                                                    W::CuPtr{Cdouble},
                                                    lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl,
                                    vu, il, iu, meig, W, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnSsygvdx(handle::cusolverDnHandle_t,
                                         itype::cusolverEigType_t, jobz::cusolverEigMode_t,
                                         range::cusolverEigRange_t, uplo::cublasFillMode_t,
                                         n::Cint, A::CuPtr{Cfloat}, lda::Cint,
                                         B::CuPtr{Cfloat}, ldb::Cint, vl::Cfloat,
                                         vu::Cfloat, il::Cint, iu::Cint, meig::Ptr{Cint},
                                         W::CuPtr{Cfloat}, work::CuPtr{Cfloat}, lwork::Cint,
                                         info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl,
                                    vu, il, iu, meig, W, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnDsygvdx(handle::cusolverDnHandle_t,
                                         itype::cusolverEigType_t, jobz::cusolverEigMode_t,
                                         range::cusolverEigRange_t, uplo::cublasFillMode_t,
                                         n::Cint, A::CuPtr{Cdouble}, lda::Cint,
                                         B::CuPtr{Cdouble}, ldb::Cint, vl::Cdouble,
                                         vu::Cdouble, il::Cint, iu::Cint, meig::Ptr{Cint},
                                         W::CuPtr{Cdouble}, work::CuPtr{Cdouble},
                                         lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnChegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl,
                                    vu, il, iu, meig, W, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnChegvdx(handle::cusolverDnHandle_t,
                                         itype::cusolverEigType_t, jobz::cusolverEigMode_t,
                                         range::cusolverEigRange_t, uplo::cublasFillMode_t,
                                         n::Cint, A::CuPtr{cuComplex}, lda::Cint,
                                         B::CuPtr{cuComplex}, ldb::Cint, vl::Cfloat,
                                         vu::Cfloat, il::Cint, iu::Cint, meig::Ptr{Cint},
                                         W::CuPtr{Cfloat}, work::CuPtr{cuComplex},
                                         lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZhegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl,
                                    vu, il, iu, meig, W, work, lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnZhegvdx(handle::cusolverDnHandle_t,
                                         itype::cusolverEigType_t, jobz::cusolverEigMode_t,
                                         range::cusolverEigRange_t, uplo::cublasFillMode_t,
                                         n::Cint, A::CuPtr{cuDoubleComplex}, lda::Cint,
                                         B::CuPtr{cuDoubleComplex}, ldb::Cint, vl::Cdouble,
                                         vu::Cdouble, il::Cint, iu::Cint, meig::Ptr{Cint},
                                         W::CuPtr{Cdouble}, work::CuPtr{cuDoubleComplex},
                                         lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnSsygvd_bufferSize(handle::cusolverDnHandle_t,
                                                   itype::cusolverEigType_t,
                                                   jobz::cusolverEigMode_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cfloat}, lda::Cint,
                                                   B::CuPtr{Cfloat}, ldb::Cint,
                                                   W::CuPtr{Cfloat},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnDsygvd_bufferSize(handle::cusolverDnHandle_t,
                                                   itype::cusolverEigType_t,
                                                   jobz::cusolverEigMode_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cdouble}, lda::Cint,
                                                   B::CuPtr{Cdouble}, ldb::Cint,
                                                   W::CuPtr{Cdouble},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnChegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnChegvd_bufferSize(handle::cusolverDnHandle_t,
                                                   itype::cusolverEigType_t,
                                                   jobz::cusolverEigMode_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuComplex}, lda::Cint,
                                                   B::CuPtr{cuComplex}, ldb::Cint,
                                                   W::CuPtr{Cfloat},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZhegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork)
    initialize_context()
    @ccall libcusolver.cusolverDnZhegvd_bufferSize(handle::cusolverDnHandle_t,
                                                   itype::cusolverEigType_t,
                                                   jobz::cusolverEigMode_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuDoubleComplex}, lda::Cint,
                                                   B::CuPtr{cuDoubleComplex}, ldb::Cint,
                                                   W::CuPtr{Cdouble},
                                                   lwork::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnSsygvd(handle::cusolverDnHandle_t,
                                        itype::cusolverEigType_t, jobz::cusolverEigMode_t,
                                        uplo::cublasFillMode_t, n::Cint, A::CuPtr{Cfloat},
                                        lda::Cint, B::CuPtr{Cfloat}, ldb::Cint,
                                        W::CuPtr{Cfloat}, work::CuPtr{Cfloat}, lwork::Cint,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnDsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnDsygvd(handle::cusolverDnHandle_t,
                                        itype::cusolverEigType_t, jobz::cusolverEigMode_t,
                                        uplo::cublasFillMode_t, n::Cint, A::CuPtr{Cdouble},
                                        lda::Cint, B::CuPtr{Cdouble}, ldb::Cint,
                                        W::CuPtr{Cdouble}, work::CuPtr{Cdouble},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnChegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnChegvd(handle::cusolverDnHandle_t,
                                        itype::cusolverEigType_t, jobz::cusolverEigMode_t,
                                        uplo::cublasFillMode_t, n::Cint,
                                        A::CuPtr{cuComplex}, lda::Cint, B::CuPtr{cuComplex},
                                        ldb::Cint, W::CuPtr{Cfloat}, work::CuPtr{cuComplex},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnZhegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info)
    initialize_context()
    @ccall libcusolver.cusolverDnZhegvd(handle::cusolverDnHandle_t,
                                        itype::cusolverEigType_t, jobz::cusolverEigMode_t,
                                        uplo::cublasFillMode_t, n::Cint,
                                        A::CuPtr{cuDoubleComplex}, lda::Cint,
                                        B::CuPtr{cuDoubleComplex}, ldb::Cint,
                                        W::CuPtr{Cdouble}, work::CuPtr{cuDoubleComplex},
                                        lwork::Cint, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnCreateSyevjInfo(info)
    initialize_context()
    @ccall libcusolver.cusolverDnCreateSyevjInfo(info::Ptr{syevjInfo_t})::cusolverStatus_t
end

@checked function cusolverDnDestroySyevjInfo(info)
    initialize_context()
    @ccall libcusolver.cusolverDnDestroySyevjInfo(info::syevjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnXsyevjSetTolerance(info, tolerance)
    initialize_context()
    @ccall libcusolver.cusolverDnXsyevjSetTolerance(info::syevjInfo_t,
                                                    tolerance::Cdouble)::cusolverStatus_t
end

@checked function cusolverDnXsyevjSetMaxSweeps(info, max_sweeps)
    initialize_context()
    @ccall libcusolver.cusolverDnXsyevjSetMaxSweeps(info::syevjInfo_t,
                                                    max_sweeps::Cint)::cusolverStatus_t
end

@checked function cusolverDnXsyevjSetSortEig(info, sort_eig)
    initialize_context()
    @ccall libcusolver.cusolverDnXsyevjSetSortEig(info::syevjInfo_t,
                                                  sort_eig::Cint)::cusolverStatus_t
end

@checked function cusolverDnXsyevjGetResidual(handle, info, residual)
    initialize_context()
    @ccall libcusolver.cusolverDnXsyevjGetResidual(handle::cusolverDnHandle_t,
                                                   info::syevjInfo_t,
                                                   residual::Ptr{Cdouble})::cusolverStatus_t
end

@checked function cusolverDnXsyevjGetSweeps(handle, info, executed_sweeps)
    initialize_context()
    @ccall libcusolver.cusolverDnXsyevjGetSweeps(handle::cusolverDnHandle_t,
                                                 info::syevjInfo_t,
                                                 executed_sweeps::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W,
                                                     lwork, params, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnSsyevjBatched_bufferSize(handle::cusolverDnHandle_t,
                                                          jobz::cusolverEigMode_t,
                                                          uplo::cublasFillMode_t, n::Cint,
                                                          A::CuPtr{Cfloat}, lda::Cint,
                                                          W::CuPtr{Cfloat},
                                                          lwork::Ptr{Cint},
                                                          params::syevjInfo_t,
                                                          batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnDsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W,
                                                     lwork, params, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnDsyevjBatched_bufferSize(handle::cusolverDnHandle_t,
                                                          jobz::cusolverEigMode_t,
                                                          uplo::cublasFillMode_t, n::Cint,
                                                          A::CuPtr{Cdouble}, lda::Cint,
                                                          W::CuPtr{Cdouble},
                                                          lwork::Ptr{Cint},
                                                          params::syevjInfo_t,
                                                          batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnCheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W,
                                                     lwork, params, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnCheevjBatched_bufferSize(handle::cusolverDnHandle_t,
                                                          jobz::cusolverEigMode_t,
                                                          uplo::cublasFillMode_t, n::Cint,
                                                          A::CuPtr{cuComplex}, lda::Cint,
                                                          W::CuPtr{Cfloat},
                                                          lwork::Ptr{Cint},
                                                          params::syevjInfo_t,
                                                          batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnZheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W,
                                                     lwork, params, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnZheevjBatched_bufferSize(handle::cusolverDnHandle_t,
                                                          jobz::cusolverEigMode_t,
                                                          uplo::cublasFillMode_t, n::Cint,
                                                          A::CuPtr{cuDoubleComplex},
                                                          lda::Cint, W::CuPtr{Cdouble},
                                                          lwork::Ptr{Cint},
                                                          params::syevjInfo_t,
                                                          batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnSsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork,
                                          info, params, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnSsyevjBatched(handle::cusolverDnHandle_t,
                                               jobz::cusolverEigMode_t,
                                               uplo::cublasFillMode_t, n::Cint,
                                               A::CuPtr{Cfloat}, lda::Cint,
                                               W::CuPtr{Cfloat}, work::CuPtr{Cfloat},
                                               lwork::Cint, info::CuPtr{Cint},
                                               params::syevjInfo_t,
                                               batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnDsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork,
                                          info, params, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnDsyevjBatched(handle::cusolverDnHandle_t,
                                               jobz::cusolverEigMode_t,
                                               uplo::cublasFillMode_t, n::Cint,
                                               A::CuPtr{Cdouble}, lda::Cint,
                                               W::CuPtr{Cdouble}, work::CuPtr{Cdouble},
                                               lwork::Cint, info::CuPtr{Cint},
                                               params::syevjInfo_t,
                                               batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnCheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork,
                                          info, params, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnCheevjBatched(handle::cusolverDnHandle_t,
                                               jobz::cusolverEigMode_t,
                                               uplo::cublasFillMode_t, n::Cint,
                                               A::CuPtr{cuComplex}, lda::Cint,
                                               W::CuPtr{Cfloat}, work::CuPtr{cuComplex},
                                               lwork::Cint, info::CuPtr{Cint},
                                               params::syevjInfo_t,
                                               batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnZheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork,
                                          info, params, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnZheevjBatched(handle::cusolverDnHandle_t,
                                               jobz::cusolverEigMode_t,
                                               uplo::cublasFillMode_t, n::Cint,
                                               A::CuPtr{cuDoubleComplex}, lda::Cint,
                                               W::CuPtr{Cdouble},
                                               work::CuPtr{cuDoubleComplex}, lwork::Cint,
                                               info::CuPtr{Cint}, params::syevjInfo_t,
                                               batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnSsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                              params)
    initialize_context()
    @ccall libcusolver.cusolverDnSsyevj_bufferSize(handle::cusolverDnHandle_t,
                                                   jobz::cusolverEigMode_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cfloat}, lda::Cint,
                                                   W::CuPtr{Cfloat}, lwork::Ptr{Cint},
                                                   params::syevjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnDsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                              params)
    initialize_context()
    @ccall libcusolver.cusolverDnDsyevj_bufferSize(handle::cusolverDnHandle_t,
                                                   jobz::cusolverEigMode_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cdouble}, lda::Cint,
                                                   W::CuPtr{Cdouble}, lwork::Ptr{Cint},
                                                   params::syevjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnCheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                              params)
    initialize_context()
    @ccall libcusolver.cusolverDnCheevj_bufferSize(handle::cusolverDnHandle_t,
                                                   jobz::cusolverEigMode_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuComplex}, lda::Cint,
                                                   W::CuPtr{Cfloat}, lwork::Ptr{Cint},
                                                   params::syevjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnZheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                              params)
    initialize_context()
    @ccall libcusolver.cusolverDnZheevj_bufferSize(handle::cusolverDnHandle_t,
                                                   jobz::cusolverEigMode_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuDoubleComplex}, lda::Cint,
                                                   W::CuPtr{Cdouble}, lwork::Ptr{Cint},
                                                   params::syevjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnSsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                   params)
    initialize_context()
    @ccall libcusolver.cusolverDnSsyevj(handle::cusolverDnHandle_t, jobz::cusolverEigMode_t,
                                        uplo::cublasFillMode_t, n::Cint, A::CuPtr{Cfloat},
                                        lda::Cint, W::CuPtr{Cfloat}, work::CuPtr{Cfloat},
                                        lwork::Cint, info::CuPtr{Cint},
                                        params::syevjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnDsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                   params)
    initialize_context()
    @ccall libcusolver.cusolverDnDsyevj(handle::cusolverDnHandle_t, jobz::cusolverEigMode_t,
                                        uplo::cublasFillMode_t, n::Cint, A::CuPtr{Cdouble},
                                        lda::Cint, W::CuPtr{Cdouble}, work::CuPtr{Cdouble},
                                        lwork::Cint, info::CuPtr{Cint},
                                        params::syevjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnCheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                   params)
    initialize_context()
    @ccall libcusolver.cusolverDnCheevj(handle::cusolverDnHandle_t, jobz::cusolverEigMode_t,
                                        uplo::cublasFillMode_t, n::Cint,
                                        A::CuPtr{cuComplex}, lda::Cint, W::CuPtr{Cfloat},
                                        work::CuPtr{cuComplex}, lwork::Cint,
                                        info::CuPtr{Cint},
                                        params::syevjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnZheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                   params)
    initialize_context()
    @ccall libcusolver.cusolverDnZheevj(handle::cusolverDnHandle_t, jobz::cusolverEigMode_t,
                                        uplo::cublasFillMode_t, n::Cint,
                                        A::CuPtr{cuDoubleComplex}, lda::Cint,
                                        W::CuPtr{Cdouble}, work::CuPtr{cuDoubleComplex},
                                        lwork::Cint, info::CuPtr{Cint},
                                        params::syevjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnSsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork, params)
    initialize_context()
    @ccall libcusolver.cusolverDnSsygvj_bufferSize(handle::cusolverDnHandle_t,
                                                   itype::cusolverEigType_t,
                                                   jobz::cusolverEigMode_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cfloat}, lda::Cint,
                                                   B::CuPtr{Cfloat}, ldb::Cint,
                                                   W::CuPtr{Cfloat}, lwork::Ptr{Cint},
                                                   params::syevjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnDsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork, params)
    initialize_context()
    @ccall libcusolver.cusolverDnDsygvj_bufferSize(handle::cusolverDnHandle_t,
                                                   itype::cusolverEigType_t,
                                                   jobz::cusolverEigMode_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{Cdouble}, lda::Cint,
                                                   B::CuPtr{Cdouble}, ldb::Cint,
                                                   W::CuPtr{Cdouble}, lwork::Ptr{Cint},
                                                   params::syevjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnChegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork, params)
    initialize_context()
    @ccall libcusolver.cusolverDnChegvj_bufferSize(handle::cusolverDnHandle_t,
                                                   itype::cusolverEigType_t,
                                                   jobz::cusolverEigMode_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuComplex}, lda::Cint,
                                                   B::CuPtr{cuComplex}, ldb::Cint,
                                                   W::CuPtr{Cfloat}, lwork::Ptr{Cint},
                                                   params::syevjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnZhegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb,
                                              W, lwork, params)
    initialize_context()
    @ccall libcusolver.cusolverDnZhegvj_bufferSize(handle::cusolverDnHandle_t,
                                                   itype::cusolverEigType_t,
                                                   jobz::cusolverEigMode_t,
                                                   uplo::cublasFillMode_t, n::Cint,
                                                   A::CuPtr{cuDoubleComplex}, lda::Cint,
                                                   B::CuPtr{cuDoubleComplex}, ldb::Cint,
                                                   W::CuPtr{Cdouble}, lwork::Ptr{Cint},
                                                   params::syevjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnSsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info, params)
    initialize_context()
    @ccall libcusolver.cusolverDnSsygvj(handle::cusolverDnHandle_t,
                                        itype::cusolverEigType_t, jobz::cusolverEigMode_t,
                                        uplo::cublasFillMode_t, n::Cint, A::CuPtr{Cfloat},
                                        lda::Cint, B::CuPtr{Cfloat}, ldb::Cint,
                                        W::CuPtr{Cfloat}, work::CuPtr{Cfloat}, lwork::Cint,
                                        info::CuPtr{Cint},
                                        params::syevjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnDsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info, params)
    initialize_context()
    @ccall libcusolver.cusolverDnDsygvj(handle::cusolverDnHandle_t,
                                        itype::cusolverEigType_t, jobz::cusolverEigMode_t,
                                        uplo::cublasFillMode_t, n::Cint, A::CuPtr{Cdouble},
                                        lda::Cint, B::CuPtr{Cdouble}, ldb::Cint,
                                        W::CuPtr{Cdouble}, work::CuPtr{Cdouble},
                                        lwork::Cint, info::CuPtr{Cint},
                                        params::syevjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnChegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info, params)
    initialize_context()
    @ccall libcusolver.cusolverDnChegvj(handle::cusolverDnHandle_t,
                                        itype::cusolverEigType_t, jobz::cusolverEigMode_t,
                                        uplo::cublasFillMode_t, n::Cint,
                                        A::CuPtr{cuComplex}, lda::Cint, B::CuPtr{cuComplex},
                                        ldb::Cint, W::CuPtr{Cfloat}, work::CuPtr{cuComplex},
                                        lwork::Cint, info::CuPtr{Cint},
                                        params::syevjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnZhegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work,
                                   lwork, info, params)
    initialize_context()
    @ccall libcusolver.cusolverDnZhegvj(handle::cusolverDnHandle_t,
                                        itype::cusolverEigType_t, jobz::cusolverEigMode_t,
                                        uplo::cublasFillMode_t, n::Cint,
                                        A::CuPtr{cuDoubleComplex}, lda::Cint,
                                        B::CuPtr{cuDoubleComplex}, ldb::Cint,
                                        W::CuPtr{Cdouble}, work::CuPtr{cuDoubleComplex},
                                        lwork::Cint, info::CuPtr{Cint},
                                        params::syevjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnCreateGesvdjInfo(info)
    initialize_context()
    @ccall libcusolver.cusolverDnCreateGesvdjInfo(info::Ptr{gesvdjInfo_t})::cusolverStatus_t
end

@checked function cusolverDnDestroyGesvdjInfo(info)
    initialize_context()
    @ccall libcusolver.cusolverDnDestroyGesvdjInfo(info::gesvdjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnXgesvdjSetTolerance(info, tolerance)
    initialize_context()
    @ccall libcusolver.cusolverDnXgesvdjSetTolerance(info::gesvdjInfo_t,
                                                     tolerance::Cdouble)::cusolverStatus_t
end

@checked function cusolverDnXgesvdjSetMaxSweeps(info, max_sweeps)
    initialize_context()
    @ccall libcusolver.cusolverDnXgesvdjSetMaxSweeps(info::gesvdjInfo_t,
                                                     max_sweeps::Cint)::cusolverStatus_t
end

@checked function cusolverDnXgesvdjSetSortEig(info, sort_svd)
    initialize_context()
    @ccall libcusolver.cusolverDnXgesvdjSetSortEig(info::gesvdjInfo_t,
                                                   sort_svd::Cint)::cusolverStatus_t
end

@checked function cusolverDnXgesvdjGetResidual(handle, info, residual)
    initialize_context()
    @ccall libcusolver.cusolverDnXgesvdjGetResidual(handle::cusolverDnHandle_t,
                                                    info::gesvdjInfo_t,
                                                    residual::Ptr{Cdouble})::cusolverStatus_t
end

@checked function cusolverDnXgesvdjGetSweeps(handle, info, executed_sweeps)
    initialize_context()
    @ccall libcusolver.cusolverDnXgesvdjGetSweeps(handle::cusolverDnHandle_t,
                                                  info::gesvdjInfo_t,
                                                  executed_sweeps::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu,
                                                      V, ldv, lwork, params, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnSgesvdjBatched_bufferSize(handle::cusolverDnHandle_t,
                                                           jobz::cusolverEigMode_t, m::Cint,
                                                           n::Cint, A::CuPtr{Cfloat},
                                                           lda::Cint, S::CuPtr{Cfloat},
                                                           U::CuPtr{Cfloat}, ldu::Cint,
                                                           V::CuPtr{Cfloat}, ldv::Cint,
                                                           lwork::Ptr{Cint},
                                                           params::gesvdjInfo_t,
                                                           batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnDgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu,
                                                      V, ldv, lwork, params, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnDgesvdjBatched_bufferSize(handle::cusolverDnHandle_t,
                                                           jobz::cusolverEigMode_t, m::Cint,
                                                           n::Cint, A::CuPtr{Cdouble},
                                                           lda::Cint, S::CuPtr{Cdouble},
                                                           U::CuPtr{Cdouble}, ldu::Cint,
                                                           V::CuPtr{Cdouble}, ldv::Cint,
                                                           lwork::Ptr{Cint},
                                                           params::gesvdjInfo_t,
                                                           batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnCgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu,
                                                      V, ldv, lwork, params, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnCgesvdjBatched_bufferSize(handle::cusolverDnHandle_t,
                                                           jobz::cusolverEigMode_t, m::Cint,
                                                           n::Cint, A::CuPtr{cuComplex},
                                                           lda::Cint, S::CuPtr{Cfloat},
                                                           U::CuPtr{cuComplex}, ldu::Cint,
                                                           V::CuPtr{cuComplex}, ldv::Cint,
                                                           lwork::Ptr{Cint},
                                                           params::gesvdjInfo_t,
                                                           batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnZgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu,
                                                      V, ldv, lwork, params, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnZgesvdjBatched_bufferSize(handle::cusolverDnHandle_t,
                                                           jobz::cusolverEigMode_t, m::Cint,
                                                           n::Cint,
                                                           A::CuPtr{cuDoubleComplex},
                                                           lda::Cint, S::CuPtr{Cdouble},
                                                           U::CuPtr{cuDoubleComplex},
                                                           ldu::Cint,
                                                           V::CuPtr{cuDoubleComplex},
                                                           ldv::Cint, lwork::Ptr{Cint},
                                                           params::gesvdjInfo_t,
                                                           batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnSgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                           work, lwork, info, params, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnSgesvdjBatched(handle::cusolverDnHandle_t,
                                                jobz::cusolverEigMode_t, m::Cint, n::Cint,
                                                A::CuPtr{Cfloat}, lda::Cint,
                                                S::CuPtr{Cfloat}, U::CuPtr{Cfloat},
                                                ldu::Cint, V::CuPtr{Cfloat}, ldv::Cint,
                                                work::CuPtr{Cfloat}, lwork::Cint,
                                                info::CuPtr{Cint}, params::gesvdjInfo_t,
                                                batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnDgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                           work, lwork, info, params, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnDgesvdjBatched(handle::cusolverDnHandle_t,
                                                jobz::cusolverEigMode_t, m::Cint, n::Cint,
                                                A::CuPtr{Cdouble}, lda::Cint,
                                                S::CuPtr{Cdouble}, U::CuPtr{Cdouble},
                                                ldu::Cint, V::CuPtr{Cdouble}, ldv::Cint,
                                                work::CuPtr{Cdouble}, lwork::Cint,
                                                info::CuPtr{Cint}, params::gesvdjInfo_t,
                                                batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnCgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                           work, lwork, info, params, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnCgesvdjBatched(handle::cusolverDnHandle_t,
                                                jobz::cusolverEigMode_t, m::Cint, n::Cint,
                                                A::CuPtr{cuComplex}, lda::Cint,
                                                S::CuPtr{Cfloat}, U::CuPtr{cuComplex},
                                                ldu::Cint, V::CuPtr{cuComplex}, ldv::Cint,
                                                work::CuPtr{cuComplex}, lwork::Cint,
                                                info::CuPtr{Cint}, params::gesvdjInfo_t,
                                                batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnZgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                           work, lwork, info, params, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnZgesvdjBatched(handle::cusolverDnHandle_t,
                                                jobz::cusolverEigMode_t, m::Cint, n::Cint,
                                                A::CuPtr{cuDoubleComplex}, lda::Cint,
                                                S::CuPtr{Cdouble},
                                                U::CuPtr{cuDoubleComplex}, ldu::Cint,
                                                V::CuPtr{cuDoubleComplex}, ldv::Cint,
                                                work::CuPtr{cuDoubleComplex}, lwork::Cint,
                                                info::CuPtr{Cint}, params::gesvdjInfo_t,
                                                batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu,
                                               V, ldv, lwork, params)
    initialize_context()
    @ccall libcusolver.cusolverDnSgesvdj_bufferSize(handle::cusolverDnHandle_t,
                                                    jobz::cusolverEigMode_t, econ::Cint,
                                                    m::Cint, n::Cint, A::CuPtr{Cfloat},
                                                    lda::Cint, S::CuPtr{Cfloat},
                                                    U::CuPtr{Cfloat}, ldu::Cint,
                                                    V::CuPtr{Cfloat}, ldv::Cint,
                                                    lwork::Ptr{Cint},
                                                    params::gesvdjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu,
                                               V, ldv, lwork, params)
    initialize_context()
    @ccall libcusolver.cusolverDnDgesvdj_bufferSize(handle::cusolverDnHandle_t,
                                                    jobz::cusolverEigMode_t, econ::Cint,
                                                    m::Cint, n::Cint, A::CuPtr{Cdouble},
                                                    lda::Cint, S::CuPtr{Cdouble},
                                                    U::CuPtr{Cdouble}, ldu::Cint,
                                                    V::CuPtr{Cdouble}, ldv::Cint,
                                                    lwork::Ptr{Cint},
                                                    params::gesvdjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnCgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu,
                                               V, ldv, lwork, params)
    initialize_context()
    @ccall libcusolver.cusolverDnCgesvdj_bufferSize(handle::cusolverDnHandle_t,
                                                    jobz::cusolverEigMode_t, econ::Cint,
                                                    m::Cint, n::Cint, A::CuPtr{cuComplex},
                                                    lda::Cint, S::CuPtr{Cfloat},
                                                    U::CuPtr{cuComplex}, ldu::Cint,
                                                    V::CuPtr{cuComplex}, ldv::Cint,
                                                    lwork::Ptr{Cint},
                                                    params::gesvdjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnZgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu,
                                               V, ldv, lwork, params)
    initialize_context()
    @ccall libcusolver.cusolverDnZgesvdj_bufferSize(handle::cusolverDnHandle_t,
                                                    jobz::cusolverEigMode_t, econ::Cint,
                                                    m::Cint, n::Cint,
                                                    A::CuPtr{cuDoubleComplex}, lda::Cint,
                                                    S::CuPtr{Cdouble},
                                                    U::CuPtr{cuDoubleComplex}, ldu::Cint,
                                                    V::CuPtr{cuDoubleComplex}, ldv::Cint,
                                                    lwork::Ptr{Cint},
                                                    params::gesvdjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnSgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                    work, lwork, info, params)
    initialize_context()
    @ccall libcusolver.cusolverDnSgesvdj(handle::cusolverDnHandle_t,
                                         jobz::cusolverEigMode_t, econ::Cint, m::Cint,
                                         n::Cint, A::CuPtr{Cfloat}, lda::Cint,
                                         S::CuPtr{Cfloat}, U::CuPtr{Cfloat}, ldu::Cint,
                                         V::CuPtr{Cfloat}, ldv::Cint, work::CuPtr{Cfloat},
                                         lwork::Cint, info::CuPtr{Cint},
                                         params::gesvdjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnDgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                    work, lwork, info, params)
    initialize_context()
    @ccall libcusolver.cusolverDnDgesvdj(handle::cusolverDnHandle_t,
                                         jobz::cusolverEigMode_t, econ::Cint, m::Cint,
                                         n::Cint, A::CuPtr{Cdouble}, lda::Cint,
                                         S::CuPtr{Cdouble}, U::CuPtr{Cdouble}, ldu::Cint,
                                         V::CuPtr{Cdouble}, ldv::Cint, work::CuPtr{Cdouble},
                                         lwork::Cint, info::CuPtr{Cint},
                                         params::gesvdjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnCgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                    work, lwork, info, params)
    initialize_context()
    @ccall libcusolver.cusolverDnCgesvdj(handle::cusolverDnHandle_t,
                                         jobz::cusolverEigMode_t, econ::Cint, m::Cint,
                                         n::Cint, A::CuPtr{cuComplex}, lda::Cint,
                                         S::CuPtr{Cfloat}, U::CuPtr{cuComplex}, ldu::Cint,
                                         V::CuPtr{cuComplex}, ldv::Cint,
                                         work::CuPtr{cuComplex}, lwork::Cint,
                                         info::CuPtr{Cint},
                                         params::gesvdjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnZgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                    work, lwork, info, params)
    initialize_context()
    @ccall libcusolver.cusolverDnZgesvdj(handle::cusolverDnHandle_t,
                                         jobz::cusolverEigMode_t, econ::Cint, m::Cint,
                                         n::Cint, A::CuPtr{cuDoubleComplex}, lda::Cint,
                                         S::CuPtr{Cdouble}, U::CuPtr{cuDoubleComplex},
                                         ldu::Cint, V::CuPtr{cuDoubleComplex}, ldv::Cint,
                                         work::CuPtr{cuDoubleComplex}, lwork::Cint,
                                         info::CuPtr{Cint},
                                         params::gesvdjInfo_t)::cusolverStatus_t
end

@checked function cusolverDnSgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A,
                                                             lda, strideA, d_S, strideS,
                                                             d_U, ldu, strideU, d_V, ldv,
                                                             strideV, lwork, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnSgesvdaStridedBatched_bufferSize(handle::cusolverDnHandle_t,
                                                                  jobz::cusolverEigMode_t,
                                                                  rank::Cint, m::Cint,
                                                                  n::Cint,
                                                                  d_A::CuPtr{Cfloat},
                                                                  lda::Cint,
                                                                  strideA::Clonglong,
                                                                  d_S::CuPtr{Cfloat},
                                                                  strideS::Clonglong,
                                                                  d_U::CuPtr{Cfloat},
                                                                  ldu::Cint,
                                                                  strideU::Clonglong,
                                                                  d_V::CuPtr{Cfloat},
                                                                  ldv::Cint,
                                                                  strideV::Clonglong,
                                                                  lwork::Ptr{Cint},
                                                                  batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnDgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A,
                                                             lda, strideA, d_S, strideS,
                                                             d_U, ldu, strideU, d_V, ldv,
                                                             strideV, lwork, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnDgesvdaStridedBatched_bufferSize(handle::cusolverDnHandle_t,
                                                                  jobz::cusolverEigMode_t,
                                                                  rank::Cint, m::Cint,
                                                                  n::Cint,
                                                                  d_A::CuPtr{Cdouble},
                                                                  lda::Cint,
                                                                  strideA::Clonglong,
                                                                  d_S::CuPtr{Cdouble},
                                                                  strideS::Clonglong,
                                                                  d_U::CuPtr{Cdouble},
                                                                  ldu::Cint,
                                                                  strideU::Clonglong,
                                                                  d_V::CuPtr{Cdouble},
                                                                  ldv::Cint,
                                                                  strideV::Clonglong,
                                                                  lwork::Ptr{Cint},
                                                                  batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnCgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A,
                                                             lda, strideA, d_S, strideS,
                                                             d_U, ldu, strideU, d_V, ldv,
                                                             strideV, lwork, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnCgesvdaStridedBatched_bufferSize(handle::cusolverDnHandle_t,
                                                                  jobz::cusolverEigMode_t,
                                                                  rank::Cint, m::Cint,
                                                                  n::Cint,
                                                                  d_A::CuPtr{cuComplex},
                                                                  lda::Cint,
                                                                  strideA::Clonglong,
                                                                  d_S::CuPtr{Cfloat},
                                                                  strideS::Clonglong,
                                                                  d_U::CuPtr{cuComplex},
                                                                  ldu::Cint,
                                                                  strideU::Clonglong,
                                                                  d_V::CuPtr{cuComplex},
                                                                  ldv::Cint,
                                                                  strideV::Clonglong,
                                                                  lwork::Ptr{Cint},
                                                                  batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnZgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A,
                                                             lda, strideA, d_S, strideS,
                                                             d_U, ldu, strideU, d_V, ldv,
                                                             strideV, lwork, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnZgesvdaStridedBatched_bufferSize(handle::cusolverDnHandle_t,
                                                                  jobz::cusolverEigMode_t,
                                                                  rank::Cint, m::Cint,
                                                                  n::Cint,
                                                                  d_A::CuPtr{cuDoubleComplex},
                                                                  lda::Cint,
                                                                  strideA::Clonglong,
                                                                  d_S::CuPtr{Cdouble},
                                                                  strideS::Clonglong,
                                                                  d_U::CuPtr{cuDoubleComplex},
                                                                  ldu::Cint,
                                                                  strideU::Clonglong,
                                                                  d_V::CuPtr{cuDoubleComplex},
                                                                  ldv::Cint,
                                                                  strideV::Clonglong,
                                                                  lwork::Ptr{Cint},
                                                                  batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnSgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda,
                                                  strideA, d_S, strideS, d_U, ldu, strideU,
                                                  d_V, ldv, strideV, d_work, lwork, d_info,
                                                  h_R_nrmF, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnSgesvdaStridedBatched(handle::cusolverDnHandle_t,
                                                       jobz::cusolverEigMode_t, rank::Cint,
                                                       m::Cint, n::Cint, d_A::CuPtr{Cfloat},
                                                       lda::Cint, strideA::Clonglong,
                                                       d_S::CuPtr{Cfloat},
                                                       strideS::Clonglong,
                                                       d_U::CuPtr{Cfloat}, ldu::Cint,
                                                       strideU::Clonglong,
                                                       d_V::CuPtr{Cfloat}, ldv::Cint,
                                                       strideV::Clonglong,
                                                       d_work::CuPtr{Cfloat}, lwork::Cint,
                                                       d_info::CuPtr{Cint},
                                                       h_R_nrmF::Ptr{Cdouble},
                                                       batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnDgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda,
                                                  strideA, d_S, strideS, d_U, ldu, strideU,
                                                  d_V, ldv, strideV, d_work, lwork, d_info,
                                                  h_R_nrmF, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnDgesvdaStridedBatched(handle::cusolverDnHandle_t,
                                                       jobz::cusolverEigMode_t, rank::Cint,
                                                       m::Cint, n::Cint,
                                                       d_A::CuPtr{Cdouble}, lda::Cint,
                                                       strideA::Clonglong,
                                                       d_S::CuPtr{Cdouble},
                                                       strideS::Clonglong,
                                                       d_U::CuPtr{Cdouble}, ldu::Cint,
                                                       strideU::Clonglong,
                                                       d_V::CuPtr{Cdouble}, ldv::Cint,
                                                       strideV::Clonglong,
                                                       d_work::CuPtr{Cdouble}, lwork::Cint,
                                                       d_info::CuPtr{Cint},
                                                       h_R_nrmF::Ptr{Cdouble},
                                                       batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnCgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda,
                                                  strideA, d_S, strideS, d_U, ldu, strideU,
                                                  d_V, ldv, strideV, d_work, lwork, d_info,
                                                  h_R_nrmF, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnCgesvdaStridedBatched(handle::cusolverDnHandle_t,
                                                       jobz::cusolverEigMode_t, rank::Cint,
                                                       m::Cint, n::Cint,
                                                       d_A::CuPtr{cuComplex}, lda::Cint,
                                                       strideA::Clonglong,
                                                       d_S::CuPtr{Cfloat},
                                                       strideS::Clonglong,
                                                       d_U::CuPtr{cuComplex}, ldu::Cint,
                                                       strideU::Clonglong,
                                                       d_V::CuPtr{cuComplex}, ldv::Cint,
                                                       strideV::Clonglong,
                                                       d_work::CuPtr{cuComplex},
                                                       lwork::Cint, d_info::CuPtr{Cint},
                                                       h_R_nrmF::Ptr{Cdouble},
                                                       batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnZgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda,
                                                  strideA, d_S, strideS, d_U, ldu, strideU,
                                                  d_V, ldv, strideV, d_work, lwork, d_info,
                                                  h_R_nrmF, batchSize)
    initialize_context()
    @ccall libcusolver.cusolverDnZgesvdaStridedBatched(handle::cusolverDnHandle_t,
                                                       jobz::cusolverEigMode_t, rank::Cint,
                                                       m::Cint, n::Cint,
                                                       d_A::CuPtr{cuDoubleComplex},
                                                       lda::Cint, strideA::Clonglong,
                                                       d_S::CuPtr{Cdouble},
                                                       strideS::Clonglong,
                                                       d_U::CuPtr{cuDoubleComplex},
                                                       ldu::Cint, strideU::Clonglong,
                                                       d_V::CuPtr{cuDoubleComplex},
                                                       ldv::Cint, strideV::Clonglong,
                                                       d_work::CuPtr{cuDoubleComplex},
                                                       lwork::Cint, d_info::CuPtr{Cint},
                                                       h_R_nrmF::Ptr{Cdouble},
                                                       batchSize::Cint)::cusolverStatus_t
end

@checked function cusolverDnCreateParams(params)
    initialize_context()
    @ccall libcusolver.cusolverDnCreateParams(params::Ptr{cusolverDnParams_t})::cusolverStatus_t
end

@checked function cusolverDnDestroyParams(params)
    initialize_context()
    @ccall libcusolver.cusolverDnDestroyParams(params::cusolverDnParams_t)::cusolverStatus_t
end

@checked function cusolverDnSetAdvOptions(params, _function, algo)
    initialize_context()
    @ccall libcusolver.cusolverDnSetAdvOptions(params::cusolverDnParams_t,
                                               _function::cusolverDnFunction_t,
                                               algo::cusolverAlgMode_t)::cusolverStatus_t
end

@checked function cusolverDnPotrf_bufferSize(handle, params, uplo, n, dataTypeA, A, lda,
                                             computeType, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverDnPotrf_bufferSize(handle::cusolverDnHandle_t,
                                                  params::cusolverDnParams_t,
                                                  uplo::cublasFillMode_t, n::Int64,
                                                  dataTypeA::cudaDataType, A::CuPtr{Cvoid},
                                                  lda::Int64, computeType::cudaDataType,
                                                  workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnPotrf(handle, params, uplo, n, dataTypeA, A, lda, computeType,
                                  pBuffer, workspaceInBytes, info)
    initialize_context()
    @ccall libcusolver.cusolverDnPotrf(handle::cusolverDnHandle_t,
                                       params::cusolverDnParams_t, uplo::cublasFillMode_t,
                                       n::Int64, dataTypeA::cudaDataType, A::CuPtr{Cvoid},
                                       lda::Int64, computeType::cudaDataType,
                                       pBuffer::CuPtr{Cvoid}, workspaceInBytes::Csize_t,
                                       info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnPotrs(handle, params, uplo, n, nrhs, dataTypeA, A, lda,
                                  dataTypeB, B, ldb, info)
    initialize_context()
    @ccall libcusolver.cusolverDnPotrs(handle::cusolverDnHandle_t,
                                       params::cusolverDnParams_t, uplo::cublasFillMode_t,
                                       n::Int64, nrhs::Int64, dataTypeA::cudaDataType,
                                       A::CuPtr{Cvoid}, lda::Int64, dataTypeB::cudaDataType,
                                       B::CuPtr{Cvoid}, ldb::Int64,
                                       info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnGeqrf_bufferSize(handle, params, m, n, dataTypeA, A, lda,
                                             dataTypeTau, tau, computeType,
                                             workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverDnGeqrf_bufferSize(handle::cusolverDnHandle_t,
                                                  params::cusolverDnParams_t, m::Int64,
                                                  n::Int64, dataTypeA::cudaDataType,
                                                  A::CuPtr{Cvoid}, lda::Int64,
                                                  dataTypeTau::cudaDataType,
                                                  tau::CuPtr{Cvoid},
                                                  computeType::cudaDataType,
                                                  workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnGeqrf(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau,
                                  computeType, pBuffer, workspaceInBytes, info)
    initialize_context()
    @ccall libcusolver.cusolverDnGeqrf(handle::cusolverDnHandle_t,
                                       params::cusolverDnParams_t, m::Int64, n::Int64,
                                       dataTypeA::cudaDataType, A::CuPtr{Cvoid}, lda::Int64,
                                       dataTypeTau::cudaDataType, tau::CuPtr{Cvoid},
                                       computeType::cudaDataType, pBuffer::CuPtr{Cvoid},
                                       workspaceInBytes::Csize_t,
                                       info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnGetrf_bufferSize(handle, params, m, n, dataTypeA, A, lda,
                                             computeType, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverDnGetrf_bufferSize(handle::cusolverDnHandle_t,
                                                  params::cusolverDnParams_t, m::Int64,
                                                  n::Int64, dataTypeA::cudaDataType,
                                                  A::CuPtr{Cvoid}, lda::Int64,
                                                  computeType::cudaDataType,
                                                  workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnGetrf(handle, params, m, n, dataTypeA, A, lda, ipiv,
                                  computeType, pBuffer, workspaceInBytes, info)
    initialize_context()
    @ccall libcusolver.cusolverDnGetrf(handle::cusolverDnHandle_t,
                                       params::cusolverDnParams_t, m::Int64, n::Int64,
                                       dataTypeA::cudaDataType, A::CuPtr{Cvoid}, lda::Int64,
                                       ipiv::CuPtr{Int64}, computeType::cudaDataType,
                                       pBuffer::CuPtr{Cvoid}, workspaceInBytes::Csize_t,
                                       info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnGetrs(handle, params, trans, n, nrhs, dataTypeA, A, lda, ipiv,
                                  dataTypeB, B, ldb, info)
    initialize_context()
    @ccall libcusolver.cusolverDnGetrs(handle::cusolverDnHandle_t,
                                       params::cusolverDnParams_t, trans::cublasOperation_t,
                                       n::Int64, nrhs::Int64, dataTypeA::cudaDataType,
                                       A::CuPtr{Cvoid}, lda::Int64, ipiv::CuPtr{Int64},
                                       dataTypeB::cudaDataType, B::CuPtr{Cvoid}, ldb::Int64,
                                       info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSyevd_bufferSize(handle, params, jobz, uplo, n, dataTypeA, A,
                                             lda, dataTypeW, W, computeType,
                                             workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverDnSyevd_bufferSize(handle::cusolverDnHandle_t,
                                                  params::cusolverDnParams_t,
                                                  jobz::cusolverEigMode_t,
                                                  uplo::cublasFillMode_t, n::Int64,
                                                  dataTypeA::cudaDataType, A::CuPtr{Cvoid},
                                                  lda::Int64, dataTypeW::cudaDataType,
                                                  W::CuPtr{Cvoid},
                                                  computeType::cudaDataType,
                                                  workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnSyevd(handle, params, jobz, uplo, n, dataTypeA, A, lda,
                                  dataTypeW, W, computeType, pBuffer, workspaceInBytes,
                                  info)
    initialize_context()
    @ccall libcusolver.cusolverDnSyevd(handle::cusolverDnHandle_t,
                                       params::cusolverDnParams_t, jobz::cusolverEigMode_t,
                                       uplo::cublasFillMode_t, n::Int64,
                                       dataTypeA::cudaDataType, A::CuPtr{Cvoid}, lda::Int64,
                                       dataTypeW::cudaDataType, W::CuPtr{Cvoid},
                                       computeType::cudaDataType, pBuffer::CuPtr{Cvoid},
                                       workspaceInBytes::Csize_t,
                                       info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnSyevdx_bufferSize(handle, params, jobz, range, uplo, n,
                                              dataTypeA, A, lda, vl, vu, il, iu, h_meig,
                                              dataTypeW, W, computeType, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverDnSyevdx_bufferSize(handle::cusolverDnHandle_t,
                                                   params::cusolverDnParams_t,
                                                   jobz::cusolverEigMode_t,
                                                   range::cusolverEigRange_t,
                                                   uplo::cublasFillMode_t, n::Int64,
                                                   dataTypeA::cudaDataType, A::CuPtr{Cvoid},
                                                   lda::Int64, vl::Ptr{Cvoid},
                                                   vu::Ptr{Cvoid}, il::Int64, iu::Int64,
                                                   h_meig::Ptr{Int64},
                                                   dataTypeW::cudaDataType, W::CuPtr{Cvoid},
                                                   computeType::cudaDataType,
                                                   workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnSyevdx(handle, params, jobz, range, uplo, n, dataTypeA, A, lda,
                                   vl, vu, il, iu, meig64, dataTypeW, W, computeType,
                                   pBuffer, workspaceInBytes, info)
    initialize_context()
    @ccall libcusolver.cusolverDnSyevdx(handle::cusolverDnHandle_t,
                                        params::cusolverDnParams_t, jobz::cusolverEigMode_t,
                                        range::cusolverEigRange_t, uplo::cublasFillMode_t,
                                        n::Int64, dataTypeA::cudaDataType, A::CuPtr{Cvoid},
                                        lda::Int64, vl::Ptr{Cvoid}, vu::Ptr{Cvoid},
                                        il::Int64, iu::Int64, meig64::Ptr{Int64},
                                        dataTypeW::cudaDataType, W::CuPtr{Cvoid},
                                        computeType::cudaDataType, pBuffer::CuPtr{Cvoid},
                                        workspaceInBytes::Csize_t,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnGesvd_bufferSize(handle, params, jobu, jobvt, m, n, dataTypeA,
                                             A, lda, dataTypeS, S, dataTypeU, U, ldu,
                                             dataTypeVT, VT, ldvt, computeType,
                                             workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverDnGesvd_bufferSize(handle::cusolverDnHandle_t,
                                                  params::cusolverDnParams_t, jobu::Int8,
                                                  jobvt::Int8, m::Int64, n::Int64,
                                                  dataTypeA::cudaDataType, A::CuPtr{Cvoid},
                                                  lda::Int64, dataTypeS::cudaDataType,
                                                  S::CuPtr{Cvoid}, dataTypeU::cudaDataType,
                                                  U::CuPtr{Cvoid}, ldu::Int64,
                                                  dataTypeVT::cudaDataType,
                                                  VT::CuPtr{Cvoid}, ldvt::Int64,
                                                  computeType::cudaDataType,
                                                  workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnGesvd(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda,
                                  dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt,
                                  computeType, pBuffer, workspaceInBytes, info)
    initialize_context()
    @ccall libcusolver.cusolverDnGesvd(handle::cusolverDnHandle_t,
                                       params::cusolverDnParams_t, jobu::Int8, jobvt::Int8,
                                       m::Int64, n::Int64, dataTypeA::cudaDataType,
                                       A::CuPtr{Cvoid}, lda::Int64, dataTypeS::cudaDataType,
                                       S::CuPtr{Cvoid}, dataTypeU::cudaDataType,
                                       U::CuPtr{Cvoid}, ldu::Int64,
                                       dataTypeVT::cudaDataType, VT::CuPtr{Cvoid},
                                       ldvt::Int64, computeType::cudaDataType,
                                       pBuffer::CuPtr{Cvoid}, workspaceInBytes::Csize_t,
                                       info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnXpotrf_bufferSize(handle, params, uplo, n, dataTypeA, A, lda,
                                              computeType, workspaceInBytesOnDevice,
                                              workspaceInBytesOnHost)
    initialize_context()
    @ccall libcusolver.cusolverDnXpotrf_bufferSize(handle::cusolverDnHandle_t,
                                                   params::cusolverDnParams_t,
                                                   uplo::cublasFillMode_t, n::Int64,
                                                   dataTypeA::cudaDataType, A::CuPtr{Cvoid},
                                                   lda::Int64, computeType::cudaDataType,
                                                   workspaceInBytesOnDevice::Ptr{Csize_t},
                                                   workspaceInBytesOnHost::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnXpotrf(handle, params, uplo, n, dataTypeA, A, lda, computeType,
                                   bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost,
                                   workspaceInBytesOnHost, info)
    initialize_context()
    @ccall libcusolver.cusolverDnXpotrf(handle::cusolverDnHandle_t,
                                        params::cusolverDnParams_t, uplo::cublasFillMode_t,
                                        n::Int64, dataTypeA::cudaDataType, A::CuPtr{Cvoid},
                                        lda::Int64, computeType::cudaDataType,
                                        bufferOnDevice::CuPtr{Cvoid},
                                        workspaceInBytesOnDevice::Csize_t,
                                        bufferOnHost::Ptr{Cvoid},
                                        workspaceInBytesOnHost::Csize_t,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnXpotrs(handle, params, uplo, n, nrhs, dataTypeA, A, lda,
                                   dataTypeB, B, ldb, info)
    initialize_context()
    @ccall libcusolver.cusolverDnXpotrs(handle::cusolverDnHandle_t,
                                        params::cusolverDnParams_t, uplo::cublasFillMode_t,
                                        n::Int64, nrhs::Int64, dataTypeA::cudaDataType,
                                        A::CuPtr{Cvoid}, lda::Int64,
                                        dataTypeB::cudaDataType, B::CuPtr{Cvoid},
                                        ldb::Int64, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnXgeqrf_bufferSize(handle, params, m, n, dataTypeA, A, lda,
                                              dataTypeTau, tau, computeType,
                                              workspaceInBytesOnDevice,
                                              workspaceInBytesOnHost)
    initialize_context()
    @ccall libcusolver.cusolverDnXgeqrf_bufferSize(handle::cusolverDnHandle_t,
                                                   params::cusolverDnParams_t, m::Int64,
                                                   n::Int64, dataTypeA::cudaDataType,
                                                   A::CuPtr{Cvoid}, lda::Int64,
                                                   dataTypeTau::cudaDataType,
                                                   tau::CuPtr{Cvoid},
                                                   computeType::cudaDataType,
                                                   workspaceInBytesOnDevice::Ptr{Csize_t},
                                                   workspaceInBytesOnHost::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnXgeqrf(handle, params, m, n, dataTypeA, A, lda, dataTypeTau,
                                   tau, computeType, bufferOnDevice,
                                   workspaceInBytesOnDevice, bufferOnHost,
                                   workspaceInBytesOnHost, info)
    initialize_context()
    @ccall libcusolver.cusolverDnXgeqrf(handle::cusolverDnHandle_t,
                                        params::cusolverDnParams_t, m::Int64, n::Int64,
                                        dataTypeA::cudaDataType, A::CuPtr{Cvoid},
                                        lda::Int64, dataTypeTau::cudaDataType,
                                        tau::CuPtr{Cvoid}, computeType::cudaDataType,
                                        bufferOnDevice::CuPtr{Cvoid},
                                        workspaceInBytesOnDevice::Csize_t,
                                        bufferOnHost::Ptr{Cvoid},
                                        workspaceInBytesOnHost::Csize_t,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnXgetrf_bufferSize(handle, params, m, n, dataTypeA, A, lda,
                                              computeType, workspaceInBytesOnDevice,
                                              workspaceInBytesOnHost)
    initialize_context()
    @ccall libcusolver.cusolverDnXgetrf_bufferSize(handle::cusolverDnHandle_t,
                                                   params::cusolverDnParams_t, m::Int64,
                                                   n::Int64, dataTypeA::cudaDataType,
                                                   A::CuPtr{Cvoid}, lda::Int64,
                                                   computeType::cudaDataType,
                                                   workspaceInBytesOnDevice::Ptr{Csize_t},
                                                   workspaceInBytesOnHost::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnXgetrf(handle, params, m, n, dataTypeA, A, lda, ipiv,
                                   computeType, bufferOnDevice, workspaceInBytesOnDevice,
                                   bufferOnHost, workspaceInBytesOnHost, info)
    initialize_context()
    @ccall libcusolver.cusolverDnXgetrf(handle::cusolverDnHandle_t,
                                        params::cusolverDnParams_t, m::Int64, n::Int64,
                                        dataTypeA::cudaDataType, A::CuPtr{Cvoid},
                                        lda::Int64, ipiv::CuPtr{Int64},
                                        computeType::cudaDataType,
                                        bufferOnDevice::CuPtr{Cvoid},
                                        workspaceInBytesOnDevice::Csize_t,
                                        bufferOnHost::Ptr{Cvoid},
                                        workspaceInBytesOnHost::Csize_t,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnXgetrs(handle, params, trans, n, nrhs, dataTypeA, A, lda, ipiv,
                                   dataTypeB, B, ldb, info)
    initialize_context()
    @ccall libcusolver.cusolverDnXgetrs(handle::cusolverDnHandle_t,
                                        params::cusolverDnParams_t,
                                        trans::cublasOperation_t, n::Int64, nrhs::Int64,
                                        dataTypeA::cudaDataType, A::CuPtr{Cvoid},
                                        lda::Int64, ipiv::CuPtr{Int64},
                                        dataTypeB::cudaDataType, B::CuPtr{Cvoid},
                                        ldb::Int64, info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnXsyevd_bufferSize(handle, params, jobz, uplo, n, dataTypeA, A,
                                              lda, dataTypeW, W, computeType,
                                              workspaceInBytesOnDevice,
                                              workspaceInBytesOnHost)
    initialize_context()
    @ccall libcusolver.cusolverDnXsyevd_bufferSize(handle::cusolverDnHandle_t,
                                                   params::cusolverDnParams_t,
                                                   jobz::cusolverEigMode_t,
                                                   uplo::cublasFillMode_t, n::Int64,
                                                   dataTypeA::cudaDataType, A::CuPtr{Cvoid},
                                                   lda::Int64, dataTypeW::cudaDataType,
                                                   W::CuPtr{Cvoid},
                                                   computeType::cudaDataType,
                                                   workspaceInBytesOnDevice::Ptr{Csize_t},
                                                   workspaceInBytesOnHost::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnXsyevd(handle, params, jobz, uplo, n, dataTypeA, A, lda,
                                   dataTypeW, W, computeType, bufferOnDevice,
                                   workspaceInBytesOnDevice, bufferOnHost,
                                   workspaceInBytesOnHost, info)
    initialize_context()
    @ccall libcusolver.cusolverDnXsyevd(handle::cusolverDnHandle_t,
                                        params::cusolverDnParams_t, jobz::cusolverEigMode_t,
                                        uplo::cublasFillMode_t, n::Int64,
                                        dataTypeA::cudaDataType, A::CuPtr{Cvoid},
                                        lda::Int64, dataTypeW::cudaDataType,
                                        W::CuPtr{Cvoid}, computeType::cudaDataType,
                                        bufferOnDevice::CuPtr{Cvoid},
                                        workspaceInBytesOnDevice::Csize_t,
                                        bufferOnHost::Ptr{Cvoid},
                                        workspaceInBytesOnHost::Csize_t,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnXsyevdx_bufferSize(handle, params, jobz, range, uplo, n,
                                               dataTypeA, A, lda, vl, vu, il, iu, h_meig,
                                               dataTypeW, W, computeType,
                                               workspaceInBytesOnDevice,
                                               workspaceInBytesOnHost)
    initialize_context()
    @ccall libcusolver.cusolverDnXsyevdx_bufferSize(handle::cusolverDnHandle_t,
                                                    params::cusolverDnParams_t,
                                                    jobz::cusolverEigMode_t,
                                                    range::cusolverEigRange_t,
                                                    uplo::cublasFillMode_t, n::Int64,
                                                    dataTypeA::cudaDataType,
                                                    A::CuPtr{Cvoid}, lda::Int64,
                                                    vl::CuPtr{Cvoid}, vu::CuPtr{Cvoid},
                                                    il::Int64, iu::Int64,
                                                    h_meig::CuPtr{Int64},
                                                    dataTypeW::cudaDataType,
                                                    W::CuPtr{Cvoid},
                                                    computeType::cudaDataType,
                                                    workspaceInBytesOnDevice::Ptr{Csize_t},
                                                    workspaceInBytesOnHost::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnXsyevdx(handle, params, jobz, range, uplo, n, dataTypeA, A, lda,
                                    vl, vu, il, iu, meig64, dataTypeW, W, computeType,
                                    bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost,
                                    workspaceInBytesOnHost, info)
    initialize_context()
    @ccall libcusolver.cusolverDnXsyevdx(handle::cusolverDnHandle_t,
                                         params::cusolverDnParams_t,
                                         jobz::cusolverEigMode_t, range::cusolverEigRange_t,
                                         uplo::cublasFillMode_t, n::Int64,
                                         dataTypeA::cudaDataType, A::CuPtr{Cvoid},
                                         lda::Int64, vl::CuPtr{Cvoid}, vu::CuPtr{Cvoid},
                                         il::Int64, iu::Int64, meig64::CuPtr{Int64},
                                         dataTypeW::cudaDataType, W::CuPtr{Cvoid},
                                         computeType::cudaDataType,
                                         bufferOnDevice::CuPtr{Cvoid},
                                         workspaceInBytesOnDevice::Csize_t,
                                         bufferOnHost::Ptr{Cvoid},
                                         workspaceInBytesOnHost::Csize_t,
                                         info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnXgesvd_bufferSize(handle, params, jobu, jobvt, m, n, dataTypeA,
                                              A, lda, dataTypeS, S, dataTypeU, U, ldu,
                                              dataTypeVT, VT, ldvt, computeType,
                                              workspaceInBytesOnDevice,
                                              workspaceInBytesOnHost)
    initialize_context()
    @ccall libcusolver.cusolverDnXgesvd_bufferSize(handle::cusolverDnHandle_t,
                                                   params::cusolverDnParams_t, jobu::Int8,
                                                   jobvt::Int8, m::Int64, n::Int64,
                                                   dataTypeA::cudaDataType, A::CuPtr{Cvoid},
                                                   lda::Int64, dataTypeS::cudaDataType,
                                                   S::CuPtr{Cvoid}, dataTypeU::cudaDataType,
                                                   U::CuPtr{Cvoid}, ldu::Int64,
                                                   dataTypeVT::cudaDataType,
                                                   VT::CuPtr{Cvoid}, ldvt::Int64,
                                                   computeType::cudaDataType,
                                                   workspaceInBytesOnDevice::Ptr{Csize_t},
                                                   workspaceInBytesOnHost::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnXgesvd(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda,
                                   dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt,
                                   computeType, bufferOnDevice, workspaceInBytesOnDevice,
                                   bufferOnHost, workspaceInBytesOnHost, info)
    initialize_context()
    @ccall libcusolver.cusolverDnXgesvd(handle::cusolverDnHandle_t,
                                        params::cusolverDnParams_t, jobu::Int8, jobvt::Int8,
                                        m::Int64, n::Int64, dataTypeA::cudaDataType,
                                        A::CuPtr{Cvoid}, lda::Int64,
                                        dataTypeS::cudaDataType, S::CuPtr{Cvoid},
                                        dataTypeU::cudaDataType, U::CuPtr{Cvoid},
                                        ldu::Int64, dataTypeVT::cudaDataType,
                                        VT::CuPtr{Cvoid}, ldvt::Int64,
                                        computeType::cudaDataType,
                                        bufferOnDevice::CuPtr{Cvoid},
                                        workspaceInBytesOnDevice::Csize_t,
                                        bufferOnHost::Ptr{Cvoid},
                                        workspaceInBytesOnHost::Csize_t,
                                        info::CuPtr{Cint})::cusolverStatus_t
end

@checked function cusolverDnXgesvdp_bufferSize(handle, params, jobz, econ, m, n, dataTypeA,
                                               A, lda, dataTypeS, S, dataTypeU, U, ldu,
                                               dataTypeV, V, ldv, computeType,
                                               workspaceInBytesOnDevice,
                                               workspaceInBytesOnHost)
    initialize_context()
    @ccall libcusolver.cusolverDnXgesvdp_bufferSize(handle::cusolverDnHandle_t,
                                                    params::cusolverDnParams_t,
                                                    jobz::cusolverEigMode_t, econ::Cint,
                                                    m::Int64, n::Int64,
                                                    dataTypeA::cudaDataType,
                                                    A::CuPtr{Cvoid}, lda::Int64,
                                                    dataTypeS::cudaDataType,
                                                    S::CuPtr{Cvoid},
                                                    dataTypeU::cudaDataType,
                                                    U::CuPtr{Cvoid}, ldu::Int64,
                                                    dataTypeV::cudaDataType,
                                                    V::CuPtr{Cvoid}, ldv::Int64,
                                                    computeType::cudaDataType,
                                                    workspaceInBytesOnDevice::Ptr{Csize_t},
                                                    workspaceInBytesOnHost::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnXgesvdp(handle, params, jobz, econ, m, n, dataTypeA, A, lda,
                                    dataTypeS, S, dataTypeU, U, ldu, dataTypeV, V, ldv,
                                    computeType, bufferOnDevice, workspaceInBytesOnDevice,
                                    bufferOnHost, workspaceInBytesOnHost, d_info,
                                    h_err_sigma)
    initialize_context()
    @ccall libcusolver.cusolverDnXgesvdp(handle::cusolverDnHandle_t,
                                         params::cusolverDnParams_t,
                                         jobz::cusolverEigMode_t, econ::Cint, m::Int64,
                                         n::Int64, dataTypeA::cudaDataType, A::CuPtr{Cvoid},
                                         lda::Int64, dataTypeS::cudaDataType,
                                         S::CuPtr{Cvoid}, dataTypeU::cudaDataType,
                                         U::CuPtr{Cvoid}, ldu::Int64,
                                         dataTypeV::cudaDataType, V::CuPtr{Cvoid},
                                         ldv::Int64, computeType::cudaDataType,
                                         bufferOnDevice::CuPtr{Cvoid},
                                         workspaceInBytesOnDevice::Csize_t,
                                         bufferOnHost::Ptr{Cvoid},
                                         workspaceInBytesOnHost::Csize_t,
                                         d_info::CuPtr{Cint},
                                         h_err_sigma::Ptr{Cdouble})::cusolverStatus_t
end

@checked function cusolverDnXgesvdr_bufferSize(handle, params, jobu, jobv, m, n, k, p,
                                               niters, dataTypeA, A, lda, dataTypeSrand,
                                               Srand, dataTypeUrand, Urand, ldUrand,
                                               dataTypeVrand, Vrand, ldVrand, computeType,
                                               workspaceInBytesOnDevice,
                                               workspaceInBytesOnHost)
    initialize_context()
    @ccall libcusolver.cusolverDnXgesvdr_bufferSize(handle::cusolverDnHandle_t,
                                                    params::cusolverDnParams_t, jobu::Int8,
                                                    jobv::Int8, m::Int64, n::Int64,
                                                    k::Int64, p::Int64, niters::Int64,
                                                    dataTypeA::cudaDataType,
                                                    A::CuPtr{Cvoid}, lda::Int64,
                                                    dataTypeSrand::cudaDataType,
                                                    Srand::CuPtr{Cvoid},
                                                    dataTypeUrand::cudaDataType,
                                                    Urand::CuPtr{Cvoid}, ldUrand::Int64,
                                                    dataTypeVrand::cudaDataType,
                                                    Vrand::CuPtr{Cvoid}, ldVrand::Int64,
                                                    computeType::cudaDataType,
                                                    workspaceInBytesOnDevice::Ptr{Csize_t},
                                                    workspaceInBytesOnHost::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverDnXgesvdr(handle, params, jobu, jobv, m, n, k, p, niters,
                                    dataTypeA, A, lda, dataTypeSrand, Srand, dataTypeUrand,
                                    Urand, ldUrand, dataTypeVrand, Vrand, ldVrand,
                                    computeType, bufferOnDevice, workspaceInBytesOnDevice,
                                    bufferOnHost, workspaceInBytesOnHost, d_info)
    initialize_context()
    @ccall libcusolver.cusolverDnXgesvdr(handle::cusolverDnHandle_t,
                                         params::cusolverDnParams_t, jobu::Int8, jobv::Int8,
                                         m::Int64, n::Int64, k::Int64, p::Int64,
                                         niters::Int64, dataTypeA::cudaDataType,
                                         A::CuPtr{Cvoid}, lda::Int64,
                                         dataTypeSrand::cudaDataType, Srand::CuPtr{Cvoid},
                                         dataTypeUrand::cudaDataType, Urand::CuPtr{Cvoid},
                                         ldUrand::Int64, dataTypeVrand::cudaDataType,
                                         Vrand::CuPtr{Cvoid}, ldVrand::Int64,
                                         computeType::cudaDataType,
                                         bufferOnDevice::CuPtr{Cvoid},
                                         workspaceInBytesOnDevice::Csize_t,
                                         bufferOnHost::Ptr{Cvoid},
                                         workspaceInBytesOnHost::Csize_t,
                                         d_info::CuPtr{Cint})::cusolverStatus_t
end

# typedef void ( * cusolverDnLoggerCallback_t ) ( int logLevel , const char * functionName , const char * message )
const cusolverDnLoggerCallback_t = Ptr{Cvoid}

@checked function cusolverDnLoggerSetCallback(callback)
    initialize_context()
    @ccall libcusolver.cusolverDnLoggerSetCallback(callback::cusolverDnLoggerCallback_t)::cusolverStatus_t
end

@checked function cusolverDnLoggerSetFile(file)
    initialize_context()
    @ccall libcusolver.cusolverDnLoggerSetFile(file::Ptr{Libc.FILE})::cusolverStatus_t
end

@checked function cusolverDnLoggerOpenFile(logFile)
    initialize_context()
    @ccall libcusolver.cusolverDnLoggerOpenFile(logFile::Cstring)::cusolverStatus_t
end

@checked function cusolverDnLoggerSetLevel(level)
    initialize_context()
    @ccall libcusolver.cusolverDnLoggerSetLevel(level::Cint)::cusolverStatus_t
end

@checked function cusolverDnLoggerSetMask(mask)
    initialize_context()
    @ccall libcusolver.cusolverDnLoggerSetMask(mask::Cint)::cusolverStatus_t
end

# no prototype is found for this function at cusolverDn.h:4868:32, please use with caution
@checked function cusolverDnLoggerForceDisable()
    initialize_context()
    @ccall libcusolver.cusolverDnLoggerForceDisable()::cusolverStatus_t
end

mutable struct cusolverSpContext end

const cusolverSpHandle_t = Ptr{cusolverSpContext}

mutable struct csrqrInfo end

const csrqrInfo_t = Ptr{csrqrInfo}

@checked function cusolverSpCreate(handle)
    initialize_context()
    @ccall libcusolver.cusolverSpCreate(handle::Ptr{cusolverSpHandle_t})::cusolverStatus_t
end

@checked function cusolverSpDestroy(handle)
    initialize_context()
    @ccall libcusolver.cusolverSpDestroy(handle::cusolverSpHandle_t)::cusolverStatus_t
end

@checked function cusolverSpSetStream(handle, streamId)
    initialize_context()
    @ccall libcusolver.cusolverSpSetStream(handle::cusolverSpHandle_t,
                                           streamId::cudaStream_t)::cusolverStatus_t
end

@checked function cusolverSpGetStream(handle, streamId)
    initialize_context()
    @ccall libcusolver.cusolverSpGetStream(handle::cusolverSpHandle_t,
                                           streamId::Ptr{cudaStream_t})::cusolverStatus_t
end

@checked function cusolverSpXcsrissymHost(handle, m, nnzA, descrA, csrRowPtrA, csrEndPtrA,
                                          csrColIndA, issym)
    initialize_context()
    @ccall libcusolver.cusolverSpXcsrissymHost(handle::cusolverSpHandle_t, m::Cint,
                                               nnzA::Cint, descrA::cusparseMatDescr_t,
                                               csrRowPtrA::Ptr{Cint}, csrEndPtrA::Ptr{Cint},
                                               csrColIndA::Ptr{Cint},
                                               issym::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpScsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrlsvluHost(handle::cusolverSpHandle_t, n::Cint,
                                               nnzA::Cint, descrA::cusparseMatDescr_t,
                                               csrValA::Ptr{Cfloat}, csrRowPtrA::Ptr{Cint},
                                               csrColIndA::Ptr{Cint}, b::Ptr{Cfloat},
                                               tol::Cfloat, reorder::Cint, x::Ptr{Cfloat},
                                               singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpDcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrlsvluHost(handle::cusolverSpHandle_t, n::Cint,
                                               nnzA::Cint, descrA::cusparseMatDescr_t,
                                               csrValA::Ptr{Cdouble}, csrRowPtrA::Ptr{Cint},
                                               csrColIndA::Ptr{Cint}, b::Ptr{Cdouble},
                                               tol::Cdouble, reorder::Cint, x::Ptr{Cdouble},
                                               singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpCcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrlsvluHost(handle::cusolverSpHandle_t, n::Cint,
                                               nnzA::Cint, descrA::cusparseMatDescr_t,
                                               csrValA::Ptr{cuComplex},
                                               csrRowPtrA::Ptr{Cint}, csrColIndA::Ptr{Cint},
                                               b::Ptr{cuComplex}, tol::Cfloat,
                                               reorder::Cint, x::Ptr{cuComplex},
                                               singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpZcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrlsvluHost(handle::cusolverSpHandle_t, n::Cint,
                                               nnzA::Cint, descrA::cusparseMatDescr_t,
                                               csrValA::Ptr{cuDoubleComplex},
                                               csrRowPtrA::Ptr{Cint}, csrColIndA::Ptr{Cint},
                                               b::Ptr{cuDoubleComplex}, tol::Cdouble,
                                               reorder::Cint, x::Ptr{cuDoubleComplex},
                                               singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpScsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd,
                                      b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrlsvqr(handle::cusolverSpHandle_t, m::Cint, nnz::Cint,
                                           descrA::cusparseMatDescr_t,
                                           csrVal::CuPtr{Cfloat}, csrRowPtr::CuPtr{Cint},
                                           csrColInd::CuPtr{Cint}, b::CuPtr{Cfloat},
                                           tol::Cfloat, reorder::Cint, x::CuPtr{Cfloat},
                                           singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpDcsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd,
                                      b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrlsvqr(handle::cusolverSpHandle_t, m::Cint, nnz::Cint,
                                           descrA::cusparseMatDescr_t,
                                           csrVal::CuPtr{Cdouble}, csrRowPtr::CuPtr{Cint},
                                           csrColInd::CuPtr{Cint}, b::CuPtr{Cdouble},
                                           tol::Cdouble, reorder::Cint, x::CuPtr{Cdouble},
                                           singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpCcsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd,
                                      b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrlsvqr(handle::cusolverSpHandle_t, m::Cint, nnz::Cint,
                                           descrA::cusparseMatDescr_t,
                                           csrVal::CuPtr{cuComplex}, csrRowPtr::CuPtr{Cint},
                                           csrColInd::CuPtr{Cint}, b::CuPtr{cuComplex},
                                           tol::Cfloat, reorder::Cint, x::CuPtr{cuComplex},
                                           singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpZcsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd,
                                      b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrlsvqr(handle::cusolverSpHandle_t, m::Cint, nnz::Cint,
                                           descrA::cusparseMatDescr_t,
                                           csrVal::CuPtr{cuDoubleComplex},
                                           csrRowPtr::CuPtr{Cint}, csrColInd::CuPtr{Cint},
                                           b::CuPtr{cuDoubleComplex}, tol::Cdouble,
                                           reorder::Cint, x::CuPtr{cuDoubleComplex},
                                           singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpScsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrlsvqrHost(handle::cusolverSpHandle_t, m::Cint,
                                               nnz::Cint, descrA::cusparseMatDescr_t,
                                               csrValA::Ptr{Cfloat}, csrRowPtrA::Ptr{Cint},
                                               csrColIndA::Ptr{Cint}, b::Ptr{Cfloat},
                                               tol::Cfloat, reorder::Cint, x::Ptr{Cfloat},
                                               singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpDcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrlsvqrHost(handle::cusolverSpHandle_t, m::Cint,
                                               nnz::Cint, descrA::cusparseMatDescr_t,
                                               csrValA::Ptr{Cdouble}, csrRowPtrA::Ptr{Cint},
                                               csrColIndA::Ptr{Cint}, b::Ptr{Cdouble},
                                               tol::Cdouble, reorder::Cint, x::Ptr{Cdouble},
                                               singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpCcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrlsvqrHost(handle::cusolverSpHandle_t, m::Cint,
                                               nnz::Cint, descrA::cusparseMatDescr_t,
                                               csrValA::Ptr{cuComplex},
                                               csrRowPtrA::Ptr{Cint}, csrColIndA::Ptr{Cint},
                                               b::Ptr{cuComplex}, tol::Cfloat,
                                               reorder::Cint, x::Ptr{cuComplex},
                                               singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpZcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                          csrColIndA, b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrlsvqrHost(handle::cusolverSpHandle_t, m::Cint,
                                               nnz::Cint, descrA::cusparseMatDescr_t,
                                               csrValA::Ptr{cuDoubleComplex},
                                               csrRowPtrA::Ptr{Cint}, csrColIndA::Ptr{Cint},
                                               b::Ptr{cuDoubleComplex}, tol::Cdouble,
                                               reorder::Cint, x::Ptr{cuDoubleComplex},
                                               singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpScsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                            csrColInd, b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrlsvcholHost(handle::cusolverSpHandle_t, m::Cint,
                                                 nnz::Cint, descrA::cusparseMatDescr_t,
                                                 csrVal::Ptr{Cfloat}, csrRowPtr::Ptr{Cint},
                                                 csrColInd::Ptr{Cint}, b::Ptr{Cfloat},
                                                 tol::Cfloat, reorder::Cint, x::Ptr{Cfloat},
                                                 singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpDcsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                            csrColInd, b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrlsvcholHost(handle::cusolverSpHandle_t, m::Cint,
                                                 nnz::Cint, descrA::cusparseMatDescr_t,
                                                 csrVal::Ptr{Cdouble}, csrRowPtr::Ptr{Cint},
                                                 csrColInd::Ptr{Cint}, b::Ptr{Cdouble},
                                                 tol::Cdouble, reorder::Cint,
                                                 x::Ptr{Cdouble},
                                                 singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpCcsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                            csrColInd, b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrlsvcholHost(handle::cusolverSpHandle_t, m::Cint,
                                                 nnz::Cint, descrA::cusparseMatDescr_t,
                                                 csrVal::Ptr{cuComplex},
                                                 csrRowPtr::Ptr{Cint}, csrColInd::Ptr{Cint},
                                                 b::Ptr{cuComplex}, tol::Cfloat,
                                                 reorder::Cint, x::Ptr{cuComplex},
                                                 singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpZcsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                            csrColInd, b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrlsvcholHost(handle::cusolverSpHandle_t, m::Cint,
                                                 nnz::Cint, descrA::cusparseMatDescr_t,
                                                 csrVal::Ptr{cuDoubleComplex},
                                                 csrRowPtr::Ptr{Cint}, csrColInd::Ptr{Cint},
                                                 b::Ptr{cuDoubleComplex}, tol::Cdouble,
                                                 reorder::Cint, x::Ptr{cuDoubleComplex},
                                                 singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpScsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                        csrColInd, b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrlsvchol(handle::cusolverSpHandle_t, m::Cint, nnz::Cint,
                                             descrA::cusparseMatDescr_t,
                                             csrVal::CuPtr{Cfloat}, csrRowPtr::CuPtr{Cint},
                                             csrColInd::CuPtr{Cint}, b::CuPtr{Cfloat},
                                             tol::Cfloat, reorder::Cint, x::CuPtr{Cfloat},
                                             singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpDcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                        csrColInd, b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrlsvchol(handle::cusolverSpHandle_t, m::Cint, nnz::Cint,
                                             descrA::cusparseMatDescr_t,
                                             csrVal::CuPtr{Cdouble}, csrRowPtr::CuPtr{Cint},
                                             csrColInd::CuPtr{Cint}, b::CuPtr{Cdouble},
                                             tol::Cdouble, reorder::Cint, x::CuPtr{Cdouble},
                                             singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpCcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                        csrColInd, b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrlsvchol(handle::cusolverSpHandle_t, m::Cint, nnz::Cint,
                                             descrA::cusparseMatDescr_t,
                                             csrVal::CuPtr{cuComplex},
                                             csrRowPtr::CuPtr{Cint}, csrColInd::CuPtr{Cint},
                                             b::CuPtr{cuComplex}, tol::Cfloat,
                                             reorder::Cint, x::CuPtr{cuComplex},
                                             singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpZcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr,
                                        csrColInd, b, tol, reorder, x, singularity)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrlsvchol(handle::cusolverSpHandle_t, m::Cint, nnz::Cint,
                                             descrA::cusparseMatDescr_t,
                                             csrVal::CuPtr{cuDoubleComplex},
                                             csrRowPtr::CuPtr{Cint}, csrColInd::CuPtr{Cint},
                                             b::CuPtr{cuDoubleComplex}, tol::Cdouble,
                                             reorder::Cint, x::CuPtr{cuDoubleComplex},
                                             singularity::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpScsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, b, tol, rankA, x, p, min_norm)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrlsqvqrHost(handle::cusolverSpHandle_t, m::Cint,
                                                n::Cint, nnz::Cint,
                                                descrA::cusparseMatDescr_t,
                                                csrValA::Ptr{Cfloat}, csrRowPtrA::Ptr{Cint},
                                                csrColIndA::Ptr{Cint}, b::Ptr{Cfloat},
                                                tol::Cfloat, rankA::Ptr{Cint},
                                                x::Ptr{Cfloat}, p::Ptr{Cint},
                                                min_norm::Ptr{Cfloat})::cusolverStatus_t
end

@checked function cusolverSpDcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, b, tol, rankA, x, p, min_norm)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrlsqvqrHost(handle::cusolverSpHandle_t, m::Cint,
                                                n::Cint, nnz::Cint,
                                                descrA::cusparseMatDescr_t,
                                                csrValA::Ptr{Cdouble},
                                                csrRowPtrA::Ptr{Cint},
                                                csrColIndA::Ptr{Cint}, b::Ptr{Cdouble},
                                                tol::Cdouble, rankA::Ptr{Cint},
                                                x::Ptr{Cdouble}, p::Ptr{Cint},
                                                min_norm::Ptr{Cdouble})::cusolverStatus_t
end

@checked function cusolverSpCcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, b, tol, rankA, x, p, min_norm)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrlsqvqrHost(handle::cusolverSpHandle_t, m::Cint,
                                                n::Cint, nnz::Cint,
                                                descrA::cusparseMatDescr_t,
                                                csrValA::Ptr{cuComplex},
                                                csrRowPtrA::Ptr{Cint},
                                                csrColIndA::Ptr{Cint}, b::Ptr{cuComplex},
                                                tol::Cfloat, rankA::Ptr{Cint},
                                                x::Ptr{cuComplex}, p::Ptr{Cint},
                                                min_norm::Ptr{Cfloat})::cusolverStatus_t
end

@checked function cusolverSpZcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, b, tol, rankA, x, p, min_norm)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrlsqvqrHost(handle::cusolverSpHandle_t, m::Cint,
                                                n::Cint, nnz::Cint,
                                                descrA::cusparseMatDescr_t,
                                                csrValA::Ptr{cuDoubleComplex},
                                                csrRowPtrA::Ptr{Cint},
                                                csrColIndA::Ptr{Cint},
                                                b::Ptr{cuDoubleComplex}, tol::Cdouble,
                                                rankA::Ptr{Cint}, x::Ptr{cuDoubleComplex},
                                                p::Ptr{Cint},
                                                min_norm::Ptr{Cdouble})::cusolverStatus_t
end

@checked function cusolverSpScsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, mu0, x0, maxite, tol, mu, x)
    initialize_context()
    @ccall libcusolver.cusolverSpScsreigvsiHost(handle::cusolverSpHandle_t, m::Cint,
                                                nnz::Cint, descrA::cusparseMatDescr_t,
                                                csrValA::Ptr{Cfloat}, csrRowPtrA::Ptr{Cint},
                                                csrColIndA::Ptr{Cint}, mu0::Cfloat,
                                                x0::Ptr{Cfloat}, maxite::Cint, tol::Cfloat,
                                                mu::Ptr{Cfloat},
                                                x::Ptr{Cfloat})::cusolverStatus_t
end

@checked function cusolverSpDcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, mu0, x0, maxite, tol, mu, x)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsreigvsiHost(handle::cusolverSpHandle_t, m::Cint,
                                                nnz::Cint, descrA::cusparseMatDescr_t,
                                                csrValA::Ptr{Cdouble},
                                                csrRowPtrA::Ptr{Cint},
                                                csrColIndA::Ptr{Cint}, mu0::Cdouble,
                                                x0::Ptr{Cdouble}, maxite::Cint,
                                                tol::Cdouble, mu::Ptr{Cdouble},
                                                x::Ptr{Cdouble})::cusolverStatus_t
end

@checked function cusolverSpCcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, mu0, x0, maxite, tol, mu, x)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsreigvsiHost(handle::cusolverSpHandle_t, m::Cint,
                                                nnz::Cint, descrA::cusparseMatDescr_t,
                                                csrValA::Ptr{cuComplex},
                                                csrRowPtrA::Ptr{Cint},
                                                csrColIndA::Ptr{Cint}, mu0::cuComplex,
                                                x0::Ptr{cuComplex}, maxite::Cint,
                                                tol::Cfloat, mu::Ptr{cuComplex},
                                                x::Ptr{cuComplex})::cusolverStatus_t
end

@checked function cusolverSpZcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, mu0, x0, maxite, tol, mu, x)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsreigvsiHost(handle::cusolverSpHandle_t, m::Cint,
                                                nnz::Cint, descrA::cusparseMatDescr_t,
                                                csrValA::Ptr{cuDoubleComplex},
                                                csrRowPtrA::Ptr{Cint},
                                                csrColIndA::Ptr{Cint}, mu0::cuDoubleComplex,
                                                x0::Ptr{cuDoubleComplex}, maxite::Cint,
                                                tol::Cdouble, mu::Ptr{cuDoubleComplex},
                                                x::Ptr{cuDoubleComplex})::cusolverStatus_t
end

@checked function cusolverSpScsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                       csrColIndA, mu0, x0, maxite, eps, mu, x)
    initialize_context()
    @ccall libcusolver.cusolverSpScsreigvsi(handle::cusolverSpHandle_t, m::Cint, nnz::Cint,
                                            descrA::cusparseMatDescr_t,
                                            csrValA::CuPtr{Cfloat}, csrRowPtrA::CuPtr{Cint},
                                            csrColIndA::CuPtr{Cint}, mu0::Cfloat,
                                            x0::CuPtr{Cfloat}, maxite::Cint, eps::Cfloat,
                                            mu::CuPtr{Cfloat},
                                            x::CuPtr{Cfloat})::cusolverStatus_t
end

@checked function cusolverSpDcsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                       csrColIndA, mu0, x0, maxite, eps, mu, x)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsreigvsi(handle::cusolverSpHandle_t, m::Cint, nnz::Cint,
                                            descrA::cusparseMatDescr_t,
                                            csrValA::CuPtr{Cdouble},
                                            csrRowPtrA::CuPtr{Cint},
                                            csrColIndA::CuPtr{Cint}, mu0::Cdouble,
                                            x0::CuPtr{Cdouble}, maxite::Cint, eps::Cdouble,
                                            mu::CuPtr{Cdouble},
                                            x::CuPtr{Cdouble})::cusolverStatus_t
end

@checked function cusolverSpCcsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                       csrColIndA, mu0, x0, maxite, eps, mu, x)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsreigvsi(handle::cusolverSpHandle_t, m::Cint, nnz::Cint,
                                            descrA::cusparseMatDescr_t,
                                            csrValA::CuPtr{cuComplex},
                                            csrRowPtrA::CuPtr{Cint},
                                            csrColIndA::CuPtr{Cint}, mu0::cuComplex,
                                            x0::CuPtr{cuComplex}, maxite::Cint, eps::Cfloat,
                                            mu::CuPtr{cuComplex},
                                            x::CuPtr{cuComplex})::cusolverStatus_t
end

@checked function cusolverSpZcsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                       csrColIndA, mu0, x0, maxite, eps, mu, x)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsreigvsi(handle::cusolverSpHandle_t, m::Cint, nnz::Cint,
                                            descrA::cusparseMatDescr_t,
                                            csrValA::CuPtr{cuDoubleComplex},
                                            csrRowPtrA::CuPtr{Cint},
                                            csrColIndA::CuPtr{Cint}, mu0::cuDoubleComplex,
                                            x0::CuPtr{cuDoubleComplex}, maxite::Cint,
                                            eps::Cdouble, mu::CuPtr{cuDoubleComplex},
                                            x::CuPtr{cuDoubleComplex})::cusolverStatus_t
end

@checked function cusolverSpScsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                         csrColIndA, left_bottom_corner, right_upper_corner,
                                         num_eigs)
    initialize_context()
    @ccall libcusolver.cusolverSpScsreigsHost(handle::cusolverSpHandle_t, m::Cint,
                                              nnz::Cint, descrA::cusparseMatDescr_t,
                                              csrValA::Ptr{Cfloat}, csrRowPtrA::Ptr{Cint},
                                              csrColIndA::Ptr{Cint},
                                              left_bottom_corner::cuComplex,
                                              right_upper_corner::cuComplex,
                                              num_eigs::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpDcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                         csrColIndA, left_bottom_corner, right_upper_corner,
                                         num_eigs)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsreigsHost(handle::cusolverSpHandle_t, m::Cint,
                                              nnz::Cint, descrA::cusparseMatDescr_t,
                                              csrValA::Ptr{Cdouble}, csrRowPtrA::Ptr{Cint},
                                              csrColIndA::Ptr{Cint},
                                              left_bottom_corner::cuDoubleComplex,
                                              right_upper_corner::cuDoubleComplex,
                                              num_eigs::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpCcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                         csrColIndA, left_bottom_corner, right_upper_corner,
                                         num_eigs)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsreigsHost(handle::cusolverSpHandle_t, m::Cint,
                                              nnz::Cint, descrA::cusparseMatDescr_t,
                                              csrValA::Ptr{cuComplex},
                                              csrRowPtrA::Ptr{Cint}, csrColIndA::Ptr{Cint},
                                              left_bottom_corner::cuComplex,
                                              right_upper_corner::cuComplex,
                                              num_eigs::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpZcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA,
                                         csrColIndA, left_bottom_corner, right_upper_corner,
                                         num_eigs)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsreigsHost(handle::cusolverSpHandle_t, m::Cint,
                                              nnz::Cint, descrA::cusparseMatDescr_t,
                                              csrValA::Ptr{cuDoubleComplex},
                                              csrRowPtrA::Ptr{Cint}, csrColIndA::Ptr{Cint},
                                              left_bottom_corner::cuDoubleComplex,
                                              right_upper_corner::cuDoubleComplex,
                                              num_eigs::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpXcsrsymrcmHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA,
                                           p)
    initialize_context()
    @ccall libcusolver.cusolverSpXcsrsymrcmHost(handle::cusolverSpHandle_t, n::Cint,
                                                nnzA::Cint, descrA::cusparseMatDescr_t,
                                                csrRowPtrA::Ptr{Cint},
                                                csrColIndA::Ptr{Cint},
                                                p::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpXcsrsymmdqHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA,
                                           p)
    initialize_context()
    @ccall libcusolver.cusolverSpXcsrsymmdqHost(handle::cusolverSpHandle_t, n::Cint,
                                                nnzA::Cint, descrA::cusparseMatDescr_t,
                                                csrRowPtrA::Ptr{Cint},
                                                csrColIndA::Ptr{Cint},
                                                p::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpXcsrsymamdHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA,
                                           p)
    initialize_context()
    @ccall libcusolver.cusolverSpXcsrsymamdHost(handle::cusolverSpHandle_t, n::Cint,
                                                nnzA::Cint, descrA::cusparseMatDescr_t,
                                                csrRowPtrA::Ptr{Cint},
                                                csrColIndA::Ptr{Cint},
                                                p::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpXcsrmetisndHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA,
                                            options, p)
    initialize_context()
    @ccall libcusolver.cusolverSpXcsrmetisndHost(handle::cusolverSpHandle_t, n::Cint,
                                                 nnzA::Cint, descrA::cusparseMatDescr_t,
                                                 csrRowPtrA::Ptr{Cint},
                                                 csrColIndA::Ptr{Cint}, options::Ptr{Int64},
                                                 p::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpScsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA,
                                        csrColIndA, P, numnz)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrzfdHost(handle::cusolverSpHandle_t, n::Cint, nnz::Cint,
                                             descrA::cusparseMatDescr_t,
                                             csrValA::Ptr{Cfloat}, csrRowPtrA::Ptr{Cint},
                                             csrColIndA::Ptr{Cint}, P::Ptr{Cint},
                                             numnz::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpDcsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA,
                                        csrColIndA, P, numnz)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrzfdHost(handle::cusolverSpHandle_t, n::Cint, nnz::Cint,
                                             descrA::cusparseMatDescr_t,
                                             csrValA::Ptr{Cdouble}, csrRowPtrA::Ptr{Cint},
                                             csrColIndA::Ptr{Cint}, P::Ptr{Cint},
                                             numnz::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpCcsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA,
                                        csrColIndA, P, numnz)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrzfdHost(handle::cusolverSpHandle_t, n::Cint, nnz::Cint,
                                             descrA::cusparseMatDescr_t,
                                             csrValA::Ptr{cuComplex}, csrRowPtrA::Ptr{Cint},
                                             csrColIndA::Ptr{Cint}, P::Ptr{Cint},
                                             numnz::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpZcsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA,
                                        csrColIndA, P, numnz)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrzfdHost(handle::cusolverSpHandle_t, n::Cint, nnz::Cint,
                                             descrA::cusparseMatDescr_t,
                                             csrValA::Ptr{cuDoubleComplex},
                                             csrRowPtrA::Ptr{Cint}, csrColIndA::Ptr{Cint},
                                             P::Ptr{Cint},
                                             numnz::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpXcsrperm_bufferSizeHost(handle, m, n, nnzA, descrA, csrRowPtrA,
                                                    csrColIndA, p, q, bufferSizeInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpXcsrperm_bufferSizeHost(handle::cusolverSpHandle_t,
                                                         m::Cint, n::Cint, nnzA::Cint,
                                                         descrA::cusparseMatDescr_t,
                                                         csrRowPtrA::Ptr{Cint},
                                                         csrColIndA::Ptr{Cint},
                                                         p::Ptr{Cint}, q::Ptr{Cint},
                                                         bufferSizeInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpXcsrpermHost(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA,
                                         p, q, map, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpXcsrpermHost(handle::cusolverSpHandle_t, m::Cint, n::Cint,
                                              nnzA::Cint, descrA::cusparseMatDescr_t,
                                              csrRowPtrA::Ptr{Cint}, csrColIndA::Ptr{Cint},
                                              p::Ptr{Cint}, q::Ptr{Cint}, map::Ptr{Cint},
                                              pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpCreateCsrqrInfo(info)
    initialize_context()
    @ccall libcusolver.cusolverSpCreateCsrqrInfo(info::Ptr{csrqrInfo_t})::cusolverStatus_t
end

@checked function cusolverSpDestroyCsrqrInfo(info)
    initialize_context()
    @ccall libcusolver.cusolverSpDestroyCsrqrInfo(info::csrqrInfo_t)::cusolverStatus_t
end

@checked function cusolverSpXcsrqrAnalysisBatched(handle, m, n, nnzA, descrA, csrRowPtrA,
                                                  csrColIndA, info)
    initialize_context()
    @ccall libcusolver.cusolverSpXcsrqrAnalysisBatched(handle::cusolverSpHandle_t, m::Cint,
                                                       n::Cint, nnzA::Cint,
                                                       descrA::cusparseMatDescr_t,
                                                       csrRowPtrA::CuPtr{Cint},
                                                       csrColIndA::CuPtr{Cint},
                                                       info::csrqrInfo_t)::cusolverStatus_t
end

@checked function cusolverSpScsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal,
                                                    csrRowPtr, csrColInd, batchSize, info,
                                                    internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrqrBufferInfoBatched(handle::cusolverSpHandle_t,
                                                         m::Cint, n::Cint, nnz::Cint,
                                                         descrA::cusparseMatDescr_t,
                                                         csrVal::CuPtr{Cfloat},
                                                         csrRowPtr::CuPtr{Cint},
                                                         csrColInd::CuPtr{Cint},
                                                         batchSize::Cint, info::csrqrInfo_t,
                                                         internalDataInBytes::Ptr{Csize_t},
                                                         workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpDcsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal,
                                                    csrRowPtr, csrColInd, batchSize, info,
                                                    internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrqrBufferInfoBatched(handle::cusolverSpHandle_t,
                                                         m::Cint, n::Cint, nnz::Cint,
                                                         descrA::cusparseMatDescr_t,
                                                         csrVal::CuPtr{Cdouble},
                                                         csrRowPtr::CuPtr{Cint},
                                                         csrColInd::CuPtr{Cint},
                                                         batchSize::Cint, info::csrqrInfo_t,
                                                         internalDataInBytes::Ptr{Csize_t},
                                                         workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpCcsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal,
                                                    csrRowPtr, csrColInd, batchSize, info,
                                                    internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrqrBufferInfoBatched(handle::cusolverSpHandle_t,
                                                         m::Cint, n::Cint, nnz::Cint,
                                                         descrA::cusparseMatDescr_t,
                                                         csrVal::CuPtr{cuComplex},
                                                         csrRowPtr::CuPtr{Cint},
                                                         csrColInd::CuPtr{Cint},
                                                         batchSize::Cint, info::csrqrInfo_t,
                                                         internalDataInBytes::Ptr{Csize_t},
                                                         workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpZcsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal,
                                                    csrRowPtr, csrColInd, batchSize, info,
                                                    internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrqrBufferInfoBatched(handle::cusolverSpHandle_t,
                                                         m::Cint, n::Cint, nnz::Cint,
                                                         descrA::cusparseMatDescr_t,
                                                         csrVal::CuPtr{cuDoubleComplex},
                                                         csrRowPtr::CuPtr{Cint},
                                                         csrColInd::CuPtr{Cint},
                                                         batchSize::Cint, info::csrqrInfo_t,
                                                         internalDataInBytes::Ptr{Csize_t},
                                                         workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpScsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                            csrColIndA, b, x, batchSize, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrqrsvBatched(handle::cusolverSpHandle_t, m::Cint,
                                                 n::Cint, nnz::Cint,
                                                 descrA::cusparseMatDescr_t,
                                                 csrValA::CuPtr{Cfloat},
                                                 csrRowPtrA::CuPtr{Cint},
                                                 csrColIndA::CuPtr{Cint}, b::CuPtr{Cfloat},
                                                 x::CuPtr{Cfloat}, batchSize::Cint,
                                                 info::csrqrInfo_t,
                                                 pBuffer::CuPtr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpDcsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                            csrColIndA, b, x, batchSize, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrqrsvBatched(handle::cusolverSpHandle_t, m::Cint,
                                                 n::Cint, nnz::Cint,
                                                 descrA::cusparseMatDescr_t,
                                                 csrValA::CuPtr{Cdouble},
                                                 csrRowPtrA::CuPtr{Cint},
                                                 csrColIndA::CuPtr{Cint}, b::CuPtr{Cdouble},
                                                 x::CuPtr{Cdouble}, batchSize::Cint,
                                                 info::csrqrInfo_t,
                                                 pBuffer::CuPtr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpCcsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                            csrColIndA, b, x, batchSize, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrqrsvBatched(handle::cusolverSpHandle_t, m::Cint,
                                                 n::Cint, nnz::Cint,
                                                 descrA::cusparseMatDescr_t,
                                                 csrValA::CuPtr{cuComplex},
                                                 csrRowPtrA::CuPtr{Cint},
                                                 csrColIndA::CuPtr{Cint},
                                                 b::CuPtr{cuComplex}, x::CuPtr{cuComplex},
                                                 batchSize::Cint, info::csrqrInfo_t,
                                                 pBuffer::CuPtr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpZcsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                            csrColIndA, b, x, batchSize, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrqrsvBatched(handle::cusolverSpHandle_t, m::Cint,
                                                 n::Cint, nnz::Cint,
                                                 descrA::cusparseMatDescr_t,
                                                 csrValA::CuPtr{cuDoubleComplex},
                                                 csrRowPtrA::CuPtr{Cint},
                                                 csrColIndA::CuPtr{Cint},
                                                 b::CuPtr{cuDoubleComplex},
                                                 x::CuPtr{cuDoubleComplex}, batchSize::Cint,
                                                 info::csrqrInfo_t,
                                                 pBuffer::CuPtr{Cvoid})::cusolverStatus_t
end

mutable struct csrluInfoHost end

const csrluInfoHost_t = Ptr{csrluInfoHost}

mutable struct csrqrInfoHost end

const csrqrInfoHost_t = Ptr{csrqrInfoHost}

mutable struct csrcholInfoHost end

const csrcholInfoHost_t = Ptr{csrcholInfoHost}

mutable struct csrcholInfo end

const csrcholInfo_t = Ptr{csrcholInfo}

@checked function cusolverSpCreateCsrluInfoHost(info)
    initialize_context()
    @ccall libcusolver.cusolverSpCreateCsrluInfoHost(info::Ptr{csrluInfoHost_t})::cusolverStatus_t
end

@checked function cusolverSpDestroyCsrluInfoHost(info)
    initialize_context()
    @ccall libcusolver.cusolverSpDestroyCsrluInfoHost(info::csrluInfoHost_t)::cusolverStatus_t
end

@checked function cusolverSpXcsrluAnalysisHost(handle, n, nnzA, descrA, csrRowPtrA,
                                               csrColIndA, info)
    initialize_context()
    @ccall libcusolver.cusolverSpXcsrluAnalysisHost(handle::cusolverSpHandle_t, n::Cint,
                                                    nnzA::Cint, descrA::cusparseMatDescr_t,
                                                    csrRowPtrA::Ptr{Cint},
                                                    csrColIndA::Ptr{Cint},
                                                    info::csrluInfoHost_t)::cusolverStatus_t
end

@checked function cusolverSpScsrluBufferInfoHost(handle, n, nnzA, descrA, csrValA,
                                                 csrRowPtrA, csrColIndA, info,
                                                 internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrluBufferInfoHost(handle::cusolverSpHandle_t, n::Cint,
                                                      nnzA::Cint,
                                                      descrA::cusparseMatDescr_t,
                                                      csrValA::Ptr{Cfloat},
                                                      csrRowPtrA::Ptr{Cint},
                                                      csrColIndA::Ptr{Cint},
                                                      info::csrluInfoHost_t,
                                                      internalDataInBytes::Ptr{Csize_t},
                                                      workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpDcsrluBufferInfoHost(handle, n, nnzA, descrA, csrValA,
                                                 csrRowPtrA, csrColIndA, info,
                                                 internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrluBufferInfoHost(handle::cusolverSpHandle_t, n::Cint,
                                                      nnzA::Cint,
                                                      descrA::cusparseMatDescr_t,
                                                      csrValA::Ptr{Cdouble},
                                                      csrRowPtrA::Ptr{Cint},
                                                      csrColIndA::Ptr{Cint},
                                                      info::csrluInfoHost_t,
                                                      internalDataInBytes::Ptr{Csize_t},
                                                      workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpCcsrluBufferInfoHost(handle, n, nnzA, descrA, csrValA,
                                                 csrRowPtrA, csrColIndA, info,
                                                 internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrluBufferInfoHost(handle::cusolverSpHandle_t, n::Cint,
                                                      nnzA::Cint,
                                                      descrA::cusparseMatDescr_t,
                                                      csrValA::Ptr{cuComplex},
                                                      csrRowPtrA::Ptr{Cint},
                                                      csrColIndA::Ptr{Cint},
                                                      info::csrluInfoHost_t,
                                                      internalDataInBytes::Ptr{Csize_t},
                                                      workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpZcsrluBufferInfoHost(handle, n, nnzA, descrA, csrValA,
                                                 csrRowPtrA, csrColIndA, info,
                                                 internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrluBufferInfoHost(handle::cusolverSpHandle_t, n::Cint,
                                                      nnzA::Cint,
                                                      descrA::cusparseMatDescr_t,
                                                      csrValA::Ptr{cuDoubleComplex},
                                                      csrRowPtrA::Ptr{Cint},
                                                      csrColIndA::Ptr{Cint},
                                                      info::csrluInfoHost_t,
                                                      internalDataInBytes::Ptr{Csize_t},
                                                      workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpScsrluFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                             csrColIndA, info, pivot_threshold, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrluFactorHost(handle::cusolverSpHandle_t, n::Cint,
                                                  nnzA::Cint, descrA::cusparseMatDescr_t,
                                                  csrValA::Ptr{Cfloat},
                                                  csrRowPtrA::Ptr{Cint},
                                                  csrColIndA::Ptr{Cint},
                                                  info::csrluInfoHost_t,
                                                  pivot_threshold::Cfloat,
                                                  pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpDcsrluFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                             csrColIndA, info, pivot_threshold, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrluFactorHost(handle::cusolverSpHandle_t, n::Cint,
                                                  nnzA::Cint, descrA::cusparseMatDescr_t,
                                                  csrValA::Ptr{Cdouble},
                                                  csrRowPtrA::Ptr{Cint},
                                                  csrColIndA::Ptr{Cint},
                                                  info::csrluInfoHost_t,
                                                  pivot_threshold::Cdouble,
                                                  pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpCcsrluFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                             csrColIndA, info, pivot_threshold, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrluFactorHost(handle::cusolverSpHandle_t, n::Cint,
                                                  nnzA::Cint, descrA::cusparseMatDescr_t,
                                                  csrValA::Ptr{cuComplex},
                                                  csrRowPtrA::Ptr{Cint},
                                                  csrColIndA::Ptr{Cint},
                                                  info::csrluInfoHost_t,
                                                  pivot_threshold::Cfloat,
                                                  pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpZcsrluFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                             csrColIndA, info, pivot_threshold, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrluFactorHost(handle::cusolverSpHandle_t, n::Cint,
                                                  nnzA::Cint, descrA::cusparseMatDescr_t,
                                                  csrValA::Ptr{cuDoubleComplex},
                                                  csrRowPtrA::Ptr{Cint},
                                                  csrColIndA::Ptr{Cint},
                                                  info::csrluInfoHost_t,
                                                  pivot_threshold::Cdouble,
                                                  pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpScsrluZeroPivotHost(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrluZeroPivotHost(handle::cusolverSpHandle_t,
                                                     info::csrluInfoHost_t, tol::Cfloat,
                                                     position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpDcsrluZeroPivotHost(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrluZeroPivotHost(handle::cusolverSpHandle_t,
                                                     info::csrluInfoHost_t, tol::Cdouble,
                                                     position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpCcsrluZeroPivotHost(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrluZeroPivotHost(handle::cusolverSpHandle_t,
                                                     info::csrluInfoHost_t, tol::Cfloat,
                                                     position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpZcsrluZeroPivotHost(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrluZeroPivotHost(handle::cusolverSpHandle_t,
                                                     info::csrluInfoHost_t, tol::Cdouble,
                                                     position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpScsrluSolveHost(handle, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrluSolveHost(handle::cusolverSpHandle_t, n::Cint,
                                                 b::Ptr{Cfloat}, x::Ptr{Cfloat},
                                                 info::csrluInfoHost_t,
                                                 pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpDcsrluSolveHost(handle, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrluSolveHost(handle::cusolverSpHandle_t, n::Cint,
                                                 b::Ptr{Cdouble}, x::Ptr{Cdouble},
                                                 info::csrluInfoHost_t,
                                                 pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpCcsrluSolveHost(handle, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrluSolveHost(handle::cusolverSpHandle_t, n::Cint,
                                                 b::Ptr{cuComplex}, x::Ptr{cuComplex},
                                                 info::csrluInfoHost_t,
                                                 pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpZcsrluSolveHost(handle, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrluSolveHost(handle::cusolverSpHandle_t, n::Cint,
                                                 b::Ptr{cuDoubleComplex},
                                                 x::Ptr{cuDoubleComplex},
                                                 info::csrluInfoHost_t,
                                                 pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpXcsrluNnzHost(handle, nnzLRef, nnzURef, info)
    initialize_context()
    @ccall libcusolver.cusolverSpXcsrluNnzHost(handle::cusolverSpHandle_t,
                                               nnzLRef::Ptr{Cint}, nnzURef::Ptr{Cint},
                                               info::csrluInfoHost_t)::cusolverStatus_t
end

@checked function cusolverSpScsrluExtractHost(handle, P, Q, descrL, csrValL, csrRowPtrL,
                                              csrColIndL, descrU, csrValU, csrRowPtrU,
                                              csrColIndU, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrluExtractHost(handle::cusolverSpHandle_t, P::Ptr{Cint},
                                                   Q::Ptr{Cint}, descrL::cusparseMatDescr_t,
                                                   csrValL::Ptr{Cfloat},
                                                   csrRowPtrL::Ptr{Cint},
                                                   csrColIndL::Ptr{Cint},
                                                   descrU::cusparseMatDescr_t,
                                                   csrValU::Ptr{Cfloat},
                                                   csrRowPtrU::Ptr{Cint},
                                                   csrColIndU::Ptr{Cint},
                                                   info::csrluInfoHost_t,
                                                   pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpDcsrluExtractHost(handle, P, Q, descrL, csrValL, csrRowPtrL,
                                              csrColIndL, descrU, csrValU, csrRowPtrU,
                                              csrColIndU, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrluExtractHost(handle::cusolverSpHandle_t, P::Ptr{Cint},
                                                   Q::Ptr{Cint}, descrL::cusparseMatDescr_t,
                                                   csrValL::Ptr{Cdouble},
                                                   csrRowPtrL::Ptr{Cint},
                                                   csrColIndL::Ptr{Cint},
                                                   descrU::cusparseMatDescr_t,
                                                   csrValU::Ptr{Cdouble},
                                                   csrRowPtrU::Ptr{Cint},
                                                   csrColIndU::Ptr{Cint},
                                                   info::csrluInfoHost_t,
                                                   pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpCcsrluExtractHost(handle, P, Q, descrL, csrValL, csrRowPtrL,
                                              csrColIndL, descrU, csrValU, csrRowPtrU,
                                              csrColIndU, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrluExtractHost(handle::cusolverSpHandle_t, P::Ptr{Cint},
                                                   Q::Ptr{Cint}, descrL::cusparseMatDescr_t,
                                                   csrValL::Ptr{cuComplex},
                                                   csrRowPtrL::Ptr{Cint},
                                                   csrColIndL::Ptr{Cint},
                                                   descrU::cusparseMatDescr_t,
                                                   csrValU::Ptr{cuComplex},
                                                   csrRowPtrU::Ptr{Cint},
                                                   csrColIndU::Ptr{Cint},
                                                   info::csrluInfoHost_t,
                                                   pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpZcsrluExtractHost(handle, P, Q, descrL, csrValL, csrRowPtrL,
                                              csrColIndL, descrU, csrValU, csrRowPtrU,
                                              csrColIndU, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrluExtractHost(handle::cusolverSpHandle_t, P::Ptr{Cint},
                                                   Q::Ptr{Cint}, descrL::cusparseMatDescr_t,
                                                   csrValL::Ptr{cuDoubleComplex},
                                                   csrRowPtrL::Ptr{Cint},
                                                   csrColIndL::Ptr{Cint},
                                                   descrU::cusparseMatDescr_t,
                                                   csrValU::Ptr{cuDoubleComplex},
                                                   csrRowPtrU::Ptr{Cint},
                                                   csrColIndU::Ptr{Cint},
                                                   info::csrluInfoHost_t,
                                                   pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpCreateCsrqrInfoHost(info)
    initialize_context()
    @ccall libcusolver.cusolverSpCreateCsrqrInfoHost(info::Ptr{csrqrInfoHost_t})::cusolverStatus_t
end

@checked function cusolverSpDestroyCsrqrInfoHost(info)
    initialize_context()
    @ccall libcusolver.cusolverSpDestroyCsrqrInfoHost(info::csrqrInfoHost_t)::cusolverStatus_t
end

@checked function cusolverSpXcsrqrAnalysisHost(handle, m, n, nnzA, descrA, csrRowPtrA,
                                               csrColIndA, info)
    initialize_context()
    @ccall libcusolver.cusolverSpXcsrqrAnalysisHost(handle::cusolverSpHandle_t, m::Cint,
                                                    n::Cint, nnzA::Cint,
                                                    descrA::cusparseMatDescr_t,
                                                    csrRowPtrA::Ptr{Cint},
                                                    csrColIndA::Ptr{Cint},
                                                    info::csrqrInfoHost_t)::cusolverStatus_t
end

@checked function cusolverSpScsrqrBufferInfoHost(handle, m, n, nnzA, descrA, csrValA,
                                                 csrRowPtrA, csrColIndA, info,
                                                 internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrqrBufferInfoHost(handle::cusolverSpHandle_t, m::Cint,
                                                      n::Cint, nnzA::Cint,
                                                      descrA::cusparseMatDescr_t,
                                                      csrValA::Ptr{Cfloat},
                                                      csrRowPtrA::Ptr{Cint},
                                                      csrColIndA::Ptr{Cint},
                                                      info::csrqrInfoHost_t,
                                                      internalDataInBytes::Ptr{Csize_t},
                                                      workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpDcsrqrBufferInfoHost(handle, m, n, nnzA, descrA, csrValA,
                                                 csrRowPtrA, csrColIndA, info,
                                                 internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrqrBufferInfoHost(handle::cusolverSpHandle_t, m::Cint,
                                                      n::Cint, nnzA::Cint,
                                                      descrA::cusparseMatDescr_t,
                                                      csrValA::Ptr{Cdouble},
                                                      csrRowPtrA::Ptr{Cint},
                                                      csrColIndA::Ptr{Cint},
                                                      info::csrqrInfoHost_t,
                                                      internalDataInBytes::Ptr{Csize_t},
                                                      workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpCcsrqrBufferInfoHost(handle, m, n, nnzA, descrA, csrValA,
                                                 csrRowPtrA, csrColIndA, info,
                                                 internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrqrBufferInfoHost(handle::cusolverSpHandle_t, m::Cint,
                                                      n::Cint, nnzA::Cint,
                                                      descrA::cusparseMatDescr_t,
                                                      csrValA::Ptr{cuComplex},
                                                      csrRowPtrA::Ptr{Cint},
                                                      csrColIndA::Ptr{Cint},
                                                      info::csrqrInfoHost_t,
                                                      internalDataInBytes::Ptr{Csize_t},
                                                      workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpZcsrqrBufferInfoHost(handle, m, n, nnzA, descrA, csrValA,
                                                 csrRowPtrA, csrColIndA, info,
                                                 internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrqrBufferInfoHost(handle::cusolverSpHandle_t, m::Cint,
                                                      n::Cint, nnzA::Cint,
                                                      descrA::cusparseMatDescr_t,
                                                      csrValA::Ptr{cuDoubleComplex},
                                                      csrRowPtrA::Ptr{Cint},
                                                      csrColIndA::Ptr{Cint},
                                                      info::csrqrInfoHost_t,
                                                      internalDataInBytes::Ptr{Csize_t},
                                                      workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpScsrqrSetupHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA,
                                            csrColIndA, mu, info)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrqrSetupHost(handle::cusolverSpHandle_t, m::Cint,
                                                 n::Cint, nnzA::Cint,
                                                 descrA::cusparseMatDescr_t,
                                                 csrValA::Ptr{Cfloat},
                                                 csrRowPtrA::Ptr{Cint},
                                                 csrColIndA::Ptr{Cint}, mu::Cfloat,
                                                 info::csrqrInfoHost_t)::cusolverStatus_t
end

@checked function cusolverSpDcsrqrSetupHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA,
                                            csrColIndA, mu, info)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrqrSetupHost(handle::cusolverSpHandle_t, m::Cint,
                                                 n::Cint, nnzA::Cint,
                                                 descrA::cusparseMatDescr_t,
                                                 csrValA::Ptr{Cdouble},
                                                 csrRowPtrA::Ptr{Cint},
                                                 csrColIndA::Ptr{Cint}, mu::Cdouble,
                                                 info::csrqrInfoHost_t)::cusolverStatus_t
end

@checked function cusolverSpCcsrqrSetupHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA,
                                            csrColIndA, mu, info)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrqrSetupHost(handle::cusolverSpHandle_t, m::Cint,
                                                 n::Cint, nnzA::Cint,
                                                 descrA::cusparseMatDescr_t,
                                                 csrValA::Ptr{cuComplex},
                                                 csrRowPtrA::Ptr{Cint},
                                                 csrColIndA::Ptr{Cint}, mu::cuComplex,
                                                 info::csrqrInfoHost_t)::cusolverStatus_t
end

@checked function cusolverSpZcsrqrSetupHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA,
                                            csrColIndA, mu, info)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrqrSetupHost(handle::cusolverSpHandle_t, m::Cint,
                                                 n::Cint, nnzA::Cint,
                                                 descrA::cusparseMatDescr_t,
                                                 csrValA::Ptr{cuDoubleComplex},
                                                 csrRowPtrA::Ptr{Cint},
                                                 csrColIndA::Ptr{Cint}, mu::cuDoubleComplex,
                                                 info::csrqrInfoHost_t)::cusolverStatus_t
end

@checked function cusolverSpScsrqrFactorHost(handle, m, n, nnzA, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrqrFactorHost(handle::cusolverSpHandle_t, m::Cint,
                                                  n::Cint, nnzA::Cint, b::Ptr{Cfloat},
                                                  x::Ptr{Cfloat}, info::csrqrInfoHost_t,
                                                  pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpDcsrqrFactorHost(handle, m, n, nnzA, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrqrFactorHost(handle::cusolverSpHandle_t, m::Cint,
                                                  n::Cint, nnzA::Cint, b::Ptr{Cdouble},
                                                  x::Ptr{Cdouble}, info::csrqrInfoHost_t,
                                                  pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpCcsrqrFactorHost(handle, m, n, nnzA, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrqrFactorHost(handle::cusolverSpHandle_t, m::Cint,
                                                  n::Cint, nnzA::Cint, b::Ptr{cuComplex},
                                                  x::Ptr{cuComplex}, info::csrqrInfoHost_t,
                                                  pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpZcsrqrFactorHost(handle, m, n, nnzA, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrqrFactorHost(handle::cusolverSpHandle_t, m::Cint,
                                                  n::Cint, nnzA::Cint,
                                                  b::Ptr{cuDoubleComplex},
                                                  x::Ptr{cuDoubleComplex},
                                                  info::csrqrInfoHost_t,
                                                  pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpScsrqrZeroPivotHost(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrqrZeroPivotHost(handle::cusolverSpHandle_t,
                                                     info::csrqrInfoHost_t, tol::Cfloat,
                                                     position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpDcsrqrZeroPivotHost(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrqrZeroPivotHost(handle::cusolverSpHandle_t,
                                                     info::csrqrInfoHost_t, tol::Cdouble,
                                                     position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpCcsrqrZeroPivotHost(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrqrZeroPivotHost(handle::cusolverSpHandle_t,
                                                     info::csrqrInfoHost_t, tol::Cfloat,
                                                     position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpZcsrqrZeroPivotHost(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrqrZeroPivotHost(handle::cusolverSpHandle_t,
                                                     info::csrqrInfoHost_t, tol::Cdouble,
                                                     position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpScsrqrSolveHost(handle, m, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrqrSolveHost(handle::cusolverSpHandle_t, m::Cint,
                                                 n::Cint, b::Ptr{Cfloat}, x::Ptr{Cfloat},
                                                 info::csrqrInfoHost_t,
                                                 pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpDcsrqrSolveHost(handle, m, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrqrSolveHost(handle::cusolverSpHandle_t, m::Cint,
                                                 n::Cint, b::Ptr{Cdouble}, x::Ptr{Cdouble},
                                                 info::csrqrInfoHost_t,
                                                 pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpCcsrqrSolveHost(handle, m, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrqrSolveHost(handle::cusolverSpHandle_t, m::Cint,
                                                 n::Cint, b::Ptr{cuComplex},
                                                 x::Ptr{cuComplex}, info::csrqrInfoHost_t,
                                                 pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpZcsrqrSolveHost(handle, m, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrqrSolveHost(handle::cusolverSpHandle_t, m::Cint,
                                                 n::Cint, b::Ptr{cuDoubleComplex},
                                                 x::Ptr{cuDoubleComplex},
                                                 info::csrqrInfoHost_t,
                                                 pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpXcsrqrAnalysis(handle, m, n, nnzA, descrA, csrRowPtrA,
                                           csrColIndA, info)
    initialize_context()
    @ccall libcusolver.cusolverSpXcsrqrAnalysis(handle::cusolverSpHandle_t, m::Cint,
                                                n::Cint, nnzA::Cint,
                                                descrA::cusparseMatDescr_t,
                                                csrRowPtrA::Ptr{Cint},
                                                csrColIndA::Ptr{Cint},
                                                info::csrqrInfo_t)::cusolverStatus_t
end

@checked function cusolverSpScsrqrBufferInfo(handle, m, n, nnzA, descrA, csrValA,
                                             csrRowPtrA, csrColIndA, info,
                                             internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrqrBufferInfo(handle::cusolverSpHandle_t, m::Cint,
                                                  n::Cint, nnzA::Cint,
                                                  descrA::cusparseMatDescr_t,
                                                  csrValA::Ptr{Cfloat},
                                                  csrRowPtrA::Ptr{Cint},
                                                  csrColIndA::Ptr{Cint}, info::csrqrInfo_t,
                                                  internalDataInBytes::Ptr{Csize_t},
                                                  workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpDcsrqrBufferInfo(handle, m, n, nnzA, descrA, csrValA,
                                             csrRowPtrA, csrColIndA, info,
                                             internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrqrBufferInfo(handle::cusolverSpHandle_t, m::Cint,
                                                  n::Cint, nnzA::Cint,
                                                  descrA::cusparseMatDescr_t,
                                                  csrValA::Ptr{Cdouble},
                                                  csrRowPtrA::Ptr{Cint},
                                                  csrColIndA::Ptr{Cint}, info::csrqrInfo_t,
                                                  internalDataInBytes::Ptr{Csize_t},
                                                  workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpCcsrqrBufferInfo(handle, m, n, nnzA, descrA, csrValA,
                                             csrRowPtrA, csrColIndA, info,
                                             internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrqrBufferInfo(handle::cusolverSpHandle_t, m::Cint,
                                                  n::Cint, nnzA::Cint,
                                                  descrA::cusparseMatDescr_t,
                                                  csrValA::Ptr{cuComplex},
                                                  csrRowPtrA::Ptr{Cint},
                                                  csrColIndA::Ptr{Cint}, info::csrqrInfo_t,
                                                  internalDataInBytes::Ptr{Csize_t},
                                                  workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpZcsrqrBufferInfo(handle, m, n, nnzA, descrA, csrValA,
                                             csrRowPtrA, csrColIndA, info,
                                             internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrqrBufferInfo(handle::cusolverSpHandle_t, m::Cint,
                                                  n::Cint, nnzA::Cint,
                                                  descrA::cusparseMatDescr_t,
                                                  csrValA::Ptr{cuDoubleComplex},
                                                  csrRowPtrA::Ptr{Cint},
                                                  csrColIndA::Ptr{Cint}, info::csrqrInfo_t,
                                                  internalDataInBytes::Ptr{Csize_t},
                                                  workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpScsrqrSetup(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA,
                                        csrColIndA, mu, info)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrqrSetup(handle::cusolverSpHandle_t, m::Cint, n::Cint,
                                             nnzA::Cint, descrA::cusparseMatDescr_t,
                                             csrValA::Ptr{Cfloat}, csrRowPtrA::Ptr{Cint},
                                             csrColIndA::Ptr{Cint}, mu::Cfloat,
                                             info::csrqrInfo_t)::cusolverStatus_t
end

@checked function cusolverSpDcsrqrSetup(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA,
                                        csrColIndA, mu, info)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrqrSetup(handle::cusolverSpHandle_t, m::Cint, n::Cint,
                                             nnzA::Cint, descrA::cusparseMatDescr_t,
                                             csrValA::Ptr{Cdouble}, csrRowPtrA::Ptr{Cint},
                                             csrColIndA::Ptr{Cint}, mu::Cdouble,
                                             info::csrqrInfo_t)::cusolverStatus_t
end

@checked function cusolverSpCcsrqrSetup(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA,
                                        csrColIndA, mu, info)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrqrSetup(handle::cusolverSpHandle_t, m::Cint, n::Cint,
                                             nnzA::Cint, descrA::cusparseMatDescr_t,
                                             csrValA::Ptr{cuComplex}, csrRowPtrA::Ptr{Cint},
                                             csrColIndA::Ptr{Cint}, mu::cuComplex,
                                             info::csrqrInfo_t)::cusolverStatus_t
end

@checked function cusolverSpZcsrqrSetup(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA,
                                        csrColIndA, mu, info)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrqrSetup(handle::cusolverSpHandle_t, m::Cint, n::Cint,
                                             nnzA::Cint, descrA::cusparseMatDescr_t,
                                             csrValA::Ptr{cuDoubleComplex},
                                             csrRowPtrA::Ptr{Cint}, csrColIndA::Ptr{Cint},
                                             mu::cuDoubleComplex,
                                             info::csrqrInfo_t)::cusolverStatus_t
end

@checked function cusolverSpScsrqrFactor(handle, m, n, nnzA, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrqrFactor(handle::cusolverSpHandle_t, m::Cint, n::Cint,
                                              nnzA::Cint, b::Ptr{Cfloat}, x::Ptr{Cfloat},
                                              info::csrqrInfo_t,
                                              pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpDcsrqrFactor(handle, m, n, nnzA, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrqrFactor(handle::cusolverSpHandle_t, m::Cint, n::Cint,
                                              nnzA::Cint, b::Ptr{Cdouble}, x::Ptr{Cdouble},
                                              info::csrqrInfo_t,
                                              pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpCcsrqrFactor(handle, m, n, nnzA, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrqrFactor(handle::cusolverSpHandle_t, m::Cint, n::Cint,
                                              nnzA::Cint, b::Ptr{cuComplex},
                                              x::Ptr{cuComplex}, info::csrqrInfo_t,
                                              pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpZcsrqrFactor(handle, m, n, nnzA, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrqrFactor(handle::cusolverSpHandle_t, m::Cint, n::Cint,
                                              nnzA::Cint, b::Ptr{cuDoubleComplex},
                                              x::Ptr{cuDoubleComplex}, info::csrqrInfo_t,
                                              pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpScsrqrZeroPivot(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrqrZeroPivot(handle::cusolverSpHandle_t,
                                                 info::csrqrInfo_t, tol::Cfloat,
                                                 position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpDcsrqrZeroPivot(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrqrZeroPivot(handle::cusolverSpHandle_t,
                                                 info::csrqrInfo_t, tol::Cdouble,
                                                 position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpCcsrqrZeroPivot(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrqrZeroPivot(handle::cusolverSpHandle_t,
                                                 info::csrqrInfo_t, tol::Cfloat,
                                                 position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpZcsrqrZeroPivot(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrqrZeroPivot(handle::cusolverSpHandle_t,
                                                 info::csrqrInfo_t, tol::Cdouble,
                                                 position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpScsrqrSolve(handle, m, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrqrSolve(handle::cusolverSpHandle_t, m::Cint, n::Cint,
                                             b::Ptr{Cfloat}, x::Ptr{Cfloat},
                                             info::csrqrInfo_t,
                                             pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpDcsrqrSolve(handle, m, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrqrSolve(handle::cusolverSpHandle_t, m::Cint, n::Cint,
                                             b::Ptr{Cdouble}, x::Ptr{Cdouble},
                                             info::csrqrInfo_t,
                                             pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpCcsrqrSolve(handle, m, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrqrSolve(handle::cusolverSpHandle_t, m::Cint, n::Cint,
                                             b::Ptr{cuComplex}, x::Ptr{cuComplex},
                                             info::csrqrInfo_t,
                                             pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpZcsrqrSolve(handle, m, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrqrSolve(handle::cusolverSpHandle_t, m::Cint, n::Cint,
                                             b::Ptr{cuDoubleComplex},
                                             x::Ptr{cuDoubleComplex}, info::csrqrInfo_t,
                                             pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpCreateCsrcholInfoHost(info)
    initialize_context()
    @ccall libcusolver.cusolverSpCreateCsrcholInfoHost(info::Ptr{csrcholInfoHost_t})::cusolverStatus_t
end

@checked function cusolverSpDestroyCsrcholInfoHost(info)
    initialize_context()
    @ccall libcusolver.cusolverSpDestroyCsrcholInfoHost(info::csrcholInfoHost_t)::cusolverStatus_t
end

@checked function cusolverSpXcsrcholAnalysisHost(handle, n, nnzA, descrA, csrRowPtrA,
                                                 csrColIndA, info)
    initialize_context()
    @ccall libcusolver.cusolverSpXcsrcholAnalysisHost(handle::cusolverSpHandle_t, n::Cint,
                                                      nnzA::Cint,
                                                      descrA::cusparseMatDescr_t,
                                                      csrRowPtrA::Ptr{Cint},
                                                      csrColIndA::Ptr{Cint},
                                                      info::csrcholInfoHost_t)::cusolverStatus_t
end

@checked function cusolverSpScsrcholBufferInfoHost(handle, n, nnzA, descrA, csrValA,
                                                   csrRowPtrA, csrColIndA, info,
                                                   internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrcholBufferInfoHost(handle::cusolverSpHandle_t, n::Cint,
                                                        nnzA::Cint,
                                                        descrA::cusparseMatDescr_t,
                                                        csrValA::Ptr{Cfloat},
                                                        csrRowPtrA::Ptr{Cint},
                                                        csrColIndA::Ptr{Cint},
                                                        info::csrcholInfoHost_t,
                                                        internalDataInBytes::Ptr{Csize_t},
                                                        workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpDcsrcholBufferInfoHost(handle, n, nnzA, descrA, csrValA,
                                                   csrRowPtrA, csrColIndA, info,
                                                   internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrcholBufferInfoHost(handle::cusolverSpHandle_t, n::Cint,
                                                        nnzA::Cint,
                                                        descrA::cusparseMatDescr_t,
                                                        csrValA::Ptr{Cdouble},
                                                        csrRowPtrA::Ptr{Cint},
                                                        csrColIndA::Ptr{Cint},
                                                        info::csrcholInfoHost_t,
                                                        internalDataInBytes::Ptr{Csize_t},
                                                        workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpCcsrcholBufferInfoHost(handle, n, nnzA, descrA, csrValA,
                                                   csrRowPtrA, csrColIndA, info,
                                                   internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrcholBufferInfoHost(handle::cusolverSpHandle_t, n::Cint,
                                                        nnzA::Cint,
                                                        descrA::cusparseMatDescr_t,
                                                        csrValA::Ptr{cuComplex},
                                                        csrRowPtrA::Ptr{Cint},
                                                        csrColIndA::Ptr{Cint},
                                                        info::csrcholInfoHost_t,
                                                        internalDataInBytes::Ptr{Csize_t},
                                                        workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpZcsrcholBufferInfoHost(handle, n, nnzA, descrA, csrValA,
                                                   csrRowPtrA, csrColIndA, info,
                                                   internalDataInBytes, workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrcholBufferInfoHost(handle::cusolverSpHandle_t, n::Cint,
                                                        nnzA::Cint,
                                                        descrA::cusparseMatDescr_t,
                                                        csrValA::Ptr{cuDoubleComplex},
                                                        csrRowPtrA::Ptr{Cint},
                                                        csrColIndA::Ptr{Cint},
                                                        info::csrcholInfoHost_t,
                                                        internalDataInBytes::Ptr{Csize_t},
                                                        workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpScsrcholFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                               csrColIndA, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrcholFactorHost(handle::cusolverSpHandle_t, n::Cint,
                                                    nnzA::Cint, descrA::cusparseMatDescr_t,
                                                    csrValA::Ptr{Cfloat},
                                                    csrRowPtrA::Ptr{Cint},
                                                    csrColIndA::Ptr{Cint},
                                                    info::csrcholInfoHost_t,
                                                    pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpDcsrcholFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                               csrColIndA, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrcholFactorHost(handle::cusolverSpHandle_t, n::Cint,
                                                    nnzA::Cint, descrA::cusparseMatDescr_t,
                                                    csrValA::Ptr{Cdouble},
                                                    csrRowPtrA::Ptr{Cint},
                                                    csrColIndA::Ptr{Cint},
                                                    info::csrcholInfoHost_t,
                                                    pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpCcsrcholFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                               csrColIndA, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrcholFactorHost(handle::cusolverSpHandle_t, n::Cint,
                                                    nnzA::Cint, descrA::cusparseMatDescr_t,
                                                    csrValA::Ptr{cuComplex},
                                                    csrRowPtrA::Ptr{Cint},
                                                    csrColIndA::Ptr{Cint},
                                                    info::csrcholInfoHost_t,
                                                    pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpZcsrcholFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                               csrColIndA, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrcholFactorHost(handle::cusolverSpHandle_t, n::Cint,
                                                    nnzA::Cint, descrA::cusparseMatDescr_t,
                                                    csrValA::Ptr{cuDoubleComplex},
                                                    csrRowPtrA::Ptr{Cint},
                                                    csrColIndA::Ptr{Cint},
                                                    info::csrcholInfoHost_t,
                                                    pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpScsrcholZeroPivotHost(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrcholZeroPivotHost(handle::cusolverSpHandle_t,
                                                       info::csrcholInfoHost_t, tol::Cfloat,
                                                       position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpDcsrcholZeroPivotHost(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrcholZeroPivotHost(handle::cusolverSpHandle_t,
                                                       info::csrcholInfoHost_t,
                                                       tol::Cdouble,
                                                       position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpCcsrcholZeroPivotHost(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrcholZeroPivotHost(handle::cusolverSpHandle_t,
                                                       info::csrcholInfoHost_t, tol::Cfloat,
                                                       position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpZcsrcholZeroPivotHost(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrcholZeroPivotHost(handle::cusolverSpHandle_t,
                                                       info::csrcholInfoHost_t,
                                                       tol::Cdouble,
                                                       position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpScsrcholSolveHost(handle, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrcholSolveHost(handle::cusolverSpHandle_t, n::Cint,
                                                   b::Ptr{Cfloat}, x::Ptr{Cfloat},
                                                   info::csrcholInfoHost_t,
                                                   pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpDcsrcholSolveHost(handle, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrcholSolveHost(handle::cusolverSpHandle_t, n::Cint,
                                                   b::Ptr{Cdouble}, x::Ptr{Cdouble},
                                                   info::csrcholInfoHost_t,
                                                   pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpCcsrcholSolveHost(handle, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrcholSolveHost(handle::cusolverSpHandle_t, n::Cint,
                                                   b::Ptr{cuComplex}, x::Ptr{cuComplex},
                                                   info::csrcholInfoHost_t,
                                                   pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpZcsrcholSolveHost(handle, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrcholSolveHost(handle::cusolverSpHandle_t, n::Cint,
                                                   b::Ptr{cuDoubleComplex},
                                                   x::Ptr{cuDoubleComplex},
                                                   info::csrcholInfoHost_t,
                                                   pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpCreateCsrcholInfo(info)
    initialize_context()
    @ccall libcusolver.cusolverSpCreateCsrcholInfo(info::Ptr{csrcholInfo_t})::cusolverStatus_t
end

@checked function cusolverSpDestroyCsrcholInfo(info)
    initialize_context()
    @ccall libcusolver.cusolverSpDestroyCsrcholInfo(info::csrcholInfo_t)::cusolverStatus_t
end

@checked function cusolverSpXcsrcholAnalysis(handle, n, nnzA, descrA, csrRowPtrA,
                                             csrColIndA, info)
    initialize_context()
    @ccall libcusolver.cusolverSpXcsrcholAnalysis(handle::cusolverSpHandle_t, n::Cint,
                                                  nnzA::Cint, descrA::cusparseMatDescr_t,
                                                  csrRowPtrA::Ptr{Cint},
                                                  csrColIndA::Ptr{Cint},
                                                  info::csrcholInfo_t)::cusolverStatus_t
end

@checked function cusolverSpScsrcholBufferInfo(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                               csrColIndA, info, internalDataInBytes,
                                               workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrcholBufferInfo(handle::cusolverSpHandle_t, n::Cint,
                                                    nnzA::Cint, descrA::cusparseMatDescr_t,
                                                    csrValA::Ptr{Cfloat},
                                                    csrRowPtrA::Ptr{Cint},
                                                    csrColIndA::Ptr{Cint},
                                                    info::csrcholInfo_t,
                                                    internalDataInBytes::Ptr{Csize_t},
                                                    workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpDcsrcholBufferInfo(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                               csrColIndA, info, internalDataInBytes,
                                               workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrcholBufferInfo(handle::cusolverSpHandle_t, n::Cint,
                                                    nnzA::Cint, descrA::cusparseMatDescr_t,
                                                    csrValA::Ptr{Cdouble},
                                                    csrRowPtrA::Ptr{Cint},
                                                    csrColIndA::Ptr{Cint},
                                                    info::csrcholInfo_t,
                                                    internalDataInBytes::Ptr{Csize_t},
                                                    workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpCcsrcholBufferInfo(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                               csrColIndA, info, internalDataInBytes,
                                               workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrcholBufferInfo(handle::cusolverSpHandle_t, n::Cint,
                                                    nnzA::Cint, descrA::cusparseMatDescr_t,
                                                    csrValA::Ptr{cuComplex},
                                                    csrRowPtrA::Ptr{Cint},
                                                    csrColIndA::Ptr{Cint},
                                                    info::csrcholInfo_t,
                                                    internalDataInBytes::Ptr{Csize_t},
                                                    workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpZcsrcholBufferInfo(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                               csrColIndA, info, internalDataInBytes,
                                               workspaceInBytes)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrcholBufferInfo(handle::cusolverSpHandle_t, n::Cint,
                                                    nnzA::Cint, descrA::cusparseMatDescr_t,
                                                    csrValA::Ptr{cuDoubleComplex},
                                                    csrRowPtrA::Ptr{Cint},
                                                    csrColIndA::Ptr{Cint},
                                                    info::csrcholInfo_t,
                                                    internalDataInBytes::Ptr{Csize_t},
                                                    workspaceInBytes::Ptr{Csize_t})::cusolverStatus_t
end

@checked function cusolverSpScsrcholFactor(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrcholFactor(handle::cusolverSpHandle_t, n::Cint,
                                                nnzA::Cint, descrA::cusparseMatDescr_t,
                                                csrValA::Ptr{Cfloat}, csrRowPtrA::Ptr{Cint},
                                                csrColIndA::Ptr{Cint}, info::csrcholInfo_t,
                                                pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpDcsrcholFactor(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrcholFactor(handle::cusolverSpHandle_t, n::Cint,
                                                nnzA::Cint, descrA::cusparseMatDescr_t,
                                                csrValA::Ptr{Cdouble},
                                                csrRowPtrA::Ptr{Cint},
                                                csrColIndA::Ptr{Cint}, info::csrcholInfo_t,
                                                pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpCcsrcholFactor(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrcholFactor(handle::cusolverSpHandle_t, n::Cint,
                                                nnzA::Cint, descrA::cusparseMatDescr_t,
                                                csrValA::Ptr{cuComplex},
                                                csrRowPtrA::Ptr{Cint},
                                                csrColIndA::Ptr{Cint}, info::csrcholInfo_t,
                                                pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpZcsrcholFactor(handle, n, nnzA, descrA, csrValA, csrRowPtrA,
                                           csrColIndA, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrcholFactor(handle::cusolverSpHandle_t, n::Cint,
                                                nnzA::Cint, descrA::cusparseMatDescr_t,
                                                csrValA::Ptr{cuDoubleComplex},
                                                csrRowPtrA::Ptr{Cint},
                                                csrColIndA::Ptr{Cint}, info::csrcholInfo_t,
                                                pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpScsrcholZeroPivot(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrcholZeroPivot(handle::cusolverSpHandle_t,
                                                   info::csrcholInfo_t, tol::Cfloat,
                                                   position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpDcsrcholZeroPivot(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrcholZeroPivot(handle::cusolverSpHandle_t,
                                                   info::csrcholInfo_t, tol::Cdouble,
                                                   position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpCcsrcholZeroPivot(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrcholZeroPivot(handle::cusolverSpHandle_t,
                                                   info::csrcholInfo_t, tol::Cfloat,
                                                   position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpZcsrcholZeroPivot(handle, info, tol, position)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrcholZeroPivot(handle::cusolverSpHandle_t,
                                                   info::csrcholInfo_t, tol::Cdouble,
                                                   position::Ptr{Cint})::cusolverStatus_t
end

@checked function cusolverSpScsrcholSolve(handle, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrcholSolve(handle::cusolverSpHandle_t, n::Cint,
                                               b::Ptr{Cfloat}, x::Ptr{Cfloat},
                                               info::csrcholInfo_t,
                                               pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpDcsrcholSolve(handle, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrcholSolve(handle::cusolverSpHandle_t, n::Cint,
                                               b::Ptr{Cdouble}, x::Ptr{Cdouble},
                                               info::csrcholInfo_t,
                                               pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpCcsrcholSolve(handle, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrcholSolve(handle::cusolverSpHandle_t, n::Cint,
                                               b::Ptr{cuComplex}, x::Ptr{cuComplex},
                                               info::csrcholInfo_t,
                                               pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpZcsrcholSolve(handle, n, b, x, info, pBuffer)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrcholSolve(handle::cusolverSpHandle_t, n::Cint,
                                               b::Ptr{cuDoubleComplex},
                                               x::Ptr{cuDoubleComplex}, info::csrcholInfo_t,
                                               pBuffer::Ptr{Cvoid})::cusolverStatus_t
end

@checked function cusolverSpScsrcholDiag(handle, info, diag)
    initialize_context()
    @ccall libcusolver.cusolverSpScsrcholDiag(handle::cusolverSpHandle_t,
                                              info::csrcholInfo_t,
                                              diag::Ptr{Cfloat})::cusolverStatus_t
end

@checked function cusolverSpDcsrcholDiag(handle, info, diag)
    initialize_context()
    @ccall libcusolver.cusolverSpDcsrcholDiag(handle::cusolverSpHandle_t,
                                              info::csrcholInfo_t,
                                              diag::Ptr{Cdouble})::cusolverStatus_t
end

@checked function cusolverSpCcsrcholDiag(handle, info, diag)
    initialize_context()
    @ccall libcusolver.cusolverSpCcsrcholDiag(handle::cusolverSpHandle_t,
                                              info::csrcholInfo_t,
                                              diag::Ptr{Cfloat})::cusolverStatus_t
end

@checked function cusolverSpZcsrcholDiag(handle, info, diag)
    initialize_context()
    @ccall libcusolver.cusolverSpZcsrcholDiag(handle::cusolverSpHandle_t,
                                              info::csrcholInfo_t,
                                              diag::Ptr{Cdouble})::cusolverStatus_t
end