# Julia wrapper for header: cusolver_common.h
# Automatically generated using Clang.jl


function cusolverGetProperty(type, value)
    @check @runtime_ccall((:cusolverGetProperty, libcusolver), cusolverStatus_t,
                 (libraryPropertyType, Ptr{Cint}),
                 type, value)
end

function cusolverGetVersion(version)
    @check @runtime_ccall((:cusolverGetVersion, libcusolver), cusolverStatus_t,
                 (Ptr{Cint},),
                 version)
end
# Julia wrapper for header: cusolverDn.h
# Automatically generated using Clang.jl


function cusolverDnCreate(handle)
    @check @runtime_ccall((:cusolverDnCreate, libcusolver), cusolverStatus_t,
                 (Ptr{cusolverDnHandle_t},),
                 handle)
end

function cusolverDnDestroy(handle)
    @check @runtime_ccall((:cusolverDnDestroy, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t,),
                 handle)
end

function cusolverDnSetStream(handle, streamId)
    @check @runtime_ccall((:cusolverDnSetStream, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, CUstream),
                 handle, streamId)
end

function cusolverDnGetStream(handle, streamId)
    @check @runtime_ccall((:cusolverDnGetStream, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Ptr{CUstream}),
                 handle, streamId)
end

function cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    @check @runtime_ccall((:cusolverDnSpotrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, A, lda, Lwork)
end

function cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    @check @runtime_ccall((:cusolverDnDpotrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, A, lda, Lwork)
end

function cusolverDnCpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    @check @runtime_ccall((:cusolverDnCpotrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, A, lda, Lwork)
end

function cusolverDnZpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    @check @runtime_ccall((:cusolverDnZpotrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, Ptr{Cint}),
                 handle, uplo, n, A, lda, Lwork)
end

function cusolverDnSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    @check @runtime_ccall((:cusolverDnSpotrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
end

function cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    @check @runtime_ccall((:cusolverDnDpotrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
end

function cusolverDnCpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    @check @runtime_ccall((:cusolverDnCpotrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
end

function cusolverDnZpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    @check @runtime_ccall((:cusolverDnZpotrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
end

function cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    @check @runtime_ccall((:cusolverDnSpotrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
end

function cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    @check @runtime_ccall((:cusolverDnDpotrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
end

function cusolverDnCpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    @check @runtime_ccall((:cusolverDnCpotrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{cuComplex},
                  Cint, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
end

function cusolverDnZpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    @check @runtime_ccall((:cusolverDnZpotrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
end

function cusolverDnSpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    @check @runtime_ccall((:cusolverDnSpotrfBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Ptr{Cfloat}}, Cint,
                  CuPtr{Cint}, Cint),
                 handle, uplo, n, Aarray, lda, infoArray, batchSize)
end

function cusolverDnDpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    @check @runtime_ccall((:cusolverDnDpotrfBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Ptr{Cdouble}}, Cint,
                  CuPtr{Cint}, Cint),
                 handle, uplo, n, Aarray, lda, infoArray, batchSize)
end

function cusolverDnCpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    @check @runtime_ccall((:cusolverDnCpotrfBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Ptr{cuComplex}}, Cint,
                  CuPtr{Cint}, Cint),
                 handle, uplo, n, Aarray, lda, infoArray, batchSize)
end

function cusolverDnZpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    @check @runtime_ccall((:cusolverDnZpotrfBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Ptr{cuDoubleComplex}},
                  Cint, CuPtr{Cint}, Cint),
                 handle, uplo, n, Aarray, lda, infoArray, batchSize)
end

function cusolverDnSpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
    @check @runtime_ccall((:cusolverDnSpotrsBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Ptr{Cfloat}},
                  Cint, CuPtr{Ptr{Cfloat}}, Cint, CuPtr{Cint}, Cint),
                 handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
end

function cusolverDnDpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
    @check @runtime_ccall((:cusolverDnDpotrsBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Ptr{Cdouble}},
                  Cint, CuPtr{Ptr{Cdouble}}, Cint, CuPtr{Cint}, Cint),
                 handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
end

function cusolverDnCpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
    @check @runtime_ccall((:cusolverDnCpotrsBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Ptr{cuComplex}},
                  Cint, CuPtr{Ptr{cuComplex}}, Cint, CuPtr{Cint}, Cint),
                 handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
end

function cusolverDnZpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
    @check @runtime_ccall((:cusolverDnZpotrsBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                  CuPtr{Ptr{cuDoubleComplex}}, Cint, CuPtr{Ptr{cuDoubleComplex}}, Cint,
                  CuPtr{Cint}, Cint),
                 handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
end

function cusolverDnSpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnSpotri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, A, lda, lwork)
end

function cusolverDnDpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnDpotri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, A, lda, lwork)
end

function cusolverDnCpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnCpotri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, A, lda, lwork)
end

function cusolverDnZpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnZpotri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, Ptr{Cint}),
                 handle, uplo, n, A, lda, lwork)
end

function cusolverDnSpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    @check @runtime_ccall((:cusolverDnSpotri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, work, lwork, devInfo)
end

function cusolverDnDpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    @check @runtime_ccall((:cusolverDnDpotri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, work, lwork, devInfo)
end

function cusolverDnCpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    @check @runtime_ccall((:cusolverDnCpotri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, work, lwork, devInfo)
end

function cusolverDnZpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    @check @runtime_ccall((:cusolverDnZpotri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, work, lwork, devInfo)
end

function cusolverDnStrtri_bufferSize(handle, uplo, diag, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnStrtri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                  CuPtr{Cfloat}, Cint, Ptr{Cint}),
                 handle, uplo, diag, n, A, lda, lwork)
end

function cusolverDnDtrtri_bufferSize(handle, uplo, diag, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnDtrtri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                  CuPtr{Cdouble}, Cint, Ptr{Cint}),
                 handle, uplo, diag, n, A, lda, lwork)
end

function cusolverDnCtrtri_bufferSize(handle, uplo, diag, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnCtrtri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                  CuPtr{cuComplex}, Cint, Ptr{Cint}),
                 handle, uplo, diag, n, A, lda, lwork)
end

function cusolverDnZtrtri_bufferSize(handle, uplo, diag, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnZtrtri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                  CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                 handle, uplo, diag, n, A, lda, lwork)
end

function cusolverDnStrtri(handle, uplo, diag, n, A, lda, work, lwork, devInfo)
    @check @runtime_ccall((:cusolverDnStrtri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, uplo, diag, n, A, lda, work, lwork, devInfo)
end

function cusolverDnDtrtri(handle, uplo, diag, n, A, lda, work, lwork, devInfo)
    @check @runtime_ccall((:cusolverDnDtrtri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, uplo, diag, n, A, lda, work, lwork, devInfo)
end

function cusolverDnCtrtri(handle, uplo, diag, n, A, lda, work, lwork, devInfo)
    @check @runtime_ccall((:cusolverDnCtrtri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, diag, n, A, lda, work, lwork, devInfo)
end

function cusolverDnZtrtri(handle, uplo, diag, n, A, lda, work, lwork, devInfo)
    @check @runtime_ccall((:cusolverDnZtrtri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, diag, n, A, lda, work, lwork, devInfo)
end

function cusolverDnSlauum_bufferSize(handle, uplo, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnSlauum_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, A, lda, lwork)
end

function cusolverDnDlauum_bufferSize(handle, uplo, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnDlauum_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, A, lda, lwork)
end

function cusolverDnClauum_bufferSize(handle, uplo, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnClauum_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, A, lda, lwork)
end

function cusolverDnZlauum_bufferSize(handle, uplo, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnZlauum_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, Ptr{Cint}),
                 handle, uplo, n, A, lda, lwork)
end

function cusolverDnSlauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    @check @runtime_ccall((:cusolverDnSlauum, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, work, lwork, devInfo)
end

function cusolverDnDlauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    @check @runtime_ccall((:cusolverDnDlauum, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, work, lwork, devInfo)
end

function cusolverDnClauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    @check @runtime_ccall((:cusolverDnClauum, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, work, lwork, devInfo)
end

function cusolverDnZlauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    @check @runtime_ccall((:cusolverDnZlauum, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, work, lwork, devInfo)
end

function cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    @check @runtime_ccall((:cusolverDnSgetrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, Ptr{Cint}),
                 handle, m, n, A, lda, Lwork)
end

function cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    @check @runtime_ccall((:cusolverDnDgetrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, Ptr{Cint}),
                 handle, m, n, A, lda, Lwork)
end

function cusolverDnCgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    @check @runtime_ccall((:cusolverDnCgetrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint, Ptr{Cint}),
                 handle, m, n, A, lda, Lwork)
end

function cusolverDnZgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    @check @runtime_ccall((:cusolverDnZgetrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                 handle, m, n, A, lda, Lwork)
end

function cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    @check @runtime_ccall((:cusolverDnSgetrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, A, lda, Workspace, devIpiv, devInfo)
end

function cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    @check @runtime_ccall((:cusolverDnDgetrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, A, lda, Workspace, devIpiv, devInfo)
end

function cusolverDnCgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    @check @runtime_ccall((:cusolverDnCgetrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, A, lda, Workspace, devIpiv, devInfo)
end

function cusolverDnZgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    @check @runtime_ccall((:cusolverDnZgetrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, A, lda, Workspace, devIpiv, devInfo)
end

function cusolverDnSlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    @check @runtime_ccall((:cusolverDnSlaswp, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, CuPtr{Cfloat}, Cint, Cint, Cint, CuPtr{Cint},
                  Cint),
                 handle, n, A, lda, k1, k2, devIpiv, incx)
end

function cusolverDnDlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    @check @runtime_ccall((:cusolverDnDlaswp, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, CuPtr{Cdouble}, Cint, Cint, Cint, CuPtr{Cint},
                  Cint),
                 handle, n, A, lda, k1, k2, devIpiv, incx)
end

function cusolverDnClaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    @check @runtime_ccall((:cusolverDnClaswp, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, CuPtr{cuComplex}, Cint, Cint, Cint,
                  CuPtr{Cint}, Cint),
                 handle, n, A, lda, k1, k2, devIpiv, incx)
end

function cusolverDnZlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    @check @runtime_ccall((:cusolverDnZlaswp, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, CuPtr{cuDoubleComplex}, Cint, Cint, Cint,
                  CuPtr{Cint}, Cint),
                 handle, n, A, lda, k1, k2, devIpiv, incx)
end

function cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    @check @runtime_ccall((:cusolverDnSgetrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasOperation_t, Cint, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
end

function cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    @check @runtime_ccall((:cusolverDnDgetrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasOperation_t, Cint, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
end

function cusolverDnCgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    @check @runtime_ccall((:cusolverDnCgetrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasOperation_t, Cint, Cint, CuPtr{cuComplex},
                  Cint, CuPtr{Cint}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
end

function cusolverDnZgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    @check @runtime_ccall((:cusolverDnZgetrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasOperation_t, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{Cint}),
                 handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
end

function cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnSgeqrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, Ptr{Cint}),
                 handle, m, n, A, lda, lwork)
end

function cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnDgeqrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, Ptr{Cint}),
                 handle, m, n, A, lda, lwork)
end

function cusolverDnCgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnCgeqrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint, Ptr{Cint}),
                 handle, m, n, A, lda, lwork)
end

function cusolverDnZgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnZgeqrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                 handle, m, n, A, lda, lwork)
end

function cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    @check @runtime_ccall((:cusolverDnSgeqrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
end

function cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    @check @runtime_ccall((:cusolverDnDgeqrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
end

function cusolverDnCgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    @check @runtime_ccall((:cusolverDnCgeqrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
end

function cusolverDnZgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    @check @runtime_ccall((:cusolverDnZgeqrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
end

function cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    @check @runtime_ccall((:cusolverDnSorgqr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Ptr{Cint}),
                 handle, m, n, k, A, lda, tau, lwork)
end

function cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    @check @runtime_ccall((:cusolverDnDorgqr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Ptr{Cint}),
                 handle, m, n, k, A, lda, tau, lwork)
end

function cusolverDnCungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    @check @runtime_ccall((:cusolverDnCungqr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Ptr{Cint}),
                 handle, m, n, k, A, lda, tau, lwork)
end

function cusolverDnZungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    @check @runtime_ccall((:cusolverDnZungqr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Ptr{Cint}),
                 handle, m, n, k, A, lda, tau, lwork)
end

function cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    @check @runtime_ccall((:cusolverDnSorgqr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, m, n, k, A, lda, tau, work, lwork, info)
end

function cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    @check @runtime_ccall((:cusolverDnDorgqr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, m, n, k, A, lda, tau, work, lwork, info)
end

function cusolverDnCungqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    @check @runtime_ccall((:cusolverDnCungqr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, m, n, k, A, lda, tau, work, lwork, info)
end

function cusolverDnZungqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    @check @runtime_ccall((:cusolverDnZungqr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, m, n, k, A, lda, tau, work, lwork, info)
end

function cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc,
                                     lwork)
    @check @runtime_ccall((:cusolverDnSormqr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                  Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, Ptr{Cint}),
                 handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
end

function cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc,
                                     lwork)
    @check @runtime_ccall((:cusolverDnDormqr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                  Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                  Ptr{Cint}),
                 handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
end

function cusolverDnCunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc,
                                     lwork)
    @check @runtime_ccall((:cusolverDnCunmqr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                  Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, CuPtr{cuComplex}, Cint,
                  Ptr{Cint}),
                 handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
end

function cusolverDnZunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc,
                                     lwork)
    @check @runtime_ccall((:cusolverDnZunmqr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                  Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                 handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
end

function cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork,
                          devInfo)
    @check @runtime_ccall((:cusolverDnSormqr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                  Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
end

function cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork,
                          devInfo)
    @check @runtime_ccall((:cusolverDnDormqr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                  Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
end

function cusolverDnCunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork,
                          devInfo)
    @check @runtime_ccall((:cusolverDnCunmqr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                  Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
end

function cusolverDnZunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork,
                          devInfo)
    @check @runtime_ccall((:cusolverDnZunmqr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                  Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
end

function cusolverDnSsytrf_bufferSize(handle, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnSsytrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, CuPtr{Cfloat}, Cint, Ptr{Cint}),
                 handle, n, A, lda, lwork)
end

function cusolverDnDsytrf_bufferSize(handle, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnDsytrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, CuPtr{Cdouble}, Cint, Ptr{Cint}),
                 handle, n, A, lda, lwork)
end

function cusolverDnCsytrf_bufferSize(handle, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnCsytrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, CuPtr{cuComplex}, Cint, Ptr{Cint}),
                 handle, n, A, lda, lwork)
end

function cusolverDnZsytrf_bufferSize(handle, n, A, lda, lwork)
    @check @runtime_ccall((:cusolverDnZsytrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                 handle, n, A, lda, lwork)
end

function cusolverDnSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    @check @runtime_ccall((:cusolverDnSsytrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

function cusolverDnDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    @check @runtime_ccall((:cusolverDnDsytrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

function cusolverDnCsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    @check @runtime_ccall((:cusolverDnCsytrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cint}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

function cusolverDnZsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    @check @runtime_ccall((:cusolverDnZsytrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

function cusolverDnSsytrs_bufferSize(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
    @check @runtime_ccall((:cusolverDnSsytrs_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}, CuPtr{Cfloat}, Cint, Ptr{Cint}),
                 handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
end

function cusolverDnDsytrs_bufferSize(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
    @check @runtime_ccall((:cusolverDnDsytrs_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}, CuPtr{Cdouble}, Cint, Ptr{Cint}),
                 handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
end

function cusolverDnCsytrs_bufferSize(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
    @check @runtime_ccall((:cusolverDnCsytrs_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{cuComplex},
                  Cint, CuPtr{Cint}, CuPtr{cuComplex}, Cint, Ptr{Cint}),
                 handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
end

function cusolverDnZsytrs_bufferSize(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
    @check @runtime_ccall((:cusolverDnZsytrs_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
end

function cusolverDnSsytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
    @check @runtime_ccall((:cusolverDnSsytrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
end

function cusolverDnDsytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
    @check @runtime_ccall((:cusolverDnDsytrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
end

function cusolverDnCsytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
    @check @runtime_ccall((:cusolverDnCsytrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{cuComplex},
                  Cint, CuPtr{Cint}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cint}),
                 handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
end

function cusolverDnZsytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
    @check @runtime_ccall((:cusolverDnZsytrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
end

function cusolverDnSsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    @check @runtime_ccall((:cusolverDnSsytri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}, Ptr{Cint}),
                 handle, uplo, n, A, lda, ipiv, lwork)
end

function cusolverDnDsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    @check @runtime_ccall((:cusolverDnDsytri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}, Ptr{Cint}),
                 handle, uplo, n, A, lda, ipiv, lwork)
end

function cusolverDnCsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    @check @runtime_ccall((:cusolverDnCsytri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cint}, Ptr{Cint}),
                 handle, uplo, n, A, lda, ipiv, lwork)
end

function cusolverDnZsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    @check @runtime_ccall((:cusolverDnZsytri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cint}, Ptr{Cint}),
                 handle, uplo, n, A, lda, ipiv, lwork)
end

function cusolverDnSsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    @check @runtime_ccall((:cusolverDnSsytri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

function cusolverDnDsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    @check @runtime_ccall((:cusolverDnDsytri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

function cusolverDnCsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    @check @runtime_ccall((:cusolverDnCsytri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cint}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

function cusolverDnZsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    @check @runtime_ccall((:cusolverDnZsytri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

function cusolverDnSgebrd_bufferSize(handle, m, n, Lwork)
    @check @runtime_ccall((:cusolverDnSgebrd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                 handle, m, n, Lwork)
end

function cusolverDnDgebrd_bufferSize(handle, m, n, Lwork)
    @check @runtime_ccall((:cusolverDnDgebrd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                 handle, m, n, Lwork)
end

function cusolverDnCgebrd_bufferSize(handle, m, n, Lwork)
    @check @runtime_ccall((:cusolverDnCgebrd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                 handle, m, n, Lwork)
end

function cusolverDnZgebrd_bufferSize(handle, m, n, Lwork)
    @check @runtime_ccall((:cusolverDnZgebrd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                 handle, m, n, Lwork)
end

function cusolverDnSgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
    @check @runtime_ccall((:cusolverDnSgebrd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                  CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}),
                 handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
end

function cusolverDnDgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
    @check @runtime_ccall((:cusolverDnDgebrd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                  CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}),
                 handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
end

function cusolverDnCgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
    @check @runtime_ccall((:cusolverDnCgebrd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint, CuPtr{Cfloat},
                  CuPtr{Cfloat}, CuPtr{cuComplex}, CuPtr{cuComplex}, CuPtr{cuComplex},
                  Cint, CuPtr{Cint}),
                 handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
end

function cusolverDnZgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
    @check @runtime_ccall((:cusolverDnZgebrd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
end

function cusolverDnSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    @check @runtime_ccall((:cusolverDnSorgbr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint, CuPtr{Cfloat},
                  Cint, CuPtr{Cfloat}, Ptr{Cint}),
                 handle, side, m, n, k, A, lda, tau, lwork)
end

function cusolverDnDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    @check @runtime_ccall((:cusolverDnDorgbr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint, CuPtr{Cdouble},
                  Cint, CuPtr{Cdouble}, Ptr{Cint}),
                 handle, side, m, n, k, A, lda, tau, lwork)
end

function cusolverDnCungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    @check @runtime_ccall((:cusolverDnCungbr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Ptr{Cint}),
                 handle, side, m, n, k, A, lda, tau, lwork)
end

function cusolverDnZungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    @check @runtime_ccall((:cusolverDnZungbr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Ptr{Cint}),
                 handle, side, m, n, k, A, lda, tau, lwork)
end

function cusolverDnSorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    @check @runtime_ccall((:cusolverDnSorgbr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint, CuPtr{Cfloat},
                  Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, side, m, n, k, A, lda, tau, work, lwork, info)
end

function cusolverDnDorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    @check @runtime_ccall((:cusolverDnDorgbr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint, CuPtr{Cdouble},
                  Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, side, m, n, k, A, lda, tau, work, lwork, info)
end

function cusolverDnCungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    @check @runtime_ccall((:cusolverDnCungbr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, CuPtr{cuComplex}, Cint,
                  CuPtr{Cint}),
                 handle, side, m, n, k, A, lda, tau, work, lwork, info)
end

function cusolverDnZungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    @check @runtime_ccall((:cusolverDnZungbr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, side, m, n, k, A, lda, tau, work, lwork, info)
end

function cusolverDnSsytrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    @check @runtime_ccall((:cusolverDnSsytrd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, Ptr{Cint}),
                 handle, uplo, n, A, lda, d, e, tau, lwork)
end

function cusolverDnDsytrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    @check @runtime_ccall((:cusolverDnDsytrd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, Ptr{Cint}),
                 handle, uplo, n, A, lda, d, e, tau, lwork)
end

function cusolverDnChetrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    @check @runtime_ccall((:cusolverDnChetrd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{cuComplex}, Ptr{Cint}),
                 handle, uplo, n, A, lda, d, e, tau, lwork)
end

function cusolverDnZhetrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    @check @runtime_ccall((:cusolverDnZhetrd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Ptr{Cint}),
                 handle, uplo, n, A, lda, d, e, tau, lwork)
end

function cusolverDnSsytrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    @check @runtime_ccall((:cusolverDnSsytrd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}),
                 handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
end

function cusolverDnDsytrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    @check @runtime_ccall((:cusolverDnDsytrd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}),
                 handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
end

function cusolverDnChetrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    @check @runtime_ccall((:cusolverDnChetrd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{cuComplex}, CuPtr{cuComplex}, Cint,
                  CuPtr{Cint}),
                 handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
end

function cusolverDnZhetrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    @check @runtime_ccall((:cusolverDnZhetrd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
end

function cusolverDnSorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    @check @runtime_ccall((:cusolverDnSorgtr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Ptr{Cint}),
                 handle, uplo, n, A, lda, tau, lwork)
end

function cusolverDnDorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    @check @runtime_ccall((:cusolverDnDorgtr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Ptr{Cint}),
                 handle, uplo, n, A, lda, tau, lwork)
end

function cusolverDnCungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    @check @runtime_ccall((:cusolverDnCungtr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Ptr{Cint}),
                 handle, uplo, n, A, lda, tau, lwork)
end

function cusolverDnZungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    @check @runtime_ccall((:cusolverDnZungtr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{cuDoubleComplex}, Ptr{Cint}),
                 handle, uplo, n, A, lda, tau, lwork)
end

function cusolverDnSorgtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    @check @runtime_ccall((:cusolverDnSorgtr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, tau, work, lwork, info)
end

function cusolverDnDorgtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    @check @runtime_ccall((:cusolverDnDorgtr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, tau, work, lwork, info)
end

function cusolverDnCungtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    @check @runtime_ccall((:cusolverDnCungtr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, tau, work, lwork, info)
end

function cusolverDnZungtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    @check @runtime_ccall((:cusolverDnZungtr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, tau, work, lwork, info)
end

function cusolverDnSormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                     lwork)
    @check @runtime_ccall((:cusolverDnSormtr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                  cublasOperation_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint, Ptr{Cint}),
                 handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
end

function cusolverDnDormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                     lwork)
    @check @runtime_ccall((:cusolverDnDormtr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                  cublasOperation_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint, Ptr{Cint}),
                 handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
end

function cusolverDnCunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                     lwork)
    @check @runtime_ccall((:cusolverDnCunmtr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                  cublasOperation_t, Cint, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint, Ptr{Cint}),
                 handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
end

function cusolverDnZunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                     lwork)
    @check @runtime_ccall((:cusolverDnZunmtr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                  cublasOperation_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                 handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
end

function cusolverDnSormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work,
                          lwork, info)
    @check @runtime_ccall((:cusolverDnSormtr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                  cublasOperation_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
end

function cusolverDnDormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work,
                          lwork, info)
    @check @runtime_ccall((:cusolverDnDormtr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                  cublasOperation_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
end

function cusolverDnCunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work,
                          lwork, info)
    @check @runtime_ccall((:cusolverDnCunmtr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                  cublasOperation_t, Cint, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
end

function cusolverDnZunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work,
                          lwork, info)
    @check @runtime_ccall((:cusolverDnZunmtr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                  cublasOperation_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
end

function cusolverDnSgesvd_bufferSize(handle, m, n, lwork)
    @check @runtime_ccall((:cusolverDnSgesvd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                 handle, m, n, lwork)
end

function cusolverDnDgesvd_bufferSize(handle, m, n, lwork)
    @check @runtime_ccall((:cusolverDnDgesvd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                 handle, m, n, lwork)
end

function cusolverDnCgesvd_bufferSize(handle, m, n, lwork)
    @check @runtime_ccall((:cusolverDnCgesvd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                 handle, m, n, lwork)
end

function cusolverDnZgesvd_bufferSize(handle, m, n, lwork)
    @check @runtime_ccall((:cusolverDnZgesvd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                 handle, m, n, lwork)
end

function cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work,
                          lwork, rwork, info)
    @check @runtime_ccall((:cusolverDnSgesvd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, UInt8, UInt8, Cint, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                  Cint, CuPtr{Cfloat}, CuPtr{Cint}),
                 handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork,
                 rwork, info)
end

function cusolverDnDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work,
                          lwork, rwork, info)
    @check @runtime_ccall((:cusolverDnDgesvd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, UInt8, UInt8, Cint, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cint}),
                 handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork,
                 rwork, info)
end

function cusolverDnCgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work,
                          lwork, rwork, info)
    @check @runtime_ccall((:cusolverDnCgesvd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, UInt8, UInt8, Cint, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{Cint}),
                 handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork,
                 rwork, info)
end

function cusolverDnZgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work,
                          lwork, rwork, info)
    @check @runtime_ccall((:cusolverDnZgesvd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, UInt8, UInt8, Cint, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cint}),
                 handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork,
                 rwork, info)
end

function cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    @check @runtime_ccall((:cusolverDnSsyevd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Ptr{Cint}),
                 handle, jobz, uplo, n, A, lda, W, lwork)
end

function cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    @check @runtime_ccall((:cusolverDnDsyevd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Ptr{Cint}),
                 handle, jobz, uplo, n, A, lda, W, lwork)
end

function cusolverDnCheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    @check @runtime_ccall((:cusolverDnCheevd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, Ptr{Cint}),
                 handle, jobz, uplo, n, A, lda, W, lwork)
end

function cusolverDnZheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    @check @runtime_ccall((:cusolverDnZheevd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}),
                 handle, jobz, uplo, n, A, lda, W, lwork)
end

function cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    @check @runtime_ccall((:cusolverDnSsyevd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info)
end

function cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    @check @runtime_ccall((:cusolverDnDsyevd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info)
end

function cusolverDnCheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    @check @runtime_ccall((:cusolverDnCheevd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                  CuPtr{Cint}),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info)
end

function cusolverDnZheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    @check @runtime_ccall((:cusolverDnZheevd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cint}),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info)
end

function cusolverDnSsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                      meig, W, lwork)
    @check @runtime_ccall((:cusolverDnSsyevdx_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                  cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, Cfloat, Cfloat, Cint, Cint,
                  Ptr{Cint}, CuPtr{Cfloat}, Ptr{Cint}),
                 handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
end

function cusolverDnDsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                      meig, W, lwork)
    @check @runtime_ccall((:cusolverDnDsyevdx_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                  cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, Cdouble, Cdouble, Cint,
                  Cint, Ptr{Cint}, CuPtr{Cdouble}, Ptr{Cint}),
                 handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
end

function cusolverDnCheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                      meig, W, lwork)
    @check @runtime_ccall((:cusolverDnCheevdx_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                  cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, Cfloat, Cfloat, Cint,
                  Cint, Ptr{Cint}, CuPtr{Cfloat}, Ptr{Cint}),
                 handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
end

function cusolverDnZheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                      meig, W, lwork)
    @check @runtime_ccall((:cusolverDnZheevdx_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                  cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint, Cdouble, Cdouble,
                  Cint, Cint, Ptr{Cint}, CuPtr{Cdouble}, Ptr{Cint}),
                 handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
end

function cusolverDnSsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W,
                           work, lwork, info)
    @check @runtime_ccall((:cusolverDnSsyevdx, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                  cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, Cfloat, Cfloat, Cint, Cint,
                  Ptr{Cint}, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work,
                 lwork, info)
end

function cusolverDnDsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W,
                           work, lwork, info)
    @check @runtime_ccall((:cusolverDnDsyevdx, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                  cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, Cdouble, Cdouble, Cint,
                  Cint, Ptr{Cint}, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work,
                 lwork, info)
end

function cusolverDnCheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W,
                           work, lwork, info)
    @check @runtime_ccall((:cusolverDnCheevdx, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                  cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, Cfloat, Cfloat, Cint,
                  Cint, Ptr{Cint}, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work,
                 lwork, info)
end

function cusolverDnZheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W,
                           work, lwork, info)
    @check @runtime_ccall((:cusolverDnZheevdx, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                  cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint, Cdouble, Cdouble,
                  Cint, Cint, Ptr{Cint}, CuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{Cint}),
                 handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work,
                 lwork, info)
end

function cusolverDnSsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb,
                                      vl, vu, il, iu, meig, W, lwork)
    @check @runtime_ccall((:cusolverDnSsygvdx_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint, Cfloat, Cfloat, Cint, Cint, Ptr{Cint},
                  CuPtr{Cfloat}, Ptr{Cint}),
                 handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig,
                 W, lwork)
end

function cusolverDnDsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb,
                                      vl, vu, il, iu, meig, W, lwork)
    @check @runtime_ccall((:cusolverDnDsygvdx_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, Cdouble, Cdouble, Cint, Cint, Ptr{Cint},
                  CuPtr{Cdouble}, Ptr{Cint}),
                 handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig,
                 W, lwork)
end

function cusolverDnChegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb,
                                      vl, vu, il, iu, meig, W, lwork)
    @check @runtime_ccall((:cusolverDnChegvdx_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, Cfloat, Cfloat, Cint, Cint, Ptr{Cint},
                  CuPtr{Cfloat}, Ptr{Cint}),
                 handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig,
                 W, lwork)
end

function cusolverDnZhegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb,
                                      vl, vu, il, iu, meig, W, lwork)
    @check @runtime_ccall((:cusolverDnZhegvdx_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, Cdouble, Cdouble, Cint, Cint, Ptr{Cint},
                  CuPtr{Cdouble}, Ptr{Cint}),
                 handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig,
                 W, lwork)
end

function cusolverDnSsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il,
                           iu, meig, W, work, lwork, info)
    @check @runtime_ccall((:cusolverDnSsygvdx, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint, Cfloat, Cfloat, Cint, Cint, Ptr{Cint},
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig,
                 W, work, lwork, info)
end

function cusolverDnDsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il,
                           iu, meig, W, work, lwork, info)
    @check @runtime_ccall((:cusolverDnDsygvdx, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, Cdouble, Cdouble, Cint, Cint, Ptr{Cint},
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig,
                 W, work, lwork, info)
end

function cusolverDnChegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il,
                           iu, meig, W, work, lwork, info)
    @check @runtime_ccall((:cusolverDnChegvdx, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, Cfloat, Cfloat, Cint, Cint, Ptr{Cint},
                  CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig,
                 W, work, lwork, info)
end

function cusolverDnZhegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il,
                           iu, meig, W, work, lwork, info)
    @check @runtime_ccall((:cusolverDnZhegvdx, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, Cdouble, Cdouble, Cint, Cint, Ptr{Cint},
                  CuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig,
                 W, work, lwork, info)
end

function cusolverDnSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
    @check @runtime_ccall((:cusolverDnSsygvd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Ptr{Cint}),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
end

function cusolverDnDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
    @check @runtime_ccall((:cusolverDnDsygvd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Ptr{Cint}),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
end

function cusolverDnChegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
    @check @runtime_ccall((:cusolverDnChegvd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cfloat}, Ptr{Cint}),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
end

function cusolverDnZhegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
    @check @runtime_ccall((:cusolverDnZhegvd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
end

function cusolverDnSsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
    @check @runtime_ccall((:cusolverDnSsygvd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
end

function cusolverDnDsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
    @check @runtime_ccall((:cusolverDnDsygvd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
end

function cusolverDnChegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
    @check @runtime_ccall((:cusolverDnChegvd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
end

function cusolverDnZhegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
    @check @runtime_ccall((:cusolverDnZhegvd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cint}),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
end

function cusolverDnCreateSyevjInfo(info)
    @check @runtime_ccall((:cusolverDnCreateSyevjInfo, libcusolver), cusolverStatus_t,
                 (Ptr{syevjInfo_t},),
                 info)
end

function cusolverDnDestroySyevjInfo(info)
    @check @runtime_ccall((:cusolverDnDestroySyevjInfo, libcusolver), cusolverStatus_t,
                 (syevjInfo_t,),
                 info)
end

function cusolverDnXsyevjSetTolerance(info, tolerance)
    @check @runtime_ccall((:cusolverDnXsyevjSetTolerance, libcusolver), cusolverStatus_t,
                 (syevjInfo_t, Cdouble),
                 info, tolerance)
end

function cusolverDnXsyevjSetMaxSweeps(info, max_sweeps)
    @check @runtime_ccall((:cusolverDnXsyevjSetMaxSweeps, libcusolver), cusolverStatus_t,
                 (syevjInfo_t, Cint),
                 info, max_sweeps)
end

function cusolverDnXsyevjSetSortEig(info, sort_eig)
    @check @runtime_ccall((:cusolverDnXsyevjSetSortEig, libcusolver), cusolverStatus_t,
                 (syevjInfo_t, Cint),
                 info, sort_eig)
end

function cusolverDnXsyevjGetResidual(handle, info, residual)
    @check @runtime_ccall((:cusolverDnXsyevjGetResidual, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, syevjInfo_t, Ptr{Cdouble}),
                 handle, info, residual)
end

function cusolverDnXsyevjGetSweeps(handle, info, executed_sweeps)
    @check @runtime_ccall((:cusolverDnXsyevjGetSweeps, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, syevjInfo_t, Ptr{Cint}),
                 handle, info, executed_sweeps)
end

function cusolverDnSsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                            params, batchSize)
    @check @runtime_ccall((:cusolverDnSsyevjBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t, Cint),
                 handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
end

function cusolverDnDsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                            params, batchSize)
    @check @runtime_ccall((:cusolverDnDsyevjBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t, Cint),
                 handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
end

function cusolverDnCheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                            params, batchSize)
    @check @runtime_ccall((:cusolverDnCheevjBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t, Cint),
                 handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
end

function cusolverDnZheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                            params, batchSize)
    @check @runtime_ccall((:cusolverDnZheevjBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t,
                  Cint),
                 handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
end

function cusolverDnSsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                 params, batchSize)
    @check @runtime_ccall((:cusolverDnSsyevjBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint},
                  syevjInfo_t, Cint),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
end

function cusolverDnDsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                 params, batchSize)
    @check @runtime_ccall((:cusolverDnDsyevjBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint},
                  syevjInfo_t, Cint),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
end

function cusolverDnCheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                 params, batchSize)
    @check @runtime_ccall((:cusolverDnCheevjBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                  CuPtr{Cint}, syevjInfo_t, Cint),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
end

function cusolverDnZheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                 params, batchSize)
    @check @runtime_ccall((:cusolverDnZheevjBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cint}, syevjInfo_t, Cint),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
end

function cusolverDnSsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params)
    @check @runtime_ccall((:cusolverDnSsyevj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t),
                 handle, jobz, uplo, n, A, lda, W, lwork, params)
end

function cusolverDnDsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params)
    @check @runtime_ccall((:cusolverDnDsyevj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t),
                 handle, jobz, uplo, n, A, lda, W, lwork, params)
end

function cusolverDnCheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params)
    @check @runtime_ccall((:cusolverDnCheevj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t),
                 handle, jobz, uplo, n, A, lda, W, lwork, params)
end

function cusolverDnZheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params)
    @check @runtime_ccall((:cusolverDnZheevj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t),
                 handle, jobz, uplo, n, A, lda, W, lwork, params)
end

function cusolverDnSsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
    @check @runtime_ccall((:cusolverDnSsyevj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint},
                  syevjInfo_t),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
end

function cusolverDnDsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
    @check @runtime_ccall((:cusolverDnDsyevj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint},
                  syevjInfo_t),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
end

function cusolverDnCheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
    @check @runtime_ccall((:cusolverDnCheevj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                  CuPtr{Cint}, syevjInfo_t),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
end

function cusolverDnZheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
    @check @runtime_ccall((:cusolverDnZheevj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cint}, syevjInfo_t),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
end

function cusolverDnSsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W,
                                     lwork, params)
    @check @runtime_ccall((:cusolverDnSsygvj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
end

function cusolverDnDsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W,
                                     lwork, params)
    @check @runtime_ccall((:cusolverDnDsygvj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
end

function cusolverDnChegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W,
                                     lwork, params)
    @check @runtime_ccall((:cusolverDnChegvj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
end

function cusolverDnZhegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W,
                                     lwork, params)
    @check @runtime_ccall((:cusolverDnZhegvj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
end

function cusolverDnSsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork,
                          info, params)
    @check @runtime_ccall((:cusolverDnSsygvj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}, syevjInfo_t),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info,
                 params)
end

function cusolverDnDsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork,
                          info, params)
    @check @runtime_ccall((:cusolverDnDsygvj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}, syevjInfo_t),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info,
                 params)
end

function cusolverDnChegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork,
                          info, params)
    @check @runtime_ccall((:cusolverDnChegvj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{Cint}, syevjInfo_t),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info,
                 params)
end

function cusolverDnZhegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork,
                          info, params)
    @check @runtime_ccall((:cusolverDnZhegvj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cint}, syevjInfo_t),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info,
                 params)
end

function cusolverDnCreateGesvdjInfo(info)
    @check @runtime_ccall((:cusolverDnCreateGesvdjInfo, libcusolver), cusolverStatus_t,
                 (Ptr{gesvdjInfo_t},),
                 info)
end

function cusolverDnDestroyGesvdjInfo(info)
    @check @runtime_ccall((:cusolverDnDestroyGesvdjInfo, libcusolver), cusolverStatus_t,
                 (gesvdjInfo_t,),
                 info)
end

function cusolverDnXgesvdjSetTolerance(info, tolerance)
    @check @runtime_ccall((:cusolverDnXgesvdjSetTolerance, libcusolver), cusolverStatus_t,
                 (gesvdjInfo_t, Cdouble),
                 info, tolerance)
end

function cusolverDnXgesvdjSetMaxSweeps(info, max_sweeps)
    @check @runtime_ccall((:cusolverDnXgesvdjSetMaxSweeps, libcusolver), cusolverStatus_t,
                 (gesvdjInfo_t, Cint),
                 info, max_sweeps)
end

function cusolverDnXgesvdjSetSortEig(info, sort_svd)
    @check @runtime_ccall((:cusolverDnXgesvdjSetSortEig, libcusolver), cusolverStatus_t,
                 (gesvdjInfo_t, Cint),
                 info, sort_svd)
end

function cusolverDnXgesvdjGetResidual(handle, info, residual)
    @check @runtime_ccall((:cusolverDnXgesvdjGetResidual, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, gesvdjInfo_t, Ptr{Cdouble}),
                 handle, info, residual)
end

function cusolverDnXgesvdjGetSweeps(handle, info, executed_sweeps)
    @check @runtime_ccall((:cusolverDnXgesvdjGetSweeps, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, gesvdjInfo_t, Ptr{Cint}),
                 handle, info, executed_sweeps)
end

function cusolverDnSgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                             lwork, params, batchSize)
    @check @runtime_ccall((:cusolverDnSgesvdjBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, Ptr{Cint},
                  gesvdjInfo_t, Cint),
                 handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
end

function cusolverDnDgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                             lwork, params, batchSize)
    @check @runtime_ccall((:cusolverDnDgesvdjBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, Ptr{Cint},
                  gesvdjInfo_t, Cint),
                 handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
end

function cusolverDnCgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                             lwork, params, batchSize)
    @check @runtime_ccall((:cusolverDnCgesvdjBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{cuComplex},
                  Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  Ptr{Cint}, gesvdjInfo_t, Cint),
                 handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
end

function cusolverDnZgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                             lwork, params, batchSize)
    @check @runtime_ccall((:cusolverDnZgesvdjBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}, gesvdjInfo_t, Cint),
                 handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
end

function cusolverDnSgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work,
                                  lwork, info, params, batchSize)
    @check @runtime_ccall((:cusolverDnSgesvdjBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                  Cint, CuPtr{Cint}, gesvdjInfo_t, Cint),
                 handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params,
                 batchSize)
end

function cusolverDnDgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work,
                                  lwork, info, params, batchSize)
    @check @runtime_ccall((:cusolverDnDgesvdjBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cint}, gesvdjInfo_t, Cint),
                 handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params,
                 batchSize)
end

function cusolverDnCgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work,
                                  lwork, info, params, batchSize)
    @check @runtime_ccall((:cusolverDnCgesvdjBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{cuComplex},
                  Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cint}, gesvdjInfo_t, Cint),
                 handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params,
                 batchSize)
end

function cusolverDnZgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work,
                                  lwork, info, params, batchSize)
    @check @runtime_ccall((:cusolverDnZgesvdjBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{Cint}, gesvdjInfo_t, Cint),
                 handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params,
                 batchSize)
end

function cusolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                      lwork, params)
    @check @runtime_ccall((:cusolverDnSgesvdj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint, CuPtr{Cfloat},
                  Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, Ptr{Cint},
                  gesvdjInfo_t),
                 handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
end

function cusolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                      lwork, params)
    @check @runtime_ccall((:cusolverDnDgesvdj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint, CuPtr{Cdouble},
                  Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  Ptr{Cint}, gesvdjInfo_t),
                 handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
end

function cusolverDnCgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                      lwork, params)
    @check @runtime_ccall((:cusolverDnCgesvdj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, Ptr{Cint}, gesvdjInfo_t),
                 handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
end

function cusolverDnZgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                      lwork, params)
    @check @runtime_ccall((:cusolverDnZgesvdj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}, gesvdjInfo_t),
                 handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
end

function cusolverDnSgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work,
                           lwork, info, params)
    @check @runtime_ccall((:cusolverDnSgesvdj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint, CuPtr{Cfloat},
                  Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cint}, gesvdjInfo_t),
                 handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                 params)
end

function cusolverDnDgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work,
                           lwork, info, params)
    @check @runtime_ccall((:cusolverDnDgesvdj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint, CuPtr{Cdouble},
                  Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cint}, gesvdjInfo_t),
                 handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                 params)
end

function cusolverDnCgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work,
                           lwork, info, params)
    @check @runtime_ccall((:cusolverDnCgesvdj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint, CuPtr{Cint}, gesvdjInfo_t),
                 handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                 params)
end

function cusolverDnZgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work,
                           lwork, info, params)
    @check @runtime_ccall((:cusolverDnZgesvdj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{Cint}, gesvdjInfo_t),
                 handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                 params)
end

function cusolverDnSgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A, lda,
                                                    strideA, d_S, strideS, d_U, ldu,
                                                    strideU, d_V, ldv, strideV, lwork,
                                                    batchSize)
    @check @runtime_ccall((:cusolverDnSgesvdaStridedBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint, CuPtr{Cfloat},
                  Cint, Clonglong, CuPtr{Cfloat}, Clonglong, CuPtr{Cfloat}, Cint,
                  Clonglong, CuPtr{Cfloat}, Cint, Clonglong, Ptr{Cint}, Cint),
                 handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu,
                 strideU, d_V, ldv, strideV, lwork, batchSize)
end

function cusolverDnDgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A, lda,
                                                    strideA, d_S, strideS, d_U, ldu,
                                                    strideU, d_V, ldv, strideV, lwork,
                                                    batchSize)
    @check @runtime_ccall((:cusolverDnDgesvdaStridedBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint, CuPtr{Cdouble},
                  Cint, Clonglong, CuPtr{Cdouble}, Clonglong, CuPtr{Cdouble}, Cint,
                  Clonglong, CuPtr{Cdouble}, Cint, Clonglong, Ptr{Cint}, Cint),
                 handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu,
                 strideU, d_V, ldv, strideV, lwork, batchSize)
end

function cusolverDnCgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A, lda,
                                                    strideA, d_S, strideS, d_U, ldu,
                                                    strideU, d_V, ldv, strideV, lwork,
                                                    batchSize)
    @check @runtime_ccall((:cusolverDnCgesvdaStridedBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                  CuPtr{cuComplex}, Cint, Clonglong, CuPtr{Cfloat}, Clonglong,
                  CuPtr{cuComplex}, Cint, Clonglong, CuPtr{cuComplex}, Cint, Clonglong,
                  Ptr{Cint}, Cint),
                 handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu,
                 strideU, d_V, ldv, strideV, lwork, batchSize)
end

function cusolverDnZgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A, lda,
                                                    strideA, d_S, strideS, d_U, ldu,
                                                    strideU, d_V, ldv, strideV, lwork,
                                                    batchSize)
    @check @runtime_ccall((:cusolverDnZgesvdaStridedBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, Clonglong, CuPtr{Cdouble}, Clonglong,
                  CuPtr{cuDoubleComplex}, Cint, Clonglong, CuPtr{cuDoubleComplex}, Cint,
                  Clonglong, Ptr{Cint}, Cint),
                 handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu,
                 strideU, d_V, ldv, strideV, lwork, batchSize)
end

function cusolverDnSgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda, strideA, d_S,
                                         strideS, d_U, ldu, strideU, d_V, ldv, strideV,
                                         d_work, lwork, d_info, h_R_nrmF, batchSize)
    @check @runtime_ccall((:cusolverDnSgesvdaStridedBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint, CuPtr{Cfloat},
                  Cint, Clonglong, CuPtr{Cfloat}, Clonglong, CuPtr{Cfloat}, Cint,
                  Clonglong, CuPtr{Cfloat}, Cint, Clonglong, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}, Ptr{Cdouble}, Cint),
                 handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu,
                 strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)
end

function cusolverDnDgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda, strideA, d_S,
                                         strideS, d_U, ldu, strideU, d_V, ldv, strideV,
                                         d_work, lwork, d_info, h_R_nrmF, batchSize)
    @check @runtime_ccall((:cusolverDnDgesvdaStridedBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint, CuPtr{Cdouble},
                  Cint, Clonglong, CuPtr{Cdouble}, Clonglong, CuPtr{Cdouble}, Cint,
                  Clonglong, CuPtr{Cdouble}, Cint, Clonglong, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}, Ptr{Cdouble}, Cint),
                 handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu,
                 strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)
end

function cusolverDnCgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda, strideA, d_S,
                                         strideS, d_U, ldu, strideU, d_V, ldv, strideV,
                                         d_work, lwork, d_info, h_R_nrmF, batchSize)
    @check @runtime_ccall((:cusolverDnCgesvdaStridedBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                  CuPtr{cuComplex}, Cint, Clonglong, CuPtr{Cfloat}, Clonglong,
                  CuPtr{cuComplex}, Cint, Clonglong, CuPtr{cuComplex}, Cint, Clonglong,
                  CuPtr{cuComplex}, Cint, CuPtr{Cint}, Ptr{Cdouble}, Cint),
                 handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu,
                 strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)
end

function cusolverDnZgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda, strideA, d_S,
                                         strideS, d_U, ldu, strideU, d_V, ldv, strideV,
                                         d_work, lwork, d_info, h_R_nrmF, batchSize)
    @check @runtime_ccall((:cusolverDnZgesvdaStridedBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, Clonglong, CuPtr{Cdouble}, Clonglong,
                  CuPtr{cuDoubleComplex}, Cint, Clonglong, CuPtr{cuDoubleComplex}, Cint,
                  Clonglong, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}, Ptr{Cdouble}, Cint),
                 handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu,
                 strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)
end
# Julia wrapper for header: cusolverSp.h
# Automatically generated using Clang.jl


function cusolverSpCreate(handle)
    @check @runtime_ccall((:cusolverSpCreate, libcusolver), cusolverStatus_t,
                 (Ptr{cusolverSpHandle_t},),
                 handle)
end

function cusolverSpDestroy(handle)
    @check @runtime_ccall((:cusolverSpDestroy, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t,),
                 handle)
end

function cusolverSpSetStream(handle, streamId)
    @check @runtime_ccall((:cusolverSpSetStream, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, CUstream),
                 handle, streamId)
end

function cusolverSpGetStream(handle, streamId)
    @check @runtime_ccall((:cusolverSpGetStream, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Ptr{CUstream}),
                 handle, streamId)
end

function cusolverSpXcsrissymHost(handle, m, nnzA, descrA, csrRowPtrA, csrEndPtrA,
                                 csrColIndA, issym)
    @check @runtime_ccall((:cusolverSpXcsrissymHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                 handle, m, nnzA, descrA, csrRowPtrA, csrEndPtrA, csrColIndA, issym)
end

function cusolverSpScsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA,
                                 b, tol, reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpScsrlsvluHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cint}),
                 handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder,
                 x, singularity)
end

function cusolverSpDcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA,
                                 b, tol, reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpDcsrlsvluHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cdouble, Cint, Ptr{Cdouble},
                  Ptr{Cint}),
                 handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder,
                 x, singularity)
end

function cusolverSpCcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA,
                                 b, tol, reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpCcsrlsvluHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                  Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cfloat, Cint, Ptr{cuComplex},
                  Ptr{Cint}),
                 handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder,
                 x, singularity)
end

function cusolverSpZcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA,
                                 b, tol, reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpZcsrlsvluHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                  Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex},
                  Cdouble, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}),
                 handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder,
                 x, singularity)
end

function cusolverSpScsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol,
                             reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpScsrlsvqr, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cfloat}, Cfloat, Cint, CuPtr{Cfloat},
                  Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpDcsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol,
                             reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpDcsrlsvqr, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cdouble, Cint, CuPtr{Cdouble},
                  Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpCcsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol,
                             reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpCcsrlsvqr, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuComplex}, Cfloat, Cint,
                  CuPtr{cuComplex}, Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpZcsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol,
                             reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpZcsrlsvqr, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                  Cdouble, Cint, CuPtr{cuDoubleComplex}, Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpScsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                 b, tol, reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpScsrlsvqrHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cint}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder,
                 x, singularity)
end

function cusolverSpDcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                 b, tol, reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpDcsrlsvqrHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cdouble, Cint, Ptr{Cdouble},
                  Ptr{Cint}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder,
                 x, singularity)
end

function cusolverSpCcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                 b, tol, reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpCcsrlsvqrHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                  Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cfloat, Cint, Ptr{cuComplex},
                  Ptr{Cint}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder,
                 x, singularity)
end

function cusolverSpZcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                 b, tol, reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpZcsrlsvqrHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                  Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex},
                  Cdouble, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder,
                 x, singularity)
end

function cusolverSpScsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b,
                                   tol, reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpScsrlsvcholHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpDcsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b,
                                   tol, reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpDcsrlsvcholHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cdouble, Cint, Ptr{Cdouble},
                  Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpCcsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b,
                                   tol, reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpCcsrlsvcholHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                  Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cfloat, Cint, Ptr{cuComplex},
                  Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpZcsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b,
                                   tol, reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpZcsrlsvcholHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                  Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex},
                  Cdouble, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpScsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b,
                               tol, reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpScsrlsvchol, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cfloat}, Cfloat, Cint, CuPtr{Cfloat},
                  Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpDcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b,
                               tol, reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpDcsrlsvchol, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cdouble, Cint, CuPtr{Cdouble},
                  Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpCcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b,
                               tol, reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpCcsrlsvchol, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuComplex}, Cfloat, Cint,
                  CuPtr{cuComplex}, Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpZcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b,
                               tol, reorder, x, singularity)
    @check @runtime_ccall((:cusolverSpZcsrlsvchol, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                  Cdouble, Cint, CuPtr{cuDoubleComplex}, Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpScsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                  csrColIndA, b, tol, rankA, x, p, min_norm)
    @check @runtime_ccall((:cusolverSpScsrlsqvqrHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cfloat, Ptr{Cint}, Ptr{Cfloat},
                  Ptr{Cint}, Ptr{Cfloat}),
                 handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA,
                 x, p, min_norm)
end

function cusolverSpDcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                  csrColIndA, b, tol, rankA, x, p, min_norm)
    @check @runtime_ccall((:cusolverSpDcsrlsqvqrHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cdouble, Ptr{Cint}, Ptr{Cdouble},
                  Ptr{Cint}, Ptr{Cdouble}),
                 handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA,
                 x, p, min_norm)
end

function cusolverSpCcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                  csrColIndA, b, tol, rankA, x, p, min_norm)
    @check @runtime_ccall((:cusolverSpCcsrlsqvqrHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cfloat, Ptr{Cint},
                  Ptr{cuComplex}, Ptr{Cint}, Ptr{Cfloat}),
                 handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA,
                 x, p, min_norm)
end

function cusolverSpZcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                  csrColIndA, b, tol, rankA, x, p, min_norm)
    @check @runtime_ccall((:cusolverSpZcsrlsqvqrHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex},
                  Cdouble, Ptr{Cint}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cdouble}),
                 handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA,
                 x, p, min_norm)
end

function cusolverSpScsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                  mu0, x0, maxite, tol, mu, x)
    @check @runtime_ccall((:cusolverSpScsreigvsiHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                  Ptr{Cint}, Ptr{Cint}, Cfloat, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat},
                  Ptr{Cfloat}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite,
                 tol, mu, x)
end

function cusolverSpDcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                  mu0, x0, maxite, tol, mu, x)
    @check @runtime_ccall((:cusolverSpDcsreigvsiHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                  Ptr{Cint}, Ptr{Cint}, Cdouble, Ptr{Cdouble}, Cint, Cdouble, Ptr{Cdouble},
                  Ptr{Cdouble}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite,
                 tol, mu, x)
end

function cusolverSpCcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                  mu0, x0, maxite, tol, mu, x)
    @check @runtime_ccall((:cusolverSpCcsreigvsiHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                  Ptr{Cint}, Ptr{Cint}, cuComplex, Ptr{cuComplex}, Cint, Cfloat,
                  Ptr{cuComplex}, Ptr{cuComplex}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite,
                 tol, mu, x)
end

function cusolverSpZcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                  mu0, x0, maxite, tol, mu, x)
    @check @runtime_ccall((:cusolverSpZcsreigvsiHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                  Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, cuDoubleComplex,
                  Ptr{cuDoubleComplex}, Cint, Cdouble, Ptr{cuDoubleComplex},
                  Ptr{cuDoubleComplex}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite,
                 tol, mu, x)
end

function cusolverSpScsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0,
                              x0, maxite, eps, mu, x)
    @check @runtime_ccall((:cusolverSpScsreigvsi, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, Cfloat, CuPtr{Cfloat}, Cint, Cfloat,
                  CuPtr{Cfloat}, CuPtr{Cfloat}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite,
                 eps, mu, x)
end

function cusolverSpDcsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0,
                              x0, maxite, eps, mu, x)
    @check @runtime_ccall((:cusolverSpDcsreigvsi, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, Cdouble, CuPtr{Cdouble}, Cint, Cdouble,
                  CuPtr{Cdouble}, CuPtr{Cdouble}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite,
                 eps, mu, x)
end

function cusolverSpCcsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0,
                              x0, maxite, eps, mu, x)
    @check @runtime_ccall((:cusolverSpCcsreigvsi, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, cuComplex, CuPtr{cuComplex}, Cint, Cfloat,
                  CuPtr{cuComplex}, CuPtr{cuComplex}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite,
                 eps, mu, x)
end

function cusolverSpZcsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0,
                              x0, maxite, eps, mu, x)
    @check @runtime_ccall((:cusolverSpZcsreigvsi, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, cuDoubleComplex,
                  CuPtr{cuDoubleComplex}, Cint, Cdouble, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite,
                 eps, mu, x)
end

function cusolverSpScsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                left_bottom_corner, right_upper_corner, num_eigs)
    @check @runtime_ccall((:cusolverSpScsreigsHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                  Ptr{Cint}, Ptr{Cint}, cuComplex, cuComplex, Ptr{Cint}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                 left_bottom_corner, right_upper_corner, num_eigs)
end

function cusolverSpDcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                left_bottom_corner, right_upper_corner, num_eigs)
    @check @runtime_ccall((:cusolverSpDcsreigsHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                  Ptr{Cint}, Ptr{Cint}, cuDoubleComplex, cuDoubleComplex, Ptr{Cint}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                 left_bottom_corner, right_upper_corner, num_eigs)
end

function cusolverSpCcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                left_bottom_corner, right_upper_corner, num_eigs)
    @check @runtime_ccall((:cusolverSpCcsreigsHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                  Ptr{Cint}, Ptr{Cint}, cuComplex, cuComplex, Ptr{Cint}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                 left_bottom_corner, right_upper_corner, num_eigs)
end

function cusolverSpZcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                left_bottom_corner, right_upper_corner, num_eigs)
    @check @runtime_ccall((:cusolverSpZcsreigsHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                  Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, cuDoubleComplex,
                  cuDoubleComplex, Ptr{Cint}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                 left_bottom_corner, right_upper_corner, num_eigs)
end

function cusolverSpXcsrsymrcmHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
    @check @runtime_ccall((:cusolverSpXcsrsymrcmHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                  Ptr{Cint}, Ptr{Cint}),
                 handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
end

function cusolverSpXcsrsymmdqHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
    @check @runtime_ccall((:cusolverSpXcsrsymmdqHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                  Ptr{Cint}, Ptr{Cint}),
                 handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
end

function cusolverSpXcsrsymamdHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
    @check @runtime_ccall((:cusolverSpXcsrsymamdHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                  Ptr{Cint}, Ptr{Cint}),
                 handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
end

function cusolverSpXcsrmetisndHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA,
                                   options, p)
    @check @runtime_ccall((:cusolverSpXcsrmetisndHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                  Ptr{Cint}, Ptr{Int64}, Ptr{Cint}),
                 handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, options, p)
end

function cusolverSpScsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P,
                               numnz)
    @check @runtime_ccall((:cusolverSpScsrzfdHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                 handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
end

function cusolverSpDcsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P,
                               numnz)
    @check @runtime_ccall((:cusolverSpDcsrzfdHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, Ptr{Cint}, Ptr{Cint}),
                 handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
end

function cusolverSpCcsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P,
                               numnz)
    @check @runtime_ccall((:cusolverSpCcsrzfdHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                 handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
end

function cusolverSpZcsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P,
                               numnz)
    @check @runtime_ccall((:cusolverSpZcsrzfdHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                  Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                 handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
end

function cusolverSpXcsrperm_bufferSizeHost(handle, m, n, nnzA, descrA, csrRowPtrA,
                                           csrColIndA, p, q, bufferSizeInBytes)
    @check @runtime_ccall((:cusolverSpXcsrperm_bufferSizeHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Csize_t}),
                 handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q,
                 bufferSizeInBytes)
end

function cusolverSpXcsrpermHost(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q,
                                map, pBuffer)
    @check @runtime_ccall((:cusolverSpXcsrpermHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}),
                 handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q, map, pBuffer)
end

function cusolverSpCreateCsrqrInfo(info)
    @check @runtime_ccall((:cusolverSpCreateCsrqrInfo, libcusolver), cusolverStatus_t,
                 (Ptr{csrqrInfo_t},),
                 info)
end

function cusolverSpDestroyCsrqrInfo(info)
    @check @runtime_ccall((:cusolverSpDestroyCsrqrInfo, libcusolver), cusolverStatus_t,
                 (csrqrInfo_t,),
                 info)
end

function cusolverSpXcsrqrAnalysisBatched(handle, m, n, nnzA, descrA, csrRowPtrA,
                                         csrColIndA, info)
    @check @runtime_ccall((:cusolverSpXcsrqrAnalysisBatched, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cint},
                  CuPtr{Cint}, csrqrInfo_t),
                 handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
end

function cusolverSpScsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal, csrRowPtr,
                                           csrColInd, batchSize, info, internalDataInBytes,
                                           workspaceInBytes)
    @check @runtime_ccall((:cusolverSpScsrqrBufferInfoBatched, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, Cint, csrqrInfo_t, Ptr{Csize_t}, Ptr{Csize_t}),
                 handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info,
                 internalDataInBytes, workspaceInBytes)
end

function cusolverSpDcsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal, csrRowPtr,
                                           csrColInd, batchSize, info, internalDataInBytes,
                                           workspaceInBytes)
    @check @runtime_ccall((:cusolverSpDcsrqrBufferInfoBatched, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, csrqrInfo_t,
                  Ptr{Csize_t}, Ptr{Csize_t}),
                 handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info,
                 internalDataInBytes, workspaceInBytes)
end

function cusolverSpCcsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal, csrRowPtr,
                                           csrColInd, batchSize, info, internalDataInBytes,
                                           workspaceInBytes)
    @check @runtime_ccall((:cusolverSpCcsrqrBufferInfoBatched, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, csrqrInfo_t,
                  Ptr{Csize_t}, Ptr{Csize_t}),
                 handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info,
                 internalDataInBytes, workspaceInBytes)
end

function cusolverSpZcsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal, csrRowPtr,
                                           csrColInd, batchSize, info, internalDataInBytes,
                                           workspaceInBytes)
    @check @runtime_ccall((:cusolverSpZcsrqrBufferInfoBatched, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, csrqrInfo_t,
                  Ptr{Csize_t}, Ptr{Csize_t}),
                 handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info,
                 internalDataInBytes, workspaceInBytes)
end

function cusolverSpScsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                   csrColIndA, b, x, batchSize, info, pBuffer)
    @check @runtime_ccall((:cusolverSpScsrqrsvBatched, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                  csrqrInfo_t, CuPtr{Cvoid}),
                 handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x,
                 batchSize, info, pBuffer)
end

function cusolverSpDcsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                   csrColIndA, b, x, batchSize, info, pBuffer)
    @check @runtime_ccall((:cusolverSpDcsrqrsvBatched, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, CuPtr{Cdouble},
                  Cint, csrqrInfo_t, CuPtr{Cvoid}),
                 handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x,
                 batchSize, info, pBuffer)
end

function cusolverSpCcsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                   csrColIndA, b, x, batchSize, info, pBuffer)
    @check @runtime_ccall((:cusolverSpCcsrqrsvBatched, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint, csrqrInfo_t, CuPtr{Cvoid}),
                 handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x,
                 batchSize, info, pBuffer)
end

function cusolverSpZcsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                   csrColIndA, b, x, batchSize, info, pBuffer)
    @check @runtime_ccall((:cusolverSpZcsrqrsvBatched, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, csrqrInfo_t, CuPtr{Cvoid}),
                 handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x,
                 batchSize, info, pBuffer)
end
