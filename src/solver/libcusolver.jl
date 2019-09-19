# Julia wrapper for header: cusolver_common.h
# Automatically generated using Clang.jl


function cusolverGetProperty(type, value)
    @check ccall((:cusolverGetProperty, libcusolver), cusolverStatus_t,
                 (libraryPropertyType, Ptr{Cint}),
                 type, value)
end

function cusolverGetVersion(version)
    @check ccall((:cusolverGetVersion, libcusolver), cusolverStatus_t,
                 (Ptr{Cint},),
                 version)
end
# Julia wrapper for header: cusolverDn.h
# Automatically generated using Clang.jl


function cusolverDnCreate(handle)
    @check ccall((:cusolverDnCreate, libcusolver), cusolverStatus_t,
                 (Ptr{cusolverDnHandle_t},),
                 handle)
end

function cusolverDnDestroy(handle)
    @check ccall((:cusolverDnDestroy, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t,),
                 handle)
end

function cusolverDnSetStream(handle, streamId)
    @check ccall((:cusolverDnSetStream, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, CuStream_t),
                 handle, streamId)
end

function cusolverDnGetStream(handle, streamId)
    @check ccall((:cusolverDnGetStream, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Ptr{CuStream_t}),
                 handle, streamId)
end

function cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    @check ccall((:cusolverDnSpotrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, A, lda, Lwork)
end

function cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    @check ccall((:cusolverDnDpotrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, A, lda, Lwork)
end

function cusolverDnCpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    @check ccall((:cusolverDnCpotrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, A, lda, Lwork)
end

function cusolverDnZpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    @check ccall((:cusolverDnZpotrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, Ptr{Cint}),
                 handle, uplo, n, A, lda, Lwork)
end

function cusolverDnSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    @check ccall((:cusolverDnSpotrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
end

function cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    @check ccall((:cusolverDnDpotrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
end

function cusolverDnCpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    @check ccall((:cusolverDnCpotrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
end

function cusolverDnZpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    @check ccall((:cusolverDnZpotrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
end

function cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    @check ccall((:cusolverDnSpotrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
end

function cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    @check ccall((:cusolverDnDpotrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
end

function cusolverDnCpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    @check ccall((:cusolverDnCpotrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{cuComplex},
                  Cint, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
end

function cusolverDnZpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    @check ccall((:cusolverDnZpotrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
end

function cusolverDnSpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    @check ccall((:cusolverDnSpotrfBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Ptr{Cfloat}}, Cint,
                  CuPtr{Cint}, Cint),
                 handle, uplo, n, Aarray, lda, infoArray, batchSize)
end

function cusolverDnDpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    @check ccall((:cusolverDnDpotrfBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Ptr{Cdouble}}, Cint,
                  CuPtr{Cint}, Cint),
                 handle, uplo, n, Aarray, lda, infoArray, batchSize)
end

function cusolverDnCpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    @check ccall((:cusolverDnCpotrfBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Ptr{cuComplex}}, Cint,
                  CuPtr{Cint}, Cint),
                 handle, uplo, n, Aarray, lda, infoArray, batchSize)
end

function cusolverDnZpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    @check ccall((:cusolverDnZpotrfBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Ptr{cuDoubleComplex}},
                  Cint, CuPtr{Cint}, Cint),
                 handle, uplo, n, Aarray, lda, infoArray, batchSize)
end

function cusolverDnSpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
    @check ccall((:cusolverDnSpotrsBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Ptr{Cfloat}},
                  Cint, CuPtr{Ptr{Cfloat}}, Cint, CuPtr{Cint}, Cint),
                 handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
end

function cusolverDnDpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
    @check ccall((:cusolverDnDpotrsBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Ptr{Cdouble}},
                  Cint, CuPtr{Ptr{Cdouble}}, Cint, CuPtr{Cint}, Cint),
                 handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
end

function cusolverDnCpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
    @check ccall((:cusolverDnCpotrsBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Ptr{cuComplex}},
                  Cint, CuPtr{Ptr{cuComplex}}, Cint, CuPtr{Cint}, Cint),
                 handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
end

function cusolverDnZpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
    @check ccall((:cusolverDnZpotrsBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                  CuPtr{Ptr{cuDoubleComplex}}, Cint, CuPtr{Ptr{cuDoubleComplex}}, Cint,
                  CuPtr{Cint}, Cint),
                 handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
end

function cusolverDnSpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    @check ccall((:cusolverDnSpotri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, A, lda, lwork)
end

function cusolverDnDpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    @check ccall((:cusolverDnDpotri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, A, lda, lwork)
end

function cusolverDnCpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    @check ccall((:cusolverDnCpotri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, A, lda, lwork)
end

function cusolverDnZpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    @check ccall((:cusolverDnZpotri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, Ptr{Cint}),
                 handle, uplo, n, A, lda, lwork)
end

function cusolverDnSpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    @check ccall((:cusolverDnSpotri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, work, lwork, devInfo)
end

function cusolverDnDpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    @check ccall((:cusolverDnDpotri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, work, lwork, devInfo)
end

function cusolverDnCpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    @check ccall((:cusolverDnCpotri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, work, lwork, devInfo)
end

function cusolverDnZpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    @check ccall((:cusolverDnZpotri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, work, lwork, devInfo)
end

function cusolverDnStrtri_bufferSize(handle, uplo, diag, n, A, lda, lwork)
    @check ccall((:cusolverDnStrtri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                  CuPtr{Cfloat}, Cint, Ptr{Cint}),
                 handle, uplo, diag, n, A, lda, lwork)
end

function cusolverDnDtrtri_bufferSize(handle, uplo, diag, n, A, lda, lwork)
    @check ccall((:cusolverDnDtrtri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                  CuPtr{Cdouble}, Cint, Ptr{Cint}),
                 handle, uplo, diag, n, A, lda, lwork)
end

function cusolverDnCtrtri_bufferSize(handle, uplo, diag, n, A, lda, lwork)
    @check ccall((:cusolverDnCtrtri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                  CuPtr{cuComplex}, Cint, Ptr{Cint}),
                 handle, uplo, diag, n, A, lda, lwork)
end

function cusolverDnZtrtri_bufferSize(handle, uplo, diag, n, A, lda, lwork)
    @check ccall((:cusolverDnZtrtri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                  CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                 handle, uplo, diag, n, A, lda, lwork)
end

function cusolverDnStrtri(handle, uplo, diag, n, A, lda, work, lwork, devInfo)
    @check ccall((:cusolverDnStrtri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, uplo, diag, n, A, lda, work, lwork, devInfo)
end

function cusolverDnDtrtri(handle, uplo, diag, n, A, lda, work, lwork, devInfo)
    @check ccall((:cusolverDnDtrtri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, uplo, diag, n, A, lda, work, lwork, devInfo)
end

function cusolverDnCtrtri(handle, uplo, diag, n, A, lda, work, lwork, devInfo)
    @check ccall((:cusolverDnCtrtri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, diag, n, A, lda, work, lwork, devInfo)
end

function cusolverDnZtrtri(handle, uplo, diag, n, A, lda, work, lwork, devInfo)
    @check ccall((:cusolverDnZtrtri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, diag, n, A, lda, work, lwork, devInfo)
end

function cusolverDnSlauum_bufferSize(handle, uplo, n, A, lda, lwork)
    @check ccall((:cusolverDnSlauum_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, A, lda, lwork)
end

function cusolverDnDlauum_bufferSize(handle, uplo, n, A, lda, lwork)
    @check ccall((:cusolverDnDlauum_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, A, lda, lwork)
end

function cusolverDnClauum_bufferSize(handle, uplo, n, A, lda, lwork)
    @check ccall((:cusolverDnClauum_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, A, lda, lwork)
end

function cusolverDnZlauum_bufferSize(handle, uplo, n, A, lda, lwork)
    @check ccall((:cusolverDnZlauum_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, Ptr{Cint}),
                 handle, uplo, n, A, lda, lwork)
end

function cusolverDnSlauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    @check ccall((:cusolverDnSlauum, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, work, lwork, devInfo)
end

function cusolverDnDlauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    @check ccall((:cusolverDnDlauum, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, work, lwork, devInfo)
end

function cusolverDnClauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    @check ccall((:cusolverDnClauum, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, work, lwork, devInfo)
end

function cusolverDnZlauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    @check ccall((:cusolverDnZlauum, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, work, lwork, devInfo)
end

function cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    @check ccall((:cusolverDnSgetrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, Ptr{Cint}),
                 handle, m, n, A, lda, Lwork)
end

function cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    @check ccall((:cusolverDnDgetrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, Ptr{Cint}),
                 handle, m, n, A, lda, Lwork)
end

function cusolverDnCgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    @check ccall((:cusolverDnCgetrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint, Ptr{Cint}),
                 handle, m, n, A, lda, Lwork)
end

function cusolverDnZgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    @check ccall((:cusolverDnZgetrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                 handle, m, n, A, lda, Lwork)
end

function cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    @check ccall((:cusolverDnSgetrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, A, lda, Workspace, devIpiv, devInfo)
end

function cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    @check ccall((:cusolverDnDgetrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, A, lda, Workspace, devIpiv, devInfo)
end

function cusolverDnCgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    @check ccall((:cusolverDnCgetrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, A, lda, Workspace, devIpiv, devInfo)
end

function cusolverDnZgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    @check ccall((:cusolverDnZgetrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, A, lda, Workspace, devIpiv, devInfo)
end

function cusolverDnSlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    @check ccall((:cusolverDnSlaswp, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, CuPtr{Cfloat}, Cint, Cint, Cint, CuPtr{Cint},
                  Cint),
                 handle, n, A, lda, k1, k2, devIpiv, incx)
end

function cusolverDnDlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    @check ccall((:cusolverDnDlaswp, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, CuPtr{Cdouble}, Cint, Cint, Cint, CuPtr{Cint},
                  Cint),
                 handle, n, A, lda, k1, k2, devIpiv, incx)
end

function cusolverDnClaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    @check ccall((:cusolverDnClaswp, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, CuPtr{cuComplex}, Cint, Cint, Cint,
                  CuPtr{Cint}, Cint),
                 handle, n, A, lda, k1, k2, devIpiv, incx)
end

function cusolverDnZlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    @check ccall((:cusolverDnZlaswp, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, CuPtr{cuDoubleComplex}, Cint, Cint, Cint,
                  CuPtr{Cint}, Cint),
                 handle, n, A, lda, k1, k2, devIpiv, incx)
end

function cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    @check ccall((:cusolverDnSgetrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasOperation_t, Cint, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
end

function cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    @check ccall((:cusolverDnDgetrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasOperation_t, Cint, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
end

function cusolverDnCgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    @check ccall((:cusolverDnCgetrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasOperation_t, Cint, Cint, CuPtr{cuComplex},
                  Cint, CuPtr{Cint}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
end

function cusolverDnZgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    @check ccall((:cusolverDnZgetrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasOperation_t, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{Cint}),
                 handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
end

function cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    @check ccall((:cusolverDnSgeqrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, Ptr{Cint}),
                 handle, m, n, A, lda, lwork)
end

function cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    @check ccall((:cusolverDnDgeqrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, Ptr{Cint}),
                 handle, m, n, A, lda, lwork)
end

function cusolverDnCgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    @check ccall((:cusolverDnCgeqrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint, Ptr{Cint}),
                 handle, m, n, A, lda, lwork)
end

function cusolverDnZgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    @check ccall((:cusolverDnZgeqrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                 handle, m, n, A, lda, lwork)
end

function cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    @check ccall((:cusolverDnSgeqrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
end

function cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    @check ccall((:cusolverDnDgeqrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
end

function cusolverDnCgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    @check ccall((:cusolverDnCgeqrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
end

function cusolverDnZgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    @check ccall((:cusolverDnZgeqrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
end

function cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    @check ccall((:cusolverDnSorgqr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Ptr{Cint}),
                 handle, m, n, k, A, lda, tau, lwork)
end

function cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    @check ccall((:cusolverDnDorgqr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Ptr{Cint}),
                 handle, m, n, k, A, lda, tau, lwork)
end

function cusolverDnCungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    @check ccall((:cusolverDnCungqr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Ptr{Cint}),
                 handle, m, n, k, A, lda, tau, lwork)
end

function cusolverDnZungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    @check ccall((:cusolverDnZungqr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Ptr{Cint}),
                 handle, m, n, k, A, lda, tau, lwork)
end

function cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    @check ccall((:cusolverDnSorgqr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, m, n, k, A, lda, tau, work, lwork, info)
end

function cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    @check ccall((:cusolverDnDorgqr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, m, n, k, A, lda, tau, work, lwork, info)
end

function cusolverDnCungqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    @check ccall((:cusolverDnCungqr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, m, n, k, A, lda, tau, work, lwork, info)
end

function cusolverDnZungqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    @check ccall((:cusolverDnZungqr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, m, n, k, A, lda, tau, work, lwork, info)
end

function cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc,
                                     lwork)
    @check ccall((:cusolverDnSormqr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                  Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, Ptr{Cint}),
                 handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
end

function cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc,
                                     lwork)
    @check ccall((:cusolverDnDormqr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                  Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                  Ptr{Cint}),
                 handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
end

function cusolverDnCunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc,
                                     lwork)
    @check ccall((:cusolverDnCunmqr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                  Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, CuPtr{cuComplex}, Cint,
                  Ptr{Cint}),
                 handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
end

function cusolverDnZunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc,
                                     lwork)
    @check ccall((:cusolverDnZunmqr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                  Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                 handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
end

function cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork,
                          devInfo)
    @check ccall((:cusolverDnSormqr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                  Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
end

function cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork,
                          devInfo)
    @check ccall((:cusolverDnDormqr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                  Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
end

function cusolverDnCunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork,
                          devInfo)
    @check ccall((:cusolverDnCunmqr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                  Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
end

function cusolverDnZunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork,
                          devInfo)
    @check ccall((:cusolverDnZunmqr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, Cint, Cint,
                  Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
end

function cusolverDnSsytrf_bufferSize(handle, n, A, lda, lwork)
    @check ccall((:cusolverDnSsytrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, CuPtr{Cfloat}, Cint, Ptr{Cint}),
                 handle, n, A, lda, lwork)
end

function cusolverDnDsytrf_bufferSize(handle, n, A, lda, lwork)
    @check ccall((:cusolverDnDsytrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, CuPtr{Cdouble}, Cint, Ptr{Cint}),
                 handle, n, A, lda, lwork)
end

function cusolverDnCsytrf_bufferSize(handle, n, A, lda, lwork)
    @check ccall((:cusolverDnCsytrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, CuPtr{cuComplex}, Cint, Ptr{Cint}),
                 handle, n, A, lda, lwork)
end

function cusolverDnZsytrf_bufferSize(handle, n, A, lda, lwork)
    @check ccall((:cusolverDnZsytrf_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                 handle, n, A, lda, lwork)
end

function cusolverDnSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    @check ccall((:cusolverDnSsytrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

function cusolverDnDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    @check ccall((:cusolverDnDsytrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

function cusolverDnCsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    @check ccall((:cusolverDnCsytrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cint}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

function cusolverDnZsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    @check ccall((:cusolverDnZsytrf, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

function cusolverDnSsytrs_bufferSize(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
    @check ccall((:cusolverDnSsytrs_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}, CuPtr{Cfloat}, Cint, Ptr{Cint}),
                 handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
end

function cusolverDnDsytrs_bufferSize(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
    @check ccall((:cusolverDnDsytrs_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}, CuPtr{Cdouble}, Cint, Ptr{Cint}),
                 handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
end

function cusolverDnCsytrs_bufferSize(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
    @check ccall((:cusolverDnCsytrs_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{cuComplex},
                  Cint, CuPtr{Cint}, CuPtr{cuComplex}, Cint, Ptr{Cint}),
                 handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
end

function cusolverDnZsytrs_bufferSize(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
    @check ccall((:cusolverDnZsytrs_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex}, Cint,
                  Ptr{Cint}),
                 handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork)
end

function cusolverDnSsytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
    @check ccall((:cusolverDnSsytrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
end

function cusolverDnDsytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
    @check ccall((:cusolverDnDsytrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
end

function cusolverDnCsytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
    @check ccall((:cusolverDnCsytrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint, CuPtr{cuComplex},
                  Cint, CuPtr{Cint}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cint}),
                 handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
end

function cusolverDnZsytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
    @check ccall((:cusolverDnZsytrs, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info)
end

function cusolverDnSsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    @check ccall((:cusolverDnSsytri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}, Ptr{Cint}),
                 handle, uplo, n, A, lda, ipiv, lwork)
end

function cusolverDnDsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    @check ccall((:cusolverDnDsytri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}, Ptr{Cint}),
                 handle, uplo, n, A, lda, ipiv, lwork)
end

function cusolverDnCsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    @check ccall((:cusolverDnCsytri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cint}, Ptr{Cint}),
                 handle, uplo, n, A, lda, ipiv, lwork)
end

function cusolverDnZsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    @check ccall((:cusolverDnZsytri_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cint}, Ptr{Cint}),
                 handle, uplo, n, A, lda, ipiv, lwork)
end

function cusolverDnSsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    @check ccall((:cusolverDnSsytri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

function cusolverDnDsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    @check ccall((:cusolverDnDsytri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

function cusolverDnCsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    @check ccall((:cusolverDnCsytri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cint}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

function cusolverDnZsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    @check ccall((:cusolverDnZsytri, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, ipiv, work, lwork, info)
end

function cusolverDnSgebrd_bufferSize(handle, m, n, Lwork)
    @check ccall((:cusolverDnSgebrd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                 handle, m, n, Lwork)
end

function cusolverDnDgebrd_bufferSize(handle, m, n, Lwork)
    @check ccall((:cusolverDnDgebrd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                 handle, m, n, Lwork)
end

function cusolverDnCgebrd_bufferSize(handle, m, n, Lwork)
    @check ccall((:cusolverDnCgebrd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                 handle, m, n, Lwork)
end

function cusolverDnZgebrd_bufferSize(handle, m, n, Lwork)
    @check ccall((:cusolverDnZgebrd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                 handle, m, n, Lwork)
end

function cusolverDnSgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
    @check ccall((:cusolverDnSgebrd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                  CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}),
                 handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
end

function cusolverDnDgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
    @check ccall((:cusolverDnDgebrd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                  CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}),
                 handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
end

function cusolverDnCgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
    @check ccall((:cusolverDnCgebrd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuComplex}, Cint, CuPtr{Cfloat},
                  CuPtr{Cfloat}, CuPtr{cuComplex}, CuPtr{cuComplex}, CuPtr{cuComplex},
                  Cint, CuPtr{Cint}),
                 handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
end

function cusolverDnZgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
    @check ccall((:cusolverDnZgebrd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
end

function cusolverDnSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    @check ccall((:cusolverDnSorgbr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint, CuPtr{Cfloat},
                  Cint, CuPtr{Cfloat}, Ptr{Cint}),
                 handle, side, m, n, k, A, lda, tau, lwork)
end

function cusolverDnDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    @check ccall((:cusolverDnDorgbr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint, CuPtr{Cdouble},
                  Cint, CuPtr{Cdouble}, Ptr{Cint}),
                 handle, side, m, n, k, A, lda, tau, lwork)
end

function cusolverDnCungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    @check ccall((:cusolverDnCungbr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Ptr{Cint}),
                 handle, side, m, n, k, A, lda, tau, lwork)
end

function cusolverDnZungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    @check ccall((:cusolverDnZungbr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Ptr{Cint}),
                 handle, side, m, n, k, A, lda, tau, lwork)
end

function cusolverDnSorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    @check ccall((:cusolverDnSorgbr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint, CuPtr{Cfloat},
                  Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, side, m, n, k, A, lda, tau, work, lwork, info)
end

function cusolverDnDorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    @check ccall((:cusolverDnDorgbr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint, CuPtr{Cdouble},
                  Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, side, m, n, k, A, lda, tau, work, lwork, info)
end

function cusolverDnCungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    @check ccall((:cusolverDnCungbr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, CuPtr{cuComplex}, Cint,
                  CuPtr{Cint}),
                 handle, side, m, n, k, A, lda, tau, work, lwork, info)
end

function cusolverDnZungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    @check ccall((:cusolverDnZungbr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, Cint, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, side, m, n, k, A, lda, tau, work, lwork, info)
end

function cusolverDnSsytrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    @check ccall((:cusolverDnSsytrd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, Ptr{Cint}),
                 handle, uplo, n, A, lda, d, e, tau, lwork)
end

function cusolverDnDsytrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    @check ccall((:cusolverDnDsytrd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, Ptr{Cint}),
                 handle, uplo, n, A, lda, d, e, tau, lwork)
end

function cusolverDnChetrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    @check ccall((:cusolverDnChetrd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{cuComplex}, Ptr{Cint}),
                 handle, uplo, n, A, lda, d, e, tau, lwork)
end

function cusolverDnZhetrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    @check ccall((:cusolverDnZhetrd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Ptr{Cint}),
                 handle, uplo, n, A, lda, d, e, tau, lwork)
end

function cusolverDnSsytrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    @check ccall((:cusolverDnSsytrd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}),
                 handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
end

function cusolverDnDsytrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    @check ccall((:cusolverDnDsytrd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}),
                 handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
end

function cusolverDnChetrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    @check ccall((:cusolverDnChetrd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{cuComplex}, CuPtr{cuComplex}, Cint,
                  CuPtr{Cint}),
                 handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
end

function cusolverDnZhetrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    @check ccall((:cusolverDnZhetrd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
end

function cusolverDnSorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    @check ccall((:cusolverDnSorgtr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Ptr{Cint}),
                 handle, uplo, n, A, lda, tau, lwork)
end

function cusolverDnDorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    @check ccall((:cusolverDnDorgtr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Ptr{Cint}),
                 handle, uplo, n, A, lda, tau, lwork)
end

function cusolverDnCungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    @check ccall((:cusolverDnCungtr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Ptr{Cint}),
                 handle, uplo, n, A, lda, tau, lwork)
end

function cusolverDnZungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    @check ccall((:cusolverDnZungtr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{cuDoubleComplex}, Ptr{Cint}),
                 handle, uplo, n, A, lda, tau, lwork)
end

function cusolverDnSorgtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    @check ccall((:cusolverDnSorgtr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, tau, work, lwork, info)
end

function cusolverDnDorgtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    @check ccall((:cusolverDnDorgtr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, tau, work, lwork, info)
end

function cusolverDnCungtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    @check ccall((:cusolverDnCungtr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, tau, work, lwork, info)
end

function cusolverDnZungtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    @check ccall((:cusolverDnZungtr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, uplo, n, A, lda, tau, work, lwork, info)
end

function cusolverDnSormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                     lwork)
    @check ccall((:cusolverDnSormtr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                  cublasOperation_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint, Ptr{Cint}),
                 handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
end

function cusolverDnDormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                     lwork)
    @check ccall((:cusolverDnDormtr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                  cublasOperation_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint, Ptr{Cint}),
                 handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
end

function cusolverDnCunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                     lwork)
    @check ccall((:cusolverDnCunmtr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                  cublasOperation_t, Cint, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint, Ptr{Cint}),
                 handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
end

function cusolverDnZunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc,
                                     lwork)
    @check ccall((:cusolverDnZunmtr_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                  cublasOperation_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}),
                 handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
end

function cusolverDnSormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work,
                          lwork, info)
    @check ccall((:cusolverDnSormtr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                  cublasOperation_t, Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
end

function cusolverDnDormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work,
                          lwork, info)
    @check ccall((:cusolverDnDormtr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                  cublasOperation_t, Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
end

function cusolverDnCunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work,
                          lwork, info)
    @check ccall((:cusolverDnCunmtr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                  cublasOperation_t, Cint, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
end

function cusolverDnZunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work,
                          lwork, info)
    @check ccall((:cusolverDnZunmtr, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t,
                  cublasOperation_t, Cint, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
end

function cusolverDnSgesvd_bufferSize(handle, m, n, lwork)
    @check ccall((:cusolverDnSgesvd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                 handle, m, n, lwork)
end

function cusolverDnDgesvd_bufferSize(handle, m, n, lwork)
    @check ccall((:cusolverDnDgesvd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                 handle, m, n, lwork)
end

function cusolverDnCgesvd_bufferSize(handle, m, n, lwork)
    @check ccall((:cusolverDnCgesvd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                 handle, m, n, lwork)
end

function cusolverDnZgesvd_bufferSize(handle, m, n, lwork)
    @check ccall((:cusolverDnZgesvd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                 handle, m, n, lwork)
end

function cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work,
                          lwork, rwork, info)
    @check ccall((:cusolverDnSgesvd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, UInt8, UInt8, Cint, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                  Cint, CuPtr{Cfloat}, CuPtr{Cint}),
                 handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork,
                 rwork, info)
end

function cusolverDnDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work,
                          lwork, rwork, info)
    @check ccall((:cusolverDnDgesvd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, UInt8, UInt8, Cint, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cint}),
                 handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork,
                 rwork, info)
end

function cusolverDnCgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work,
                          lwork, rwork, info)
    @check ccall((:cusolverDnCgesvd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, UInt8, UInt8, Cint, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{Cint}),
                 handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork,
                 rwork, info)
end

function cusolverDnZgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work,
                          lwork, rwork, info)
    @check ccall((:cusolverDnZgesvd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, UInt8, UInt8, Cint, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cint}),
                 handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork,
                 rwork, info)
end

function cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    @check ccall((:cusolverDnSsyevd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Ptr{Cint}),
                 handle, jobz, uplo, n, A, lda, W, lwork)
end

function cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    @check ccall((:cusolverDnDsyevd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Ptr{Cint}),
                 handle, jobz, uplo, n, A, lda, W, lwork)
end

function cusolverDnCheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    @check ccall((:cusolverDnCheevd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, Ptr{Cint}),
                 handle, jobz, uplo, n, A, lda, W, lwork)
end

function cusolverDnZheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    @check ccall((:cusolverDnZheevd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}),
                 handle, jobz, uplo, n, A, lda, W, lwork)
end

function cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    @check ccall((:cusolverDnSsyevd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info)
end

function cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    @check ccall((:cusolverDnDsyevd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info)
end

function cusolverDnCheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    @check ccall((:cusolverDnCheevd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                  CuPtr{Cint}),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info)
end

function cusolverDnZheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    @check ccall((:cusolverDnZheevd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cint}),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info)
end

function cusolverDnSsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                      meig, W, lwork)
    @check ccall((:cusolverDnSsyevdx_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                  cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, Cfloat, Cfloat, Cint, Cint,
                  Ptr{Cint}, CuPtr{Cfloat}, Ptr{Cint}),
                 handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
end

function cusolverDnDsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                      meig, W, lwork)
    @check ccall((:cusolverDnDsyevdx_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                  cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, Cdouble, Cdouble, Cint,
                  Cint, Ptr{Cint}, CuPtr{Cdouble}, Ptr{Cint}),
                 handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
end

function cusolverDnCheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                      meig, W, lwork)
    @check ccall((:cusolverDnCheevdx_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                  cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, Cfloat, Cfloat, Cint,
                  Cint, Ptr{Cint}, CuPtr{Cfloat}, Ptr{Cint}),
                 handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
end

function cusolverDnZheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                                      meig, W, lwork)
    @check ccall((:cusolverDnZheevdx_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                  cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint, Cdouble, Cdouble,
                  Cint, Cint, Ptr{Cint}, CuPtr{Cdouble}, Ptr{Cint}),
                 handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
end

function cusolverDnSsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W,
                           work, lwork, info)
    @check ccall((:cusolverDnSsyevdx, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                  cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, Cfloat, Cfloat, Cint, Cint,
                  Ptr{Cint}, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work,
                 lwork, info)
end

function cusolverDnDsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W,
                           work, lwork, info)
    @check ccall((:cusolverDnDsyevdx, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                  cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, Cdouble, Cdouble, Cint,
                  Cint, Ptr{Cint}, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work,
                 lwork, info)
end

function cusolverDnCheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W,
                           work, lwork, info)
    @check ccall((:cusolverDnCheevdx, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                  cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, Cfloat, Cfloat, Cint,
                  Cint, Ptr{Cint}, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work,
                 lwork, info)
end

function cusolverDnZheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W,
                           work, lwork, info)
    @check ccall((:cusolverDnZheevdx, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
                  cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint, Cdouble, Cdouble,
                  Cint, Cint, Ptr{Cint}, CuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{Cint}),
                 handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work,
                 lwork, info)
end

function cusolverDnSsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb,
                                      vl, vu, il, iu, meig, W, lwork)
    @check ccall((:cusolverDnSsygvdx_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint, Cfloat, Cfloat, Cint, Cint, Ptr{Cint},
                  CuPtr{Cfloat}, Ptr{Cint}),
                 handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig,
                 W, lwork)
end

function cusolverDnDsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb,
                                      vl, vu, il, iu, meig, W, lwork)
    @check ccall((:cusolverDnDsygvdx_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, Cdouble, Cdouble, Cint, Cint, Ptr{Cint},
                  CuPtr{Cdouble}, Ptr{Cint}),
                 handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig,
                 W, lwork)
end

function cusolverDnChegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb,
                                      vl, vu, il, iu, meig, W, lwork)
    @check ccall((:cusolverDnChegvdx_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, Cfloat, Cfloat, Cint, Cint, Ptr{Cint},
                  CuPtr{Cfloat}, Ptr{Cint}),
                 handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig,
                 W, lwork)
end

function cusolverDnZhegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb,
                                      vl, vu, il, iu, meig, W, lwork)
    @check ccall((:cusolverDnZhegvdx_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, Cdouble, Cdouble, Cint, Cint, Ptr{Cint},
                  CuPtr{Cdouble}, Ptr{Cint}),
                 handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig,
                 W, lwork)
end

function cusolverDnSsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il,
                           iu, meig, W, work, lwork, info)
    @check ccall((:cusolverDnSsygvdx, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint, Cfloat, Cfloat, Cint, Cint, Ptr{Cint},
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig,
                 W, work, lwork, info)
end

function cusolverDnDsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il,
                           iu, meig, W, work, lwork, info)
    @check ccall((:cusolverDnDsygvdx, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, Cdouble, Cdouble, Cint, Cint, Ptr{Cint},
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig,
                 W, work, lwork, info)
end

function cusolverDnChegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il,
                           iu, meig, W, work, lwork, info)
    @check ccall((:cusolverDnChegvdx, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, Cfloat, Cfloat, Cint, Cint, Ptr{Cint},
                  CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig,
                 W, work, lwork, info)
end

function cusolverDnZhegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il,
                           iu, meig, W, work, lwork, info)
    @check ccall((:cusolverDnZhegvdx, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cusolverEigRange_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, Cdouble, Cdouble, Cint, Cint, Ptr{Cint},
                  CuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}),
                 handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig,
                 W, work, lwork, info)
end

function cusolverDnSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
    @check ccall((:cusolverDnSsygvd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Ptr{Cint}),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
end

function cusolverDnDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
    @check ccall((:cusolverDnDsygvd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Ptr{Cint}),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
end

function cusolverDnChegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
    @check ccall((:cusolverDnChegvd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cfloat}, Ptr{Cint}),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
end

function cusolverDnZhegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
    @check ccall((:cusolverDnZhegvd_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
end

function cusolverDnSsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
    @check ccall((:cusolverDnSsygvd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
end

function cusolverDnDsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
    @check ccall((:cusolverDnDsygvd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
end

function cusolverDnChegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
    @check ccall((:cusolverDnChegvd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{Cint}),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
end

function cusolverDnZhegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
    @check ccall((:cusolverDnZhegvd, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cint}),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
end

function cusolverDnCreateSyevjInfo(info)
    @check ccall((:cusolverDnCreateSyevjInfo, libcusolver), cusolverStatus_t,
                 (Ptr{syevjInfo_t},),
                 info)
end

function cusolverDnDestroySyevjInfo(info)
    @check ccall((:cusolverDnDestroySyevjInfo, libcusolver), cusolverStatus_t,
                 (syevjInfo_t,),
                 info)
end

function cusolverDnXsyevjSetTolerance(info, tolerance)
    @check ccall((:cusolverDnXsyevjSetTolerance, libcusolver), cusolverStatus_t,
                 (syevjInfo_t, Cdouble),
                 info, tolerance)
end

function cusolverDnXsyevjSetMaxSweeps(info, max_sweeps)
    @check ccall((:cusolverDnXsyevjSetMaxSweeps, libcusolver), cusolverStatus_t,
                 (syevjInfo_t, Cint),
                 info, max_sweeps)
end

function cusolverDnXsyevjSetSortEig(info, sort_eig)
    @check ccall((:cusolverDnXsyevjSetSortEig, libcusolver), cusolverStatus_t,
                 (syevjInfo_t, Cint),
                 info, sort_eig)
end

function cusolverDnXsyevjGetResidual(handle, info, residual)
    @check ccall((:cusolverDnXsyevjGetResidual, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, syevjInfo_t, Ptr{Cdouble}),
                 handle, info, residual)
end

function cusolverDnXsyevjGetSweeps(handle, info, executed_sweeps)
    @check ccall((:cusolverDnXsyevjGetSweeps, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, syevjInfo_t, Ptr{Cint}),
                 handle, info, executed_sweeps)
end

function cusolverDnSsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                            params, batchSize)
    @check ccall((:cusolverDnSsyevjBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t, Cint),
                 handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
end

function cusolverDnDsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                            params, batchSize)
    @check ccall((:cusolverDnDsyevjBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t, Cint),
                 handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
end

function cusolverDnCheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                            params, batchSize)
    @check ccall((:cusolverDnCheevjBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t, Cint),
                 handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
end

function cusolverDnZheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                            params, batchSize)
    @check ccall((:cusolverDnZheevjBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t,
                  Cint),
                 handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
end

function cusolverDnSsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                 params, batchSize)
    @check ccall((:cusolverDnSsyevjBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint},
                  syevjInfo_t, Cint),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
end

function cusolverDnDsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                 params, batchSize)
    @check ccall((:cusolverDnDsyevjBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint},
                  syevjInfo_t, Cint),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
end

function cusolverDnCheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                 params, batchSize)
    @check ccall((:cusolverDnCheevjBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                  CuPtr{Cint}, syevjInfo_t, Cint),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
end

function cusolverDnZheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                                 params, batchSize)
    @check ccall((:cusolverDnZheevjBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cint}, syevjInfo_t, Cint),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
end

function cusolverDnSsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params)
    @check ccall((:cusolverDnSsyevj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t),
                 handle, jobz, uplo, n, A, lda, W, lwork, params)
end

function cusolverDnDsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params)
    @check ccall((:cusolverDnDsyevj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t),
                 handle, jobz, uplo, n, A, lda, W, lwork, params)
end

function cusolverDnCheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params)
    @check ccall((:cusolverDnCheevj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t),
                 handle, jobz, uplo, n, A, lda, W, lwork, params)
end

function cusolverDnZheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params)
    @check ccall((:cusolverDnZheevj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t),
                 handle, jobz, uplo, n, A, lda, W, lwork, params)
end

function cusolverDnSsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
    @check ccall((:cusolverDnSsyevj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint},
                  syevjInfo_t),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
end

function cusolverDnDsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
    @check ccall((:cusolverDnDsyevj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint},
                  syevjInfo_t),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
end

function cusolverDnCheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
    @check ccall((:cusolverDnCheevj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                  CuPtr{Cint}, syevjInfo_t),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
end

function cusolverDnZheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
    @check ccall((:cusolverDnZheevj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cint}, syevjInfo_t),
                 handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
end

function cusolverDnSsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W,
                                     lwork, params)
    @check ccall((:cusolverDnSsygvj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
end

function cusolverDnDsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W,
                                     lwork, params)
    @check ccall((:cusolverDnDsygvj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
end

function cusolverDnChegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W,
                                     lwork, params)
    @check ccall((:cusolverDnChegvj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cfloat}, Ptr{Cint}, syevjInfo_t),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
end

function cusolverDnZhegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W,
                                     lwork, params)
    @check ccall((:cusolverDnZhegvj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, Ptr{Cint}, syevjInfo_t),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
end

function cusolverDnSsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork,
                          info, params)
    @check ccall((:cusolverDnSsygvj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cint}, syevjInfo_t),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info,
                 params)
end

function cusolverDnDsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork,
                          info, params)
    @check ccall((:cusolverDnDsygvj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cint}, syevjInfo_t),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info,
                 params)
end

function cusolverDnChegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork,
                          info, params)
    @check ccall((:cusolverDnChegvj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{Cint}, syevjInfo_t),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info,
                 params)
end

function cusolverDnZhegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork,
                          info, params)
    @check ccall((:cusolverDnZhegvj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
                  cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cint}, syevjInfo_t),
                 handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info,
                 params)
end

function cusolverDnCreateGesvdjInfo(info)
    @check ccall((:cusolverDnCreateGesvdjInfo, libcusolver), cusolverStatus_t,
                 (Ptr{gesvdjInfo_t},),
                 info)
end

function cusolverDnDestroyGesvdjInfo(info)
    @check ccall((:cusolverDnDestroyGesvdjInfo, libcusolver), cusolverStatus_t,
                 (gesvdjInfo_t,),
                 info)
end

function cusolverDnXgesvdjSetTolerance(info, tolerance)
    @check ccall((:cusolverDnXgesvdjSetTolerance, libcusolver), cusolverStatus_t,
                 (gesvdjInfo_t, Cdouble),
                 info, tolerance)
end

function cusolverDnXgesvdjSetMaxSweeps(info, max_sweeps)
    @check ccall((:cusolverDnXgesvdjSetMaxSweeps, libcusolver), cusolverStatus_t,
                 (gesvdjInfo_t, Cint),
                 info, max_sweeps)
end

function cusolverDnXgesvdjSetSortEig(info, sort_svd)
    @check ccall((:cusolverDnXgesvdjSetSortEig, libcusolver), cusolverStatus_t,
                 (gesvdjInfo_t, Cint),
                 info, sort_svd)
end

function cusolverDnXgesvdjGetResidual(handle, info, residual)
    @check ccall((:cusolverDnXgesvdjGetResidual, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, gesvdjInfo_t, Ptr{Cdouble}),
                 handle, info, residual)
end

function cusolverDnXgesvdjGetSweeps(handle, info, executed_sweeps)
    @check ccall((:cusolverDnXgesvdjGetSweeps, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, gesvdjInfo_t, Ptr{Cint}),
                 handle, info, executed_sweeps)
end

function cusolverDnSgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                             lwork, params, batchSize)
    @check ccall((:cusolverDnSgesvdjBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, Ptr{Cint},
                  gesvdjInfo_t, Cint),
                 handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
end

function cusolverDnDgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                             lwork, params, batchSize)
    @check ccall((:cusolverDnDgesvdjBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, Ptr{Cint},
                  gesvdjInfo_t, Cint),
                 handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
end

function cusolverDnCgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                             lwork, params, batchSize)
    @check ccall((:cusolverDnCgesvdjBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{cuComplex},
                  Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  Ptr{Cint}, gesvdjInfo_t, Cint),
                 handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
end

function cusolverDnZgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
                                             lwork, params, batchSize)
    @check ccall((:cusolverDnZgesvdjBatched_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}, gesvdjInfo_t, Cint),
                 handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
end

function cusolverDnSgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work,
                                  lwork, info, params, batchSize)
    @check ccall((:cusolverDnSgesvdjBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat},
                  Cint, CuPtr{Cint}, gesvdjInfo_t, Cint),
                 handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params,
                 batchSize)
end

function cusolverDnDgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work,
                                  lwork, info, params, batchSize)
    @check ccall((:cusolverDnDgesvdjBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cint}, gesvdjInfo_t, Cint),
                 handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params,
                 batchSize)
end

function cusolverDnCgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work,
                                  lwork, info, params, batchSize)
    @check ccall((:cusolverDnCgesvdjBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, CuPtr{cuComplex},
                  Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cint}, gesvdjInfo_t, Cint),
                 handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params,
                 batchSize)
end

function cusolverDnZgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work,
                                  lwork, info, params, batchSize)
    @check ccall((:cusolverDnZgesvdjBatched, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{Cint}, gesvdjInfo_t, Cint),
                 handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params,
                 batchSize)
end

function cusolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                      lwork, params)
    @check ccall((:cusolverDnSgesvdj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint, CuPtr{Cfloat},
                  Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, Ptr{Cint},
                  gesvdjInfo_t),
                 handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
end

function cusolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                      lwork, params)
    @check ccall((:cusolverDnDgesvdj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint, CuPtr{Cdouble},
                  Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  Ptr{Cint}, gesvdjInfo_t),
                 handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
end

function cusolverDnCgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                      lwork, params)
    @check ccall((:cusolverDnCgesvdj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, Ptr{Cint}, gesvdjInfo_t),
                 handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
end

function cusolverDnZgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                                      lwork, params)
    @check ccall((:cusolverDnZgesvdj_bufferSize, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cdouble}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{cuDoubleComplex}, Cint, Ptr{Cint}, gesvdjInfo_t),
                 handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
end

function cusolverDnSgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work,
                           lwork, info, params)
    @check ccall((:cusolverDnSgesvdj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint, CuPtr{Cfloat},
                  Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cint}, gesvdjInfo_t),
                 handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                 params)
end

function cusolverDnDgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work,
                           lwork, info, params)
    @check ccall((:cusolverDnDgesvdj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint, CuPtr{Cdouble},
                  Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cint}, gesvdjInfo_t),
                 handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                 params)
end

function cusolverDnCgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work,
                           lwork, info, params)
    @check ccall((:cusolverDnCgesvdj, libcusolver), cusolverStatus_t,
                 (cusolverDnHandle_t, cusolverEigMode_t, Cint, Cint, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{Cfloat}, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint, CuPtr{Cint}, gesvdjInfo_t),
                 handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info,
                 params)
end

function cusolverDnZgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work,
                           lwork, info, params)
    @check ccall((:cusolverDnZgesvdj, libcusolver), cusolverStatus_t,
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
    @check ccall((:cusolverDnSgesvdaStridedBatched_bufferSize, libcusolver), cusolverStatus_t,
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
    @check ccall((:cusolverDnDgesvdaStridedBatched_bufferSize, libcusolver), cusolverStatus_t,
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
    @check ccall((:cusolverDnCgesvdaStridedBatched_bufferSize, libcusolver), cusolverStatus_t,
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
    @check ccall((:cusolverDnZgesvdaStridedBatched_bufferSize, libcusolver), cusolverStatus_t,
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
    @check ccall((:cusolverDnSgesvdaStridedBatched, libcusolver), cusolverStatus_t,
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
    @check ccall((:cusolverDnDgesvdaStridedBatched, libcusolver), cusolverStatus_t,
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
    @check ccall((:cusolverDnCgesvdaStridedBatched, libcusolver), cusolverStatus_t,
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
    @check ccall((:cusolverDnZgesvdaStridedBatched, libcusolver), cusolverStatus_t,
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
    @check ccall((:cusolverSpCreate, libcusolver), cusolverStatus_t,
                 (Ptr{cusolverSpHandle_t},),
                 handle)
end

function cusolverSpDestroy(handle)
    @check ccall((:cusolverSpDestroy, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t,),
                 handle)
end

function cusolverSpSetStream(handle, streamId)
    @check ccall((:cusolverSpSetStream, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, CuStream_t),
                 handle, streamId)
end

function cusolverSpGetStream(handle, streamId)
    @check ccall((:cusolverSpGetStream, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Ptr{CuStream_t}),
                 handle, streamId)
end

function cusolverSpXcsrissymHost(handle, m, nnzA, descrA, csrRowPtrA, csrEndPtrA,
                                 csrColIndA, issym)
    @check ccall((:cusolverSpXcsrissymHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                 handle, m, nnzA, descrA, csrRowPtrA, csrEndPtrA, csrColIndA, issym)
end

function cusolverSpScsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA,
                                 b, tol, reorder, x, singularity)
    @check ccall((:cusolverSpScsrlsvluHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cint}),
                 handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder,
                 x, singularity)
end

function cusolverSpDcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA,
                                 b, tol, reorder, x, singularity)
    @check ccall((:cusolverSpDcsrlsvluHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cdouble, Cint, Ptr{Cdouble},
                  Ptr{Cint}),
                 handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder,
                 x, singularity)
end

function cusolverSpCcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA,
                                 b, tol, reorder, x, singularity)
    @check ccall((:cusolverSpCcsrlsvluHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                  Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cfloat, Cint, Ptr{cuComplex},
                  Ptr{Cint}),
                 handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder,
                 x, singularity)
end

function cusolverSpZcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA,
                                 b, tol, reorder, x, singularity)
    @check ccall((:cusolverSpZcsrlsvluHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                  Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex},
                  Cdouble, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}),
                 handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder,
                 x, singularity)
end

function cusolverSpScsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol,
                             reorder, x, singularity)
    @check ccall((:cusolverSpScsrlsvqr, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cfloat}, Cfloat, Cint, CuPtr{Cfloat},
                  Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpDcsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol,
                             reorder, x, singularity)
    @check ccall((:cusolverSpDcsrlsvqr, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cdouble, Cint, CuPtr{Cdouble},
                  Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpCcsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol,
                             reorder, x, singularity)
    @check ccall((:cusolverSpCcsrlsvqr, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuComplex}, Cfloat, Cint,
                  CuPtr{cuComplex}, Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpZcsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol,
                             reorder, x, singularity)
    @check ccall((:cusolverSpZcsrlsvqr, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                  Cdouble, Cint, CuPtr{cuDoubleComplex}, Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpScsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                 b, tol, reorder, x, singularity)
    @check ccall((:cusolverSpScsrlsvqrHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cint}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder,
                 x, singularity)
end

function cusolverSpDcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                 b, tol, reorder, x, singularity)
    @check ccall((:cusolverSpDcsrlsvqrHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cdouble, Cint, Ptr{Cdouble},
                  Ptr{Cint}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder,
                 x, singularity)
end

function cusolverSpCcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                 b, tol, reorder, x, singularity)
    @check ccall((:cusolverSpCcsrlsvqrHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                  Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cfloat, Cint, Ptr{cuComplex},
                  Ptr{Cint}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder,
                 x, singularity)
end

function cusolverSpZcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                 b, tol, reorder, x, singularity)
    @check ccall((:cusolverSpZcsrlsvqrHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                  Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex},
                  Cdouble, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder,
                 x, singularity)
end

function cusolverSpScsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b,
                                   tol, reorder, x, singularity)
    @check ccall((:cusolverSpScsrlsvcholHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpDcsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b,
                                   tol, reorder, x, singularity)
    @check ccall((:cusolverSpDcsrlsvcholHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cdouble, Cint, Ptr{Cdouble},
                  Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpCcsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b,
                                   tol, reorder, x, singularity)
    @check ccall((:cusolverSpCcsrlsvcholHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                  Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cfloat, Cint, Ptr{cuComplex},
                  Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpZcsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b,
                                   tol, reorder, x, singularity)
    @check ccall((:cusolverSpZcsrlsvcholHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                  Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex},
                  Cdouble, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpScsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b,
                               tol, reorder, x, singularity)
    @check ccall((:cusolverSpScsrlsvchol, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cfloat}, Cfloat, Cint, CuPtr{Cfloat},
                  Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpDcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b,
                               tol, reorder, x, singularity)
    @check ccall((:cusolverSpDcsrlsvchol, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cdouble, Cint, CuPtr{Cdouble},
                  Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpCcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b,
                               tol, reorder, x, singularity)
    @check ccall((:cusolverSpCcsrlsvchol, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuComplex}, Cfloat, Cint,
                  CuPtr{cuComplex}, Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpZcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b,
                               tol, reorder, x, singularity)
    @check ccall((:cusolverSpZcsrlsvchol, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                  Cdouble, Cint, CuPtr{cuDoubleComplex}, Ptr{Cint}),
                 handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x,
                 singularity)
end

function cusolverSpScsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                  csrColIndA, b, tol, rankA, x, p, min_norm)
    @check ccall((:cusolverSpScsrlsqvqrHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cfloat, Ptr{Cint}, Ptr{Cfloat},
                  Ptr{Cint}, Ptr{Cfloat}),
                 handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA,
                 x, p, min_norm)
end

function cusolverSpDcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                  csrColIndA, b, tol, rankA, x, p, min_norm)
    @check ccall((:cusolverSpDcsrlsqvqrHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cdouble, Ptr{Cint}, Ptr{Cdouble},
                  Ptr{Cint}, Ptr{Cdouble}),
                 handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA,
                 x, p, min_norm)
end

function cusolverSpCcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                  csrColIndA, b, tol, rankA, x, p, min_norm)
    @check ccall((:cusolverSpCcsrlsqvqrHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cfloat, Ptr{Cint},
                  Ptr{cuComplex}, Ptr{Cint}, Ptr{Cfloat}),
                 handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA,
                 x, p, min_norm)
end

function cusolverSpZcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                  csrColIndA, b, tol, rankA, x, p, min_norm)
    @check ccall((:cusolverSpZcsrlsqvqrHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex},
                  Cdouble, Ptr{Cint}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cdouble}),
                 handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA,
                 x, p, min_norm)
end

function cusolverSpScsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                  mu0, x0, maxite, tol, mu, x)
    @check ccall((:cusolverSpScsreigvsiHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                  Ptr{Cint}, Ptr{Cint}, Cfloat, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat},
                  Ptr{Cfloat}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite,
                 tol, mu, x)
end

function cusolverSpDcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                  mu0, x0, maxite, tol, mu, x)
    @check ccall((:cusolverSpDcsreigvsiHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                  Ptr{Cint}, Ptr{Cint}, Cdouble, Ptr{Cdouble}, Cint, Cdouble, Ptr{Cdouble},
                  Ptr{Cdouble}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite,
                 tol, mu, x)
end

function cusolverSpCcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                  mu0, x0, maxite, tol, mu, x)
    @check ccall((:cusolverSpCcsreigvsiHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                  Ptr{Cint}, Ptr{Cint}, cuComplex, Ptr{cuComplex}, Cint, Cfloat,
                  Ptr{cuComplex}, Ptr{cuComplex}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite,
                 tol, mu, x)
end

function cusolverSpZcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                  mu0, x0, maxite, tol, mu, x)
    @check ccall((:cusolverSpZcsreigvsiHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                  Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, cuDoubleComplex,
                  Ptr{cuDoubleComplex}, Cint, Cdouble, Ptr{cuDoubleComplex},
                  Ptr{cuDoubleComplex}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite,
                 tol, mu, x)
end

function cusolverSpScsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0,
                              x0, maxite, eps, mu, x)
    @check ccall((:cusolverSpScsreigvsi, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, Cfloat, CuPtr{Cfloat}, Cint, Cfloat,
                  CuPtr{Cfloat}, CuPtr{Cfloat}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite,
                 eps, mu, x)
end

function cusolverSpDcsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0,
                              x0, maxite, eps, mu, x)
    @check ccall((:cusolverSpDcsreigvsi, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, Cdouble, CuPtr{Cdouble}, Cint, Cdouble,
                  CuPtr{Cdouble}, CuPtr{Cdouble}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite,
                 eps, mu, x)
end

function cusolverSpCcsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0,
                              x0, maxite, eps, mu, x)
    @check ccall((:cusolverSpCcsreigvsi, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, cuComplex, CuPtr{cuComplex}, Cint, Cfloat,
                  CuPtr{cuComplex}, CuPtr{cuComplex}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite,
                 eps, mu, x)
end

function cusolverSpZcsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0,
                              x0, maxite, eps, mu, x)
    @check ccall((:cusolverSpZcsreigvsi, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, cuDoubleComplex,
                  CuPtr{cuDoubleComplex}, Cint, Cdouble, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite,
                 eps, mu, x)
end

function cusolverSpScsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                left_bottom_corner, right_upper_corner, num_eigs)
    @check ccall((:cusolverSpScsreigsHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                  Ptr{Cint}, Ptr{Cint}, cuComplex, cuComplex, Ptr{Cint}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                 left_bottom_corner, right_upper_corner, num_eigs)
end

function cusolverSpDcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                left_bottom_corner, right_upper_corner, num_eigs)
    @check ccall((:cusolverSpDcsreigsHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                  Ptr{Cint}, Ptr{Cint}, cuDoubleComplex, cuDoubleComplex, Ptr{Cint}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                 left_bottom_corner, right_upper_corner, num_eigs)
end

function cusolverSpCcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                left_bottom_corner, right_upper_corner, num_eigs)
    @check ccall((:cusolverSpCcsreigsHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                  Ptr{Cint}, Ptr{Cint}, cuComplex, cuComplex, Ptr{Cint}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                 left_bottom_corner, right_upper_corner, num_eigs)
end

function cusolverSpZcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                left_bottom_corner, right_upper_corner, num_eigs)
    @check ccall((:cusolverSpZcsreigsHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                  Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, cuDoubleComplex,
                  cuDoubleComplex, Ptr{Cint}),
                 handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                 left_bottom_corner, right_upper_corner, num_eigs)
end

function cusolverSpXcsrsymrcmHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
    @check ccall((:cusolverSpXcsrsymrcmHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                  Ptr{Cint}, Ptr{Cint}),
                 handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
end

function cusolverSpXcsrsymmdqHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
    @check ccall((:cusolverSpXcsrsymmdqHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                  Ptr{Cint}, Ptr{Cint}),
                 handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
end

function cusolverSpXcsrsymamdHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
    @check ccall((:cusolverSpXcsrsymamdHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                  Ptr{Cint}, Ptr{Cint}),
                 handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
end

function cusolverSpXcsrmetisndHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA,
                                   options, p)
    @check ccall((:cusolverSpXcsrmetisndHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                  Ptr{Cint}, Ptr{Int64}, Ptr{Cint}),
                 handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, options, p)
end

function cusolverSpScsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P,
                               numnz)
    @check ccall((:cusolverSpScsrzfdHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                 handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
end

function cusolverSpDcsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P,
                               numnz)
    @check ccall((:cusolverSpDcsrzfdHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, Ptr{Cint}, Ptr{Cint}),
                 handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
end

function cusolverSpCcsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P,
                               numnz)
    @check ccall((:cusolverSpCcsrzfdHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                 handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
end

function cusolverSpZcsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P,
                               numnz)
    @check ccall((:cusolverSpZcsrzfdHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                  Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                 handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
end

function cusolverSpXcsrperm_bufferSizeHost(handle, m, n, nnzA, descrA, csrRowPtrA,
                                           csrColIndA, p, q, bufferSizeInBytes)
    @check ccall((:cusolverSpXcsrperm_bufferSizeHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Csize_t}),
                 handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q,
                 bufferSizeInBytes)
end

function cusolverSpXcsrpermHost(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q,
                                map, pBuffer)
    @check ccall((:cusolverSpXcsrpermHost, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cint},
                  Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}),
                 handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q, map, pBuffer)
end

function cusolverSpCreateCsrqrInfo(info)
    @check ccall((:cusolverSpCreateCsrqrInfo, libcusolver), cusolverStatus_t,
                 (Ptr{csrqrInfo_t},),
                 info)
end

function cusolverSpDestroyCsrqrInfo(info)
    @check ccall((:cusolverSpDestroyCsrqrInfo, libcusolver), cusolverStatus_t,
                 (csrqrInfo_t,),
                 info)
end

function cusolverSpXcsrqrAnalysisBatched(handle, m, n, nnzA, descrA, csrRowPtrA,
                                         csrColIndA, info)
    @check ccall((:cusolverSpXcsrqrAnalysisBatched, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cint},
                  CuPtr{Cint}, csrqrInfo_t),
                 handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
end

function cusolverSpScsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal, csrRowPtr,
                                           csrColInd, batchSize, info, internalDataInBytes,
                                           workspaceInBytes)
    @check ccall((:cusolverSpScsrqrBufferInfoBatched, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, Cint, csrqrInfo_t, Ptr{Csize_t}, Ptr{Csize_t}),
                 handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info,
                 internalDataInBytes, workspaceInBytes)
end

function cusolverSpDcsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal, csrRowPtr,
                                           csrColInd, batchSize, info, internalDataInBytes,
                                           workspaceInBytes)
    @check ccall((:cusolverSpDcsrqrBufferInfoBatched, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, csrqrInfo_t,
                  Ptr{Csize_t}, Ptr{Csize_t}),
                 handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info,
                 internalDataInBytes, workspaceInBytes)
end

function cusolverSpCcsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal, csrRowPtr,
                                           csrColInd, batchSize, info, internalDataInBytes,
                                           workspaceInBytes)
    @check ccall((:cusolverSpCcsrqrBufferInfoBatched, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, csrqrInfo_t,
                  Ptr{Csize_t}, Ptr{Csize_t}),
                 handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info,
                 internalDataInBytes, workspaceInBytes)
end

function cusolverSpZcsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal, csrRowPtr,
                                           csrColInd, batchSize, info, internalDataInBytes,
                                           workspaceInBytes)
    @check ccall((:cusolverSpZcsrqrBufferInfoBatched, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, csrqrInfo_t,
                  Ptr{Csize_t}, Ptr{Csize_t}),
                 handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info,
                 internalDataInBytes, workspaceInBytes)
end

function cusolverSpScsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                   csrColIndA, b, x, batchSize, info, pBuffer)
    @check ccall((:cusolverSpScsrqrsvBatched, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                  csrqrInfo_t, CuPtr{Cvoid}),
                 handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x,
                 batchSize, info, pBuffer)
end

function cusolverSpDcsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                   csrColIndA, b, x, batchSize, info, pBuffer)
    @check ccall((:cusolverSpDcsrqrsvBatched, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, CuPtr{Cdouble},
                  Cint, csrqrInfo_t, CuPtr{Cvoid}),
                 handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x,
                 batchSize, info, pBuffer)
end

function cusolverSpCcsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                   csrColIndA, b, x, batchSize, info, pBuffer)
    @check ccall((:cusolverSpCcsrqrsvBatched, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint, csrqrInfo_t, CuPtr{Cvoid}),
                 handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x,
                 batchSize, info, pBuffer)
end

function cusolverSpZcsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA,
                                   csrColIndA, b, x, batchSize, info, pBuffer)
    @check ccall((:cusolverSpZcsrqrsvBatched, libcusolver), cusolverStatus_t,
                 (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, csrqrInfo_t, CuPtr{Cvoid}),
                 handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x,
                 batchSize, info, pBuffer)
end
