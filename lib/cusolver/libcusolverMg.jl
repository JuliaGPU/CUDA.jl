using CEnum

mutable struct cusolverMgContext end

const cusolverMgHandle_t = Ptr{cusolverMgContext}

@cenum cusolverMgGridMapping_t::UInt32 begin
    CUDALIBMG_GRID_MAPPING_ROW_MAJOR = 1
    CUDALIBMG_GRID_MAPPING_COL_MAJOR = 0
end

const cudaLibMgGrid_t = Ptr{Cvoid}

const cudaLibMgMatrixDesc_t = Ptr{Cvoid}

@checked function cusolverMgCreate(handle)
        initialize_context()
        ccall((:cusolverMgCreate, libcusolverMg), cusolverStatus_t, (Ptr{cusolverMgHandle_t},), handle)
    end

@checked function cusolverMgDestroy(handle)
        initialize_context()
        ccall((:cusolverMgDestroy, libcusolverMg), cusolverStatus_t, (cusolverMgHandle_t,), handle)
    end

@checked function cusolverMgDeviceSelect(handle, nbDevices, deviceId)
        initialize_context()
        ccall((:cusolverMgDeviceSelect, libcusolverMg), cusolverStatus_t, (cusolverMgHandle_t, Cint, Ptr{Cint}), handle, nbDevices, deviceId)
    end

@checked function cusolverMgCreateDeviceGrid(grid, numRowDevices, numColDevices, deviceId, mapping)
        initialize_context()
        ccall((:cusolverMgCreateDeviceGrid, libcusolverMg), cusolverStatus_t, (Ptr{cudaLibMgGrid_t}, Int32, Int32, Ptr{Int32}, cusolverMgGridMapping_t), grid, numRowDevices, numColDevices, deviceId, mapping)
    end

@checked function cusolverMgDestroyGrid(grid)
        initialize_context()
        ccall((:cusolverMgDestroyGrid, libcusolverMg), cusolverStatus_t, (cudaLibMgGrid_t,), grid)
    end

@checked function cusolverMgCreateMatrixDesc(desc, numRows, numCols, rowBlockSize, colBlockSize, dataType, grid)
        initialize_context()
        ccall((:cusolverMgCreateMatrixDesc, libcusolverMg), cusolverStatus_t, (Ptr{cudaLibMgMatrixDesc_t}, Int64, Int64, Int64, Int64, cudaDataType, cudaLibMgGrid_t), desc, numRows, numCols, rowBlockSize, colBlockSize, dataType, grid)
    end

@checked function cusolverMgDestroyMatrixDesc(desc)
        initialize_context()
        ccall((:cusolverMgDestroyMatrixDesc, libcusolverMg), cusolverStatus_t, (cudaLibMgMatrixDesc_t,), desc)
    end

@checked function cusolverMgSyevd_bufferSize(handle, jobz, uplo, N, array_d_A, IA, JA, descrA, W, dataTypeW, computeType, lwork)
        initialize_context()
        ccall((:cusolverMgSyevd_bufferSize, libcusolverMg), cusolverStatus_t, (cusolverMgHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint, Ptr{CuPtr{Cvoid}}, Cint, Cint, cudaLibMgMatrixDesc_t, Ptr{Cvoid}, cudaDataType, cudaDataType, Ptr{Int64}), handle, jobz, uplo, N, array_d_A, IA, JA, descrA, W, dataTypeW, computeType, lwork)
    end

@checked function cusolverMgSyevd(handle, jobz, uplo, N, array_d_A, IA, JA, descrA, W, dataTypeW, computeType, array_d_work, lwork, info)
        initialize_context()
        ccall((:cusolverMgSyevd, libcusolverMg), cusolverStatus_t, (cusolverMgHandle_t, cusolverEigMode_t, cublasFillMode_t, Cint, Ptr{CuPtr{Cvoid}}, Cint, Cint, cudaLibMgMatrixDesc_t, Ptr{Cvoid}, cudaDataType, cudaDataType, Ptr{CuPtr{Cvoid}}, Int64, Ptr{Cint}), handle, jobz, uplo, N, array_d_A, IA, JA, descrA, W, dataTypeW, computeType, array_d_work, lwork, info)
    end

@checked function cusolverMgGetrf_bufferSize(handle, M, N, array_d_A, IA, JA, descrA, array_d_IPIV, computeType, lwork)
        initialize_context()
        ccall((:cusolverMgGetrf_bufferSize, libcusolverMg), cusolverStatus_t, (cusolverMgHandle_t, Cint, Cint, Ptr{CuPtr{Cvoid}}, Cint, Cint, cudaLibMgMatrixDesc_t, Ptr{CuPtr{Cint}}, cudaDataType, Ptr{Int64}), handle, M, N, array_d_A, IA, JA, descrA, array_d_IPIV, computeType, lwork)
    end

@checked function cusolverMgGetrf(handle, M, N, array_d_A, IA, JA, descrA, array_d_IPIV, computeType, array_d_work, lwork, info)
        initialize_context()
        ccall((:cusolverMgGetrf, libcusolverMg), cusolverStatus_t, (cusolverMgHandle_t, Cint, Cint, Ptr{CuPtr{Cvoid}}, Cint, Cint, cudaLibMgMatrixDesc_t, Ptr{CuPtr{Cint}}, cudaDataType, Ptr{CuPtr{Cvoid}}, Int64, Ptr{Cint}), handle, M, N, array_d_A, IA, JA, descrA, array_d_IPIV, computeType, array_d_work, lwork, info)
    end

@checked function cusolverMgGetrs_bufferSize(handle, TRANS, N, NRHS, array_d_A, IA, JA, descrA, array_d_IPIV, array_d_B, IB, JB, descrB, computeType, lwork)
        initialize_context()
        ccall((:cusolverMgGetrs_bufferSize, libcusolverMg), cusolverStatus_t, (cusolverMgHandle_t, cublasOperation_t, Cint, Cint, Ptr{CuPtr{Cvoid}}, Cint, Cint, cudaLibMgMatrixDesc_t, Ptr{CuPtr{Cint}}, Ptr{CuPtr{Cvoid}}, Cint, Cint, cudaLibMgMatrixDesc_t, cudaDataType, Ptr{Int64}), handle, TRANS, N, NRHS, array_d_A, IA, JA, descrA, array_d_IPIV, array_d_B, IB, JB, descrB, computeType, lwork)
    end

@checked function cusolverMgGetrs(handle, TRANS, N, NRHS, array_d_A, IA, JA, descrA, array_d_IPIV, array_d_B, IB, JB, descrB, computeType, array_d_work, lwork, info)
        initialize_context()
        ccall((:cusolverMgGetrs, libcusolverMg), cusolverStatus_t, (cusolverMgHandle_t, cublasOperation_t, Cint, Cint, Ptr{CuPtr{Cvoid}}, Cint, Cint, cudaLibMgMatrixDesc_t, Ptr{CuPtr{Cint}}, Ptr{CuPtr{Cvoid}}, Cint, Cint, cudaLibMgMatrixDesc_t, cudaDataType, Ptr{CuPtr{Cvoid}}, Int64, Ptr{Cint}), handle, TRANS, N, NRHS, array_d_A, IA, JA, descrA, array_d_IPIV, array_d_B, IB, JB, descrB, computeType, array_d_work, lwork, info)
    end

@checked function cusolverMgPotrf_bufferSize(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, lwork)
        initialize_context()
        ccall((:cusolverMgPotrf_bufferSize, libcusolverMg), cusolverStatus_t, (cusolverMgHandle_t, cublasFillMode_t, Cint, Ptr{CuPtr{Cvoid}}, Cint, Cint, cudaLibMgMatrixDesc_t, cudaDataType, Ptr{Int64}), handle, uplo, N, array_d_A, IA, JA, descrA, computeType, lwork)
    end

@checked function cusolverMgPotrf(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, array_d_work, lwork, h_info)
        initialize_context()
        ccall((:cusolverMgPotrf, libcusolverMg), cusolverStatus_t, (cusolverMgHandle_t, cublasFillMode_t, Cint, Ptr{CuPtr{Cvoid}}, Cint, Cint, cudaLibMgMatrixDesc_t, cudaDataType, Ptr{CuPtr{Cvoid}}, Int64, Ptr{Cint}), handle, uplo, N, array_d_A, IA, JA, descrA, computeType, array_d_work, lwork, h_info)
    end

@checked function cusolverMgPotrs_bufferSize(handle, uplo, n, nrhs, array_d_A, IA, JA, descrA, array_d_B, IB, JB, descrB, computeType, lwork)
        initialize_context()
        ccall((:cusolverMgPotrs_bufferSize, libcusolverMg), cusolverStatus_t, (cusolverMgHandle_t, cublasFillMode_t, Cint, Cint, Ptr{CuPtr{Cvoid}}, Cint, Cint, cudaLibMgMatrixDesc_t, Ptr{CuPtr{Cvoid}}, Cint, Cint, cudaLibMgMatrixDesc_t, cudaDataType, Ptr{Int64}), handle, uplo, n, nrhs, array_d_A, IA, JA, descrA, array_d_B, IB, JB, descrB, computeType, lwork)
    end

@checked function cusolverMgPotrs(handle, uplo, n, nrhs, array_d_A, IA, JA, descrA, array_d_B, IB, JB, descrB, computeType, array_d_work, lwork, h_info)
        initialize_context()
        ccall((:cusolverMgPotrs, libcusolverMg), cusolverStatus_t, (cusolverMgHandle_t, cublasFillMode_t, Cint, Cint, Ptr{CuPtr{Cvoid}}, Cint, Cint, cudaLibMgMatrixDesc_t, Ptr{CuPtr{Cvoid}}, Cint, Cint, cudaLibMgMatrixDesc_t, cudaDataType, Ptr{CuPtr{Cvoid}}, Int64, Ptr{Cint}), handle, uplo, n, nrhs, array_d_A, IA, JA, descrA, array_d_B, IB, JB, descrB, computeType, array_d_work, lwork, h_info)
    end

@checked function cusolverMgPotri_bufferSize(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, lwork)
        initialize_context()
        ccall((:cusolverMgPotri_bufferSize, libcusolverMg), cusolverStatus_t, (cusolverMgHandle_t, cublasFillMode_t, Cint, Ptr{CuPtr{Cvoid}}, Cint, Cint, cudaLibMgMatrixDesc_t, cudaDataType, Ptr{Int64}), handle, uplo, N, array_d_A, IA, JA, descrA, computeType, lwork)
    end

@checked function cusolverMgPotri(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, array_d_work, lwork, h_info)
        initialize_context()
        ccall((:cusolverMgPotri, libcusolverMg), cusolverStatus_t, (cusolverMgHandle_t, cublasFillMode_t, Cint, Ptr{CuPtr{Cvoid}}, Cint, Cint, cudaLibMgMatrixDesc_t, cudaDataType, Ptr{CuPtr{Cvoid}}, Int64, Ptr{Cint}), handle, uplo, N, array_d_A, IA, JA, descrA, computeType, array_d_work, lwork, h_info)
    end

