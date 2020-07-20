# Julia wrapper for header: cublasMg.h
# Automatically generated using Clang.jl

@checked function cublasMgGemm(handle, transA, transB, alpha, descA, A, llda, descB, B, lldb, beta, descC, C, lldc, descD, D, lldd, computeType, workspace, lwork, streams)
    initialize_api()
    @runtime_ccall((:cublasMgGemm, libcublasmg()), cublasStatus_t, (cublasMgHandle_t, cublasOperation_t, cublasOperation_t, Ptr{Cvoid}, cudaLibMgMatrixDesc_t, Ptr{PtrOrCuPtr{Cvoid}}, Ptr{Int64}, cudaLibMgMatrixDesc_t, Ptr{PtrOrCuPtr{Cvoid}}, Ptr{Int64}, Ptr{Cvoid}, cudaLibMgMatrixDesc_t, Ptr{PtrOrCuPtr{Cvoid}}, Ptr{Int64}, cudaLibMgMatrixDesc_t, Ptr{PtrOrCuPtr{Cvoid}}, Ptr{Int64}, cudaDataType_t, Ptr{CuPtr{Cvoid}}, Ptr{Csize_t}, Ptr{CUstream}), handle, transA, transB, alpha, descA, A, llda, descB, B, lldb, beta, descC, C, lldc, descD, D, lldd, computeType, workspace, lwork, streams)
end

@checked function cublasMgGemmWorkspace(handle, transA, transB, alpha, descA, A, llda, descB, B, lldb, beta, descC, C, lldc, descD, D, lldd, computeType, workspace, lwork)
    initialize_api()
    @runtime_ccall((:cublasMgGemmWorkspace, libcublasmg()), cublasStatus_t, (cublasMgHandle_t, cublasOperation_t, cublasOperation_t, Ptr{Cvoid}, cudaLibMgMatrixDesc_t, Ptr{PtrOrCuPtr{Cvoid}}, Ptr{Int64}, cudaLibMgMatrixDesc_t, Ptr{PtrOrCuPtr{Cvoid}}, Ptr{Int64}, Ptr{Cvoid}, cudaLibMgMatrixDesc_t, Ptr{PtrOrCuPtr{Cvoid}}, Ptr{Int64}, cudaLibMgMatrixDesc_t, Ptr{PtrOrCuPtr{Cvoid}}, Ptr{Int64}, cudaDataType_t, Ptr{CuPtr{Cvoid}}, Ptr{Csize_t}), handle, transA, transB, alpha, descA, A, llda, descB, B, lldb, beta, descC, C, lldc, descD, D, lldd, computeType, workspace, lwork)
end

function cublasMgCreate(handle)
    initialize_api()
    @runtime_ccall((:cublasMgCreate, libcublasmg()), cublasStatus_t, (Ptr{cublasMgHandle_t},), handle)
end

function cublasMgDestroy(handle)
    initialize_api()
    @runtime_ccall((:cublasMgDestroy, libcublasmg()), cublasStatus_t, (cublasMgHandle_t,), handle)
end

@checked function cublasMgDeviceSelect(handle, nbDevices, deviceIds)
    initialize_api()
    @runtime_ccall((:cublasMgDeviceSelect, libcublasmg()), cublasStatus_t, (cublasMgHandle_t, Cint, Ptr{Cint}), handle, nbDevices, deviceIds)
end

function cublasMgDeviceCount(handle, nbDevices)
    initialize_api()
    @runtime_ccall((:cublasMgDeviceCount, libcublasmg()), cublasStatus_t, (cublasMgHandle_t, Ptr{Cint}), handle, nbDevices)
end

function cublasMgGetVersion()
    initialize_api()
    @runtime_ccall((:cublasMgGetVersion, libcublasmg()), Csize_t, ())
end

function cublasMgGetCudartVersion()
    initialize_api()
    @runtime_ccall((:cublasMgGetCudartVersion, libcublasmg()), Csize_t, ())
end
