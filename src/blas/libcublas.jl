# Julia wrapper for header: cublas_v2.h
# Automatically generated using Clang.jl

# Julia wrapper for header: cublas_api.h
# Automatically generated using Clang.jl


function cublasCreate_v2(handle)
    @check @runtime_ccall((:cublasCreate_v2, libcublas[]), cublasStatus_t,
                 (Ptr{cublasHandle_t},),
                 handle)
end

function cublasDestroy_v2(handle)
    @check @runtime_ccall((:cublasDestroy_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t,),
                 handle)
end

function cublasGetVersion_v2(handle, version)
    @check @runtime_ccall((:cublasGetVersion_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Ptr{Cint}),
                 handle, version)
end

function cublasGetProperty(type, value)
    @check @runtime_ccall((:cublasGetProperty, libcublas[]), cublasStatus_t,
                 (libraryPropertyType, Ptr{Cint}),
                 type, value)
end

function cublasGetCudartVersion()
    @runtime_ccall((:cublasGetCudartVersion, libcublas[]), Csize_t, ())
end

function cublasSetStream_v2(handle, streamId)
    @check @runtime_ccall((:cublasSetStream_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, CUstream),
                 handle, streamId)
end

function cublasGetStream_v2(handle, streamId)
    @check @runtime_ccall((:cublasGetStream_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Ptr{CUstream}),
                 handle, streamId)
end

function cublasGetPointerMode_v2(handle, mode)
    @check @runtime_ccall((:cublasGetPointerMode_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Ptr{cublasPointerMode_t}),
                 handle, mode)
end

function cublasSetPointerMode_v2(handle, mode)
    @check @runtime_ccall((:cublasSetPointerMode_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasPointerMode_t),
                 handle, mode)
end

function cublasGetAtomicsMode(handle, mode)
    @check @runtime_ccall((:cublasGetAtomicsMode, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Ptr{cublasAtomicsMode_t}),
                 handle, mode)
end

function cublasSetAtomicsMode(handle, mode)
    @check @runtime_ccall((:cublasSetAtomicsMode, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasAtomicsMode_t),
                 handle, mode)
end

function cublasGetMathMode(handle, mode)
    @check @runtime_ccall((:cublasGetMathMode, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Ptr{cublasMath_t}),
                 handle, mode)
end

function cublasSetMathMode(handle, mode)
    @check @runtime_ccall((:cublasSetMathMode, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasMath_t),
                 handle, mode)
end

function cublasLoggerConfigure(logIsOn, logToStdOut, logToStdErr, logFileName)
    @check @runtime_ccall((:cublasLoggerConfigure, libcublas[]), cublasStatus_t,
                 (Cint, Cint, Cint, Cstring),
                 logIsOn, logToStdOut, logToStdErr, logFileName)
end

function cublasSetLoggerCallback(userCallback)
    @check @runtime_ccall((:cublasSetLoggerCallback, libcublas[]), cublasStatus_t,
                 (cublasLogCallback,),
                 userCallback)
end

function cublasGetLoggerCallback(userCallback)
    @check @runtime_ccall((:cublasGetLoggerCallback, libcublas[]), cublasStatus_t,
                 (Ptr{cublasLogCallback},),
                 userCallback)
end

function cublasSetVector(n, elemSize, x, incx, devicePtr, incy)
    @check @runtime_ccall((:cublasSetVector, libcublas[]), cublasStatus_t,
                 (Cint, Cint, Ptr{Cvoid}, Cint, CuPtr{Cvoid}, Cint),
                 n, elemSize, x, incx, devicePtr, incy)
end

function cublasGetVector(n, elemSize, x, incx, y, incy)
    @check @runtime_ccall((:cublasGetVector, libcublas[]), cublasStatus_t,
                 (Cint, Cint, CuPtr{Cvoid}, Cint, Ptr{Cvoid}, Cint),
                 n, elemSize, x, incx, y, incy)
end

function cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb)
    @check @runtime_ccall((:cublasSetMatrix, libcublas[]), cublasStatus_t,
                 (Cint, Cint, Cint, Ptr{Cvoid}, Cint, CuPtr{Cvoid}, Cint),
                 rows, cols, elemSize, A, lda, B, ldb)
end

function cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb)
    @check @runtime_ccall((:cublasGetMatrix, libcublas[]), cublasStatus_t,
                 (Cint, Cint, Cint, CuPtr{Cvoid}, Cint, Ptr{Cvoid}, Cint),
                 rows, cols, elemSize, A, lda, B, ldb)
end

function cublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, stream)
    @check @runtime_ccall((:cublasSetVectorAsync, libcublas[]), cublasStatus_t,
                 (Cint, Cint, Ptr{Cvoid}, Cint, CuPtr{Cvoid}, Cint, CUstream),
                 n, elemSize, hostPtr, incx, devicePtr, incy, stream)
end

function cublasGetVectorAsync(n, elemSize, devicePtr, incx, hostPtr, incy, stream)
    @check @runtime_ccall((:cublasGetVectorAsync, libcublas[]), cublasStatus_t,
                 (Cint, Cint, CuPtr{Cvoid}, Cint, Ptr{Cvoid}, Cint, CUstream),
                 n, elemSize, devicePtr, incx, hostPtr, incy, stream)
end

function cublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream)
    @check @runtime_ccall((:cublasSetMatrixAsync, libcublas[]), cublasStatus_t,
                 (Cint, Cint, Cint, Ptr{Cvoid}, Cint, CuPtr{Cvoid}, Cint, CUstream),
                 rows, cols, elemSize, A, lda, B, ldb, stream)
end

function cublasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream)
    @check @runtime_ccall((:cublasGetMatrixAsync, libcublas[]), cublasStatus_t,
                 (Cint, Cint, Cint, CuPtr{Cvoid}, Cint, Ptr{Cvoid}, Cint, CUstream),
                 rows, cols, elemSize, A, lda, B, ldb, stream)
end

function cublasXerbla(srName, info)
    @runtime_ccall((:cublasXerbla, libcublas[]), Cvoid,
          (Cstring, Cint),
          srName, info)
end

function cublasNrm2Ex(handle, n, x, xType, incx, result, resultType, executionType)
    @check @runtime_ccall((:cublasNrm2Ex, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cvoid}, cudaDataType, Cint,
                  PtrOrCuPtr{Cvoid}, cudaDataType, cudaDataType),
                 handle, n, x, xType, incx, result, resultType, executionType)
end

function cublasSnrm2_v2(handle, n, x, incx, result)
    @check @runtime_ccall((:cublasSnrm2_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cfloat}, Cint, PtrOrCuPtr{Cfloat}),
                 handle, n, x, incx, result)
end

function cublasDnrm2_v2(handle, n, x, incx, result)
    @check @runtime_ccall((:cublasDnrm2_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cdouble}, Cint, PtrOrCuPtr{Cdouble}),
                 handle, n, x, incx, result)
end

function cublasScnrm2_v2(handle, n, x, incx, result)
    @check @runtime_ccall((:cublasScnrm2_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuComplex}, Cint, PtrOrCuPtr{Cfloat}),
                 handle, n, x, incx, result)
end

function cublasDznrm2_v2(handle, n, x, incx, result)
    @check @runtime_ccall((:cublasDznrm2_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{Cdouble}),
                 handle, n, x, incx, result)
end

function cublasDotEx(handle, n, x, xType, incx, y, yType, incy, result, resultType,
                     executionType)
    @check @runtime_ccall((:cublasDotEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cvoid}, cudaDataType, Cint, CuPtr{Cvoid},
                  cudaDataType, Cint, PtrOrCuPtr{Cvoid}, cudaDataType, cudaDataType),
                 handle, n, x, xType, incx, y, yType, incy, result, resultType,
                 executionType)
end

function cublasDotcEx(handle, n, x, xType, incx, y, yType, incy, result, resultType,
                      executionType)
    @check @runtime_ccall((:cublasDotcEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cvoid}, cudaDataType, Cint, CuPtr{Cvoid},
                  cudaDataType, Cint, PtrOrCuPtr{Cvoid}, cudaDataType, cudaDataType),
                 handle, n, x, xType, incx, y, yType, incy, result, resultType,
                 executionType)
end

function cublasSdot_v2(handle, n, x, incx, y, incy, result)
    @check @runtime_ccall((:cublasSdot_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                  PtrOrCuPtr{Cfloat}),
                 handle, n, x, incx, y, incy, result)
end

function cublasDdot_v2(handle, n, x, incx, y, incy, result)
    @check @runtime_ccall((:cublasDdot_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  PtrOrCuPtr{Cdouble}),
                 handle, n, x, incx, y, incy, result)
end

function cublasCdotu_v2(handle, n, x, incx, y, incy, result)
    @check @runtime_ccall((:cublasCdotu_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  PtrOrCuPtr{cuComplex}),
                 handle, n, x, incx, y, incy, result)
end

function cublasCdotc_v2(handle, n, x, incx, y, incy, result)
    @check @runtime_ccall((:cublasCdotc_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  PtrOrCuPtr{cuComplex}),
                 handle, n, x, incx, y, incy, result)
end

function cublasZdotu_v2(handle, n, x, incx, y, incy, result)
    @check @runtime_ccall((:cublasZdotu_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{cuDoubleComplex}),
                 handle, n, x, incx, y, incy, result)
end

function cublasZdotc_v2(handle, n, x, incx, y, incy, result)
    @check @runtime_ccall((:cublasZdotc_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{cuDoubleComplex}),
                 handle, n, x, incx, y, incy, result)
end

function cublasScalEx(handle, n, alpha, alphaType, x, xType, incx, executionType)
    @check @runtime_ccall((:cublasScalEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, PtrOrCuPtr{Cvoid}, cudaDataType, CuPtr{Cvoid},
                  cudaDataType, Cint, cudaDataType),
                 handle, n, alpha, alphaType, x, xType, incx, executionType)
end

function cublasSscal_v2(handle, n, alpha, x, incx)
    @check @runtime_ccall((:cublasSscal_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Cint),
                 handle, n, alpha, x, incx)
end

function cublasDscal_v2(handle, n, alpha, x, incx)
    @check @runtime_ccall((:cublasDscal_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Cint),
                 handle, n, alpha, x, incx)
end

function cublasCscal_v2(handle, n, alpha, x, incx)
    @check @runtime_ccall((:cublasCscal_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint),
                 handle, n, alpha, x, incx)
end

function cublasCsscal_v2(handle, n, alpha, x, incx)
    @check @runtime_ccall((:cublasCsscal_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, PtrOrCuPtr{Cfloat}, CuPtr{cuComplex}, Cint),
                 handle, n, alpha, x, incx)
end

function cublasZscal_v2(handle, n, alpha, x, incx)
    @check @runtime_ccall((:cublasZscal_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, n, alpha, x, incx)
end

function cublasZdscal_v2(handle, n, alpha, x, incx)
    @check @runtime_ccall((:cublasZdscal_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, PtrOrCuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Cint),
                 handle, n, alpha, x, incx)
end

function cublasAxpyEx(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy,
                      executiontype)
    @check @runtime_ccall((:cublasAxpyEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, PtrOrCuPtr{Cvoid}, cudaDataType, CuPtr{Cvoid},
                  cudaDataType, Cint, CuPtr{Cvoid}, cudaDataType, Cint, cudaDataType),
                 handle, n, alpha, alphaType, x, xType, incx, y, yType, incy,
                 executiontype)
end

function cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy)
    @check @runtime_ccall((:cublasSaxpy_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint),
                 handle, n, alpha, x, incx, y, incy)
end

function cublasDaxpy_v2(handle, n, alpha, x, incx, y, incy)
    @check @runtime_ccall((:cublasDaxpy_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint),
                 handle, n, alpha, x, incx, y, incy)
end

function cublasCaxpy_v2(handle, n, alpha, x, incx, y, incy)
    @check @runtime_ccall((:cublasCaxpy_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint),
                 handle, n, alpha, x, incx, y, incy)
end

function cublasZaxpy_v2(handle, n, alpha, x, incx, y, incy)
    @check @runtime_ccall((:cublasZaxpy_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint),
                 handle, n, alpha, x, incx, y, incy)
end

function cublasCopyEx(handle, n, x, xType, incx, y, yType, incy)
    @check @runtime_ccall((:cublasCopyEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cvoid}, cudaDataType, Cint, CuPtr{Cvoid},
                  cudaDataType, Cint),
                 handle, n, x, xType, incx, y, yType, incy)
end

function cublasScopy_v2(handle, n, x, incx, y, incy)
    @check @runtime_ccall((:cublasScopy_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint),
                 handle, n, x, incx, y, incy)
end

function cublasDcopy_v2(handle, n, x, incx, y, incy)
    @check @runtime_ccall((:cublasDcopy_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint),
                 handle, n, x, incx, y, incy)
end

function cublasCcopy_v2(handle, n, x, incx, y, incy)
    @check @runtime_ccall((:cublasCcopy_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint),
                 handle, n, x, incx, y, incy)
end

function cublasZcopy_v2(handle, n, x, incx, y, incy)
    @check @runtime_ccall((:cublasZcopy_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, n, x, incx, y, incy)
end

function cublasSswap_v2(handle, n, x, incx, y, incy)
    @check @runtime_ccall((:cublasSswap_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint),
                 handle, n, x, incx, y, incy)
end

function cublasDswap_v2(handle, n, x, incx, y, incy)
    @check @runtime_ccall((:cublasDswap_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint),
                 handle, n, x, incx, y, incy)
end

function cublasCswap_v2(handle, n, x, incx, y, incy)
    @check @runtime_ccall((:cublasCswap_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint),
                 handle, n, x, incx, y, incy)
end

function cublasZswap_v2(handle, n, x, incx, y, incy)
    @check @runtime_ccall((:cublasZswap_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, n, x, incx, y, incy)
end

function cublasSwapEx(handle, n, x, xType, incx, y, yType, incy)
    @check @runtime_ccall((:cublasSwapEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cvoid}, cudaDataType, Cint, CuPtr{Cvoid},
                  cudaDataType, Cint),
                 handle, n, x, xType, incx, y, yType, incy)
end

function cublasIsamax_v2(handle, n, x, incx, result)
    @check @runtime_ccall((:cublasIsamax_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cfloat}, Cint, PtrOrCuPtr{Cint}),
                 handle, n, x, incx, result)
end

function cublasIdamax_v2(handle, n, x, incx, result)
    @check @runtime_ccall((:cublasIdamax_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cdouble}, Cint, PtrOrCuPtr{Cint}),
                 handle, n, x, incx, result)
end

function cublasIcamax_v2(handle, n, x, incx, result)
    @check @runtime_ccall((:cublasIcamax_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuComplex}, Cint, PtrOrCuPtr{Cint}),
                 handle, n, x, incx, result)
end

function cublasIzamax_v2(handle, n, x, incx, result)
    @check @runtime_ccall((:cublasIzamax_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{Cint}),
                 handle, n, x, incx, result)
end

function cublasIamaxEx(handle, n, x, xType, incx, result)
    @check @runtime_ccall((:cublasIamaxEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cvoid}, cudaDataType, Cint, PtrOrCuPtr{Cint}),
                 handle, n, x, xType, incx, result)
end

function cublasIsamin_v2(handle, n, x, incx, result)
    @check @runtime_ccall((:cublasIsamin_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cfloat}, Cint, PtrOrCuPtr{Cint}),
                 handle, n, x, incx, result)
end

function cublasIdamin_v2(handle, n, x, incx, result)
    @check @runtime_ccall((:cublasIdamin_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cdouble}, Cint, PtrOrCuPtr{Cint}),
                 handle, n, x, incx, result)
end

function cublasIcamin_v2(handle, n, x, incx, result)
    @check @runtime_ccall((:cublasIcamin_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuComplex}, Cint, PtrOrCuPtr{Cint}),
                 handle, n, x, incx, result)
end

function cublasIzamin_v2(handle, n, x, incx, result)
    @check @runtime_ccall((:cublasIzamin_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{Cint}),
                 handle, n, x, incx, result)
end

function cublasIaminEx(handle, n, x, xType, incx, result)
    @check @runtime_ccall((:cublasIaminEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cvoid}, cudaDataType, Cint, PtrOrCuPtr{Cint}),
                 handle, n, x, xType, incx, result)
end

function cublasAsumEx(handle, n, x, xType, incx, result, resultType, executiontype)
    @check @runtime_ccall((:cublasAsumEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cvoid}, cudaDataType, Cint,
                  PtrOrCuPtr{Cvoid}, cudaDataType, cudaDataType),
                 handle, n, x, xType, incx, result, resultType, executiontype)
end

function cublasSasum_v2(handle, n, x, incx, result)
    @check @runtime_ccall((:cublasSasum_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cfloat}, Cint, PtrOrCuPtr{Cfloat}),
                 handle, n, x, incx, result)
end

function cublasDasum_v2(handle, n, x, incx, result)
    @check @runtime_ccall((:cublasDasum_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cdouble}, Cint, PtrOrCuPtr{Cdouble}),
                 handle, n, x, incx, result)
end

function cublasScasum_v2(handle, n, x, incx, result)
    @check @runtime_ccall((:cublasScasum_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuComplex}, Cint, PtrOrCuPtr{Cfloat}),
                 handle, n, x, incx, result)
end

function cublasDzasum_v2(handle, n, x, incx, result)
    @check @runtime_ccall((:cublasDzasum_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{Cdouble}),
                 handle, n, x, incx, result)
end

function cublasSrot_v2(handle, n, x, incx, y, incy, c, s)
    @check @runtime_ccall((:cublasSrot_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                  PtrOrCuPtr{Cfloat}, PtrOrCuPtr{Cfloat}),
                 handle, n, x, incx, y, incy, c, s)
end

function cublasDrot_v2(handle, n, x, incx, y, incy, c, s)
    @check @runtime_ccall((:cublasDrot_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  PtrOrCuPtr{Cdouble}, PtrOrCuPtr{Cdouble}),
                 handle, n, x, incx, y, incy, c, s)
end

function cublasCrot_v2(handle, n, x, incx, y, incy, c, s)
    @check @runtime_ccall((:cublasCrot_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  PtrOrCuPtr{Cfloat}, PtrOrCuPtr{cuComplex}),
                 handle, n, x, incx, y, incy, c, s)
end

function cublasCsrot_v2(handle, n, x, incx, y, incy, c, s)
    @check @runtime_ccall((:cublasCsrot_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  PtrOrCuPtr{Cfloat}, PtrOrCuPtr{Cfloat}),
                 handle, n, x, incx, y, incy, c, s)
end

function cublasZrot_v2(handle, n, x, incx, y, incy, c, s)
    @check @runtime_ccall((:cublasZrot_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{Cdouble},
                  PtrOrCuPtr{cuDoubleComplex}),
                 handle, n, x, incx, y, incy, c, s)
end

function cublasZdrot_v2(handle, n, x, incx, y, incy, c, s)
    @check @runtime_ccall((:cublasZdrot_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{Cdouble}, PtrOrCuPtr{Cdouble}),
                 handle, n, x, incx, y, incy, c, s)
end

function cublasRotEx(handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype)
    @check @runtime_ccall((:cublasRotEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cvoid}, cudaDataType, Cint, CuPtr{Cvoid},
                  cudaDataType, Cint, PtrOrCuPtr{Cvoid}, PtrOrCuPtr{Cvoid}, cudaDataType,
                  cudaDataType),
                 handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype)
end

function cublasSrotg_v2(handle, a, b, c, s)
    @check @runtime_ccall((:cublasSrotg_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, PtrOrCuPtr{Cfloat}, PtrOrCuPtr{Cfloat},
                  PtrOrCuPtr{Cfloat}, PtrOrCuPtr{Cfloat}),
                 handle, a, b, c, s)
end

function cublasDrotg_v2(handle, a, b, c, s)
    @check @runtime_ccall((:cublasDrotg_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, PtrOrCuPtr{Cdouble}, PtrOrCuPtr{Cdouble},
                  PtrOrCuPtr{Cdouble}, PtrOrCuPtr{Cdouble}),
                 handle, a, b, c, s)
end

function cublasCrotg_v2(handle, a, b, c, s)
    @check @runtime_ccall((:cublasCrotg_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, PtrOrCuPtr{cuComplex}, PtrOrCuPtr{cuComplex},
                  PtrOrCuPtr{Cfloat}, PtrOrCuPtr{cuComplex}),
                 handle, a, b, c, s)
end

function cublasZrotg_v2(handle, a, b, c, s)
    @check @runtime_ccall((:cublasZrotg_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, PtrOrCuPtr{cuDoubleComplex},
                  PtrOrCuPtr{cuDoubleComplex}, PtrOrCuPtr{Cdouble},
                  PtrOrCuPtr{cuDoubleComplex}),
                 handle, a, b, c, s)
end

function cublasRotgEx(handle, a, b, abType, c, s, csType, executiontype)
    @check @runtime_ccall((:cublasRotgEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, PtrOrCuPtr{Cvoid}, PtrOrCuPtr{Cvoid}, cudaDataType,
                  PtrOrCuPtr{Cvoid}, PtrOrCuPtr{Cvoid}, cudaDataType, cudaDataType),
                 handle, a, b, abType, c, s, csType, executiontype)
end

function cublasSrotm_v2(handle, n, x, incx, y, incy, param)
    @check @runtime_ccall((:cublasSrotm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                  PtrOrCuPtr{Cfloat}),
                 handle, n, x, incx, y, incy, param)
end

function cublasDrotm_v2(handle, n, x, incx, y, incy, param)
    @check @runtime_ccall((:cublasDrotm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  PtrOrCuPtr{Cdouble}),
                 handle, n, x, incx, y, incy, param)
end

function cublasRotmEx(handle, n, x, xType, incx, y, yType, incy, param, paramType,
                      executiontype)
    @check @runtime_ccall((:cublasRotmEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Cvoid}, cudaDataType, Cint, CuPtr{Cvoid},
                  cudaDataType, Cint, PtrOrCuPtr{Cvoid}, cudaDataType, cudaDataType),
                 handle, n, x, xType, incx, y, yType, incy, param, paramType,
                 executiontype)
end

function cublasSrotmg_v2(handle, d1, d2, x1, y1, param)
    @check @runtime_ccall((:cublasSrotmg_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, PtrOrCuPtr{Cfloat}, PtrOrCuPtr{Cfloat},
                  PtrOrCuPtr{Cfloat}, PtrOrCuPtr{Cfloat}, PtrOrCuPtr{Cfloat}),
                 handle, d1, d2, x1, y1, param)
end

function cublasDrotmg_v2(handle, d1, d2, x1, y1, param)
    @check @runtime_ccall((:cublasDrotmg_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, PtrOrCuPtr{Cdouble}, PtrOrCuPtr{Cdouble},
                  PtrOrCuPtr{Cdouble}, PtrOrCuPtr{Cdouble}, PtrOrCuPtr{Cdouble}),
                 handle, d1, d2, x1, y1, param)
end

function cublasRotmgEx(handle, d1, d1Type, d2, d2Type, x1, x1Type, y1, y1Type, param,
                       paramType, executiontype)
    @check @runtime_ccall((:cublasRotmgEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, PtrOrCuPtr{Cvoid}, cudaDataType, PtrOrCuPtr{Cvoid},
                  cudaDataType, PtrOrCuPtr{Cvoid}, cudaDataType, PtrOrCuPtr{Cvoid},
                  cudaDataType, PtrOrCuPtr{Cvoid}, cudaDataType, cudaDataType),
                 handle, d1, d1Type, d2, d2Type, x1, x1Type, y1, y1Type, param, paramType,
                 executiontype)
end

function cublasSgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasSgemv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, Cint, Cint, PtrOrCuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, PtrOrCuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint),
                 handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasDgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasDgemv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, Cint, Cint, PtrOrCuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, PtrOrCuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint),
                 handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasCgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasCgemv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, Cint, Cint, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint),
                 handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasZgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasZgemv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasSgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasSgbmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Cint),
                 handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasDgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasDgbmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Cint),
                 handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasCgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasCgbmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint),
                 handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasZgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasZgbmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasStrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    @check @runtime_ccall((:cublasStrmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint),
                 handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasDtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    @check @runtime_ccall((:cublasDtrmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint),
                 handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasCtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    @check @runtime_ccall((:cublasCtrmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint),
                 handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasZtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    @check @runtime_ccall((:cublasZtrmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasStbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    @check @runtime_ccall((:cublasStbmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint),
                 handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasDtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    @check @runtime_ccall((:cublasDtbmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint),
                 handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasCtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    @check @runtime_ccall((:cublasCtbmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint),
                 handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasZtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    @check @runtime_ccall((:cublasZtbmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasStpmv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    @check @runtime_ccall((:cublasStpmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint),
                 handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasDtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    @check @runtime_ccall((:cublasDtpmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint),
                 handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasCtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    @check @runtime_ccall((:cublasCtpmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, CuPtr{cuComplex}, CuPtr{cuComplex}, Cint),
                 handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasZtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    @check @runtime_ccall((:cublasZtpmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasStrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    @check @runtime_ccall((:cublasStrsv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint),
                 handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasDtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    @check @runtime_ccall((:cublasDtrsv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint),
                 handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasCtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    @check @runtime_ccall((:cublasCtrsv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint),
                 handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasZtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    @check @runtime_ccall((:cublasZtrsv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasStpsv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    @check @runtime_ccall((:cublasStpsv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint),
                 handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasDtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    @check @runtime_ccall((:cublasDtpsv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint),
                 handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasCtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    @check @runtime_ccall((:cublasCtpsv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, CuPtr{cuComplex}, CuPtr{cuComplex}, Cint),
                 handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasZtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    @check @runtime_ccall((:cublasZtpsv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasStbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    @check @runtime_ccall((:cublasStbsv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint),
                 handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasDtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    @check @runtime_ccall((:cublasDtbsv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint),
                 handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasCtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    @check @runtime_ccall((:cublasCtbsv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint),
                 handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasZtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    @check @runtime_ccall((:cublasZtbsv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                  Cint, Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasSsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasSsymv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, PtrOrCuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint),
                 handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasDsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasDsymv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, PtrOrCuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint),
                 handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasCsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasCsymv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint),
                 handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasZsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasZsymv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasChemv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasChemv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint),
                 handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasZhemv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasZhemv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasSsbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasSsbmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, Cint, PtrOrCuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, PtrOrCuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint),
                 handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasDsbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasDsbmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, Cint, PtrOrCuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, PtrOrCuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint),
                 handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasChbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasChbmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, Cint, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint),
                 handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasZhbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasZhbmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasSspmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasSspmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{Cfloat},
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, PtrOrCuPtr{Cfloat}, CuPtr{Cfloat},
                  Cint),
                 handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
end

function cublasDspmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasDspmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{Cdouble},
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, PtrOrCuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint),
                 handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
end

function cublasChpmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasChpmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint),
                 handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
end

function cublasZhpmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    @check @runtime_ccall((:cublasZhpmv_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
end

function cublasSger_v2(handle, m, n, alpha, x, incx, y, incy, A, lda)
    @check @runtime_ccall((:cublasSger_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, Cint, PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint),
                 handle, m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasDger_v2(handle, m, n, alpha, x, incx, y, incy, A, lda)
    @check @runtime_ccall((:cublasDger_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, Cint, PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint),
                 handle, m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasCgeru_v2(handle, m, n, alpha, x, incx, y, incy, A, lda)
    @check @runtime_ccall((:cublasCgeru_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, Cint, PtrOrCuPtr{cuComplex}, CuPtr{cuComplex},
                  Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint),
                 handle, m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasCgerc_v2(handle, m, n, alpha, x, incx, y, incy, A, lda)
    @check @runtime_ccall((:cublasCgerc_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, Cint, PtrOrCuPtr{cuComplex}, CuPtr{cuComplex},
                  Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint),
                 handle, m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasZgeru_v2(handle, m, n, alpha, x, incx, y, incy, A, lda)
    @check @runtime_ccall((:cublasZgeru_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasZgerc_v2(handle, m, n, alpha, x, incx, y, incy, A, lda)
    @check @runtime_ccall((:cublasZgerc_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasSsyr_v2(handle, uplo, n, alpha, x, incx, A, lda)
    @check @runtime_ccall((:cublasSsyr_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint),
                 handle, uplo, n, alpha, x, incx, A, lda)
end

function cublasDsyr_v2(handle, uplo, n, alpha, x, incx, A, lda)
    @check @runtime_ccall((:cublasDsyr_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint),
                 handle, uplo, n, alpha, x, incx, A, lda)
end

function cublasCsyr_v2(handle, uplo, n, alpha, x, incx, A, lda)
    @check @runtime_ccall((:cublasCsyr_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint),
                 handle, uplo, n, alpha, x, incx, A, lda)
end

function cublasZsyr_v2(handle, uplo, n, alpha, x, incx, A, lda)
    @check @runtime_ccall((:cublasZsyr_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, n, alpha, x, incx, A, lda)
end

function cublasCher_v2(handle, uplo, n, alpha, x, incx, A, lda)
    @check @runtime_ccall((:cublasCher_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{Cfloat},
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint),
                 handle, uplo, n, alpha, x, incx, A, lda)
end

function cublasZher_v2(handle, uplo, n, alpha, x, incx, A, lda)
    @check @runtime_ccall((:cublasZher_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{Cdouble},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, n, alpha, x, incx, A, lda)
end

function cublasSspr_v2(handle, uplo, n, alpha, x, incx, AP)
    @check @runtime_ccall((:cublasSspr_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}),
                 handle, uplo, n, alpha, x, incx, AP)
end

function cublasDspr_v2(handle, uplo, n, alpha, x, incx, AP)
    @check @runtime_ccall((:cublasDspr_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}),
                 handle, uplo, n, alpha, x, incx, AP)
end

function cublasChpr_v2(handle, uplo, n, alpha, x, incx, AP)
    @check @runtime_ccall((:cublasChpr_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{Cfloat},
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}),
                 handle, uplo, n, alpha, x, incx, AP)
end

function cublasZhpr_v2(handle, uplo, n, alpha, x, incx, AP)
    @check @runtime_ccall((:cublasZhpr_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{Cdouble},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}),
                 handle, uplo, n, alpha, x, incx, AP)
end

function cublasSsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    @check @runtime_ccall((:cublasSsyr2_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint),
                 handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasDsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    @check @runtime_ccall((:cublasDsyr2_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint),
                 handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasCsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    @check @runtime_ccall((:cublasCsyr2_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint),
                 handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasZsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    @check @runtime_ccall((:cublasZsyr2_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasCher2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    @check @runtime_ccall((:cublasCher2_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint),
                 handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasZher2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    @check @runtime_ccall((:cublasZher2_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasSspr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP)
    @check @runtime_ccall((:cublasSspr2_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}),
                 handle, uplo, n, alpha, x, incx, y, incy, AP)
end

function cublasDspr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP)
    @check @runtime_ccall((:cublasDspr2_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}),
                 handle, uplo, n, alpha, x, incx, y, incy, AP)
end

function cublasChpr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP)
    @check @runtime_ccall((:cublasChpr2_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}),
                 handle, uplo, n, alpha, x, incx, y, incy, AP)
end

function cublasZhpr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP)
    @check @runtime_ccall((:cublasZhpr2_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}),
                 handle, uplo, n, alpha, x, incx, y, incy, AP)
end

function cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasSgemm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Cint),
                 handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasDgemm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Cint),
                 handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasCgemm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint),
                 handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasCgemm3m, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint),
                 handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCgemm3mEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype,
                         ldb, beta, C, Ctype, ldc)
    @check @runtime_ccall((:cublasCgemm3mEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{Cvoid}, cudaDataType, Cint, CuPtr{Cvoid},
                  cudaDataType, Cint, PtrOrCuPtr{cuComplex}, CuPtr{Cvoid}, cudaDataType,
                  Cint),
                 handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
                 beta, C, Ctype, ldc)
end

function cublasZgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasZgemm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasZgemm3m, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasSgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype,
                       ldb, beta, C, Ctype, ldc)
    @check @runtime_ccall((:cublasSgemmEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cvoid}, cudaDataType, Cint, CuPtr{Cvoid},
                  cudaDataType, Cint, PtrOrCuPtr{Cfloat}, CuPtr{Cvoid}, cudaDataType, Cint),
                 handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
                 beta, C, Ctype, ldc)
end

function cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
                      beta, C, Ctype, ldc, computeType, algo)
    @check @runtime_ccall((:cublasGemmEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{Cvoid}, CuPtr{Cvoid}, cudaDataType, Cint, CuPtr{Cvoid},
                  cudaDataType, Cint, PtrOrCuPtr{Cvoid}, CuPtr{Cvoid}, cudaDataType, Cint,
                  cudaDataType, cublasGemmAlgo_t),
                 handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
                 beta, C, Ctype, ldc, computeType, algo)
end

function cublasCgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype,
                       ldb, beta, C, Ctype, ldc)
    @check @runtime_ccall((:cublasCgemmEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{Cvoid}, cudaDataType, Cint, CuPtr{Cvoid},
                  cudaDataType, Cint, PtrOrCuPtr{cuComplex}, CuPtr{Cvoid}, cudaDataType,
                  Cint),
                 handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
                 beta, C, Ctype, ldc)
end

function cublasUint8gemmBias(handle, transa, transb, transc, m, n, k, A, A_bias, lda, B,
                             B_bias, ldb, C, C_bias, ldc, C_mult, C_shift)
    @check @runtime_ccall((:cublasUint8gemmBias, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, cublasOperation_t,
                  Cint, Cint, Cint, CuPtr{Cuchar}, Cint, Cint, CuPtr{Cuchar}, Cint, Cint,
                  CuPtr{Cuchar}, Cint, Cint, Cint, Cint),
                 handle, transa, transb, transc, m, n, k, A, A_bias, lda, B, B_bias, ldb,
                 C, C_bias, ldc, C_mult, C_shift)
end

function cublasSsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    @check @runtime_ccall((:cublasSsyrk_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Cint, PtrOrCuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint),
                 handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasDsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    @check @runtime_ccall((:cublasDsyrk_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Cint, PtrOrCuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint),
                 handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasCsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    @check @runtime_ccall((:cublasCsyrk_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint),
                 handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasZsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    @check @runtime_ccall((:cublasZsyrk_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasCsyrkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    @check @runtime_ccall((:cublasCsyrkEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{Cvoid}, cudaDataType, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{Cvoid}, cudaDataType, Cint),
                 handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
end

function cublasCsyrk3mEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype,
                         ldc)
    @check @runtime_ccall((:cublasCsyrk3mEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{Cvoid}, cudaDataType, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{Cvoid}, cudaDataType, Cint),
                 handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
end

function cublasCherk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    @check @runtime_ccall((:cublasCherk_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{cuComplex}, Cint, PtrOrCuPtr{Cfloat},
                  CuPtr{cuComplex}, Cint),
                 handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasZherk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    @check @runtime_ccall((:cublasZherk_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{Cdouble},
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasCherkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    @check @runtime_ccall((:cublasCherkEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cvoid}, cudaDataType, Cint, PtrOrCuPtr{Cfloat},
                  CuPtr{Cvoid}, cudaDataType, Cint),
                 handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
end

function cublasCherk3mEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype,
                         ldc)
    @check @runtime_ccall((:cublasCherk3mEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cvoid}, cudaDataType, Cint, PtrOrCuPtr{Cfloat},
                  CuPtr{Cvoid}, cudaDataType, Cint),
                 handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
end

function cublasSsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasSsyr2k_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Cint),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasDsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasDsyr2k_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Cint),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasCsyr2k_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasZsyr2k_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCher2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasCher2k_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{cuComplex}, Cint),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZher2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasZher2k_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{Cdouble},
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasSsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasSsyrkx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Cint),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasDsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasDsyrkx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Cint),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasCsyrkx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasZsyrkx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasCherkx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{cuComplex}, Cint),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasZherkx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{Cdouble},
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasSsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasSsymm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Cint),
                 handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasDsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasDsymm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Cint),
                 handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasCsymm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint),
                 handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasZsymm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasChemm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasChemm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint),
                 handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZhemm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasZhemm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasStrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    @check @runtime_ccall((:cublasStrsm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Cint, Cint, PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasDtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    @check @runtime_ccall((:cublasDtrsm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Cint, Cint, PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasCtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    @check @runtime_ccall((:cublasCtrsm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Cint, Cint, PtrOrCuPtr{cuComplex}, CuPtr{cuComplex},
                  Cint, CuPtr{cuComplex}, Cint),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasZtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    @check @runtime_ccall((:cublasZtrsm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Cint, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasStrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    @check @runtime_ccall((:cublasStrmm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Cint, Cint, PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end

function cublasDtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    @check @runtime_ccall((:cublasDtrmm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Cint, Cint, PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end

function cublasCtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    @check @runtime_ccall((:cublasCtrmm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Cint, Cint, PtrOrCuPtr{cuComplex}, CuPtr{cuComplex},
                  Cint, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end

function cublasZtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    @check @runtime_ccall((:cublasZtrmm_v2, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Cint, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end

function cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray,
                            ldb, beta, Carray, ldc, batchCount)
    @check @runtime_ccall((:cublasSgemmBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{Ptr{Cfloat}}, Cint, CuPtr{Ptr{Cfloat}}, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{Ptr{Cfloat}}, Cint, Cint),
                 handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta,
                 Carray, ldc, batchCount)
end

function cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray,
                            ldb, beta, Carray, ldc, batchCount)
    @check @runtime_ccall((:cublasDgemmBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{Cdouble}, CuPtr{Ptr{Cdouble}}, Cint, CuPtr{Ptr{Cdouble}},
                  Cint, PtrOrCuPtr{Cdouble}, CuPtr{Ptr{Cdouble}}, Cint, Cint),
                 handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta,
                 Carray, ldc, batchCount)
end

function cublasCgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray,
                            ldb, beta, Carray, ldc, batchCount)
    @check @runtime_ccall((:cublasCgemmBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{Ptr{cuComplex}}, Cint,
                  CuPtr{Ptr{cuComplex}}, Cint, PtrOrCuPtr{cuComplex},
                  CuPtr{Ptr{cuComplex}}, Cint, Cint),
                 handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta,
                 Carray, ldc, batchCount)
end

function cublasCgemm3mBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray,
                              ldb, beta, Carray, ldc, batchCount)
    @check @runtime_ccall((:cublasCgemm3mBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{Ptr{cuComplex}}, Cint,
                  CuPtr{Ptr{cuComplex}}, Cint, PtrOrCuPtr{cuComplex},
                  CuPtr{Ptr{cuComplex}}, Cint, Cint),
                 handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta,
                 Carray, ldc, batchCount)
end

function cublasZgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray,
                            ldb, beta, Carray, ldc, batchCount)
    @check @runtime_ccall((:cublasZgemmBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{Ptr{cuDoubleComplex}}, Cint,
                  CuPtr{Ptr{cuDoubleComplex}}, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{Ptr{cuDoubleComplex}}, Cint, Cint),
                 handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta,
                 Carray, ldc, batchCount)
end

function cublasGemmBatchedEx(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda,
                             Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount,
                             computeType, algo)
    @check @runtime_ccall((:cublasGemmBatchedEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{Cvoid}, CuPtr{Ptr{Cvoid}}, cudaDataType, Cint,
                  CuPtr{Ptr{Cvoid}}, cudaDataType, Cint, PtrOrCuPtr{Cvoid},
                  CuPtr{Ptr{Cvoid}}, cudaDataType, Cint, Cint, cudaDataType,
                  cublasGemmAlgo_t),
                 handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype,
                 ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo)
end

function cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda,
                                    strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc,
                                    strideC, batchCount, computeType, algo)
    @check @runtime_ccall((:cublasGemmStridedBatchedEx, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{Cvoid}, CuPtr{Cvoid}, cudaDataType, Cint, Clonglong,
                  CuPtr{Cvoid}, cudaDataType, Cint, Clonglong, PtrOrCuPtr{Cvoid},
                  CuPtr{Cvoid}, cudaDataType, Cint, Clonglong, Cint, cudaDataType,
                  cublasGemmAlgo_t),
                 handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype,
                 ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo)
end

function cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA,
                                   B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    @check @runtime_ccall((:cublasSgemmStridedBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Cint, Clonglong, CuPtr{Cfloat}, Cint,
                  Clonglong, PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Cint, Clonglong, Cint),
                 handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB,
                 beta, C, ldc, strideC, batchCount)
end

function cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA,
                                   B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    @check @runtime_ccall((:cublasDgemmStridedBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Cint, Clonglong, CuPtr{Cdouble},
                  Cint, Clonglong, PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Cint, Clonglong,
                  Cint),
                 handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB,
                 beta, C, ldc, strideC, batchCount)
end

function cublasCgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA,
                                   B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    @check @runtime_ccall((:cublasCgemmStridedBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint, Clonglong,
                  CuPtr{cuComplex}, Cint, Clonglong, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint, Clonglong, Cint),
                 handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB,
                 beta, C, ldc, strideC, batchCount)
end

function cublasCgemm3mStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda,
                                     strideA, B, ldb, strideB, beta, C, ldc, strideC,
                                     batchCount)
    @check @runtime_ccall((:cublasCgemm3mStridedBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint, Clonglong,
                  CuPtr{cuComplex}, Cint, Clonglong, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint, Clonglong, Cint),
                 handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB,
                 beta, C, ldc, strideC, batchCount)
end

function cublasZgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA,
                                   B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    @check @runtime_ccall((:cublasZgemmStridedBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, Clonglong,
                  CuPtr{cuDoubleComplex}, Cint, Clonglong, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, Clonglong, Cint),
                 handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB,
                 beta, C, ldc, strideC, batchCount)
end

function cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    @check @runtime_ccall((:cublasSgeam, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Cint, PtrOrCuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint),
                 handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
end

function cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    @check @runtime_ccall((:cublasDgeam, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Cint, PtrOrCuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint),
                 handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
end

function cublasCgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    @check @runtime_ccall((:cublasCgeam, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Cint, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint),
                 handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
end

function cublasZgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    @check @runtime_ccall((:cublasZgeam, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
end

function cublasSgetrfBatched(handle, n, A, lda, P, info, batchSize)
    @check @runtime_ccall((:cublasSgetrfBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Ptr{Cfloat}}, Cint, CuPtr{Cint},
                  CuPtr{Cint}, Cint),
                 handle, n, A, lda, P, info, batchSize)
end

function cublasDgetrfBatched(handle, n, A, lda, P, info, batchSize)
    @check @runtime_ccall((:cublasDgetrfBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Ptr{Cdouble}}, Cint, CuPtr{Cint},
                  CuPtr{Cint}, Cint),
                 handle, n, A, lda, P, info, batchSize)
end

function cublasCgetrfBatched(handle, n, A, lda, P, info, batchSize)
    @check @runtime_ccall((:cublasCgetrfBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Ptr{cuComplex}}, Cint, CuPtr{Cint},
                  CuPtr{Cint}, Cint),
                 handle, n, A, lda, P, info, batchSize)
end

function cublasZgetrfBatched(handle, n, A, lda, P, info, batchSize)
    @check @runtime_ccall((:cublasZgetrfBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Ptr{cuDoubleComplex}}, Cint, CuPtr{Cint},
                  CuPtr{Cint}, Cint),
                 handle, n, A, lda, P, info, batchSize)
end

function cublasSgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize)
    @check @runtime_ccall((:cublasSgetriBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Ptr{Cfloat}}, Cint, CuPtr{Cint},
                  CuPtr{Ptr{Cfloat}}, Cint, CuPtr{Cint}, Cint),
                 handle, n, A, lda, P, C, ldc, info, batchSize)
end

function cublasDgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize)
    @check @runtime_ccall((:cublasDgetriBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Ptr{Cdouble}}, Cint, CuPtr{Cint},
                  CuPtr{Ptr{Cdouble}}, Cint, CuPtr{Cint}, Cint),
                 handle, n, A, lda, P, C, ldc, info, batchSize)
end

function cublasCgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize)
    @check @runtime_ccall((:cublasCgetriBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Ptr{cuComplex}}, Cint, CuPtr{Cint},
                  CuPtr{Ptr{cuComplex}}, Cint, CuPtr{Cint}, Cint),
                 handle, n, A, lda, P, C, ldc, info, batchSize)
end

function cublasZgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize)
    @check @runtime_ccall((:cublasZgetriBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Ptr{cuDoubleComplex}}, Cint, CuPtr{Cint},
                  CuPtr{Ptr{cuDoubleComplex}}, Cint, CuPtr{Cint}, Cint),
                 handle, n, A, lda, P, C, ldc, info, batchSize)
end

function cublasSgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb,
                             info, batchSize)
    @check @runtime_ccall((:cublasSgetrsBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, Cint, Cint, CuPtr{Ptr{Cfloat}}, Cint,
                  CuPtr{Cint}, CuPtr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Cint),
                 handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info,
                 batchSize)
end

function cublasDgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb,
                             info, batchSize)
    @check @runtime_ccall((:cublasDgetrsBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, Cint, Cint, CuPtr{Ptr{Cdouble}},
                  Cint, CuPtr{Cint}, CuPtr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Cint),
                 handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info,
                 batchSize)
end

function cublasCgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb,
                             info, batchSize)
    @check @runtime_ccall((:cublasCgetrsBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, Cint, Cint, CuPtr{Ptr{cuComplex}},
                  Cint, CuPtr{Cint}, CuPtr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Cint),
                 handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info,
                 batchSize)
end

function cublasZgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb,
                             info, batchSize)
    @check @runtime_ccall((:cublasZgetrsBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, Cint, Cint,
                  CuPtr{Ptr{cuDoubleComplex}}, Cint, CuPtr{Cint},
                  CuPtr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Cint),
                 handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info,
                 batchSize)
end

function cublasStrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb,
                            batchCount)
    @check @runtime_ccall((:cublasStrsmBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Cint, Cint, PtrOrCuPtr{Cfloat}, CuPtr{Ptr{Cfloat}},
                  Cint, CuPtr{Ptr{Cfloat}}, Cint, Cint),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
end

function cublasDtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb,
                            batchCount)
    @check @runtime_ccall((:cublasDtrsmBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Cint, Cint, PtrOrCuPtr{Cdouble}, CuPtr{Ptr{Cdouble}},
                  Cint, CuPtr{Ptr{Cdouble}}, Cint, Cint),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
end

function cublasCtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb,
                            batchCount)
    @check @runtime_ccall((:cublasCtrsmBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Cint, Cint, PtrOrCuPtr{cuComplex},
                  CuPtr{Ptr{cuComplex}}, Cint, CuPtr{Ptr{cuComplex}}, Cint, Cint),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
end

function cublasZtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb,
                            batchCount)
    @check @runtime_ccall((:cublasZtrsmBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Cint, Cint, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{Ptr{cuDoubleComplex}}, Cint, CuPtr{Ptr{cuDoubleComplex}}, Cint,
                  Cint),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
end

function cublasSmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize)
    @check @runtime_ccall((:cublasSmatinvBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Ptr{Cfloat}}, Cint, CuPtr{Ptr{Cfloat}},
                  Cint, CuPtr{Cint}, Cint),
                 handle, n, A, lda, Ainv, lda_inv, info, batchSize)
end

function cublasDmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize)
    @check @runtime_ccall((:cublasDmatinvBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Ptr{Cdouble}}, Cint, CuPtr{Ptr{Cdouble}},
                  Cint, CuPtr{Cint}, Cint),
                 handle, n, A, lda, Ainv, lda_inv, info, batchSize)
end

function cublasCmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize)
    @check @runtime_ccall((:cublasCmatinvBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Ptr{cuComplex}}, Cint,
                  CuPtr{Ptr{cuComplex}}, Cint, CuPtr{Cint}, Cint),
                 handle, n, A, lda, Ainv, lda_inv, info, batchSize)
end

function cublasZmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize)
    @check @runtime_ccall((:cublasZmatinvBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, CuPtr{Ptr{cuDoubleComplex}}, Cint,
                  CuPtr{Ptr{cuDoubleComplex}}, Cint, CuPtr{Cint}, Cint),
                 handle, n, A, lda, Ainv, lda_inv, info, batchSize)
end

function cublasSgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize)
    @check @runtime_ccall((:cublasSgeqrfBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, Cint, CuPtr{Ptr{Cfloat}}, Cint,
                  CuPtr{Ptr{Cfloat}}, Ptr{Cint}, Cint),
                 handle, m, n, Aarray, lda, TauArray, info, batchSize)
end

function cublasDgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize)
    @check @runtime_ccall((:cublasDgeqrfBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, Cint, CuPtr{Ptr{Cdouble}}, Cint,
                  CuPtr{Ptr{Cdouble}}, Ptr{Cint}, Cint),
                 handle, m, n, Aarray, lda, TauArray, info, batchSize)
end

function cublasCgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize)
    @check @runtime_ccall((:cublasCgeqrfBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, Cint, CuPtr{Ptr{cuComplex}}, Cint,
                  CuPtr{Ptr{cuComplex}}, Ptr{Cint}, Cint),
                 handle, m, n, Aarray, lda, TauArray, info, batchSize)
end

function cublasZgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize)
    @check @runtime_ccall((:cublasZgeqrfBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, Cint, Cint, CuPtr{Ptr{cuDoubleComplex}}, Cint,
                  CuPtr{Ptr{cuDoubleComplex}}, Ptr{Cint}, Cint),
                 handle, m, n, Aarray, lda, TauArray, info, batchSize)
end

function cublasSgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info,
                            devInfoArray, batchSize)
    @check @runtime_ccall((:cublasSgelsBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, CuPtr{Ptr{Cfloat}},
                  Cint, CuPtr{Ptr{Cfloat}}, Cint, Ptr{Cint}, CuPtr{Cint}, Cint),
                 handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray,
                 batchSize)
end

function cublasDgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info,
                            devInfoArray, batchSize)
    @check @runtime_ccall((:cublasDgelsBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint,
                  CuPtr{Ptr{Cdouble}}, Cint, CuPtr{Ptr{Cdouble}}, Cint, Ptr{Cint},
                  CuPtr{Cint}, Cint),
                 handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray,
                 batchSize)
end

function cublasCgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info,
                            devInfoArray, batchSize)
    @check @runtime_ccall((:cublasCgelsBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint,
                  CuPtr{Ptr{cuComplex}}, Cint, CuPtr{Ptr{cuComplex}}, Cint, Ptr{Cint},
                  CuPtr{Cint}, Cint),
                 handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray,
                 batchSize)
end

function cublasZgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info,
                            devInfoArray, batchSize)
    @check @runtime_ccall((:cublasZgelsBatched, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint,
                  CuPtr{Ptr{cuDoubleComplex}}, Cint, CuPtr{Ptr{cuDoubleComplex}}, Cint,
                  Ptr{Cint}, CuPtr{Cint}, Cint),
                 handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray,
                 batchSize)
end

function cublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc)
    @check @runtime_ccall((:cublasSdgmm, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, Cint, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint),
                 handle, mode, m, n, A, lda, x, incx, C, ldc)
end

function cublasDdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc)
    @check @runtime_ccall((:cublasDdgmm, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, Cint, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint),
                 handle, mode, m, n, A, lda, x, incx, C, ldc)
end

function cublasCdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc)
    @check @runtime_ccall((:cublasCdgmm, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, Cint, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint),
                 handle, mode, m, n, A, lda, x, incx, C, ldc)
end

function cublasZdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc)
    @check @runtime_ccall((:cublasZdgmm, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasSideMode_t, Cint, Cint, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, Cint),
                 handle, mode, m, n, A, lda, x, incx, C, ldc)
end

function cublasStpttr(handle, uplo, n, AP, A, lda)
    @check @runtime_ccall((:cublasStpttr, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, CuPtr{Cfloat},
                  Cint),
                 handle, uplo, n, AP, A, lda)
end

function cublasDtpttr(handle, uplo, n, AP, A, lda)
    @check @runtime_ccall((:cublasDtpttr, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, CuPtr{Cdouble},
                  Cint),
                 handle, uplo, n, AP, A, lda)
end

function cublasCtpttr(handle, uplo, n, AP, A, lda)
    @check @runtime_ccall((:cublasCtpttr, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex},
                  CuPtr{cuComplex}, Cint),
                 handle, uplo, n, AP, A, lda)
end

function cublasZtpttr(handle, uplo, n, AP, A, lda)
    @check @runtime_ccall((:cublasZtpttr, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, uplo, n, AP, A, lda)
end

function cublasStrttp(handle, uplo, n, A, lda, AP)
    @check @runtime_ccall((:cublasStrttp, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, CuPtr{Cfloat}, Cint,
                  CuPtr{Cfloat}),
                 handle, uplo, n, A, lda, AP)
end

function cublasDtrttp(handle, uplo, n, A, lda, AP)
    @check @runtime_ccall((:cublasDtrttp, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, CuPtr{Cdouble}, Cint,
                  CuPtr{Cdouble}),
                 handle, uplo, n, A, lda, AP)
end

function cublasCtrttp(handle, uplo, n, A, lda, AP)
    @check @runtime_ccall((:cublasCtrttp, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, CuPtr{cuComplex}, Cint,
                  CuPtr{cuComplex}),
                 handle, uplo, n, A, lda, AP)
end

function cublasZtrttp(handle, uplo, n, A, lda, AP)
    @check @runtime_ccall((:cublasZtrttp, libcublas[]), cublasStatus_t,
                 (cublasHandle_t, cublasFillMode_t, Cint, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}),
                 handle, uplo, n, A, lda, AP)
end
# Julia wrapper for header: cublasXt.h
# Automatically generated using Clang.jl


function cublasXtCreate(handle)
    @check @runtime_ccall((:cublasXtCreate, libcublas[]), cublasStatus_t,
                 (Ptr{cublasXtHandle_t},),
                 handle)
end

function cublasXtDestroy(handle)
    @check @runtime_ccall((:cublasXtDestroy, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t,),
                 handle)
end

function cublasXtGetNumBoards(nbDevices, deviceId, nbBoards)
    @check @runtime_ccall((:cublasXtGetNumBoards, libcublas[]), cublasStatus_t,
                 (Cint, Ptr{Cint}, Ptr{Cint}),
                 nbDevices, deviceId, nbBoards)
end

function cublasXtMaxBoards(nbGpuBoards)
    @check @runtime_ccall((:cublasXtMaxBoards, libcublas[]), cublasStatus_t,
                 (Ptr{Cint},),
                 nbGpuBoards)
end

function cublasXtDeviceSelect(handle, nbDevices, deviceId)
    @check @runtime_ccall((:cublasXtDeviceSelect, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, Cint, Ptr{Cint}),
                 handle, nbDevices, deviceId)
end

function cublasXtSetBlockDim(handle, blockDim)
    @check @runtime_ccall((:cublasXtSetBlockDim, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, Cint),
                 handle, blockDim)
end

function cublasXtGetBlockDim(handle, blockDim)
    @check @runtime_ccall((:cublasXtGetBlockDim, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, Ptr{Cint}),
                 handle, blockDim)
end

function cublasXtGetPinningMemMode(handle, mode)
    @check @runtime_ccall((:cublasXtGetPinningMemMode, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, Ptr{cublasXtPinnedMemMode_t}),
                 handle, mode)
end

function cublasXtSetPinningMemMode(handle, mode)
    @check @runtime_ccall((:cublasXtSetPinningMemMode, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasXtPinnedMemMode_t),
                 handle, mode)
end

function cublasXtSetCpuRoutine(handle, blasOp, type, blasFunctor)
    @check @runtime_ccall((:cublasXtSetCpuRoutine, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasXtBlasOp_t, cublasXtOpType_t, Ptr{Cvoid}),
                 handle, blasOp, type, blasFunctor)
end

function cublasXtSetCpuRatio(handle, blasOp, type, ratio)
    @check @runtime_ccall((:cublasXtSetCpuRatio, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasXtBlasOp_t, cublasXtOpType_t, Cfloat),
                 handle, blasOp, type, ratio)
end

function cublasXtSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtSgemm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasOperation_t, cublasOperation_t, Csize_t,
                  Csize_t, Csize_t, PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Csize_t,
                  CuPtr{Cfloat}, Csize_t, PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Csize_t),
                 handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtDgemm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasOperation_t, cublasOperation_t, Csize_t,
                  Csize_t, Csize_t, PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Csize_t,
                  CuPtr{Cdouble}, Csize_t, PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Csize_t),
                 handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtCgemm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasOperation_t, cublasOperation_t, Csize_t,
                  Csize_t, Csize_t, PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Csize_t,
                  CuPtr{cuComplex}, Csize_t, PtrOrCuPtr{cuComplex}, CuPtr{cuComplex},
                  Csize_t),
                 handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtZgemm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasOperation_t, cublasOperation_t, Csize_t,
                  Csize_t, Csize_t, PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex},
                  Csize_t, CuPtr{cuDoubleComplex}, Csize_t, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Csize_t),
                 handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtSsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    @check @runtime_ccall((:cublasXtSsyrk, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Csize_t, PtrOrCuPtr{Cfloat},
                  CuPtr{Cfloat}, Csize_t),
                 handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasXtDsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    @check @runtime_ccall((:cublasXtDsyrk, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Csize_t, PtrOrCuPtr{Cdouble},
                  CuPtr{Cdouble}, Csize_t),
                 handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasXtCsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    @check @runtime_ccall((:cublasXtCsyrk, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Csize_t, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Csize_t),
                 handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasXtZsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    @check @runtime_ccall((:cublasXtZsyrk, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Csize_t,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Csize_t),
                 handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasXtCherk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    @check @runtime_ccall((:cublasXtCherk, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t,
                  PtrOrCuPtr{Cfloat}, CuPtr{cuComplex}, Csize_t, PtrOrCuPtr{Cfloat},
                  CuPtr{cuComplex}, Csize_t),
                 handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasXtZherk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    @check @runtime_ccall((:cublasXtZherk, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t,
                  PtrOrCuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Csize_t,
                  PtrOrCuPtr{Cdouble}, CuPtr{cuDoubleComplex}, Csize_t),
                 handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasXtSsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtSsyr2k, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Csize_t, CuPtr{Cfloat}, Csize_t,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Csize_t),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtDsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtDsyr2k, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Csize_t, CuPtr{Cdouble}, Csize_t,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Csize_t),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtCsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtCsyr2k, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Csize_t, CuPtr{cuComplex},
                  Csize_t, PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Csize_t),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtZsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtZsyr2k, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Csize_t,
                  CuPtr{cuDoubleComplex}, Csize_t, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Csize_t),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtCherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtCherkx, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Csize_t, CuPtr{cuComplex},
                  Csize_t, PtrOrCuPtr{Cfloat}, CuPtr{cuComplex}, Csize_t),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtZherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtZherkx, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Csize_t,
                  CuPtr{cuDoubleComplex}, Csize_t, PtrOrCuPtr{Cdouble},
                  CuPtr{cuDoubleComplex}, Csize_t),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    @check @runtime_ccall((:cublasXtStrsm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Csize_t, Csize_t, PtrOrCuPtr{Cfloat}, CuPtr{Cfloat},
                  Csize_t, CuPtr{Cfloat}, Csize_t),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasXtDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    @check @runtime_ccall((:cublasXtDtrsm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Csize_t, Csize_t, PtrOrCuPtr{Cdouble}, CuPtr{Cdouble},
                  Csize_t, CuPtr{Cdouble}, Csize_t),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasXtCtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    @check @runtime_ccall((:cublasXtCtrsm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Csize_t, Csize_t, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Csize_t, CuPtr{cuComplex}, Csize_t),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasXtZtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    @check @runtime_ccall((:cublasXtZtrsm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Csize_t, Csize_t, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Csize_t, CuPtr{cuDoubleComplex}, Csize_t),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasXtSsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtSsymm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Csize_t, CuPtr{Cfloat}, Csize_t,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Csize_t),
                 handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtDsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtDsymm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Csize_t, CuPtr{Cdouble}, Csize_t,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Csize_t),
                 handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtCsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtCsymm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Csize_t, CuPtr{cuComplex},
                  Csize_t, PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Csize_t),
                 handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtZsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtZsymm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Csize_t,
                  CuPtr{cuDoubleComplex}, Csize_t, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Csize_t),
                 handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtChemm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtChemm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Csize_t, CuPtr{cuComplex},
                  Csize_t, PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Csize_t),
                 handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtZhemm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtZhemm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Csize_t,
                  CuPtr{cuDoubleComplex}, Csize_t, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Csize_t),
                 handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtSsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtSsyrkx, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Csize_t, CuPtr{Cfloat}, Csize_t,
                  PtrOrCuPtr{Cfloat}, CuPtr{Cfloat}, Csize_t),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtDsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtDsyrkx, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Csize_t, CuPtr{Cdouble}, Csize_t,
                  PtrOrCuPtr{Cdouble}, CuPtr{Cdouble}, Csize_t),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtCsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtCsyrkx, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Csize_t, CuPtr{cuComplex},
                  Csize_t, PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Csize_t),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtZsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtZsyrkx, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Csize_t,
                  CuPtr{cuDoubleComplex}, Csize_t, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Csize_t),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtCher2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtCher2k, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t,
                  PtrOrCuPtr{cuComplex}, CuPtr{cuComplex}, Csize_t, CuPtr{cuComplex},
                  Csize_t, PtrOrCuPtr{Cfloat}, CuPtr{cuComplex}, Csize_t),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtZher2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtZher2k, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t,
                  PtrOrCuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Csize_t,
                  CuPtr{cuDoubleComplex}, Csize_t, PtrOrCuPtr{Cdouble},
                  CuPtr{cuDoubleComplex}, Csize_t),
                 handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtSspmm(handle, side, uplo, m, n, alpha, AP, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtSspmm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t,
                  Ptr{Cfloat}, Ptr{Cfloat}, PtrOrCuPtr{Cfloat}, Csize_t, Ptr{Cfloat},
                  PtrOrCuPtr{Cfloat}, Csize_t),
                 handle, side, uplo, m, n, alpha, AP, B, ldb, beta, C, ldc)
end

function cublasXtDspmm(handle, side, uplo, m, n, alpha, AP, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtDspmm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t,
                  Ptr{Cdouble}, Ptr{Cdouble}, PtrOrCuPtr{Cdouble}, Csize_t, Ptr{Cdouble},
                  PtrOrCuPtr{Cdouble}, Csize_t),
                 handle, side, uplo, m, n, alpha, AP, B, ldb, beta, C, ldc)
end

function cublasXtCspmm(handle, side, uplo, m, n, alpha, AP, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtCspmm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t,
                  Ptr{cuComplex}, Ptr{cuComplex}, PtrOrCuPtr{cuComplex}, Csize_t,
                  Ptr{cuComplex}, PtrOrCuPtr{cuComplex}, Csize_t),
                 handle, side, uplo, m, n, alpha, AP, B, ldb, beta, C, ldc)
end

function cublasXtZspmm(handle, side, uplo, m, n, alpha, AP, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cublasXtZspmm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t,
                  Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, PtrOrCuPtr{cuDoubleComplex},
                  Csize_t, Ptr{cuDoubleComplex}, PtrOrCuPtr{cuDoubleComplex}, Csize_t),
                 handle, side, uplo, m, n, alpha, AP, B, ldb, beta, C, ldc)
end

function cublasXtStrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    @check @runtime_ccall((:cublasXtStrmm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Csize_t, Csize_t, PtrOrCuPtr{Cfloat}, CuPtr{Cfloat},
                  Csize_t, CuPtr{Cfloat}, Csize_t, CuPtr{Cfloat}, Csize_t),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end

function cublasXtDtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    @check @runtime_ccall((:cublasXtDtrmm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Csize_t, Csize_t, PtrOrCuPtr{Cdouble}, CuPtr{Cdouble},
                  Csize_t, CuPtr{Cdouble}, Csize_t, CuPtr{Cdouble}, Csize_t),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end

function cublasXtCtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    @check @runtime_ccall((:cublasXtCtrmm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Csize_t, Csize_t, PtrOrCuPtr{cuComplex},
                  CuPtr{cuComplex}, Csize_t, CuPtr{cuComplex}, Csize_t, CuPtr{cuComplex},
                  Csize_t),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end

function cublasXtZtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    @check @runtime_ccall((:cublasXtZtrmm, libcublas[]), cublasStatus_t,
                 (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                  cublasDiagType_t, Csize_t, Csize_t, PtrOrCuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Csize_t, CuPtr{cuDoubleComplex}, Csize_t,
                  CuPtr{cuDoubleComplex}, Csize_t),
                 handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end
