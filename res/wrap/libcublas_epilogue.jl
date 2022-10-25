# Float16 functionality is only enabled when using C++ (and defining __cplusplus breaks things)

@checked function cublasHSHgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
        initialize_context()
        ccall((:cublasHSHgemvBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{Float32}, Ptr{Ptr{Float16}}, Cint, Ptr{Ptr{Float16}}, Cint, Ptr{Float32}, Ptr{Ptr{Float16}}, Cint, Cint), handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
    end

@checked function cublasHSSgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
        initialize_context()
        ccall((:cublasHSSgemvBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{Float32}, Ptr{Ptr{Float16}}, Cint, Ptr{Ptr{Float16}}, Cint, Ptr{Float32}, Ptr{Ptr{Float32}}, Cint, Cint), handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
    end

@checked function cublasHSHgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
        initialize_context()
        ccall((:cublasHSHgemvStridedBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{Float32}, Ptr{Float16}, Cint, Clonglong, Ptr{Float16}, Cint, Clonglong, Ptr{Float32}, Ptr{Float16}, Cint, Clonglong, Cint), handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
    end

@checked function cublasHSSgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
        initialize_context()
        ccall((:cublasHSSgemvStridedBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{Float32}, Ptr{Float16}, Cint, Clonglong, Ptr{Float16}, Cint, Clonglong, Ptr{Float32}, Ptr{Float32}, Cint, Clonglong, Cint), handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
    end

@checked function cublasHgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                                     Barray, ldb, beta, Carray, ldc, batchCount)
    initialize_context()
    ccall((:cublasHgemmBatched, libcublas), cublasStatus_t,
                   (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint,
                    Cint, RefOrCuRef{Float16}, CuPtr{Ptr{Float16}}, Cint, CuPtr{Ptr{Float16}},
                    Cint, RefOrCuRef{Float16}, CuPtr{Ptr{Float16}}, Cint, Cint),
                   handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta,
                   Carray, ldc, batchCount)
end

@checked function cublasHgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda,
                                            strideA, B, ldb, strideB, beta, C, ldc,
                                            strideC, batchCount)
    initialize_context()
    ccall((:cublasHgemmStridedBatched, libcublas), cublasStatus_t,
                   (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint,
                    Cint, RefOrCuRef{Float16}, CuPtr{Float16}, Cint, Clonglong,
                    CuPtr{Float16}, Cint, Clonglong, RefOrCuRef{Float16}, CuPtr{Float16},
                    Cint, Clonglong, Cint),
                   handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb,
                   strideB, beta, C, ldc, strideC, batchCount)
end
