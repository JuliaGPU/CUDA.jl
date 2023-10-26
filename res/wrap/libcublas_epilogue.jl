# Float16 functionality is only enabled when using C++ (defining __cplusplus breaks things)

@checked function cublasHSHgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray,
                                       incx, beta, yarray, incy, batchCount)
    initialize_context()
    @ccall libcublas.cublasHSHgemvBatched(handle::cublasHandle_t, trans::cublasOperation_t,
                                          m::Cint, n::Cint, alpha::RefOrCuRef{Cfloat},
                                          Aarray::CuPtr{Ptr{Float16}}, lda::Cint,
                                          xarray::CuPtr{Ptr{Float16}}, incx::Cint,
                                          beta::RefOrCuRef{Cfloat}, yarray::CuPtr{Ptr{Float16}},
                                          incy::Cint, batchCount::Cint)::cublasStatus_t
end

@checked function cublasHSSgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray,
                                       incx, beta, yarray, incy, batchCount)
    initialize_context()
    @ccall libcublas.cublasHSSgemvBatched(handle::cublasHandle_t, trans::cublasOperation_t,
                                          m::Cint, n::Cint, alpha::RefOrCuRef{Cfloat},
                                          Aarray::CuPtr{Ptr{Float16}}, lda::Cint,
                                          xarray::CuPtr{Ptr{Float16}}, incx::Cint,
                                          beta::RefOrCuRef{Cfloat}, yarray::CuPtr{Ptr{Cfloat}},
                                          incy::Cint, batchCount::Cint)::cublasStatus_t
end

@checked function cublasTSTgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray,
                                       incx, beta, yarray, incy, batchCount)
    initialize_context()
    @ccall libcublas.cublasTSTgemvBatched(handle::cublasHandle_t, trans::cublasOperation_t,
                                          m::Cint, n::Cint, alpha::Ptr{Cfloat},
                                          Aarray::Ptr{Ptr{BFloat16}}, lda::Cint,
                                          xarray::Ptr{Ptr{BFloat16}}, incx::Cint,
                                          beta::Ptr{Cfloat}, yarray::Ptr{Ptr{BFloat16}},
                                          incy::Cint, batchCount::Cint)::cublasStatus_t
end

@checked function cublasTSSgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray,
                                       incx, beta, yarray, incy, batchCount)
    initialize_context()
    @ccall libcublas.cublasTSSgemvBatched(handle::cublasHandle_t, trans::cublasOperation_t,
                                          m::Cint, n::Cint, alpha::Ptr{Cfloat},
                                          Aarray::Ptr{Ptr{BFloat16}}, lda::Cint,
                                          xarray::Ptr{Ptr{BFloat16}}, incx::Cint,
                                          beta::Ptr{Cfloat}, yarray::Ptr{Ptr{Cfloat}},
                                          incy::Cint, batchCount::Cint)::cublasStatus_t
end

@checked function cublasHSHgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA,
                                              x, incx, stridex, beta, y, incy, stridey,
                                              batchCount)
    initialize_context()
    @ccall libcublas.cublasHSHgemvStridedBatched(handle::cublasHandle_t,
                                                 trans::cublasOperation_t, m::Cint, n::Cint,
                                                 alpha::RefOrCuRef{Cfloat}, A::CuPtr{Float16},
                                                 lda::Cint, strideA::Clonglong,
                                                 x::CuPtr{Float16}, incx::Cint,
                                                 stridex::Clonglong, beta::RefOrCuRef{Cfloat},
                                                 y::CuPtr{Float16}, incy::Cint,
                                                 stridey::Clonglong,
                                                 batchCount::Cint)::cublasStatus_t
end

@checked function cublasHSSgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA,
                                              x, incx, stridex, beta, y, incy, stridey,
                                              batchCount)
    initialize_context()
    @ccall libcublas.cublasHSSgemvStridedBatched(handle::cublasHandle_t,
                                                 trans::cublasOperation_t, m::Cint, n::Cint,
                                                 alpha::RefOrCuRef{Cfloat}, A::CuPtr{Float16},
                                                 lda::Cint, strideA::Clonglong,
                                                 x::CuPtr{Float16}, incx::Cint,
                                                 stridex::Clonglong, beta::RefOrCuRef{Cfloat},
                                                 y::CuPtr{Cfloat}, incy::Cint,
                                                 stridey::Clonglong,
                                                 batchCount::Cint)::cublasStatus_t
end

@checked function cublasTSTgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA,
                                              x, incx, stridex, beta, y, incy, stridey,
                                              batchCount)
    initialize_context()
    @ccall libcublas.cublasTSTgemvStridedBatched(handle::cublasHandle_t,
                                                 trans::cublasOperation_t, m::Cint, n::Cint,
                                                 alpha::RefOrCuRef{Cfloat}, A::CuPtr{BFloat16},
                                                 lda::Cint, strideA::Clonglong,
                                                 x::CuPtr{BFloat16}, incx::Cint,
                                                 stridex::Clonglong, beta::RefOrCuRef{Cfloat},
                                                 y::CuPtr{BFloat16}, incy::Cint,
                                                 stridey::Clonglong,
                                                 batchCount::Cint)::cublasStatus_t
end

@checked function cublasTSSgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA,
                                              x, incx, stridex, beta, y, incy, stridey,
                                              batchCount)
    initialize_context()
    @ccall libcublas.cublasTSSgemvStridedBatched(handle::cublasHandle_t,
                                                 trans::cublasOperation_t, m::Cint, n::Cint,
                                                 alpha::RefOrCuRef{Cfloat}, A::CuPtr{BFloat16},
                                                 lda::Cint, strideA::Clonglong,
                                                 x::CuPtr{BFloat16}, incx::Cint,
                                                 stridex::Clonglong, beta::RefOrCuRef{Cfloat},
                                                 y::CuPtr{Cfloat}, incy::Cint,
                                                 stridey::Clonglong,
                                                 batchCount::Cint)::cublasStatus_t
end

@checked function cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta,
                              C, ldc)
    initialize_context()
    @ccall libcublas.cublasHgemm(handle::cublasHandle_t, transa::cublasOperation_t,
                                 transb::cublasOperation_t, m::Cint, n::Cint, k::Cint,
                                 alpha::Ptr{Float16}, A::Ptr{Float16}, lda::Cint,
                                 B::Ptr{Float16}, ldb::Cint, beta::Ptr{Float16},
                                 C::Ptr{Float16}, ldc::Cint)::cublasStatus_t
end

@checked function cublasHgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                                     Barray, ldb, beta, Carray, ldc, batchCount)
    initialize_context()
    @ccall libcublas.cublasHgemmBatched(handle::cublasHandle_t, transa::cublasOperation_t,
                                        transb::cublasOperation_t, m::Cint, n::Cint,
                                        k::Cint, alpha::RefOrCuRef{Float16},
                                        Aarray::CuPtr{Ptr{Float16}}, lda::Cint,
                                        Barray::CuPtr{Ptr{Float16}}, ldb::Cint,
                                        beta::RefOrCuRef{Float16},
                                        Carray::CuPtr{Ptr{Float16}}, ldc::Cint,
                                        batchCount::Cint)::cublasStatus_t
end

@checked function cublasHgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda,
                                            strideA, B, ldb, strideB, beta, C, ldc, strideC,
                                            batchCount)
    initialize_context()
    @ccall libcublas.cublasHgemmStridedBatched(handle::cublasHandle_t,
                                               transa::cublasOperation_t,
                                               transb::cublasOperation_t, m::Cint, n::Cint,
                                               k::Cint, alpha::RefOrCuRef{Float16},
                                               A::CuPtr{Float16}, lda::Cint,
                                               strideA::Clonglong, B::CuPtr{Float16},
                                               ldb::Cint, strideB::Clonglong,
                                               beta::RefOrCuRef{Float16}, C::CuPtr{Float16},
                                               ldc::Cint, strideC::Clonglong,
                                               batchCount::Cint)::cublasStatus_t
end
