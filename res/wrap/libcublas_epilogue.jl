# Float16 functionality is only enabled when using C++ (defining __cplusplus breaks things)

@checked function cublasHSHgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray,
                                       incx, beta, yarray, incy, batchCount)
    initialize_context()
    @ccall libcublas.cublasHSHgemvBatched(handle::cublasHandle_t, trans::cublasOperation_t,
                                          m::Cint, n::Cint, alpha::CuRef{Cfloat},
                                          Aarray::CuPtr{Ptr{Float16}}, lda::Cint,
                                          xarray::CuPtr{Ptr{Float16}}, incx::Cint,
                                          beta::CuRef{Cfloat}, yarray::CuPtr{Ptr{Float16}},
                                          incy::Cint, batchCount::Cint)::cublasStatus_t
end

@checked function cublasHSHgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray,
                                          incx, beta, yarray, incy, batchCount)
    initialize_context()
    @ccall libcublas.cublasHSHgemvBatched_64(handle::cublasHandle_t, trans::cublasOperation_t,
                                             m::Int64, n::Int64, alpha::CuRef{Cfloat},
                                             Aarray::CuPtr{Ptr{Float16}}, lda::Int64,
                                             xarray::CuPtr{Ptr{Float16}}, incx::Int64,
                                             beta::CuRef{Cfloat}, yarray::CuPtr{Ptr{Float16}},
                                             incy::Int64, batchCount::Int64)::cublasStatus_t
end

@checked function cublasHSSgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray,
                                       incx, beta, yarray, incy, batchCount)
    initialize_context()
    @ccall libcublas.cublasHSSgemvBatched(handle::cublasHandle_t, trans::cublasOperation_t,
                                          m::Cint, n::Cint, alpha::CuRef{Cfloat},
                                          Aarray::CuPtr{Ptr{Float16}}, lda::Cint,
                                          xarray::CuPtr{Ptr{Float16}}, incx::Cint,
                                          beta::CuRef{Cfloat}, yarray::CuPtr{Ptr{Cfloat}},
                                          incy::Cint, batchCount::Cint)::cublasStatus_t
end

@checked function cublasHSSgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray,
                                          incx, beta, yarray, incy, batchCount)
    initialize_context()
    @ccall libcublas.cublasHSSgemvBatched_64(handle::cublasHandle_t, trans::cublasOperation_t,
                                             m::Int64, n::Int64, alpha::CuRef{Cfloat},
                                             Aarray::CuPtr{Ptr{Float16}}, lda::Int64,
                                             xarray::CuPtr{Ptr{Float16}}, incx::Int64,
                                             beta::CuRef{Cfloat}, yarray::CuPtr{Ptr{Cfloat}},
                                             incy::Int64, batchCount::Int64)::cublasStatus_t
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

@checked function cublasTSTgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray,
                                          incx, beta, yarray, incy, batchCount)
    initialize_context()
    @ccall libcublas.cublasTSTgemvBatched_64(handle::cublasHandle_t, trans::cublasOperation_t,
                                             m::Int64, n::Int64, alpha::Ptr{Cfloat},
                                             Aarray::Ptr{Ptr{BFloat16}}, lda::Int64,
                                             xarray::Ptr{Ptr{BFloat16}}, incx::Int64,
                                             beta::Ptr{Cfloat}, yarray::Ptr{Ptr{BFloat16}},
                                             incy::Int64, batchCount::Int64)::cublasStatus_t
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

@checked function cublasTSSgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray,
                                          incx, beta, yarray, incy, batchCount)
    initialize_context()
    @ccall libcublas.cublasTSSgemvBatched_64(handle::cublasHandle_t, trans::cublasOperation_t,
                                             m::Int64, n::Int64, alpha::Ptr{Cfloat},
                                             Aarray::Ptr{Ptr{BFloat16}}, lda::Int64,
                                             xarray::Ptr{Ptr{BFloat16}}, incx::Int64,
                                             beta::Ptr{Cfloat}, yarray::Ptr{Ptr{Cfloat}},
                                             incy::Int64, batchCount::Int64)::cublasStatus_t
end

@checked function cublasHSHgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA,
                                              x, incx, stridex, beta, y, incy, stridey,
                                              batchCount)
    initialize_context()
    @ccall libcublas.cublasHSHgemvStridedBatched(handle::cublasHandle_t,
                                                 trans::cublasOperation_t, m::Cint, n::Cint,
                                                 alpha::CuRef{Cfloat}, A::CuPtr{Float16},
                                                 lda::Cint, strideA::Clonglong,
                                                 x::CuPtr{Float16}, incx::Cint,
                                                 stridex::Clonglong, beta::CuRef{Cfloat},
                                                 y::CuPtr{Float16}, incy::Cint,
                                                 stridey::Clonglong,
                                                 batchCount::Cint)::cublasStatus_t
end

@checked function cublasHSHgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA,
                                                 x, incx, stridex, beta, y, incy, stridey,
                                                 batchCount)
    initialize_context()
    @ccall libcublas.cublasHSHgemvStridedBatched_64(handle::cublasHandle_t,
                                                    trans::cublasOperation_t, m::Int64, n::Int64,
                                                    alpha::CuRef{Cfloat}, A::CuPtr{Float16},
                                                    lda::Int64, strideA::Clonglong,
                                                    x::CuPtr{Float16}, incx::Int64,
                                                    stridex::Clonglong, beta::CuRef{Cfloat},
                                                    y::CuPtr{Float16}, incy::Int64,
                                                    stridey::Clonglong,
                                                    batchCount::Int64)::cublasStatus_t
end

@checked function cublasHSSgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA,
                                              x, incx, stridex, beta, y, incy, stridey,
                                              batchCount)
    initialize_context()
    @ccall libcublas.cublasHSSgemvStridedBatched(handle::cublasHandle_t,
                                                 trans::cublasOperation_t, m::Cint, n::Cint,
                                                 alpha::CuRef{Cfloat}, A::CuPtr{Float16},
                                                 lda::Cint, strideA::Clonglong,
                                                 x::CuPtr{Float16}, incx::Cint,
                                                 stridex::Clonglong, beta::CuRef{Cfloat},
                                                 y::CuPtr{Cfloat}, incy::Cint,
                                                 stridey::Clonglong,
                                                 batchCount::Cint)::cublasStatus_t
end

@checked function cublasHSSgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA,
                                                 x, incx, stridex, beta, y, incy, stridey,
                                                 batchCount)
    initialize_context()
    @ccall libcublas.cublasHSSgemvStridedBatched_64(handle::cublasHandle_t,
                                                    trans::cublasOperation_t, m::Int64, n::Int64,
                                                    alpha::CuRef{Cfloat}, A::CuPtr{Float16},
                                                    lda::Int64, strideA::Clonglong,
                                                    x::CuPtr{Float16}, incx::Int64,
                                                    stridex::Clonglong, beta::CuRef{Cfloat},
                                                    y::CuPtr{Cfloat}, incy::Int64,
                                                    stridey::Clonglong,
                                                    batchCount::Int64)::cublasStatus_t
end

@checked function cublasTSTgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA,
                                              x, incx, stridex, beta, y, incy, stridey,
                                              batchCount)
    initialize_context()
    @ccall libcublas.cublasTSTgemvStridedBatched(handle::cublasHandle_t,
                                                 trans::cublasOperation_t, m::Cint, n::Cint,
                                                 alpha::CuRef{Cfloat}, A::CuPtr{BFloat16},
                                                 lda::Cint, strideA::Clonglong,
                                                 x::CuPtr{BFloat16}, incx::Cint,
                                                 stridex::Clonglong, beta::CuRef{Cfloat},
                                                 y::CuPtr{BFloat16}, incy::Cint,
                                                 stridey::Clonglong,
                                                 batchCount::Cint)::cublasStatus_t
end

@checked function cublasTSTgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA,
                                                 x, incx, stridex, beta, y, incy, stridey,
                                                 batchCount)
    initialize_context()
    @ccall libcublas.cublasTSTgemvStridedBatched_64(handle::cublasHandle_t,
                                                    trans::cublasOperation_t, m::Int64, n::Int64,
                                                    alpha::CuRef{Cfloat}, A::CuPtr{BFloat16},
                                                    lda::Int64, strideA::Clonglong,
                                                    x::CuPtr{BFloat16}, incx::Int64,
                                                    stridex::Clonglong, beta::CuRef{Cfloat},
                                                    y::CuPtr{BFloat16}, incy::Int64,
                                                    stridey::Clonglong,
                                                    batchCount::Int64)::cublasStatus_t
end

@checked function cublasTSSgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA,
                                              x, incx, stridex, beta, y, incy, stridey,
                                              batchCount)
    initialize_context()
    @ccall libcublas.cublasTSSgemvStridedBatched(handle::cublasHandle_t,
                                                 trans::cublasOperation_t, m::Cint, n::Cint,
                                                 alpha::CuRef{Cfloat}, A::CuPtr{BFloat16},
                                                 lda::Cint, strideA::Clonglong,
                                                 x::CuPtr{BFloat16}, incx::Cint,
                                                 stridex::Clonglong, beta::CuRef{Cfloat},
                                                 y::CuPtr{Cfloat}, incy::Cint,
                                                 stridey::Clonglong,
                                                 batchCount::Cint)::cublasStatus_t
end

@checked function cublasTSSgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA,
                                                 x, incx, stridex, beta, y, incy, stridey,
                                                 batchCount)
    initialize_context()
    @ccall libcublas.cublasTSSgemvStridedBatched_64(handle::cublasHandle_t,
                                                    trans::cublasOperation_t, m::Int64, n::Int64,
                                                    alpha::CuRef{Cfloat}, A::CuPtr{BFloat16},
                                                    lda::Int64, strideA::Clonglong,
                                                    x::CuPtr{BFloat16}, incx::Int64,
                                                    stridex::Clonglong, beta::CuRef{Cfloat},
                                                    y::CuPtr{Cfloat}, incy::Int64,
                                                    stridey::Clonglong,
                                                    batchCount::Int64)::cublasStatus_t
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

@checked function cublasHgemm_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta,
                                 C, ldc)
    initialize_context()
    @ccall libcublas.cublasHgemm_64(handle::cublasHandle_t, transa::cublasOperation_t,
                                    transb::cublasOperation_t, m::Int64, n::Int64, k::Int64,
                                    alpha::Ptr{Float16}, A::Ptr{Float16}, lda::Int64,
                                    B::Ptr{Float16}, ldb::Int64, beta::Ptr{Float16},
                                    C::Ptr{Float16}, ldc::Int64)::cublasStatus_t
end

@checked function cublasHgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                                     Barray, ldb, beta, Carray, ldc, batchCount)
    initialize_context()
    @ccall libcublas.cublasHgemmBatched(handle::cublasHandle_t, transa::cublasOperation_t,
                                        transb::cublasOperation_t, m::Cint, n::Cint,
                                        k::Cint, alpha::CuRef{Float16},
                                        Aarray::CuPtr{Ptr{Float16}}, lda::Cint,
                                        Barray::CuPtr{Ptr{Float16}}, ldb::Cint,
                                        beta::CuRef{Float16},
                                        Carray::CuPtr{Ptr{Float16}}, ldc::Cint,
                                        batchCount::Cint)::cublasStatus_t
end

@checked function cublasHgemmBatched_64(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                                        Barray, ldb, beta, Carray, ldc, batchCount)
    initialize_context()
    @ccall libcublas.cublasHgemmBatched_64(handle::cublasHandle_t, transa::cublasOperation_t,
                                           transb::cublasOperation_t, m::Int64, n::Int64,
                                           k::Int64, alpha::CuRef{Float16},
                                           Aarray::CuPtr{Ptr{Float16}}, lda::Int64,
                                           Barray::CuPtr{Ptr{Float16}}, ldb::Int64,
                                           beta::CuRef{Float16},
                                           Carray::CuPtr{Ptr{Float16}}, ldc::Int64,
                                           batchCount::Int64)::cublasStatus_t
end

@checked function cublasHgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda,
                                            strideA, B, ldb, strideB, beta, C, ldc, strideC,
                                            batchCount)
    initialize_context()
    @ccall libcublas.cublasHgemmStridedBatched(handle::cublasHandle_t,
                                               transa::cublasOperation_t,
                                               transb::cublasOperation_t, m::Cint, n::Cint,
                                               k::Cint, alpha::CuRef{Float16},
                                               A::CuPtr{Float16}, lda::Cint,
                                               strideA::Clonglong, B::CuPtr{Float16},
                                               ldb::Cint, strideB::Clonglong,
                                               beta::CuRef{Float16}, C::CuPtr{Float16},
                                               ldc::Cint, strideC::Clonglong,
                                               batchCount::Cint)::cublasStatus_t
end

@checked function cublasHgemmStridedBatched_64(handle, transa, transb, m, n, k, alpha, A, lda,
                                               strideA, B, ldb, strideB, beta, C, ldc, strideC,
                                               batchCount)
    initialize_context()
    @ccall libcublas.cublasHgemmStridedBatched_64(handle::cublasHandle_t,
                                                  transa::cublasOperation_t,
                                                  transb::cublasOperation_t, m::Int64, n::Int64,
                                                  k::Int64, alpha::CuRef{Float16},
                                                  A::CuPtr{Float16}, lda::Int64,
                                                  strideA::Clonglong, B::CuPtr{Float16},
                                                  ldb::Int64, strideB::Clonglong,
                                                  beta::CuRef{Float16}, C::CuPtr{Float16},
                                                  ldc::Int64, strideC::Clonglong,
                                                  batchCount::Int64)::cublasStatus_t
end
