# different signature before CUDA 11.0

@checked function cublasGemmEx_old(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B,
                                   Btype, ldb, beta, C, Ctype, ldc, computeType, algo)
    initialize_context()
    @ccall libcublas.cublasGemmEx(handle::cublasHandle_t, transa::cublasOperation_t,
                                  transb::cublasOperation_t, m::Cint, n::Cint, k::Cint,
                                  alpha::PtrOrCuPtr{Cvoid}, A::CuPtr{Cvoid},
                                  Atype::cudaDataType, lda::Cint, B::CuPtr{Cvoid},
                                  Btype::cudaDataType, ldb::Cint, beta::PtrOrCuPtr{Cvoid},
                                  C::CuPtr{Cvoid}, Ctype::cudaDataType, ldc::Cint,
                                  computeType::cudaDataType,
                                  algo::cublasGemmAlgo_t)::cublasStatus_t
end

@checked function cublasGemmBatchedEx_old(handle, transa, transb, m, n, k, alpha, Aarray,
                                          Atype, lda, Barray, Btype, ldb, beta, Carray,
                                          Ctype, ldc, batchCount, computeType, algo)
    initialize_context()
    @gcsafe_ccall libcublas.cublasGemmBatchedEx(handle::cublasHandle_t,
                                                transa::cublasOperation_t,
                                                transb::cublasOperation_t, m::Cint, n::Cint,
                                                k::Cint, alpha::PtrOrCuPtr{Cvoid},
                                                Aarray::CuPtr{Ptr{Cvoid}},
                                                Atype::cudaDataType, lda::Cint,
                                                Barray::CuPtr{Ptr{Cvoid}},
                                                Btype::cudaDataType, ldb::Cint,
                                                beta::PtrOrCuPtr{Cvoid},
                                                Carray::CuPtr{Ptr{Cvoid}},
                                                Ctype::cudaDataType, ldc::Cint,
                                                batchCount::Cint, computeType::cudaDataType,
                                                algo::cublasGemmAlgo_t)::cublasStatus_t
end

@checked function cublasGemmStridedBatchedEx_old(handle, transa, transb, m, n, k, alpha, A,
                                                 Atype, lda, strideA, B, Btype, ldb,
                                                 strideB, beta, C, Ctype, ldc, strideC,
                                                 batchCount, computeType, algo)
    initialize_context()
    @gcsafe_ccall libcublas.cublasGemmStridedBatchedEx(handle::cublasHandle_t,
                                                       transa::cublasOperation_t,
                                                       transb::cublasOperation_t, m::Cint,
                                                       n::Cint, k::Cint,
                                                       alpha::PtrOrCuPtr{Cvoid},
                                                       A::CuPtr{Cvoid}, Atype::cudaDataType,
                                                       lda::Cint, strideA::Clonglong,
                                                       B::CuPtr{Cvoid}, Btype::cudaDataType,
                                                       ldb::Cint, strideB::Clonglong,
                                                       beta::PtrOrCuPtr{Cvoid},
                                                       C::CuPtr{Cvoid}, Ctype::cudaDataType,
                                                       ldc::Cint, strideC::Clonglong,
                                                       batchCount::Cint,
                                                       computeType::cudaDataType,
                                                       algo::cublasGemmAlgo_t)::cublasStatus_t
end
