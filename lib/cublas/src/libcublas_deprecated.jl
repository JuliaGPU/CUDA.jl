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
