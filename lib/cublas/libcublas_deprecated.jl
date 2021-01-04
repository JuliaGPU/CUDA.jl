# Removed in CUDA 11.0

@checked function cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                              beta, C, ldc)
    initialize_api()
    ccall((:cublasDgemm_v2, libcublas()), cublasStatus_t,
                   (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint,
                    Cint, RefOrCuRef{Float16}, CuPtr{Float16}, Cint, CuPtr{Float16}, Cint,
                    RefOrCuRef{Float16}, CuPtr{Float16}, Cint),
                   handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end
