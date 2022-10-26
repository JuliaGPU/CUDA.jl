# Removed in CUDA 11.0

@checked function cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                              beta, C, ldc)
    initialize_context()
    ccall((:cublasHgemm, libcublas), cublasStatus_t,
                   (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint,
                    Cint, RefOrCuRef{Float16}, CuPtr{Float16}, Cint, CuPtr{Float16}, Cint,
                    RefOrCuRef{Float16}, CuPtr{Float16}, Cint),
                   handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

# different signature before CUDA 11.0

@checked function cublasGemmEx_old(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo)
        initialize_context()
        ccall((:cublasGemmEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, PtrOrCuPtr{Cvoid}, CuPtr{Cvoid}, cudaDataType, Cint, CuPtr{Cvoid}, cudaDataType, Cint, PtrOrCuPtr{Cvoid}, CuPtr{Cvoid}, cudaDataType, Cint, cudaDataType, cublasGemmAlgo_t), handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo)
    end
