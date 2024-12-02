# generic APIs

export gather!, scatter!, axpby!, rot!
export vv!, sv!, sm!, gemv, gemm, gemm!, sddmm!
export bmm!

## API functions

function sparsetodense(A::Union{CuSparseMatrixCSC{T},CuSparseMatrixCSR{T},CuSparseMatrixCOO{T}}, index::SparseChar, algo::cusparseSparseToDenseAlg_t=CUSPARSE_SPARSETODENSE_ALG_DEFAULT) where {T}
    m,n = size(A)
    B = CuMatrix{T}(undef, m, n)
    desc_sparse = CuSparseMatrixDescriptor(A, index)
    desc_dense = CuDenseMatrixDescriptor(B)

    function bufferSize()
        out = Ref{Csize_t}()
        cusparseSparseToDense_bufferSize(handle(), desc_sparse, desc_dense, algo, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        cusparseSparseToDense(handle(), desc_sparse, desc_dense, algo, buffer)
    end
    return B
end

function densetosparse(A::CuMatrix{T}, fmt::Symbol, index::SparseChar, algo::cusparseDenseToSparseAlg_t=CUSPARSE_DENSETOSPARSE_ALG_DEFAULT) where {T}
    m,n = size(A)
    local rowPtr, colPtr, desc_sparse, B
    if fmt == :coo
        desc_sparse = CuSparseMatrixDescriptor(CuSparseMatrixCOO, T, Cint, m, n, index)
    elseif fmt == :csr
        rowPtr = CuVector{Cint}(undef, m+1)
        desc_sparse = CuSparseMatrixDescriptor(CuSparseMatrixCSR, rowPtr, T, Cint, m, n, index)
    elseif fmt == :csc
        colPtr = CuVector{Cint}(undef, n+1)
        desc_sparse = CuSparseMatrixDescriptor(CuSparseMatrixCSC, colPtr, T, Cint, m, n, index)
    else
        error("Format :$fmt not available, use :csc, :csr or :coo.")
    end
    desc_dense = CuDenseMatrixDescriptor(A)

    function bufferSize()
        out = Ref{Csize_t}()
        cusparseDenseToSparse_bufferSize(handle(), desc_dense, desc_sparse, algo, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        cusparseDenseToSparse_analysis(handle(), desc_dense, desc_sparse, algo, buffer)
        nnzB = Ref{Int64}()
        cusparseSpMatGetSize(desc_sparse, Ref{Int64}(), Ref{Int64}(), nnzB)
        if fmt == :coo
            rowInd = CuVector{Cint}(undef, nnzB[])
            colInd = CuVector{Cint}(undef, nnzB[])
            nzVal = CuVector{T}(undef, nnzB[])
            B = CuSparseMatrixCOO{T, Cint}(rowInd, colInd, nzVal, (m,n))
            cusparseCooSetPointers(desc_sparse, B.rowInd, B.colInd, B.nzVal)
        elseif fmt == :csr
            colVal = CuVector{Cint}(undef, nnzB[])
            nzVal = CuVector{T}(undef, nnzB[])
            B = CuSparseMatrixCSR{T, Cint}(rowPtr, colVal, nzVal, (m,n))
            cusparseCsrSetPointers(desc_sparse, B.rowPtr, B.colVal, B.nzVal)
        elseif fmt == :csc
            rowVal = CuVector{Cint}(undef, nnzB[])
            nzVal = CuVector{T}(undef, nnzB[])
            B = CuSparseMatrixCSC{T, Cint}(colPtr, rowVal, nzVal, (m,n))
            cusparseCscSetPointers(desc_sparse, B.colPtr, B.rowVal, B.nzVal)
        else
            error("Format :$fmt not available, use :csc, :csr or :coo.")
        end
        cusparseDenseToSparse_convert(handle(), desc_dense, desc_sparse, algo, buffer)
    end
    # Issue with the CUDA routine `cusparseDenseToSparse_convert`
    if (index == 'O') && (fmt == :csc) && (n == 0)
        B.colPtr = CuVector{Cint}([1])
    end
    if (index == 'O') && (fmt == :csr) && (m == 0)
        B.rowPtr = CuVector{Cint}([1])
    end
    return B
end

"""
    gather!(X::CuSparseVector, Y::CuVector, index::SparseChar)

Sets the nonzero elements of `X` equal to the nonzero elements of `Y` at the same indices.
"""
function gather!(X::CuSparseVector, Y::CuVector, index::SparseChar)
    descX = CuSparseVectorDescriptor(X, index)
    descY = CuDenseVectorDescriptor(Y)
    cusparseGather(handle(), descY, descX)
    return X
end

"""
    scatter!(Y::CuVector, X::CuSparseVector, index::SparseChar)

Set `Y[:] = X[:]` for dense `Y` and sparse `X`.
"""
function scatter!(Y::CuVector, X::CuSparseVector, index::SparseChar)
    descX = CuSparseVectorDescriptor(X, index)
    descY = CuDenseVectorDescriptor(Y)
    cusparseScatter(handle(), descX, descY)
    return Y
end

"""
    axpby!(alpha::Number, X::CuSparseVector, beta::Number, Y::CuVector, index::SparseChar)

Computes `alpha * X + beta * Y` for sparse `X` and dense `Y`.
"""
function axpby!(alpha::Number, X::CuSparseVector{T}, beta::Number, Y::CuVector{T}, index::SparseChar) where {T}
    descX = CuSparseVectorDescriptor(X, index)
    descY = CuDenseVectorDescriptor(Y)
    cusparseAxpby(handle(), Ref{T}(alpha), descX, Ref{T}(beta), descY)
    return Y
end

"""
    rot!(X::CuSparseVector, Y::CuVector, c::Number, s::Number, index::SparseChar)

Performs the Givens rotation specified by `c` and `s` to sparse `X` and dense `Y`.
"""
function rot!(X::CuSparseVector{T}, Y::CuVector{T}, c::Number, s::Number, index::SparseChar) where {T}
    descX = CuSparseVectorDescriptor(X, index)
    descY = CuDenseVectorDescriptor(Y)
    cusparseRot(handle(), Ref{T}(c), Ref{T}(s), descX, descY)
    return X, Y
end

function vv!(transx::SparseChar, X::CuSparseVector{T}, Y::DenseCuVector{T}, index::SparseChar) where {T}
    descX = CuSparseVectorDescriptor(X, index)
    descY = CuDenseVectorDescriptor(Y)
    result = Ref{T}()

    function bufferSize()
        out = Ref{Csize_t}()
        cusparseSpVV_bufferSize(handle(), transx, descX, descY, result, T, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        cusparseSpVV(handle(), transx, descX, descY, result, T, buffer)
    end
    return result[]
end

function mv!(transa::SparseChar, alpha::Number, A::Union{CuSparseMatrixCSC{TA},CuSparseMatrixCSR{TA},CuSparseMatrixCOO{TA}}, X::DenseCuVector{T},
             beta::Number, Y::DenseCuVector{T}, index::SparseChar, algo::cusparseSpMVAlg_t=CUSPARSE_SPMV_ALG_DEFAULT) where {TA, T}

    # Support transa = 'C' for real matrices
    transa = T <: Real && transa == 'C' ? 'T' : transa

    if CUSPARSE.version() < v"12.0" && isa(A, CuSparseMatrixCSC) && transa == 'C' && TA <: Complex
        throw(ArgumentError("Matrix-vector multiplication with the adjoint of a complex CSC matrix" *
                            " is not supported by the current CUDA version. Use a CSR or COO matrix instead."))
    end

    if CUSPARSE.version() < v"12.0" && isa(A, CuSparseMatrixCSC)
        # cusparseSpMV completely supports CSC matrices with CUSPARSE.version() ≥ v"12.0".
        # We use Aᵀ to model them as CSR matrices for older versions of CUSPARSE.
        descA = CuSparseMatrixDescriptor(A, index, transposed=true)
        n,m = size(A)
        transa = transa == 'N' ? 'T' : 'N'
    else
        descA = CuSparseMatrixDescriptor(A, index)
        m,n = size(A)
    end

    if transa == 'N'
        chkmvdims(X,n,Y,m)
    elseif transa == 'T' || transa == 'C'
        chkmvdims(X,m,Y,n)
    end

    descX = CuDenseVectorDescriptor(X)
    descY = CuDenseVectorDescriptor(Y)

    # operations with 16-bit numbers always imply mixed-precision computation
    # TODO: we should better model the supported combinations here,
    #       and error if using an unsupported one (like with gemmEx!)
    compute_type = if version() >= v"11.4" && T == Float16
        Float32
    elseif version() >= v"11.7.2" && T == ComplexF16
        ComplexF32
    else
        T
    end

    function bufferSize()
        out = Ref{Csize_t}()
        cusparseSpMV_bufferSize(handle(), transa, Ref{compute_type}(alpha), descA, descX, Ref{compute_type}(beta),
                                descY, compute_type, algo, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        cusparseSpMV(handle(), transa, Ref{compute_type}(alpha), descA, descX, Ref{compute_type}(beta),
                     descY, compute_type, algo, buffer)
    end
    return Y
end

function mm!(transa::SparseChar, transb::SparseChar, alpha::Number, A::Union{CuSparseMatrixCSC{T},CuSparseMatrixCSR{T},CuSparseMatrixCOO{T}},
             B::DenseCuMatrix{T}, beta::Number, C::DenseCuMatrix{T}, index::SparseChar, algo::cusparseSpMMAlg_t=CUSPARSE_SPMM_ALG_DEFAULT) where {T}

    # Support transa = 'C' and `transb = 'C' for real matrices
    transa = T <: Real && transa == 'C' ? 'T' : transa
    transb = T <: Real && transb == 'C' ? 'T' : transb

    if CUSPARSE.version() < v"12.0" && isa(A, CuSparseMatrixCSC) && transa == 'C' && T <: Complex
        throw(ArgumentError("Matrix-matrix multiplication with the adjoint of a complex CSC matrix" *
                            " is not supported by the current CUDA version. Use a CSR or COO matrix instead."))
    end

    if CUSPARSE.version() < v"12.0" && isa(A, CuSparseMatrixCSC)
        # cusparseSpMM completely supports CSC matrices with CUSPARSE.version() ≥ v"12.0".
        # We use Aᵀ to model them as CSR matrices for older versions of CUSPARSE.
        descA = CuSparseMatrixDescriptor(A, index, transposed=true)
        k,m = size(A)
        transa = transa == 'N' ? 'T' : 'N'
    else
        descA = CuSparseMatrixDescriptor(A, index)
        m,k = size(A)
    end
    n = size(C)[2]

    if transa == 'N' && transb == 'N'
        chkmmdims(B,C,k,n,m,n)
    elseif transa == 'N' && transb != 'N'
        chkmmdims(B,C,n,k,m,n)
    elseif transa != 'N' && transb == 'N'
        chkmmdims(B,C,m,n,k,n)
    elseif transa != 'N' && transb != 'N'
        chkmmdims(B,C,n,m,k,n)
    end

    descB = CuDenseMatrixDescriptor(B)
    descC = CuDenseMatrixDescriptor(C)

    # cusparseDnMatSetStridedBatch(descB, size(B,3), size(B,1)*size(B,2))
    # cusparseDnMatSetStridedBatch(descB, size(B,3), size(B,1)*size(B,2))
    # batchsize = length(nonzeros(A)) ÷ nnz(A)
    # if batchsize > 1
    #     cusparseCsrSetStridedBatch(obj, batchsize, 0, nnz(A))
    # end

    # Set default buffer for small matrices (10000 chosen arbitrarly)
    # Otherwise tries to allocate 120TB of memory (see #2296)
    function bufferSize()
        out = Ref{Csize_t}(10000) 
        cusparseSpMM_bufferSize(
            handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, algo, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        # Uncomment if we find a way to reuse the buffer (issue #1362)
        # cusparseSpMM_preprocess(
        #     handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
        #     descC, T, algo, buffer)
        # end
        cusparseSpMM(
            handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, algo, buffer)
    end
    return C
end

function bmm!(transa::SparseChar, transb::SparseChar, alpha::Number, A::CuSparseArrayCSR{T,Ti,N},
              B::DenseCuArray{T,N}, beta::Number, C::DenseCuArray{T,N}, index::SparseChar, algo::cusparseSpMMAlg_t=CUSPARSE_SPMM_ALG_DEFAULT) where {T,Ti,N}
    Ar = reshape(A, :, :, :)
    Br = reshape(B, size(B,1), size(B,2), :)
    Cr = reshape(C, size(C,1), size(C,2), :)
    bmm!(transa, transb, alpha, Ar, Br, beta, Cr, index, algo)
    return C
end

# batched sparse * dense -> dense matmul
function bmm!(transa::SparseChar, transb::SparseChar, alpha::Number, A::CuSparseArrayCSR{T,Ti,3},
              B::DenseCuArray{T,3}, beta::Number, C::DenseCuArray{T,3}, index::SparseChar, algo::cusparseSpMMAlg_t=CUSPARSE_SPMM_ALG_DEFAULT) where {T,Ti}

    if CUSPARSE.version() < v"11.7.2"
        throw(ErrorException("Batched dense-matrix times batched sparse-matrix (bmm!) requires a CUSPARSE version ≥ 11.7.2 (yours: $(CUSPARSE.version()))."))
    end


    # Support transa = 'C' and `transb = 'C' for real matrices
    transa = T <: Real && transa == 'C' ? 'T' : transa
    transb = T <: Real && transb == 'C' ? 'T' : transb

    m, k = size(A)[1:2]
    n, bc = size(C)[2:3]
    b = max(size(A, 3), size(B, 3))

    if b != bc
        throw(ArgumentError("C must have same batch-dimension as max(size(A,3)=$(size(A,3)), size(B,3)=$(size(B,3))), got $(size(C,3))."))
    end

    if n == 1 && b > 1
        throw(ArgumentError("bmm! does not work for n==1 and b>1 due to CUDA error."))
    end

    if transa == 'N' && transb == 'N'
        chkbmmdims(B,C,k,n,m,n)
    elseif transa == 'N' && transb != 'N'
        chkbmmdims(B,C,n,k,m,n)
    elseif transa != 'N' && transb == 'N'
        chkbmmdims(B,C,m,n,k,n)
    elseif transa != 'N' && transb != 'N'
        chkbmmdims(B,C,n,m,k,n)
    end

    descA = CuSparseMatrixDescriptor(A, index)
    descB = CuDenseMatrixDescriptor(B)
    descC = CuDenseMatrixDescriptor(C)

    cusparseCsrSetStridedBatch(descA, b, ptrstride(A), valstride(A))

    strideB = stride(B, 3)
    cusparseDnMatSetStridedBatch(descB, b, strideB)

    strideC = stride(C, 3)
    cusparseDnMatSetStridedBatch(descC, b, strideC)

    # Set default buffer for small matrices (10000 chosen arbitrarly)
    # Otherwise tries to allocate 120TB of memory (see #2296)
    function bufferSize()
        out = Ref{Csize_t}(10000)
        cusparseSpMM_bufferSize(
            handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, algo, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        # We should find a way to reuse the buffer (issue #1362)
        if !(A isa CuSparseMatrixCOO) && (CUSPARSE.version() ≥ v"11.7.2")
            cusparseSpMM_preprocess(
                handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
                descC, T, algo, buffer)
        end
        cusparseSpMM(
            handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, algo, buffer)
    end
    return C
end

function mm!(transa::SparseChar, transb::SparseChar, alpha::Number, A::DenseCuMatrix{T},
             B::Union{CuSparseMatrixCSC{T},CuSparseMatrixCSR{T},CuSparseMatrixCOO{T}},
             beta::Number, C::DenseCuMatrix{T}, index::SparseChar, algo::cusparseSpMMAlg_t=CUSPARSE_SPMM_ALG_DEFAULT) where {T}

    CUSPARSE.version() < v"11.7.4" && throw(ErrorException("This operation is not supported by the current CUDA version."))
    # Support transa = 'C' and `transb = 'C' for real matrices
    transa = T <: Real && transa == 'C' ? 'T' : transa
    transb = T <: Real && transb == 'C' ? 'T' : transb

    # cusparseSpMM can be also used to perform the multiplication of a dense matrix and a sparse matrix by switching the dense matrices layout:
    # Cc = α * Ac * B  + β * Cc → α * Bᵀ * Ar + β * Cr
    # Cc = α * Ac * Bᵀ + β * Cc → α * B  * Ar + β * Cr
    # Cc = α * Ac * Bᴴ + β * Cc → α * B̅  * Ar + β * Cr
    # where B is a sparse matrix, Ac and Cc indicate column-major layout, while Ar and Cr refer to row-major layout.

    if CUSPARSE.version() < v"12.0" && isa(B, CuSparseMatrixCSR) && transb == 'C' && T <: Complex
        throw(ArgumentError("Matrix-matrix multiplication with the adjoint of a complex CSR matrix" *
                            " is not supported by the current CUDA version. Use a CSC or COO matrix instead."))
    end

    m,k = size(A)
    n = size(C)[2]

    if transa == 'N' && transb == 'N'
        chkmmdims(B,C,k,n,m,n)
    elseif transa == 'N' && transb != 'N'
        chkmmdims(B,C,n,k,m,n)
    elseif transa != 'N' && transb == 'N'
        chkmmdims(B,C,m,n,k,n)
    elseif transa != 'N' && transb != 'N'
        chkmmdims(B,C,n,m,k,n)
    end

    descA = CuDenseMatrixDescriptor(A, transposed=true)
    descB = CuSparseMatrixDescriptor(B, index, transposed=true)
    descC = CuDenseMatrixDescriptor(C, transposed=true)

    # Set default buffer for small matrices (10000 chosen arbitrarly)
    # Otherwise tries to allocate 120TB of memory (see #2296)
    function bufferSize()
        out = Ref{Csize_t}(10000)
        cusparseSpMM_bufferSize(
            handle(), transb, transa, Ref{T}(alpha), descB, descA, Ref{T}(beta),
            descC, T, algo, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        # We should find a way to reuse the buffer (issue #1362)
        if !(B isa CuSparseMatrixCOO) && (CUSPARSE.version() ≥ v"11.7.2")
            cusparseSpMM_preprocess(
                handle(), transb, transa, Ref{T}(alpha), descB, descA, Ref{T}(beta),
                descC, T, algo, buffer)
        end
        cusparseSpMM(
            handle(), transb, transa, Ref{T}(alpha), descB, descA, Ref{T}(beta),
            descC, T, algo, buffer)
    end
    return C
end

# AB and C must have the same sparsity pattern if β ≠ 0.
function gemm!(transa::SparseChar, transb::SparseChar, alpha::Number, A::CuSparseMatrixCSR{T}, B::CuSparseMatrixCSR{T},
               beta::Number, C::CuSparseMatrixCSR{T}, index::SparseChar, algo::cusparseSpGEMMAlg_t=CUSPARSE_SPGEMM_DEFAULT) where {T}

    m,k = size(A)
    n = size(C)[2]

    if transa == 'N' && transb == 'N'
        chkmmdims(B,C,k,n,m,n)
    else
        throw(ArgumentError("Sparse matrix-matrix multiplication only supports transa ($transa) = 'N' and transb ($transb) = 'N'"))
    end

    descA = CuSparseMatrixDescriptor(A, index)
    descB = CuSparseMatrixDescriptor(B, index)
    descC = CuSparseMatrixDescriptor(C, index)

    spgemm_desc = CuSpGEMMDescriptor()

    buffer1 = CuVector{UInt8}(undef, 0)
    buffer2 = CuVector{UInt8}(undef, 0)
    GC.@preserve buffer1 buffer2 begin
        # compute an upper bound of the memory required for the intermediate products
        function buffer1Size()
            out = Ref{Csize_t}(0)
            cusparseSpGEMM_workEstimation(
                handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
                descC, T, algo, spgemm_desc, out, CU_NULL)
            return out[]
        end
        with_workspace(buffer1, buffer1Size) do buffer
            out = Ref{Csize_t}(sizeof(buffer))
            cusparseSpGEMM_workEstimation(
                handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
                descC, T, algo, spgemm_desc, out, buffer)
        end

        # compute the structure of the output matrix and its values in a temporary buffer
        function buffer2Size()
            out = Ref{Csize_t}(0)
            cusparseSpGEMM_compute(
                handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
                descC, T, algo, spgemm_desc, out, CU_NULL)
            return out[]
        end
        with_workspace(buffer2, buffer2Size) do buffer
            out = Ref{Csize_t}(sizeof(buffer))
            cusparseSpGEMM_compute(
                handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
                descC, T, algo, spgemm_desc, out, buffer)
        end
        CUDA.unsafe_free!(buffer1)

        # retrieve the size of the output matrix and the number of non-zero elements
        nnzC = Ref{Int64}()
        cusparseSpMatGetSize(descC, Ref{Int64}(), Ref{Int64}(), nnzC)
        # assume that AB and C have the same sparsity pattern if they have the same number
        # of non-zero elements.
        #
        # XXX: even if AB and C have the same number of non-zero elements they could have
        #      different sparsity patterns and the result will be wrong with β ≠ 0...
        if nnz(C) ≠ nnzC[]
            beta ≠ zero(T) && throw(ErrorException("AB and C must have the same sparsity pattern."))
            # AB and C don't have the same sparsity pattern but we are still able to
            # reallocate the memory to store the result of αAB in C because β = 0.
            unsafe_free!(C.rowPtr)
            unsafe_free!(C.colVal)
            unsafe_free!(C.nzVal)
            C.rowPtr = CuVector{Cint}(undef, m+1)
            C.colVal = CuVector{Cint}(undef, nnzC[])
            C.nzVal  = CuVector{T}(undef, nnzC[])
            C.nnz    = nnzC[]
            ## update the descriptor
            cusparseCsrSetPointers(descC, C.rowPtr, C.colVal, C.nzVal)
        end

        # copy the offsets, column indices, and values to the output matrix
        cusparseSpGEMM_copy(handle(), transa, transb, Ref{T}(alpha), descA, descB,
                            Ref{T}(beta), descC, T, algo, spgemm_desc)
        CUDA.unsafe_free!(buffer2)
    end

    return C
end

function gemm!(transa::SparseChar, transb::SparseChar, alpha::Number, A::CuSparseMatrixCSC{T}, B::CuSparseMatrixCSC{T},
               beta::Number, C::CuSparseMatrixCSC{T}, index::SparseChar, algo::cusparseSpGEMMAlg_t=CUSPARSE_SPGEMM_DEFAULT) where {T}
    # C = AB <---> Cᵀ = BᵀAᵀ
    Aᵀ = CuSparseMatrixCSR(A.colPtr, A.rowVal, A.nzVal, reverse(size(A)))
    Bᵀ = CuSparseMatrixCSR(B.colPtr, B.rowVal, B.nzVal, reverse(size(B)))
    Cᵀ = CuSparseMatrixCSR(C.colPtr, C.rowVal, C.nzVal, reverse(size(C)))
    gemm!(transb, transa, alpha, Bᵀ, Aᵀ, beta, Cᵀ, index, algo)
    # If BᵀAᵀ and Cᵀ have the same sparsity pattern, C is already updated after the gemm! call.
    # If BᵀAᵀ and Cᵀ don't have the same sparsity pattern, Cᵀ is reallocated and C must be updated.
    C = CuSparseMatrixCSC(Cᵀ.rowPtr, Cᵀ.colVal, Cᵀ.nzVal, size(C))
    return C
end

function gemm(transa::SparseChar, transb::SparseChar, alpha::Number, A::CuSparseMatrixCSR{T},
              B::CuSparseMatrixCSR{T}, index::SparseChar, algo::cusparseSpGEMMAlg_t=CUSPARSE_SPGEMM_DEFAULT) where {T}

    m,k = size(A)
    l,n = size(B)

    (k != l) && throw(DimensionMismatch("A must have the same the number of columns that B has as rows, but A has $k columns and B has $l columns"))
    !(transa == 'N' && transb == 'N') && throw(ArgumentError("Sparse matrix-matrix multiplication only supports transa ($transa) = 'N' and transb ($transb) = 'N'"))

    descA = CuSparseMatrixDescriptor(A, index)
    descB = CuSparseMatrixDescriptor(B, index)

    rowPtr = CuVector{Cint}(undef, m+1)
    descC = CuSparseMatrixDescriptor(CuSparseMatrixCSR, rowPtr, T, Cint, m, n, index)

    spgemm_desc = CuSpGEMMDescriptor()

    buffer1 = CuVector{UInt8}(undef, 0)
    buffer2 = CuVector{UInt8}(undef, 0)
    GC.@preserve buffer1 buffer1 begin
        # compute an upper bound of the memory required for the intermediate products.
        function buffer1Size()
            out = Ref{Csize_t}(0)
            cusparseSpGEMM_workEstimation(
                handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(0),
                descC, T, algo, spgemm_desc, out, CU_NULL)
            return out[]
        end
        with_workspace(buffer1, buffer1Size) do buffer
            out = Ref{Csize_t}(sizeof(buffer))
            cusparseSpGEMM_workEstimation(
                handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(0),
                descC, T, algo, spgemm_desc, out, buffer)
        end

        # compute the structure of the output matrix and its values in a temporary buffer
        function buffer2Size()
            out = Ref{Csize_t}(0)
            cusparseSpGEMM_compute(
                handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(0),
                descC, T, algo, spgemm_desc, out, CU_NULL)
            return out[]
        end
        with_workspace(buffer2, buffer2Size) do buffer
            out = Ref{Csize_t}(sizeof(buffer))
            cusparseSpGEMM_compute(
                handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(0),
                descC, T, algo, spgemm_desc, out, buffer)
        end
        CUDA.unsafe_free!(buffer1)

        # retrieve the size of the output matrix and the number of non-zero elements
        nnzC = Ref{Int64}()
        cusparseSpMatGetSize(descC, Ref{Int64}(), Ref{Int64}(), nnzC)
        ## allocate the memory to store the result of αAB
        colVal = CuVector{Cint}(undef, nnzC[])
        nzVal = CuVector{T}(undef, nnzC[])
        C = CuSparseMatrixCSR{T, Cint}(rowPtr, colVal, nzVal, (m,n))
        ## update the descriptor
        cusparseCsrSetPointers(descC, C.rowPtr, C.colVal, C.nzVal)

        # copy the offsets, column indices, and values to the output matrix
        cusparseSpGEMM_copy(handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(0),
                            descC, T, algo, spgemm_desc)
        CUDA.unsafe_free!(buffer2)
    end

    return C
end

function gemm(transa::SparseChar, transb::SparseChar, alpha::Number, A::CuSparseMatrixCSC{T},
              B::CuSparseMatrixCSC{T}, index::SparseChar, algo::cusparseSpGEMMAlg_t=CUSPARSE_SPGEMM_DEFAULT) where {T}
    # C = AB <---> Cᵀ = BᵀAᵀ
    Aᵀ = CuSparseMatrixCSR(A.colPtr, A.rowVal, A.nzVal, reverse(size(A)))
    Bᵀ = CuSparseMatrixCSR(B.colPtr, B.rowVal, B.nzVal, reverse(size(B)))
    Cᵀ = gemm(transb, transa, alpha, Bᵀ, Aᵀ, index, algo)
    C = CuSparseMatrixCSC(Cᵀ.rowPtr, Cᵀ.colVal, Cᵀ.nzVal, reverse(size(Cᵀ)))
    return C
end

"""
    y = gemv(transa, alpha, A, x, index, [algo])

Perform a product between a `CuSparseMatrix` and a `CuSparseVector`, returning a `CuSparseVector`.
This function should only be used for highly sparse matrices and vectors, as the result is expected
to have many non-zeros in practice.
For this reason, high-level functions like `mul!` and `*` internally convert the sparse vector into a
dense vector to use a more efficient CUSPARSE routine.

Supported formats for the sparse matrix are `CuSparseMatrixCSC` and `CuSparseMatrixCSR`.
"""
function gemv end

function gemv(transa::SparseChar, alpha::Number, A::CuSparseMatrixCSC{T},
              x::CuSparseVector{T}, index::SparseChar, algo::cusparseSpGEMMAlg_t=CUSPARSE_SPGEMM_DEFAULT) where {T}
    m, n = size(A)
    p = length(x)
    p == n || throw(DimensionMismatch("dimensions must match: x has length $p, A has length $m × $n"))
    # we model x as a CuSparseMatrixCSC with one column.
    B = CuSparseMatrixCSC(x)
    C = gemm(transa, 'N', alpha, A, B, index, algo)
    y = CuSparseVector(C)
    return y
end

function gemv(transa::SparseChar, alpha::Number, A::CuSparseMatrixCSR{T},
              x::CuSparseVector{T}, index::SparseChar, algo::cusparseSpGEMMAlg_t=CUSPARSE_SPGEMM_DEFAULT) where {T}
    m, n = size(A)
    p = length(x)
    p == n || throw(DimensionMismatch("dimensions must match: x has length $p, A has length $m × $n"))
    # we model x as a CuSparseMatrixCSR with one column.
    B = CuSparseMatrixCSR(x)
    C = gemm(transa, 'N', alpha, A, B, index, algo)
    y = CuSparseVector(C)
    return y
end

for SparseMatrixType in (:CuSparseMatrixCSC, :CuSparseMatrixCSR)
    @eval begin
        function gemm(transa::SparseChar, transb::SparseChar, alpha::Number, A::$SparseMatrixType{T}, B::$SparseMatrixType{T},
                      beta::Number, C::$SparseMatrixType{T}, index::SparseChar, algo::cusparseSpGEMMAlg_t=CUSPARSE_SPGEMM_DEFAULT; same_pattern::Bool=false) where {T}
            if same_pattern
                D = copy(C)
                gemm!(transa, transb, alpha, A, B, beta, D, index, algo)
            else
                AB = gemm(transa, transb, one(T), A, B, index, algo)
                D = geam(alpha, AB, beta, C, index)
            end
            return D
        end
    end
end

function sv!(transa::SparseChar, uplo::SparseChar, diag::SparseChar,
             alpha::Number, A::Union{CuSparseMatrixCSC{T},CuSparseMatrixCSR{T},CuSparseMatrixCOO{T}}, X::DenseCuVector{T},
             Y::DenseCuVector{T}, index::SparseChar, algo::cusparseSpSVAlg_t=CUSPARSE_SPSV_ALG_DEFAULT) where {T}

    # Support transa = 'C' for real matrices
    transa = T <: Real && transa == 'C' ? 'T' : transa

    if isa(A, CuSparseMatrixCSC) && transa == 'C' && T <: Complex
        throw(ArgumentError("Backward and forward sweeps with the adjoint of a complex CSC matrix is not supported. Use a CSR or COO matrix instead."))
    end

    mA,nA = size(A)
    mX = length(X)
    mY = length(Y)

    (mA != nA) && throw(DimensionMismatch("A must be square, but has dimensions ($mA,$nA)!"))
    (mX != mA) && throw(DimensionMismatch("X must have length $mA, but has length $mX"))
    (mY != mA) && throw(DimensionMismatch("Y must have length $mA, but has length $mY"))

    if isa(A, CuSparseMatrixCSC)
        # cusparseSpSV doesn't support CSC matrices so we use Aᵀ to model them as CSR matrices.
        descA = CuSparseMatrixDescriptor(A, index, transposed=true)
        transa = transa == 'N' ? 'T' : 'N'
        uplo = uplo == 'U' ? 'L' : 'U'
    else
        descA = CuSparseMatrixDescriptor(A, index)
    end

    cusparse_uplo = Ref{cusparseFillMode_t}(uplo)
    cusparse_diag = Ref{cusparseDiagType_t}(diag)

    cusparseSpMatSetAttribute(descA, 'F', cusparse_uplo, Csize_t(sizeof(cusparse_uplo)))
    cusparseSpMatSetAttribute(descA, 'D', cusparse_diag, Csize_t(sizeof(cusparse_diag)))

    descX = CuDenseVectorDescriptor(X)
    descY = CuDenseVectorDescriptor(Y)

    spsv_desc = CuSparseSpSVDescriptor()
    function bufferSize()
        out = Ref{Csize_t}()
        cusparseSpSV_bufferSize(handle(), transa, Ref{T}(alpha), descA, descX, descY, T, algo, spsv_desc, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        cusparseSpSV_analysis(handle(), transa, Ref{T}(alpha), descA, descX, descY, T, algo, spsv_desc, buffer)
        cusparseSpSV_solve(handle(), transa, Ref{T}(alpha), descA, descX, descY, T, algo, spsv_desc)
    end
    return Y
end

function sm!(transa::SparseChar, transb::SparseChar, uplo::SparseChar, diag::SparseChar,
             alpha::Number, A::Union{CuSparseMatrixCSC{T},CuSparseMatrixCSR{T},CuSparseMatrixCOO{T}}, B::DenseCuMatrix{T},
             C::DenseCuMatrix{T}, index::SparseChar, algo::cusparseSpSMAlg_t=CUSPARSE_SPSM_ALG_DEFAULT) where {T}

    # Support transa = 'C' and `transb = 'C' for real matrices
    transa = T <: Real && transa == 'C' ? 'T' : transa
    transb = T <: Real && transb == 'C' ? 'T' : transb

    # Check if we solve a triangular system in-place with transb != 'N'.
    # In that case we need to update the descriptor of C such that it represents Bᵀ.
    is_C_transposed = (B === C) && (transb != 'N')

    if isa(A, CuSparseMatrixCSC) && transa == 'C' && T <: Complex
        throw(ArgumentError("Backward and forward sweeps with the adjoint of a complex CSC matrix is not supported. Use a CSR or COO matrix instead."))
    end

    mA,nA = size(A)
    mB,nB = size(B)
    mC,nC = !is_C_transposed ? size(C) : reverse(size(C))

    (mA != nA) && throw(DimensionMismatch("A must be square, but has dimensions ($mA,$nA)!"))
    (mC != mA) && throw(DimensionMismatch("C must have $mA rows, but has $mC rows"))
    (mB != mA) && (transb == 'N') && throw(DimensionMismatch("B must have $mA rows, but has $mB rows"))
    (nB != mA) && (transb != 'N') && throw(DimensionMismatch("B must have $mA columns, but has $nB columns"))
    (nB != nC) && (transb == 'N') && throw(DimensionMismatch("B and C must have the same number of columns, but B has $nB columns and C has $nC columns"))
    (mB != nC) && (transb != 'N') && throw(DimensionMismatch("B must have the same the number of rows that C has as columns, but B has $mB rows and C has $nC columns"))

    if isa(A, CuSparseMatrixCSC)
        # cusparseSpSM doesn't support CSC matrices so we use Aᵀ to model them as CSR matrices.
        descA = CuSparseMatrixDescriptor(A, index, transposed=true)
        transa = transa == 'N' ? 'T' : 'N'
        uplo = uplo == 'U' ? 'L' : 'U'
    else
        descA = CuSparseMatrixDescriptor(A, index)
    end

    cusparse_uplo = Ref{cusparseFillMode_t}(uplo)
    cusparse_diag = Ref{cusparseDiagType_t}(diag)

    cusparseSpMatSetAttribute(descA, 'F', cusparse_uplo, Csize_t(sizeof(cusparse_uplo)))
    cusparseSpMatSetAttribute(descA, 'D', cusparse_diag, Csize_t(sizeof(cusparse_diag)))

    descB = CuDenseMatrixDescriptor(B)
    descC = CuDenseMatrixDescriptor(C, transposed=is_C_transposed)

    spsm_desc = CuSparseSpSMDescriptor()
    function bufferSize()
        out = Ref{Csize_t}()
        cusparseSpSM_bufferSize(handle(), transa, transb, Ref{T}(alpha), descA, descB, descC, T, algo, spsm_desc, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        cusparseSpSM_analysis(handle(), transa, transb, Ref{T}(alpha), descA, descB, descC, T, algo, spsm_desc, buffer)
        cusparseSpSM_solve(handle(), transa, transb, Ref{T}(alpha), descA, descB, descC, T, algo, spsm_desc)
    end
    return C
end

function sddmm!(transa::SparseChar, transb::SparseChar, alpha::Number, A::DenseCuMatrix{T}, B::DenseCuMatrix{T},
                beta::Number, C::CuSparseMatrixCSR{T}, index::SparseChar, algo::cusparseSDDMMAlg_t=CUSPARSE_SDDMM_ALG_DEFAULT) where {T}

    CUSPARSE.version() < v"11.4.1" && throw(ErrorException("This operation is not supported by the current CUDA version."))

    # Support transa = 'C' and `transb = 'C' for real matrices
    transa = T <: Real && transa == 'C' ? 'T' : transa
    transb = T <: Real && transb == 'C' ? 'T' : transb

    m,k = size(A)
    n = size(C)[2]

    if transa == 'N' && transb == 'N'
        chkmmdims(B,C,k,n,m,n)
    elseif transa == 'N' && transb != 'N'
        chkmmdims(B,C,n,k,m,n)
    elseif transa != 'N' && transb == 'N'
        chkmmdims(B,C,m,n,k,n)
    elseif transa != 'N' && transb != 'N'
        chkmmdims(B,C,n,m,k,n)
    end

    descA = CuDenseMatrixDescriptor(A)
    descB = CuDenseMatrixDescriptor(B)
    descC = CuSparseMatrixDescriptor(C, index)

    function bufferSize()
        out = Ref{Csize_t}()
        cusparseSDDMM_bufferSize(handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta), descC, T, algo, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        # We should find a way to reuse the buffer (issue #1362)
        cusparseSDDMM_preprocess(handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta), descC, T, algo, buffer)
        cusparseSDDMM(handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta), descC, T, algo, buffer)
    end
    return C
end
