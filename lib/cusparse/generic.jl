# generic APIs

export vv!, sv!, sm!

## API functions

function sparsetodense(A::Union{CuSparseMatrixCSC{T},CuSparseMatrixCSR{T},CuSparseMatrixCOO{T}}, index::SparseChar, algo::cusparseSparseToDenseAlg_t=CUSPARSE_SPARSETODENSE_ALG_DEFAULT) where {T}
    m,n = size(A)
    B = CUDA.zeros(T,m,n)
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
    local rowPtr, colPtr, B
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
    return B
end

function gather!(X::CuSparseVector, Y::CuVector, index::SparseChar)
    descX = CuSparseVectorDescriptor(X, index)
    descY = CuDenseVectorDescriptor(Y)
    cusparseGather(handle(), descY, descX)
    return X
end

function scatter!(Y::CuVector, X::CuSparseVector, index::SparseChar)
    descX = CuSparseVectorDescriptor(X, index)
    descY = CuDenseVectorDescriptor(Y)
    cusparseScatter(handle(), descX, descY)
    return Y
end

function axpby!(alpha::Number, X::CuSparseVector{T}, beta::Number, Y::CuVector{T}, index::SparseChar) where {T}
    descX = CuSparseVectorDescriptor(X, index)
    descY = CuDenseVectorDescriptor(Y)
    cusparseAxpby(handle(), Ref{T}(alpha), descX, Ref{T}(beta), descY)
    return Y
end

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
             beta::Number, Y::DenseCuVector{T}, index::SparseChar, algo::cusparseSpMVAlg_t=CUSPARSE_MV_ALG_DEFAULT) where {TA, T}

    # Support transa = 'C' for real matrices
    transa = T <: Real && transa == 'C' ? 'T' : transa

    if isa(A, CuSparseMatrixCSC) && transa == 'C' && TA <: Complex
        throw(ArgumentError("Matrix-vector multiplication with the adjoint of a complex CSC matrix" *
                            " is not supported. Use a CSR or COO matrix instead."))
    end

    if isa(A, CuSparseMatrixCSC)
        # cusparseSpMV doesn't support CSC format with CUSPARSE.version() < v"11.6.1"
        # cusparseSpMV supports the CSC format with CUSPARSE.version() ≥ v"11.6.1"
        # but it doesn't work for complex numbers when transa == 'C'
        descA = CuSparseMatrixDescriptor(A, index, convert=true)
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
             B::DenseCuMatrix{T}, beta::Number, C::DenseCuMatrix{T}, index::SparseChar, algo::cusparseSpMMAlg_t=CUSPARSE_MM_ALG_DEFAULT) where {T}

    # Support transa = 'C' and `transb = 'C' for real matrices
    transa = T <: Real && transa == 'C' ? 'T' : transa
    transb = T <: Real && transb == 'C' ? 'T' : transb

    if isa(A, CuSparseMatrixCSC) && transa == 'C' && T <: Complex
        throw(ArgumentError("Matrix-matrix multiplication with the adjoint of a complex CSC matrix" *
                            " is not supported. Use a CSR and COO matrix instead."))
    end

    if isa(A, CuSparseMatrixCSC)
        # cusparseSpMM doesn't support CSC format with CUSPARSE.version() < v"11.6.1"
        # cusparseSpMM supports the CSC format with CUSPARSE.version() ≥ v"11.6.1"
        # but it doesn't work for complex numbers when transa == 'C'
        descA = CuSparseMatrixDescriptor(A, index, convert=true)
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

    function bufferSize()
        out = Ref{Csize_t}()
        cusparseSpMM_bufferSize(
            handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, algo, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        cusparseSpMM(
            handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, algo, buffer)
    end
    return C
end

function mm!(transa::SparseChar, transb::SparseChar, α::Number, A::CuSparseMatrixCSR{T},
             B::CuSparseMatrixCSR{T}, β::Number, C::CuSparseMatrixCSR{T}, index::SparseChar) where {T}
    m,k = size(A)
    n = size(C)[2]
    alpha = convert(T, α)
    beta  = convert(T, β)

    if transa == 'N' && transb == 'N'
        chkmmdims(B,C,k,n,m,n)
    else
        throw(ArgumentError("Sparse mm! only supports transa ($transa) = 'N' and transb ($transb) = 'N'"))
    end

    descA = CuSparseMatrixDescriptor(A, index)
    descB = CuSparseMatrixDescriptor(B, index)
    descC = CuSparseMatrixDescriptor(C, index)

    spgemm_Desc = CuSpGEMMDescriptor()
    function buffer1Size()
        out = Ref{Csize_t}(0)
        cusparseSpGEMM_workEstimation(
            handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, CUSPARSE_SPGEMM_DEFAULT, spgemm_Desc, out, CU_NULL)
        return out[]
    end
    with_workspace(buffer1Size; keep=true) do buffer
        out = Ref{Csize_t}(sizeof(buffer))
        cusparseSpGEMM_workEstimation(
            handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, CUSPARSE_SPGEMM_DEFAULT, spgemm_Desc, out, buffer)
    end
    function buffer2Size()
        out = Ref{Csize_t}(0)
        cusparseSpGEMM_compute(
            handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, CUSPARSE_SPGEMM_DEFAULT, spgemm_Desc, out, CU_NULL)
        return out[]
    end
    with_workspace(buffer2Size; keep=true) do buffer
        out = Ref{Csize_t}(sizeof(buffer))
        cusparseSpGEMM_compute(
            handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, CUSPARSE_SPGEMM_DEFAULT, spgemm_Desc, out, buffer)
    end
    Cm   = Ref{Int64}()
    Cn   = Ref{Int64}()
    Cnnz1 = Ref{Int64}()
    cusparseSpMatGetSize(descC, Cm, Cn, Cnnz1)
    # SpGEMM_copy assumes A*B and C have the same sparsity pattern if
    # beta is not zero. If that isn't the case, we must use broadcasted
    # add to get the correct result.
    if beta == zero(beta)
        unsafe_free!(C.rowPtr)
        unsafe_free!(C.colVal)
        unsafe_free!(C.nzVal)
        C.rowPtr = CuVector{Cint}(undef, Cm[] + 1)
        C.colVal = CuVector{Cint}(undef, Cnnz1[])
        C.nzVal  = CuVector{T}(undef, Cnnz1[])
        C.nnz    = Cnnz1[]
        cusparseCsrSetPointers(descC, C.rowPtr, C.colVal, C.nzVal)
        cusparseSpGEMM_copy(handle(), transa, transb, Ref{T}(alpha), descA, descB,
                            Ref{T}(beta), descC, T, CUSPARSE_SPGEMM_DEFAULT, spgemm_Desc)
        return C
    else
        newC = CuSparseMatrixCSR(CUDA.zeros(Cint, Cm[] + 1), CUDA.zeros(Cint, Cnnz1[]), CUDA.zeros(T, Cnnz1[]), (Cm[], Cn[]))
        descnewC = CuSparseMatrixDescriptor(newC, index)
        cusparseSpGEMM_copy(handle(), transa, transb, Ref{T}(alpha), descA, descB,
                            Ref{T}(beta), descnewC, T, CUSPARSE_SPGEMM_DEFAULT, spgemm_Desc)
        D = beta.*C .+ newC
        return D
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
        # cusparseSpSV doesn't support CSC format
        descA = CuSparseMatrixDescriptor(A, index, convert=true)
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

    if isa(A, CuSparseMatrixCSC) && transa == 'C' && T <: Complex
        throw(ArgumentError("Backward and forward sweeps with the adjoint of a complex CSC matrix is not supported. Use a CSR or COO matrix instead."))
    end

    mA,nA = size(A)
    mB,nB = size(B)
    mC,nC = size(C)

    (mA != nA) && throw(DimensionMismatch("A must be square, but has dimensions ($mA,$nA)!"))
    (mC != mA) && throw(DimensionMismatch("C must have $mA rows, but has $mC rows"))
    (mB != mA) && (transb == 'N') && throw(DimensionMismatch("B must have $mA rows, but has $mB rows"))
    (nB != mA) && (transb != 'N') && throw(DimensionMismatch("B must have $mA columns, but has $nB columns"))
    (nB != nC) && (transb == 'N') && throw(DimensionMismatch("B and C must have the same number of columns, but B has $nB columns and C has $nC columns"))
    (mB != nC) && (transb != 'N') && throw(DimensionMismatch("B must have the same the number of rows that C has as columns, but B has $mB rows and C has $nC columns"))

    if isa(A, CuSparseMatrixCSC)
        # cusparseSpSM doesn't support CSC format
        descA = CuSparseMatrixDescriptor(A, index, convert=true)
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
    descC = CuDenseMatrixDescriptor(C)

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
