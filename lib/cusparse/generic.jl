# generic APIs

## dense vector descriptor

mutable struct CuDenseVectorDescriptor
    handle::cusparseDnVecDescr_t

    function CuDenseVectorDescriptor(x::DenseCuVector)
        desc_ref = Ref{cusparseDnVecDescr_t}()
        cusparseCreateDnVec(desc_ref, length(x), x, eltype(x))
        obj = new(desc_ref[])
        finalizer(cusparseDestroyDnVec, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cusparseDnVecDescr_t}, desc::CuDenseVectorDescriptor) = desc.handle


## sparse vector descriptor

mutable struct CuSparseVectorDescriptor
    handle::cusparseSpVecDescr_t

    function CuSparseVectorDescriptor(x::CuSparseVector, IndexBase::Char)
        desc_ref = Ref{cusparseSpVecDescr_t}()
        cusparseCreateSpVec(desc_ref, length(x), nnz(x), nonzeroinds(x), nonzeros(x),
                            eltype(nonzeroinds(x)), IndexBase, eltype(x))
        obj = new(desc_ref[])
        finalizer(cusparseDestroySpVec, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cusparseSpVecDescr_t}, desc::CuSparseVectorDescriptor) = desc.handle


## dense matrix descriptor

mutable struct CuDenseMatrixDescriptor
    handle::cusparseDnMatDescr_t

    function CuDenseMatrixDescriptor(x::DenseCuMatrix)
        desc_ref = Ref{cusparseDnMatDescr_t}()
        cusparseCreateDnMat(desc_ref, size(x)..., stride(x,2), x, eltype(x), CUSPARSE_ORDER_COL)
        obj = new(desc_ref[])
        finalizer(cusparseDestroyDnMat, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cusparseDnMatDescr_t}, desc::CuDenseMatrixDescriptor) = desc.handle


## sparse matrix descriptor

mutable struct CuSparseMatrixDescriptor
    handle::cusparseSpMatDescr_t

    function CuSparseMatrixDescriptor(A::CuSparseMatrixCSR, IndexBase::Char)
        desc_ref = Ref{cusparseSpMatDescr_t}()
        cusparseCreateCsr(
            desc_ref,
            size(A)..., length(nonzeros(A)),
            A.rowPtr, A.colVal, nonzeros(A),
            eltype(A.rowPtr), eltype(A.colVal), IndexBase, eltype(nonzeros(A))
        )
        obj = new(desc_ref[])
        finalizer(cusparseDestroySpMat, obj)
        return obj
    end

    function CuSparseMatrixDescriptor(A::CuSparseMatrixCSC, IndexBase::Char; convert=true)
        desc_ref = Ref{cusparseSpMatDescr_t}()
        if convert
            # many algorithms, e.g. mv! and mm!, do not support CSC sparse format
            # so we eagerly convert this to a CSR matrix.
            cusparseCreateCsr(
                desc_ref,
                reverse(size(A))..., length(nonzeros(A)),
                A.colPtr, rowvals(A), nonzeros(A),
                eltype(A.colPtr), eltype(rowvals(A)), IndexBase, eltype(nonzeros(A))
            )
        else
            cusparseCreateCsc(
                desc_ref,
                size(A)..., length(nonzeros(A)),
                A.colPtr, rowvals(A), nonzeros(A),
                eltype(A.colPtr), eltype(rowvals(A)), IndexBase, eltype(nonzeros(A))
            )
        end
        obj = new(desc_ref[])
        finalizer(cusparseDestroySpMat, obj)
        return obj
    end
end

Base.unsafe_convert(::Type{cusparseSpMatDescr_t}, desc::CuSparseMatrixDescriptor) = desc.handle


## API functions

function gather!(X::CuSparseVector, Y::CuVector, index::SparseChar)
    descX = CuSparseVectorDescriptor(X, index)
    descY = CuDenseVectorDescriptor(Y)

    cusparseGather(handle(), descY, descX)

    X
end

function mv!(transa::SparseChar, alpha::Number, A::Union{CuSparseMatrixBSR{TA},CuSparseMatrixCSR{TA}},
             X::DenseCuVector{T}, beta::Number, Y::DenseCuVector{T}, index::SparseChar, algo::cusparseSpMVAlg_t=CUSPARSE_MV_ALG_DEFAULT) where {TA, T}
    m,n = size(A)

    if transa == 'N'
        chkmvdims(X,n,Y,m)
    elseif transa == 'T' || transa == 'C'
        chkmvdims(X,m,Y,n)
    end

    descA = CuSparseMatrixDescriptor(A, index)
    descX = CuDenseVectorDescriptor(X)
    descY = CuDenseVectorDescriptor(Y)
    compute_type = T == Float16 && version() >= v"11.4.0" ? Float32 : T

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
    Y
end

function mv!(transa::SparseChar, alpha::Number, A::CuSparseMatrixCSC{TA}, X::DenseCuVector{T},
             beta::Number, Y::DenseCuVector{T}, index::SparseChar, algo::cusparseSpMVAlg_t=CUSPARSE_MV_ALG_DEFAULT) where {TA, T}
    ctransa = 'N'
    if transa == 'N'
        ctransa = 'T'
    elseif transa == 'C' && TA <: Complex
        throw(ArgumentError("Matrix-vector multiplication with the adjoint of a CSC matrix" *
                            " is not supported. Use a CSR matrix instead."))
    end

    n,m = size(A)

    if ctransa == 'N'
        chkmvdims(X,n,Y,m)
    elseif ctransa == 'T' || ctransa == 'C'
        chkmvdims(X,m,Y,n)
    end

    descA = CuSparseMatrixDescriptor(A, index)
    descX = CuDenseVectorDescriptor(X)
    descY = CuDenseVectorDescriptor(Y)
    compute_type = T == Float16 && version() >= v"11.4.0" ? Float32 : T

    function bufferSize()
        out = Ref{Csize_t}()
        cusparseSpMV_bufferSize(handle(), ctransa, Ref{compute_type}(alpha), descA, descX, Ref{compute_type}(beta),
                                descY, compute_type, algo, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        cusparseSpMV(handle(), ctransa, Ref{compute_type}(alpha), descA, descX, Ref{compute_type}(beta),
                     descY, compute_type, algo, buffer)
    end

    return Y
end

function mm!(transa::SparseChar, transb::SparseChar, alpha::Number, A::CuSparseMatrixCSR{T},
             B::DenseCuMatrix{T}, beta::Number, C::DenseCuMatrix{T}, index::SparseChar, algo::cusparseSpMMAlg_t=CUSPARSE_MM_ALG_DEFAULT) where {T}
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

    descA = CuSparseMatrixDescriptor(A, index)
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

function mm!(transa::SparseChar, transb::SparseChar, alpha::Number, A::CuSparseMatrixCSC{T},
             B::DenseCuMatrix{T}, beta::Number, C::DenseCuMatrix{T}, index::SparseChar, algo::cusparseSpMMAlg_t=CUSPARSE_MM_ALG_DEFAULT) where {T}
    ctransa = 'N'
    if transa == 'N'
        ctransa = 'T'
    elseif transa == 'C' && T <: Complex
        throw(ArgumentError("Matrix-matrix multiplication with the adjoint of a CSC matrix" *
                            " is not supported. Use a CSR matrix instead."))
    end

    k,m = size(A)
    n = size(C)[2]

    if ctransa == 'N' && transb == 'N'
        chkmmdims(B,C,k,n,m,n)
    elseif ctransa == 'N' && transb != 'N'
        chkmmdims(B,C,n,k,m,n)
    elseif ctransa != 'N' && transb == 'N'
        chkmmdims(B,C,m,n,k,n)
    elseif ctransa != 'N' && transb != 'N'
        chkmmdims(B,C,n,m,k,n)
    end

    descA = CuSparseMatrixDescriptor(A, index)
    descB = CuDenseMatrixDescriptor(B)
    descC = CuDenseMatrixDescriptor(C)

    function bufferSize()
        out = Ref{Csize_t}()
        cusparseSpMM_bufferSize(
            handle(), ctransa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, algo, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        cusparseSpMM(
            handle(), ctransa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
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

