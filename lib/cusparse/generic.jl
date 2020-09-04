# generic APIs

## dense vector descriptor

mutable struct CuDenseVectorDescriptor
    handle::cusparseDnVecDescr_t

    function CuDenseVectorDescriptor(x::CuVector)
        desc_ref = Ref{cusparseDnVecDescr_t}()
        cusparseCreateDnVec(desc_ref, length(x), x, eltype(x))
        obj = new(desc_ref[])
        finalizer(cusparseDestroyDnVec, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cusparseDnVecDescr_t}, desc::CuDenseVectorDescriptor) = desc.handle


## dense matrix descriptor

mutable struct CuDenseMatrixDescriptor
    handle::cusparseDnMatDescr_t

    function CuDenseMatrixDescriptor(x::CuMatrix)
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

    function CuSparseMatrixDescriptor(A::CuSparseMatrixCSR)
        desc_ref = Ref{cusparseSpMatDescr_t}()
        cusparseCreateCsr(
            desc_ref,
            A.dims..., length(A.nzVal),
            A.rowPtr, A.colVal, A.nzVal,
            eltype(A.rowPtr), eltype(A.colVal), 'O', eltype(A.nzVal)
        )
        obj = new(desc_ref[])
        finalizer(cusparseDestroySpMat, obj)
        return obj
    end

    function CuSparseMatrixDescriptor(A::CuSparseMatrixCSC)
        desc_ref = Ref{cusparseSpMatDescr_t}()
        cusparseCreateCsr(
            desc_ref,
            reverse(A.dims)..., length(A.nzVal),
            A.colPtr, A.rowVal, A.nzVal,
            eltype(A.colPtr), eltype(A.rowVal), 'O', eltype(A.nzVal)
        )
        obj = new(desc_ref[])
        finalizer(cusparseDestroySpMat, obj)
        return obj
    end
end

Base.unsafe_convert(::Type{cusparseSpMatDescr_t}, desc::CuSparseMatrixDescriptor) = desc.handle


## SpMV

function mv!(
    transa::SparseChar,
    alpha::T,
    A::CuSparseMatrixCSR{T},
    X::CuVector{T},
    beta::T,
    Y::CuVector{T},
    index::SparseChar
) where {T}

    m,n = size(A)

    if transa == 'N'
        chkmvdims(X,n,Y,m)
    end
    if transa == 'T' || transa == 'C'
        chkmvdims(X,m,Y,n)
    end

    cusparseSpMV(
        handle(),
        transa,
        [alpha],
        CuSparseMatrixDescriptor(A),
        CuDenseVectorDescriptor(X),
        [beta],
        CuDenseVectorDescriptor(Y),
        T,
        CUSPARSE_MV_ALG_DEFAULT,
        CU_NULL
    )

    Y
end
