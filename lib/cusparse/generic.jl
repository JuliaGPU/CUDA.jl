# generic APIs

## dense vector descriptor

mutable struct CuDenseVectorDescriptor
    handle::cusparseDnVecDescr_t

    function CuDenseVectorDescriptor(v::CuVector{T}) where {T}
        vec_ref = Ref{cusparseDnVecDescr_t}()
        cusparseCreateDnVec(vec_ref, length(v), v, T)
        obj = new(vec_ref[])
        finalizer(cusparseDestroyDnVec, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cusparseDnVecDescr_t}, desc::CuDenseVectorDescriptor) = desc.handle


## sparse matrix descriptor

mutable struct CuSparseMatrixDescriptor
    handle::cusparseSpMatDescr_t
end

function CuSparseMatrixDescriptor(A::CuSparseMatrixCSR{T}) where {T}
    desc_ref = Ref{cusparseSpMatDescr_t}()
    cusparseCreateCsr(
        desc_ref,
        A.dims..., length(A.nzVal),
        A.rowPtr, A.colVal, A.nzVal,
        eltype(A.rowPtr), eltype(A.colVal), 'O', eltype(A.nzVal)
    )
    obj = CuSparseMatrixDescriptor(desc_ref[])
    finalizer(cusparseDestroySpMat, obj)
    return obj
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
