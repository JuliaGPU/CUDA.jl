# cuSPARSE helper functions


## matrix descriptor

mutable struct CuMatrixDescriptor
    handle::cusparseMatDescr_t

    function CuMatrixDescriptor()
        descr_ref = Ref{cusparseMatDescr_t}()
        cusparseCreateMatDescr(descr_ref)
        obj = new(descr_ref[])
        finalizer(cusparseDestroyMatDescr, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cusparseMatDescr_t}, desc::CuMatrixDescriptor) = desc.handle

function CuMatrixDescriptor(MatrixType, FillMode, DiagType, IndexBase)
    desc = CuMatrixDescriptor()
    if MatrixType != CUSPARSE_MATRIX_TYPE_GENERAL
        cusparseSetMatType(desc, MatrixType)
    end
    cusparseSetMatFillMode(desc, FillMode)
    cusparseSetMatDiagType(desc, DiagType)
    if IndexBase != CUSPARSE_INDEX_BASE_ZERO
        cusparseSetMatIndexBase(desc, IndexBase)
    end
    return desc
end
