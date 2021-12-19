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

mutable struct CuSpGEMMDescriptor
    handle::cusparseSpGEMMDescr_t

    function CuSpGEMMDescriptor()
        descr_ref = Ref{cusparseSpGEMMDescr_t}()
        cusparseSpGEMM_createDescr(descr_ref)
        obj = new(descr_ref[])
        finalizer(cusparseSpGEMM_destroyDescr, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cusparseSpGEMMDescr_t}, desc::CuSpGEMMDescriptor) = desc.handle


function CuMatrixDescriptor(MatrixType::Char, FillMode::Char, DiagType::Char, IndexBase::Char)
    desc = CuMatrixDescriptor()
    if MatrixType != 'G'
        cusparseSetMatType(desc, MatrixType)
    end
    cusparseSetMatFillMode(desc, FillMode)
    cusparseSetMatDiagType(desc, DiagType)
    if IndexBase != 'Z'
        cusparseSetMatIndexBase(desc, IndexBase)
    end
    return desc
end
