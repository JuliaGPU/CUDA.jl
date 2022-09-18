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

mutable struct CuSparseSpSVDescriptor
    handle::cusparseSpSVDescr_t

    function CuSparseSpSVDescriptor()
        descr_ref = Ref{cusparseSpSVDescr_t}()
        cusparseSpSV_createDescr(descr_ref)
        obj = new(descr_ref[])
        finalizer(cusparseSpSV_destroyDescr, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cusparseSpSVDescr_t}, desc::CuSparseSpSVDescriptor) = desc.handle

mutable struct CuSparseSpSMDescriptor
    handle::cusparseSpSMDescr_t

    function CuSparseSpSMDescriptor()
        descr_ref = Ref{cusparseSpSMDescr_t}()
        cusparseSpSM_createDescr(descr_ref)
        obj = new(descr_ref[])
        finalizer(cusparseSpSM_destroyDescr, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cusparseSpSMDescr_t}, desc::CuSparseSpSMDescriptor) = desc.handle
