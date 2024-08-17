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


## dense vector descriptor

mutable struct CuDenseVectorDescriptor
    handle::cusparseDnVecDescr_t

    function CuDenseVectorDescriptor(T::DataType, n::Integer)
        desc_ref = Ref{cusparseDnVecDescr_t}()
        cusparseCreateDnVec(desc_ref, n, CU_NULL, T)
        obj = new(desc_ref[])
        finalizer(cusparseDestroyDnVec, obj)
        obj
    end

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

    function CuDenseMatrixDescriptor(T::DataType, m::Integer, n::Integer; transposed::Bool=false)
        desc_ref = Ref{cusparseDnMatDescr_t}()
        if transposed
            cusparseCreateDnMat(desc_ref, n, m, m, CU_NULL, T, 'R')
        else
            cusparseCreateDnMat(desc_ref, m, n, m, CU_NULL, T, 'C')
        end
        obj = new(desc_ref[])
        finalizer(cusparseDestroyDnMat, obj)
        obj
    end

    function CuDenseMatrixDescriptor(A::DenseCuMatrix; transposed::Bool=false)
        desc_ref = Ref{cusparseDnMatDescr_t}()
        if transposed
            cusparseCreateDnMat(desc_ref, reverse(size(A))..., stride(A,2), A, eltype(A), 'R')
        else
            cusparseCreateDnMat(desc_ref, size(A)..., stride(A,2), A, eltype(A), 'C')
        end
        obj = new(desc_ref[])
        finalizer(cusparseDestroyDnMat, obj)
        obj
    end

    function CuDenseMatrixDescriptor(A::DenseCuArray{T, 3}; transposed::Bool=false) where T
        desc_ref = Ref{cusparseDnMatDescr_t}()
        if transposed
            cusparseCreateDnMat(desc_ref, size(A,2), size(A,1), stride(A,2), A, eltype(A), 'R')
        else
            cusparseCreateDnMat(desc_ref, size(A,1), size(A,2), stride(A,2), A, eltype(A), 'C')
        end
        obj = new(desc_ref[])
        finalizer(cusparseDestroyDnMat, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cusparseDnMatDescr_t}, desc::CuDenseMatrixDescriptor) = desc.handle


## sparse matrix descriptor

mutable struct CuSparseMatrixDescriptor
    handle::cusparseSpMatDescr_t

    function CuSparseMatrixDescriptor(A::CuSparseMatrixCOO, IndexBase::Char; transposed::Bool=false)
        desc_ref = Ref{cusparseSpMatDescr_t}()
        if transposed
            cusparseCreateCoo(
                desc_ref,
                reverse(size(A))..., nnz(A),
                A.colInd, A.rowInd, nonzeros(A),
                eltype(A.colInd), IndexBase, eltype(nonzeros(A))
            )
        else
            cusparseCreateCoo(
                desc_ref,
                size(A)..., nnz(A),
                A.rowInd, A.colInd, nonzeros(A),
                eltype(A.rowInd), IndexBase, eltype(nonzeros(A))
            )
        end
        obj = new(desc_ref[])
        finalizer(cusparseDestroySpMat, obj)
        return obj
    end

    function CuSparseMatrixDescriptor(::Type{CuSparseMatrixCOO}, Tv::DataType, Ti::DataType, m::Integer, n::Integer, IndexBase::Char)
        desc_ref = Ref{cusparseSpMatDescr_t}()
        cusparseCreateCoo(desc_ref, m, n, Ti(0), CU_NULL, CU_NULL, CU_NULL, Ti, IndexBase, Tv)
        obj = new(desc_ref[])
        finalizer(cusparseDestroySpMat, obj)
        return obj
    end

    function CuSparseMatrixDescriptor(A::CuSparseArrayCSR, IndexBase::Char; transposed::Bool=false)
        desc_ref = Ref{cusparseSpMatDescr_t}()
        if transposed
            cusparseCreateCsc(
                desc_ref,
                reverse(size(A)[1:2])..., nnz(A) ÷ size(A,3),
                A.rowPtr, A.colVal, nonzeros(A),
                eltype(A.rowPtr), eltype(A.colVal), IndexBase, eltype(nonzeros(A))
            )
        else
            cusparseCreateCsr(
                desc_ref,
                size(A)[1:2]..., nnz(A) ÷ size(A,3),
                A.rowPtr, A.colVal, nonzeros(A),
                eltype(A.rowPtr), eltype(A.colVal), IndexBase, eltype(nonzeros(A))
            )
        end
        obj = new(desc_ref[])
        finalizer(cusparseDestroySpMat, obj)
        return obj
    end

    function CuSparseMatrixDescriptor(A::CuSparseMatrixCSR, IndexBase::Char; transposed::Bool=false)
        desc_ref = Ref{cusparseSpMatDescr_t}()
        if transposed
            cusparseCreateCsc(
                desc_ref,
                reverse(size(A))..., nnz(A),
                A.rowPtr, A.colVal, nonzeros(A),
                eltype(A.rowPtr), eltype(A.colVal), IndexBase, eltype(nonzeros(A))
            )
        else
            cusparseCreateCsr(
                desc_ref,
                size(A)..., nnz(A),
                A.rowPtr, A.colVal, nonzeros(A),
                eltype(A.rowPtr), eltype(A.colVal), IndexBase, eltype(nonzeros(A))
            )
        end
        obj = new(desc_ref[])
        finalizer(cusparseDestroySpMat, obj)
        return obj
    end

    function CuSparseMatrixDescriptor(::Type{CuSparseMatrixCSR}, rowPtr::CuVector, Tv::DataType, Ti::DataType, m::Integer, n::Integer, IndexBase::Char)
        desc_ref = Ref{cusparseSpMatDescr_t}()
        cusparseCreateCsr(desc_ref, m, n, Ti(0), rowPtr, CU_NULL, CU_NULL, Ti, Ti, IndexBase, Tv)
        obj = new(desc_ref[])
        finalizer(cusparseDestroySpMat, obj)
        return obj
    end

    function CuSparseMatrixDescriptor(A::CuSparseMatrixCSC, IndexBase::Char; transposed::Bool=false)
        desc_ref = Ref{cusparseSpMatDescr_t}()
        if transposed
            # many algorithms, e.g. mv!, mm!, sv! and sm! do not support CSC matrices but we can use CSR matrices that model their transposed.
            cusparseCreateCsr(
                desc_ref,
                reverse(size(A))..., nnz(A),
                A.colPtr, rowvals(A), nonzeros(A),
                eltype(A.colPtr), eltype(rowvals(A)), IndexBase, eltype(nonzeros(A))
            )
        else
            cusparseCreateCsc(
                desc_ref,
                size(A)..., nnz(A),
                A.colPtr, rowvals(A), nonzeros(A),
                eltype(A.colPtr), eltype(rowvals(A)), IndexBase, eltype(nonzeros(A))
            )
        end
        obj = new(desc_ref[])
        finalizer(cusparseDestroySpMat, obj)
        return obj
    end

    function CuSparseMatrixDescriptor(::Type{CuSparseMatrixCSC}, colPtr::CuVector, Tv::DataType, Ti::DataType, m::Integer, n::Integer, IndexBase::Char)
        desc_ref = Ref{cusparseSpMatDescr_t}()
        cusparseCreateCsc(desc_ref, m, n, Ti(0), colPtr, CU_NULL, CU_NULL, Ti, Ti, IndexBase, Tv)
        obj = new(desc_ref[])
        finalizer(cusparseDestroySpMat, obj)
        return obj
    end

    function CuSparseMatrixDescriptor(A::CuSparseMatrixBSR, IndexBase::Char)
        desc_ref = Ref{cusparseSpMatDescr_t}()
        cusparseCreateBsr(
            desc_ref,
            size(A)..., nnz(A),
            A.blockDim, A.blockDim,
            A.rowPtr, A.colVal, nonzeros(A),
            eltype(A.rowPtr), eltype(A.colVal), IndexBase,
            eltype(nonzeros(A)), A.dir
        )
        obj = new(desc_ref[])
        finalizer(cusparseDestroySpMat, obj)
        return obj
    end
end

Base.unsafe_convert(::Type{cusparseSpMatDescr_t}, desc::CuSparseMatrixDescriptor) = desc.handle

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

mutable struct IC0Info
    info::csric02Info_t

    function IC0Info()
        info_ref = Ref{csric02Info_t}()
        cusparseCreateCsric02Info(info_ref)
        obj = new(info_ref[])
        finalizer(cusparseDestroyCsric02Info, obj)
        obj
    end
end

Base.unsafe_convert(::Type{csric02Info_t}, info::IC0Info) = info.info

mutable struct IC0InfoBSR
    info::bsric02Info_t

    function IC0InfoBSR()
        info_ref = Ref{bsric02Info_t}()
        cusparseCreateBsric02Info(info_ref)
        obj = new(info_ref[])
        finalizer(cusparseDestroyBsric02Info, obj)
        obj
    end
end

Base.unsafe_convert(::Type{bsric02Info_t}, info::IC0InfoBSR) = info.info

mutable struct ILU0Info
    info::csrilu02Info_t

    function ILU0Info()
        info_ref = Ref{csrilu02Info_t}()
        cusparseCreateCsrilu02Info(info_ref)
        obj = new(info_ref[])
        finalizer(cusparseDestroyCsrilu02Info, obj)
        obj
    end
end

Base.unsafe_convert(::Type{csrilu02Info_t}, info::ILU0Info) = info.info

mutable struct ILU0InfoBSR
    info::bsrilu02Info_t

    function ILU0InfoBSR()
        info_ref = Ref{bsrilu02Info_t}()
        cusparseCreateBsrilu02Info(info_ref)
        obj = new(info_ref[])
        finalizer(cusparseDestroyBsrilu02Info, obj)
        obj
    end
end

Base.unsafe_convert(::Type{bsrilu02Info_t}, info::ILU0InfoBSR) = info.info
