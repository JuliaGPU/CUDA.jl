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

    function CuSparseMatrixDescriptor(A::CuSparseMatrixCSR)
        desc_ref = Ref{cusparseSpMatDescr_t}()
        cusparseCreateCsr(
            desc_ref,
            A.dims..., length(nonzeros(A)),
            A.rowPtr, A.colVal, nonzeros(A),
            eltype(A.rowPtr), eltype(A.colVal), 'O', eltype(nonzeros(A))
        )
        obj = new(desc_ref[])
        finalizer(cusparseDestroySpMat, obj)
        return obj
    end

    function CuSparseMatrixDescriptor(A::CuSparseMatrixCSC)
        desc_ref = Ref{cusparseSpMatDescr_t}()
        cusparseCreateCsr(
            desc_ref,
            reverse(A.dims)..., length(nonzeros(A)),
            A.colPtr, rowvals(A), nonzeros(A),
            eltype(A.colPtr), eltype(rowvals(A)), 'O', eltype(nonzeros(A))
        )
        obj = new(desc_ref[])
        finalizer(cusparseDestroySpMat, obj)
        return obj
    end
end

Base.unsafe_convert(::Type{cusparseSpMatDescr_t}, desc::CuSparseMatrixDescriptor) = desc.handle


## API functions

function mv!(transa::SparseChar, alpha::Number, A::Union{CuSparseMatrixBSR{T},CuSparseMatrixCSR{T}},
             X::DenseCuVector{T}, beta::Number, Y::DenseCuVector{T}, index::SparseChar) where {T}
    m,n = size(A)

    if transa == 'N'
        chkmvdims(X,n,Y,m)
    elseif transa == 'T' || transa == 'C'
        chkmvdims(X,m,Y,n)
    end

    descA = CuSparseMatrixDescriptor(A)
    descX = CuDenseVectorDescriptor(X)
    descY = CuDenseVectorDescriptor(Y)

    @workspace size=@argout(
            cusparseSpMV_bufferSize(handle(), transa, Ref{T}(alpha), descA, descX, Ref{T}(beta),
                                    descY, T, CUSPARSE_MV_ALG_DEFAULT, out(Ref{Csize_t}()))
        )[] buffer->begin
            cusparseSpMV(handle(), transa, Ref{T}(alpha), descA, descX, Ref{T}(beta),
                         descY, T, CUSPARSE_MV_ALG_DEFAULT, buffer)
        end
    Y
end

function mv!(transa::SparseChar, alpha::Number, A::CuSparseMatrixCSC{T}, X::DenseCuVector{T},
             beta::Number, Y::DenseCuVector{T}, index::SparseChar) where {T}
    ctransa = 'N'
    if transa == 'N'
        ctransa = 'T'
    elseif transa == 'C' && T <: Complex
        throw(ArgumentError("Matrix-vector multiplication with the adjoint of a CSC matrix" *
                            " is not supported. Use a CSR matrix instead."))
    end

    n,m = size(A)

    if ctransa == 'N'
        chkmvdims(X,n,Y,m)
    elseif ctransa == 'T' || ctransa == 'C'
        chkmvdims(X,m,Y,n)
    end

    descA = CuSparseMatrixDescriptor(A)
    descX = CuDenseVectorDescriptor(X)
    descY = CuDenseVectorDescriptor(Y)

    @workspace size=@argout(
            cusparseSpMV_bufferSize(handle(), ctransa, Ref{T}(alpha), descA, descX, Ref{T}(beta),
                                    descY, T, CUSPARSE_MV_ALG_DEFAULT, out(Ref{Csize_t}()))
        )[] buffer->begin
            cusparseSpMV(handle(), ctransa, Ref{T}(alpha), descA, descX, Ref{T}(beta),
                         descY, T, CUSPARSE_MV_ALG_DEFAULT, buffer)
        end

    return Y
end

function mm!(transa::SparseChar, transb::SparseChar, alpha::Number, A::CuSparseMatrixCSR{T},
             B::DenseCuMatrix{T}, beta::Number, C::DenseCuMatrix{T}, index::SparseChar) where {T}
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

    descA = CuSparseMatrixDescriptor(A)
    descB = CuDenseMatrixDescriptor(B)
    descC = CuDenseMatrixDescriptor(C)

    @workspace size=@argout(
        cusparseSpMM_bufferSize(
            handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, CUSPARSE_MM_ALG_DEFAULT, out(Ref{Csize_t}()))
    )[] buffer->begin
        cusparseSpMM(
            handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, CUSPARSE_MM_ALG_DEFAULT, buffer)
    end

    return C
end

function mm!(transa::SparseChar, transb::SparseChar, alpha::Number, A::CuSparseMatrixCSC{T},
             B::DenseCuMatrix{T}, beta::Number, C::DenseCuMatrix{T}, index::SparseChar) where {T}
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

    descA = CuSparseMatrixDescriptor(A)
    descB = CuDenseMatrixDescriptor(B)
    descC = CuDenseMatrixDescriptor(C)

    @workspace size=@argout(
        cusparseSpMM_bufferSize(
            handle(), ctransa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, CUSPARSE_MM_ALG_DEFAULT, out(Ref{Csize_t}()))
    )[] buffer->begin
        cusparseSpMM(
            handle(), ctransa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, CUSPARSE_MM_ALG_DEFAULT, buffer)
    end

    return C
end
