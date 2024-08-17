# cuSPARSE types

## index type

function Base.convert(::Type{cusparseIndexType_t}, T::DataType)
    if T == Int16
        return CUSPARSE_INDEX_16U
    elseif T == Int32
        return CUSPARSE_INDEX_32I
    elseif T == Int64
        return CUSPARSE_INDEX_64I
    else
        throw(ArgumentError("CUSPARSE type equivalent for index type $T does not exist!"))
    end
end

function Base.convert(::Type{Type}, T::cusparseIndexType_t)
    if T == CUSPARSE_INDEX_16U
        return Int16
    elseif T == CUSPARSE_INDEX_32I
        return Int32
    elseif T == CUSPARSE_INDEX_64I
        return Int64
    else
        throw(ArgumentError("Julia type equivalent for index type $T does not exist!"))
    end
end


## index base

function Base.convert(::Type{cusparseIndexBase_t}, base::Integer)
    if base == 0
        return CUSPARSE_INDEX_BASE_ZERO
    elseif base == 1
        return CUSPARSE_INDEX_BASE_ONE
    else
        throw(ArgumentError("CUSPARSE does not support index base $base!"))
    end
end

function Base.convert(T::Type{<:Integer}, base::cusparseIndexBase_t)
    if base == CUSPARSE_INDEX_BASE_ZERO
        return T(0)
    elseif base == CUSPARSE_INDEX_BASE_ONE
        return T(1)
    else
        throw(ArgumentError("Unknown index base $base!"))
    end
end


## SparseChar conversions

function Base.convert(::Type{cusparseOperation_t}, trans::SparseChar)
    if trans == 'N'
        CUSPARSE_OPERATION_NON_TRANSPOSE
    elseif trans == 'T'
        CUSPARSE_OPERATION_TRANSPOSE
    elseif trans == 'C'
        CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
    else
        throw(ArgumentError("Unknown operation $trans"))
    end
end

function Base.convert(::Type{cusparseMatrixType_t}, mattype::SparseChar)
    if mattype == 'G'
        CUSPARSE_MATRIX_TYPE_GENERAL
    elseif mattype == 'T'
        CUSPARSE_MATRIX_TYPE_TRIANGULAR
    elseif mattype == 'S'
        CUSPARSE_MATRIX_TYPE_SYMMETRIC
    elseif mattype == 'H'
        CUSPARSE_MATRIX_TYPE_HERMITIAN
    else
        throw(ArgumentError("Unknown matrix type $mattype"))
    end
end

function Base.convert(::Type{cusparseSpMatAttribute_t}, attribute::SparseChar)
    if attribute == 'F'
        CUSPARSE_SPMAT_FILL_MODE
    elseif attribute == 'D'
        CUSPARSE_SPMAT_DIAG_TYPE
    else
        throw(ArgumentError("Unknown attribute $attribute"))
    end
end

function Base.convert(::Type{cusparseFillMode_t}, uplo::SparseChar)
    if uplo == 'U'
        CUSPARSE_FILL_MODE_UPPER
    elseif uplo == 'L'
        CUSPARSE_FILL_MODE_LOWER
    else
        throw(ArgumentError("Unknown fill mode $uplo"))
    end
end

function Base.convert(::Type{cusparseDiagType_t}, diag::SparseChar)
    if diag == 'U'
        CUSPARSE_DIAG_TYPE_UNIT
    elseif diag == 'N'
        CUSPARSE_DIAG_TYPE_NON_UNIT
    else
        throw(ArgumentError("Unknown diag type $diag"))
    end
end

function Base.convert(::Type{cusparseIndexBase_t}, index::SparseChar)
    if index == 'Z'
        CUSPARSE_INDEX_BASE_ZERO
    elseif index == 'O'
        CUSPARSE_INDEX_BASE_ONE
    else
        throw(ArgumentError("Unknown index base"))
    end
end

function Base.convert(::Type{cusparseDirection_t}, dir::SparseChar)
    if dir == 'R'
        CUSPARSE_DIRECTION_ROW
    elseif dir == 'C'
        CUSPARSE_DIRECTION_COLUMN
    else
        throw(ArgumentError("Unknown direction $dir"))
    end
end

function Base.convert(::Type{cusparseOrder_t}, order::SparseChar)
    if order == 'R'
        CUSPARSE_ORDER_ROW
    elseif order == 'C'
        CUSPARSE_ORDER_COL
    else
        throw(ArgumentError("Unknown order $order"))
    end
end

function Base.convert(::Type{cusparseSpSVUpdate_t}, update::SparseChar)
    if update == 'G'
        CUSPARSE_SPSV_UPDATE_GENERAL
    elseif update == 'D'
        CUSPARSE_SPSV_UPDATE_DIAGONAL
    else
        throw(ArgumentError("Unknown update $update"))
    end
end

function Base.convert(::Type{cusparseSpSMUpdate_t}, update::SparseChar)
    if update == 'G'
        CUSPARSE_SPSM_UPDATE_GENERAL
    elseif update == 'D'
        CUSPARSE_SPSM_UPDATE_DIAGONAL
    else
        throw(ArgumentError("Unknown update $update"))
    end
end
