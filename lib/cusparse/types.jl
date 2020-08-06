# type conversions

Base.convert(::Type{cusparseIndexType_t}, ::Type{UInt16}) = CUSPARSE_INDEX_16U
Base.convert(::Type{cusparseIndexType_t}, ::Type{Int32})  = CUSPARSE_INDEX_32I
Base.convert(::Type{cusparseIndexType_t}, ::Type{Int64})  = CUSPARSE_INDEX_64I


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
        throw(ArgumentError("Unknown diag mode $diag"))
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
