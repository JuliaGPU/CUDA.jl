export CUSPARSEError

struct CUSPARSEError <: Exception
    code::cusparseStatus_t
    msg::AbstractString
end
Base.show(io::IO, err::CUSPARSEError) = print(io, "CUSPARSEError(code $(err.code), $(err.msg))")

function CUSPARSEError(code::cusparseStatus_t)
    msg = status_message(code)
    return CUSPARSEError(code, msg)
end

function status_message( status )
    if status == CUSPARSE_STATUS_SUCCESS
        return "success"
    end
    if status == CUSPARSE_STATUS_NOT_INITIALIZED
        return "not initialized"
    end
    if status == CUSPARSE_STATUS_ALLOC_FAILED
        return "allocation failed"
    end
    if status == CUSPARSE_STATUS_INVALID_VALUE
        return "invalid value"
    end
    if status == CUSPARSE_STATUS_ARCH_MISMATCH
        return "architecture mismatch"
    end
    if status == CUSPARSE_STATUS_MAPPING_ERROR
        return "mapping error"
    end
    if status == CUSPARSE_STATUS_EXECUTION_FAILED
        return "execution failed"
    end
    if status == CUSPARSE_STATUS_INTERNAL_ERROR
        return "internal error"
    end
    if status == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
        return "matrix type not supported"
    end
end

macro check(sparse_func)
    quote
        local err = $(esc(sparse_func::Expr))
        if err != CUSPARSE_STATUS_SUCCESS
            throw(CUSPARSEError(cusparseStatus_t(err)))
        end
        err
    end
end

