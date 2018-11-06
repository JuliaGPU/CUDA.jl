export CUSPARSError

struct CUSPARSEError <: Exception
    code::cusparseStatus_t
    msg::AbstractString
end
Base.show(io::IO, err::CUSPARSEError) = print(io, "CUSPARSError(code $(err.code), $(err.msg))")

function CUSPARSError(code::cusparseStatus_t)
    msg = status_message(code)
    return CUSPARSEError(code, msg)
end


function statusmessage( status )
    if status == CUSPARSE_STATUS_SUCCESS
        return "cusparse success"
    end
    if status == CUSPARSE_STATUS_NOT_INITIALIZED
        return "cusparse not initialized"
    end
    if status == CUSPARSE_STATUS_ALLOC_FAILED
        return "cusparse allocation failed"
    end
    if status == CUSPARSE_STATUS_INVALID_VALUE
        return "cusparse invalid value"
    end
    if status == CUSPARSE_STATUS_ARCH_MISMATCH
        return "cusparse architecture mismatch"
    end
    if status == CUSPARSE_STATUS_MAPPING_ERROR
        return "cusparse mapping error"
    end
    if status == CUSPARSE_STATUS_EXECUTION_FAILED
        return "cusparse execution failed"
    end
    if status == CUSPARSE_STATUS_INTERNAL_ERROR
        return "cusparse internal error"
    end
    if status == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
        return "cusparse matrix type not supported"
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

