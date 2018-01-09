export CUSOLVERError

struct CUSOLVERError <: Exception
    code::cusolverStatus_t
    msg::AbstractString
end
Base.show(io::IO, err::CUSOLVERError) = print(io, "CUSOLVERError(code $(err.code), $(err.msg))")

function CUSOLVERError(code::cusolverStatus_t)
    msg = status_message(code)
    return CUSOLVERError(code, msg)
end

function status_message(status)
    if status == CUSOLVER_STATUS_SUCCESS
        return "the operation completed successfully"
    elseif status == CUSOLVER_STATUS_NOT_INITIALIZED
        return "the library was not initialized"
    elseif status == CUSOLVER_STATUS_ALLOC_FAILED
        return "the resource allocation failed"
    elseif status == CUSOLVER_STATUS_INVALID_VALUE
        return "an invalid value was used as an argument"
    elseif status == CUSOLVER_STATUS_ARCH_MISMATCH
        return "an absent device architectural feature is required"
    elseif status == CUSOLVER_STATUS_EXECUTION_FAILED
        return "the GPU program failed to execute"
    elseif status == CUSOLVER_STATUS_INTERNAL_ERROR
        return "an internal operation failed"
    elseif status == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED
        return "the matrix type is not supported."
    else
        return "unknown status"
    end
end

macro check(solver_func)
    quote
        local err::cusolverStatus_t
        err = $(esc(solver_func::Expr))
        if err != CUSOLVER_STATUS_SUCCESS
            throw(CUSOLVERError(err))
        end
        err
    end
end