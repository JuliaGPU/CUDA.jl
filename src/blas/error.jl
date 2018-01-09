export CUBLASError

struct CUBLASError <: Exception
    code::cublasStatus_t
    msg::AbstractString
end
Base.show(io::IO, err::CUBLASError) = print(io, "CUBLASError(code $(err.code), $(err.msg))")

function CUBLASError(code::cublasStatus_t)
    msg = status_message(code)
    return CUBLASError(code, msg)
end

function status_message(status)
    if status == CUBLAS_STATUS_SUCCESS
        return "the operation completed successfully"
    elseif status == CUBLAS_STATUS_NOT_INITIALIZED
        return "the library was not initialized"
    elseif status == CUBLAS_STATUS_ALLOC_FAILED
        return "the resource allocation failed"
    elseif status == CUBLAS_STATUS_INVALID_VALUE
        return "an invalid value was used as an argument"
    elseif status == CUBLAS_STATUS_ARCH_MISMATCH
        return "an absent device architectural feature is required"
    elseif status == CUBLAS_STATUS_MAPPING_ERROR
        return "an access to GPU memory space failed"
    elseif status == CUBLAS_STATUS_EXECUTION_FAILED
        return "the GPU program failed to execute"
    elseif status == CUBLAS_STATUS_INTERNAL_ERROR
        return "an internal operation failed"
    elseif status == CUBLAS_STATUS_NOT_SUPPORTED
        return "the requested feature is not supported"
    elseif status == CUBLAS_STATUS_LICENSE_ERROR
        return "error detected trying to check the license"
    else
        return "unknown status"
    end
end

macro check(blas_func)
    quote
        local err::cublasStatus_t
        err = $(esc(blas_func::Expr))
        if err != CUBLAS_STATUS_SUCCESS
            throw(CUBLASError(err))
        end
        err
    end
end