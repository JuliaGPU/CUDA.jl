export CUTENSORrror

struct CUTENSORError <: Exception
    code::cutensorStatus_t
    msg::AbstractString
end
Base.show(io::IO, err::CUTENSORError) = print(io, "CUTENSORrror(code $(err.code), $(err.msg))")

function CUTENSORError(code::cutensorStatus_t)
    msg = statusmessage(code)
    return CUTENSORError(code, msg)
end

function statusmessage( status )
    if status == CUTENSOR_STATUS_SUCCESS
        return "the operation completed successfully"
    elseif status == CUTENSOR_STATUS_NOT_INITIALIZED
        return "the library was not initialized"
    elseif status == CUTENSOR_STATUS_ALLOC_FAILED
        return "the resource allocation failed"
    elseif status == CUTENSOR_STATUS_INVALID_VALUE
        return "an invalid value was used as an argument"
    elseif status == CUTENSOR_STATUS_ARCH_MISMATCH
        return "an absent device architectural feature is required"
    elseif status == CUTENSOR_STATUS_MAPPING_ERROR
        return "an access to GPU memory space failed"
    elseif status == CUTENSOR_STATUS_EXECUTION_FAILED
        return "the GPU program failed to execute"
    elseif status == CUTENSOR_STATUS_INTERNAL_ERROR
        return "an internal operation failed"
    elseif status == CUTENSOR_STATUS_LICENSE_ERROR
        return "error detected trying to check the license"
    elseif status == CUTENSOR_STATUS_CUBLAS_ERROR
        return "error occurred during a CUBLAS operation"
    elseif status == CUTENSOR_STATUS_CUDA_ERROR
        return "error occurred during a CUDA operation"
    elseif status == CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE
        return "insufficient workspace memory for this operation"
    else
        return "unknown status"
    end
end

macro check(tensor_func)
    quote
        local err = $(esc(tensor_func::Expr))
        if err != CUTENSOR_STATUS_SUCCESS
            throw(CUTENSORError(cutensorStatus_t(err)))
        end
        err
    end
end

