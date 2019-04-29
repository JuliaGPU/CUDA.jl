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
        return "cutensor success"
    end
    if status == CUTENSOR_STATUS_NOT_INITIALIZED
        return "cutensor not initialized"
    end
    if status == CUTENSOR_STATUS_ALLOC_FAILED
        return "cutensor allocation failed"
    end
    if status == CUTENSOR_STATUS_INVALID_VALUE
        return "cutensor invalid value"
    end
    if status == CUTENSOR_STATUS_ARCH_MISMATCH
        return "cutensor architecture mismatch"
    end
    if status == CUTENSOR_STATUS_MAPPING_ERROR
        return "cutensor mapping error"
    end
    if status == CUTENSOR_STATUS_EXECUTION_FAILED
        return "cutensor execution failed"
    end
    if status == CUTENSOR_STATUS_INTERNAL_ERROR
        return "cutensor internal error"
    end
    if status == CUTENSOR_STATUS_LICENSE_ERROR
        return "cutensor license error"
    end
    if status == CUTENSOR_STATUS_CUBLAS_ERROR
        return "cutensor cublas error"
    end
    if status == CUTENSOR_STATUS_CUDA_ERROR
        return "cutensor cuda error"
    end
    if status == CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE
        return "cutensor insufficient workspace error"
    end
    return "unknown cutensor error"
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

