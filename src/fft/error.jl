export CUFFTError

struct CUFFTError <: Exception
    code::cufftResult
    msg::AbstractString
end
Base.show(io::IO, err::CUFFTError) = print(io, "CUFFTError(code $(err.code), $(err.msg))")

function CUFFTError(code::cufftResult)
    msg = status_message(code)
    return CUFFTError(code, msg)
end

function status_message(status)
    if status == CUFFT_STATUS_SUCCESS
        return "the operation completed successfully"
    elseif status == CUFFT_STATUS_INVALID_PLAN
        return "cuFFT was passed an invalid plan handle"
    elseif status == CUFFT_STATUS_ALLOC_FAILED
        return "cuFFT failed to allocate GPU or CPU memory"
    elseif status == CUFFT_STATUS_INVALID_TYPE
        return "cuFFT invalid type " # No longer used
    elseif status == CUFFT_STATUS_INVALID_VALUE
        return "User specified an invalid pointer or parameter"
    elseif status == CUFFT_STATUS_INTERNAL_ERROR
        return "Driver or internal cuFFT library error"
    elseif status == CUFFT_STATUS_EXEC_FAILED
        return "Failed to execute an FFT on the GPU"
    elseif status == CUFFT_STATUS_SETUP_FAILED
        return "The cuFFT library failed to initialize"
    elseif status == CUFFT_STATUS_INVALID_SIZE
        return "User specified an invalid transform size"
    elseif status == CUFFT_STATUS_UNALIGNED_DATA
        return "cuFFT unaligned data" # No longer used
    elseif status == CUFFT_STATUS_INCOMPLETE_PARAMETER_LIST
        return "Missing parameters in call"
    elseif status == CUFFT_STATUS_INVALID_DEVICE
        return "Execution of a plan was on different GPU than plan creation"
    elseif status == CUFFT_STATUS_PARSE_ERROR
        return "Internal plan database error"
    elseif status == CUFFT_STATUS_NO_WORKSPACE
        return "No workspace has been provided prior to plan execution"
    elseif status == CUFFT_STATUS_NOT_IMPLEMENTED
        return "Function does not implement functionality for parameters given."
    elseif status == CUFFT_STATUS_LICENSE_ERROR
        return "cuFFT license error" # Used in previous versions.
    elseif status == CUFFT_STATUS_NOT_SUPPORTED
        return "Operation is not supported for parameters given."
    else
        return "unknown status"
    end
end

macro check(fft_func)
    quote
        local err::cufftResult
        err = $(esc(fft_func::Expr))
        if err != CUFFT_STATUS_SUCCESS
            throw(CUFFTError(err))
        end
        err
    end
end
