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
    if status == CUFFT_SUCCESS
        return "the operation completed successfully"
    elseif status == CUFFT_INVALID_PLAN
        return "cuFFT was passed an invalid plan handle"
    elseif status == CUFFT_ALLOC_FAILED
        return "cuFFT failed to allocate GPU or CPU memory"
    elseif status == CUFFT_INVALID_TYPE
        return "cuFFT invalid type " # No longer used
    elseif status == CUFFT_INVALID_VALUE
        return "User specified an invalid pointer or parameter"
    elseif status == CUFFT_INTERNAL_ERROR
        return "Driver or internal cuFFT library error"
    elseif status == CUFFT_EXEC_FAILED
        return "Failed to execute an FFT on the GPU"
    elseif status == CUFFT_SETUP_FAILED
        return "The cuFFT library failed to initialize"
    elseif status == CUFFT_INVALID_SIZE
        return "User specified an invalid transform size"
    elseif status == CUFFT_UNALIGNED_DATA
        return "cuFFT unaligned data" # No longer used
    elseif status == CUFFT_INCOMPLETE_PARAMETER_LIST
        return "Missing parameters in call"
    elseif status == CUFFT_INVALID_DEVICE
        return "Execution of a plan was on different GPU than plan creation"
    elseif status == CUFFT_PARSE_ERROR
        return "Internal plan database error"
    elseif status == CUFFT_NO_WORKSPACE
        return "No workspace has been provided prior to plan execution"
    elseif status == CUFFT_NOT_IMPLEMENTED
        return "Function does not implement functionality for parameters given."
    elseif status == CUFFT_LICENSE_ERROR
        return "cuFFT license error" # Used in previous versions.
    elseif status == CUFFT_NOT_SUPPORTED
        return "Operation is not supported for parameters given."
    else
        return "unknown status"
    end
end


## API call wrapper

# API calls that are allowed without a functional context
const preinit_apicalls = Set{Symbol}([
    :cufftGetVersion,
    :cufftGetProperty,
])

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(CUFFTError(res))
end

macro check(ex)
    fun = Symbol(decode_ccall_function(ex))
    init = if !in(fun, preinit_apicalls)
        :(CUDAnative.maybe_initialize())
    end
    quote
        $init

        res = $(esc(ex))
        if res != CUFFT_SUCCESS
            throw_api_error(res)
        end

        return
    end
end
