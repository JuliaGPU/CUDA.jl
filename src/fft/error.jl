export CUFFTError

struct CUFFTError <: Exception
    code::cufftResult
end

Base.convert(::Type{cufftResult}, err::CUFFTError) = err.code

Base.showerror(io::IO, err::CUFFTError) =
    print(io, "CUFFTError: ", description(err), " (code $(reinterpret(Int32, err.code)), $(name(err)))")

name(err::CUFFTError) = string(err.code)

## COV_EXCL_START
function description(err::CUFFTError)
    if err.code == CUFFT_SUCCESS
        "the operation completed successfully"
    elseif err.code == CUFFT_INVALID_PLAN
        "cuFFT was passed an invalid plan handle"
    elseif err.code == CUFFT_ALLOC_FAILED
        "cuFFT failed to allocate GPU or CPU memory"
    elseif err.code == CUFFT_INVALID_TYPE
        "cuFFT invalid type " # No longer used
    elseif err.code == CUFFT_INVALID_VALUE
        "user specified an invalid pointer or parameter"
    elseif err.code == CUFFT_INTERNAL_ERROR
        "driver or internal cuFFT library error"
    elseif err.code == CUFFT_EXEC_FAILED
        "failed to execute an FFT on the GPU"
    elseif err.code == CUFFT_SETUP_FAILED
        "the cuFFT library failed to initialize"
    elseif err.code == CUFFT_INVALID_SIZE
        "user specified an invalid transform size"
    elseif err.code == CUFFT_UNALIGNED_DATA
        "cuFFT unaligned data" # No longer used
    elseif err.code == CUFFT_INCOMPLETE_PARAMETER_LIST
        "missing parameters in call"
    elseif err.code == CUFFT_INVALID_DEVICE
        "execution of a plan was on different GPU than plan creation"
    elseif err.code == CUFFT_PARSE_ERROR
        "internal plan database error"
    elseif err.code == CUFFT_NO_WORKSPACE
        "no workspace has been provided prior to plan execution"
    elseif err.code == CUFFT_NOT_IMPLEMENTED
        "function does not implement functionality for parameters given."
    elseif err.code == CUFFT_LICENSE_ERROR
        "cuFFT license error" # Used in previous versions.
    elseif err.code == CUFFT_NOT_SUPPORTED
        "operation is not supported for parameters given."
    else
        "no description for this error"
    end
end
## COV_EXCL_STOP


## API call wrapper

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(CUFFTError(res))
end

function initialize_api()
    # make sure the calling thread has an active context
    CUDAnative.initialize_context()
end

macro check(ex)
    quote
        res = $(esc(ex))
        if res != CUFFT_SUCCESS
            throw_api_error(res)
        end

        return
    end
end
