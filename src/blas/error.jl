export CUBLASError

struct CUBLASError <: Exception
    code::cublasStatus_t
end

Base.convert(::Type{cublasStatus_t}, err::CUBLASError) = err.code

Base.showerror(io::IO, err::CUBLASError) =
    print(io, "CUBLASError: ", description(err), " (code $(reinterpret(Int32, err.code)), $(name(err)))")

name(err::CUBLASError) = string(err.code)

## COV_EXCL_START
function description(err)
    if err.code == CUBLAS_STATUS_SUCCESS
        "the operation completed successfully"
    elseif err.code == CUBLAS_STATUS_NOT_INITIALIZED
        "the library was not initialized"
    elseif err.code == CUBLAS_STATUS_ALLOC_FAILED
        "the resource allocation failed"
    elseif err.code == CUBLAS_STATUS_INVALID_VALUE
        "an invalid value was used as an argument"
    elseif err.code == CUBLAS_STATUS_ARCH_MISMATCH
        "an absent device architectural feature is required"
    elseif err.code == CUBLAS_STATUS_MAPPING_ERROR
        "an access to GPU memory space failed"
    elseif err.code == CUBLAS_STATUS_EXECUTION_FAILED
        "the GPU program failed to execute"
    elseif err.code == CUBLAS_STATUS_INTERNAL_ERROR
        "an internal operation failed"
    elseif err.code == CUBLAS_STATUS_NOT_SUPPORTED
        "the requested feature is not supported"
    elseif err.code == CUBLAS_STATUS_LICENSE_ERROR
        "error detected trying to check the license"
    else
        "no description for this error"
    end
end
## COV_EXCL_STOP


## API call wrapper

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(CUBLASError(res))
end

function initialize_api()
    # make sure the calling thread has an active context
    CUDAnative.initialize_context()
end

macro check(ex)
    quote
        res = @retry_reclaim CUBLAS_STATUS_ALLOC_FAILED $(esc(ex))
        if res != CUBLAS_STATUS_SUCCESS
            throw_api_error(res)
        end

        return
    end
end
