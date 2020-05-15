export CUTENSORError

struct CUTENSORError <: Exception
    code::cutensorStatus_t
end

Base.convert(::Type{cutensorStatus_t}, err::CUTENSORError) = err.code

Base.showerror(io::IO, err::CUTENSORError) =
    print(io, "CUTENSORError: ", description(err), " (code $(reinterpret(Int32, err.code)), $(name(err)))")

name(err::CUTENSORError) = unsafe_string(cutensorGetErrorString(err))

## COV_EXCL_START
function description(err::CUTENSORError)
    if err.code == CUTENSOR_STATUS_SUCCESS
        "the operation completed successfully"
    elseif err.code == CUTENSOR_STATUS_NOT_INITIALIZED
        "the library was not initialized"
    elseif err.code == CUTENSOR_STATUS_ALLOC_FAILED
        "the resource allocation failed"
    elseif err.code == CUTENSOR_STATUS_INVALID_VALUE
        "an invalid value was used as an argument"
    elseif err.code == CUTENSOR_STATUS_ARCH_MISMATCH
        "an absent device architectural feature is required"
    elseif err.code == CUTENSOR_STATUS_MAPPING_ERROR
        "an access to GPU memory space failed"
    elseif err.code == CUTENSOR_STATUS_EXECUTION_FAILED
        "the GPU program failed to execute"
    elseif err.code == CUTENSOR_STATUS_INTERNAL_ERROR
        "an internal operation failed"
    elseif err.code == CUTENSOR_STATUS_NOT_SUPPORTED
        "operation not supported (yet)"
    elseif err.code == CUTENSOR_STATUS_LICENSE_ERROR
        "error detected trying to check the license"
    elseif err.code == CUTENSOR_STATUS_CUBLAS_ERROR
        "error occurred during a CUBLAS operation"
    elseif err.code == CUTENSOR_STATUS_CUDA_ERROR
        "error occurred during a CUDA operation"
    elseif err.code == CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE
        "insufficient workspace memory for this operation"
    elseif err.code == CUTENSOR_STATUS_INSUFFICIENT_DRIVER
        "insufficient driver version"
    else
        "no description for this error"
    end
end
## COV_EXCL_STOP


## API call wrapper

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(CUTENSORError(res))
end

function initialize_api()
    CUDAnative.prepare_cuda_call()
end

macro check(ex)
    quote
        res = @retry_reclaim CUTENSOR_STATUS_ALLOC_FAILED $(esc(ex))
        if res != CUTENSOR_STATUS_SUCCESS
            throw_api_error(res)
        end

        return
    end
end
