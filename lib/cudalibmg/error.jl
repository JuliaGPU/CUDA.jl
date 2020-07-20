export CUDALIBMGError

struct CUDALIBMGError <: Exception
    code::cudaLibMgStatus_t
end

Base.convert(::Type{cudaLibMgStatus_t}, err::CUDALIBMGError) = err.code

Base.showerror(io::IO, err::CUDALIBMGError) =
    print(io, "CUDALIBMGError: ", description(err), " (code $(reinterpret(Int32, err.code))")

## COV_EXCL_START
function description(err::CUDALIBMGError)
    if err.code == CUDALIBMG_STATUS_SUCCESS
        "the operation completed successfully"
    elseif err.code == CUDALIBMG_STATUS_NOT_INITIALIZED
        "the library was not initialized"
    elseif err.code == CUDALIBMG_STATUS_ALLOC_FAILED
        "the resource allocation failed"
    elseif err.code == CUDALIBMG_STATUS_INVALID_VALUE
        "an invalid value was used as an argument"
    elseif err.code == CUDALIBMG_STATUS_ARCH_MISMATCH
        "an absent device architectural feature is required"
    elseif err.code == CUDALIBMG_STATUS_MAPPING_ERROR
        "an access to GPU memory space failed"
    elseif err.code == CUDALIBMG_STATUS_EXECUTION_FAILED
        "the GPU program failed to execute"
    elseif err.code == CUDALIBMG_STATUS_INTERNAL_ERROR
        "an internal operation failed"
    elseif err.code == CUDALIBMG_STATUS_NOT_SUPPORTED
        "operation not supported (yet)"
    elseif err.code == CUDALIBMG_STATUS_LICENSE_ERROR
        "error detected trying to check the license"
    elseif err.code == CUDALIBMG_STATUS_CUBLAS_ERROR
        "error occurred during a CUBLAS operation"
    elseif err.code == CUDALIBMG_STATUS_CUDA_ERROR
        "error occurred during a CUDA operation"
    elseif err.code == CUDALIBMG_STATUS_INSUFFICIENT_WORKSPACE
        "insufficient workspace memory for this operation"
    elseif err.code == CUDALIBMG_STATUS_INSUFFICIENT_DRIVER
        "insufficient driver version"
    else
        "no description for this error"
    end
end
## COV_EXCL_STOP


## API call wrapper

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(CUDALIBMGError(res))
end

function initialize_api()
    CUDAnative.prepare_cuda_call()
end

macro check(ex)
    quote
        res = @retry_reclaim CUDALIBMG_STATUS_ALLOC_FAILED $(esc(ex))
        if res != CUDALIBMG_STATUS_SUCCESS
            throw_api_error(res)
        end

        return
    end
end
