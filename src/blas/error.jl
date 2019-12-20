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


## API call wrapper

# API calls that are allowed without a functional context
const preinit_apicalls = Set{Symbol}([
    :cublasGetVersion,
    :cublasGetProperty,
    :cublasGetCudartVersion
])

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(CuError(res))
end

macro check(ex)
    fun = Symbol(decode_ccall_function(ex))
    init = if !in(fun, preinit_apicalls)
        :(CUDAnative.maybe_initialize($(QuoteNode(fun))))
    end
    quote
        $init

        res = $(esc(ex))
        if res != CUBLAS_STATUS_SUCCESS
            throw_api_error(res)
        end

        return
    end
end
