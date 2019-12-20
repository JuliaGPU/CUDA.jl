export CUTENSORError

struct CUTENSORError <: Exception
    code::cutensorStatus_t
    msg::AbstractString
end
Base.show(io::IO, err::CUTENSORError) = print(io, "CUTENSORError(code $(err.code), $(err.msg))")

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
    elseif status == CUTENSOR_STATUS_NOT_SUPPORTED
        return "operation not supported (yet)"
    elseif status == CUTENSOR_STATUS_LICENSE_ERROR
        return "error detected trying to check the license"
    elseif status == CUTENSOR_STATUS_CUBLAS_ERROR
        return "error occurred during a CUBLAS operation"
    elseif status == CUTENSOR_STATUS_CUDA_ERROR
        return "error occurred during a CUDA operation"
    elseif status == CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE
        return "insufficient workspace memory for this operation"
    elseif status == CUTENSOR_STATUS_INSUFFICIENT_DRIVER
        return "insufficient driver version"
    else
        return "unknown status"
    end
end


## API call wrapper

# API calls that are allowed without a functional context
const preinit_apicalls = Set{Symbol}([
    :cutensorGetVersion,
    :cutensorGetCudartVersion,
    :cutensorGetErrorString,
])

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(CUTENSORError(res))
end

macro check(ex)
    fun = Symbol(decode_ccall_function(ex))
    init = if !in(fun, preinit_apicalls)
        :(CUDAnative.maybe_initialize())
    end
    quote
        $init

        res = $(esc(ex))
        if res != CUTENSOR_STATUS_SUCCESS
            throw_api_error(res)
        end

        return
    end
end
