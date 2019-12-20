export CURANDError

struct CURANDError <: Exception
    code::curandStatus_t
    msg::AbstractString
end
Base.show(io::IO, err::CURANDError) = print(io, "CURANDError(code $(err.code), $(err.msg))")

function CURANDError(code::curandStatus_t)
    msg = status_message(code)
    return CURANDError(code, msg)
end

function status_message(status)
    if status == CURAND_STATUS_SUCCESS
        return "generator was created successfully"
    elseif status == CURAND_STATUS_VERSION_MISMATCH
        return "Header file and linked library version do not match"
    elseif status == CURAND_STATUS_NOT_INITIALIZED
        return "Generator not initialized"
    elseif status == CURAND_STATUS_ALLOCATION_FAILED
        return "Memory allocation failed"
    elseif status == CURAND_STATUS_TYPE_ERROR
        return "Generator is wrong type"
    elseif status == CURAND_STATUS_OUT_OF_RANGE
        return "Argument out of range"
    elseif status == CURAND_STATUS_LENGTH_NOT_MULTIPLE
        return "Length requested is not a multple of dimension"
    elseif status == CURAND_STATUS_DOUBLE_PRECISION_REQUIRED
        return "GPU does not have double precision required by MRG32k3a"
    elseif status == CURAND_STATUS_LAUNCH_FAILURE
        return "Kernel launch failure"
    elseif status == CURAND_STATUS_PREEXISTING_FAILURE
        return "Preexisting failure on library entry"
    elseif status == CURAND_STATUS_INITIALIZATION_FAILED
        return "Initialization of CUDA failed"
    elseif status == CURAND_STATUS_ARCH_MISMATCH
        return "Architecture mismatch, GPU does not support requested feature"
    elseif status == CURAND_STATUS_INTERNAL_ERROR
        return "Internal library error"
    else
        return "unknown status"
    end
end


## API call wrapper

# API calls that are allowed without a functional context
const preinit_apicalls = Set{Symbol}([
    :curandGetVersion,
    :curandGetProperty
])

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(CURANDError(res))
end

macro check(ex)
    fun = Symbol(decode_ccall_function(ex))
    init = if !in(fun, preinit_apicalls)
        :(CUDAnative.maybe_initialize())
    end
    quote
        $init

        res = $(esc(ex))
        if res != CURAND_STATUS_SUCCESS
            throw_api_error(res)
        end

        return
    end
end
