export NVPerfError

struct NVPerfError <: Exception
    code::NVPA_Status
    msg::AbstractString
end
Base.show(io::IO, err::NVPerfError) = print(io, "NVPerfError(code $(err.code), $(err.msg))")

function NVPerfError(code)
    msg = status_message(code)
    return NVPerfError(code, msg)
end

function status_message(status)
    if status == NVPA_STATUS_SUCCESS
        "success"
    elseif status == NVPA_STATUS_ERROR
        "generic error"
    elseif status == NVPA_STATUS_INTERNAL_ERROR
        "internal error"
    elseif status == NVPA_STATUS_NOT_INITIALIZED
        "nvpa_init() has not been called yet"
    elseif status == NVPA_STATUS_NOT_LOADED
        "the NvPerfAPI DLL/DSO could not be loaded during init"
    elseif status == NVPA_STATUS_FUNCTION_NOT_FOUND
        "the function was not found in this version of the NvPerfAPI DLL/DSO"
    elseif status == NVPA_STATUS_NOT_SUPPORTED
        "the request is intentionally not supported by NvPerfAPI"
    elseif status == NVPA_STATUS_NOT_IMPLEMENTED
        "the request is not implemented by this version of NvPerfAPI"
    elseif status == NVPA_STATUS_INVALID_ARGUMENT
        "invalid argument"
    elseif status == NVPA_STATUS_INVALID_METRIC_ID
        "a MetricId argument does not belong to the specified NVPA_Activity or NVPA_Config"
    elseif status == NVPA_STATUS_DRIVER_NOT_LOADED
        "no driver has been loaded via NVPA_*_LoadDriver()"
    elseif status == NVPA_STATUS_OUT_OF_MEMORY
        "failed memory allocation"
    elseif status == NVPA_STATUS_INVALID_THREAD_STATE
        "the request could not be fulfilled due to the state of the current thread"
    elseif status == NVPA_STATUS_FAILED_CONTEXT_ALLOC
        "allocation of context object failed"
    elseif status == NVPA_STATUS_UNSUPPORTED_GPU
        "the specified GPU is not supported"
    elseif status == NVPA_STATUS_INSUFFICIENT_DRIVER_VERSION
        "the installed NVIDIA driver is too old"
    elseif status == NVPA_STATUS_OBJECT_NOT_REGISTERED
        "graphics object has not been registered via NVPA_Register*()"
    elseif status == NVPA_STATUS_INSUFFICIENT_PRIVILEGE
        "the operation failed due to a security check"
    elseif status == NVPA_STATUS_INVALID_CONTEXT_STATE
        "the request could not be fulfilled due to the state of the context"
    elseif status == NVPA_STATUS_INVALID_OBJECT_STATE
        "the request could not be fulfilled due to the state of the object"
    elseif status == NVPA_STATUS_RESOURCE_UNAVAILABLE
        "the request could not be fulfilled because a system resource is already in use"
    elseif status == NVPA_STATUS_DRIVER_LOADED_TOO_LATE
        "the NVPA_*_LoadDriver() is called after the context, command queue or device is created"
    elseif status == NVPA_STATUS_INSUFFICIENT_SPACE
        "the provided buffer is not large enough"
    elseif status == NVPA_STATUS_OBJECT_MISMATCH
        "the API object passed to NVPA_[API]_BeginPass/NVPA_[API]_EndPass and NVPA_[API]_PushRange/NVPA_[API"
    else
        "unknown status"
    end
end


## API call wrapper

# API calls that are allowed without a functional context
const preinit_apicalls = Set{Symbol}([
])

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(NVPerfError(res))
end

macro check(ex)
    fun = Symbol(decode_ccall_function(ex))
    init = if !in(fun, preinit_apicalls)
        :(CUDAnative.maybe_initialize())
    end
    quote
        $init

        res = $(esc(ex))
        if res != NVPA_STATUS_SUCCESS
            throw_api_error(res)
        end

        return
    end
end
