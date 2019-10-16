export CUPTIError

struct CUPTIError <: Exception
    code::CUptiResult
    msg::AbstractString
end
Base.show(io::IO, err::CUPTIError) = print(io, "CUPTIError(code $(err.code), $(err.msg))")

function CUPTIError(code)
    msg = status_message(code)
    return CUPTIError(code, msg)
end

function status_message(status)
    if status == CUPTI_SUCCESS
        "no error"
    elseif status == CUPTI_ERROR_INVALID_PARAMETER
        "one or more of the parameters is invalid"
    elseif status == CUPTI_ERROR_INVALID_DEVICE
        "the device does not correspond to a valid CUDA device"
    elseif status == CUPTI_ERROR_INVALID_CONTEXT
        "the context is NULL or not valid"
    elseif status == CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID
        "the event domain id is invalid"
    elseif status == CUPTI_ERROR_INVALID_EVENT_ID
        "the event id is invalid"
    elseif status == CUPTI_ERROR_INVALID_EVENT_NAME
        "the event name is invalid"
    elseif status == CUPTI_ERROR_INVALID_OPERATION
        "the current operation cannot be performed due to dependency on other factors"
    elseif status == CUPTI_ERROR_OUT_OF_MEMORY
        "unable to allocate enough memory to perform the requested operation"
    elseif status == CUPTI_ERROR_HARDWARE
        "an error occurred on the performance monitoring hardware"
    elseif status == CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT
        "the output buffer size is not sufficient to return all requested data"
    elseif status == CUPTI_ERROR_API_NOT_IMPLEMENTED
        "aPI is not implemented"
    elseif status == CUPTI_ERROR_MAX_LIMIT_REACHED
        "the maximum limit is reached"
    elseif status == CUPTI_ERROR_NOT_READY
        "the object is not yet ready to perform the requested operation"
    elseif status == CUPTI_ERROR_NOT_COMPATIBLE
        "The current operation is not compatible with the current state of the object"
    elseif status == CUPTI_ERROR_NOT_INITIALIZED
        "CUPTI is unable to initialize its connection to the CUDA driver"
    elseif status == CUPTI_ERROR_INVALID_METRIC_ID
        "the metric id is invalid"
    elseif status == CUPTI_ERROR_INVALID_METRIC_NAME
        "the metric name is invalid"
    elseif status == CUPTI_ERROR_QUEUE_EMPTY
        "the queue is empty"
    elseif status == CUPTI_ERROR_INVALID_HANDLE
        "invalid handle (internal?)"
    elseif status == CUPTI_ERROR_INVALID_STREAM
        "invalid stream"
    elseif status == CUPTI_ERROR_INVALID_KIND
        "invalid kind"
    elseif status == CUPTI_ERROR_INVALID_EVENT_VALUE
        "invalid event value"
    elseif status == CUPTI_ERROR_DISABLED
        "CUPTI is disabled due to conflicts with other enabled profilers"
    elseif status == CUPTI_ERROR_INVALID_MODULE
        "invalid module"
    elseif status == CUPTI_ERROR_INVALID_METRIC_VALUE
        "invalid metric value"
    elseif status == CUPTI_ERROR_HARDWARE_BUSY
        "the performance monitoring hardware is in use by other client"
    elseif status == CUPTI_ERROR_NOT_SUPPORTED
        "the attempted operation is not supported on the current system or device"
    elseif status == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED
        "unified memory profiling is not supported on the system. Potential reason could be unsupported OS or architecture"
    elseif status == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE
        "Unified memory profiling is not supported on the device"
    elseif status == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES
        "Unified memory profiling is not supported on a multi-GPU configuration without P2P support between any pair of devices"
    elseif status == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_WITH_MPS
        "unified memory profiling is not supported under the Multi-Process Service (MPS) environment with CUDA 7.5"
    elseif status == CUPTI_ERROR_CDP_TRACING_NOT_SUPPORTED
        "Devices with compute capability 7.0 don't support CDP tracing"
    elseif status == CUPTI_ERROR_VIRTUALIZED_DEVICE_NOT_SUPPORTED
        "profiling on virtualized GPU is not supported"
    elseif status == CUPTI_ERROR_CUDA_COMPILER_NOT_COMPATIBLE
        "Profiling results might be incorrect for CUDA applications compiled with nvcc version older than 9.0 for devices with compute capability 6.0 and 6.1"
    elseif status == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES
        "user doesn't have sufficient privileges which are required to start the profiling session"
    elseif status == CUPTI_ERROR_OLD_PROFILER_API_INITIALIZE
        "Old profiling api's are not supported with new profiling api's"
    elseif status == CUPTI_ERROR_OPENACC_UNDEFINED_ROUTINE
        "missing definition of the OpenACC API routine in the linked OpenACC library"
    elseif status == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED
        "an unknown internal error has occurred. Legacy CUPTI Profiling is not supported on devices with Compute Capability 7.5 or higher (Turing+)"
    elseif status == CUPTI_ERROR_UNKNOWN
        "an unknown error has occurred"
    else
        "unknown status"
    end
end


## API call wrapper

# API calls that are allowed without a functional context
const preinit_apicalls = Set{Symbol}([
    :cuptiGetVersion,
    :cuptiGetResultString,
])

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(CUPTIError(res))
end

macro check(ex)
    fun = Symbol(decode_ccall_function(ex))
    init = if !in(fun, preinit_apicalls)
        :(CUDAnative.maybe_initialize())
    end
    quote
        $init

        res = $(esc(ex))
        if res != CUPTI_SUCCESS
            throw_api_error(res)
        end

        return
    end
end
