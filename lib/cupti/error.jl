export CUPTIError

struct CUPTIError <: Exception
    code::CUptiResult
end

Base.convert(::Type{CUptiResult}, err::CUPTIError) = err.code

Base.showerror(io::IO, err::CUPTIError) =
    print(io, "CUPTIError: ", description(err), " (code $(reinterpret(Int32, err.code)), $(name(err)))")

function name(err::CUPTIError)
    str_ref = Ref{Cstring}()
    cuptiGetResultString(err, str_ref)
    unsafe_string(str_ref[])
end

@enum_without_prefix CUptiResult CUPTI_

## COV_EXCL_START
function description(err)
    if err.code == SUCCESS
        "no error"
    elseif err.code == ERROR_INVALID_PARAMETER
        "one or more of the parameters is invalid"
    elseif err.code == ERROR_INVALID_DEVICE
        "the device does not correspond to a valid CUDA device"
    elseif err.code == ERROR_INVALID_CONTEXT
        "the context is NULL or not valid"
    elseif err.code == ERROR_INVALID_EVENT_DOMAIN_ID
        "the event domain id is invalid"
    elseif err.code == ERROR_INVALID_EVENT_ID
        "the event id is invalid"
    elseif err.code == ERROR_INVALID_EVENT_NAME
        "the event name is invalid"
    elseif err.code == ERROR_INVALID_OPERATION
        "the current operation cannot be performed due to dependency on other factors"
    elseif err.code == ERROR_OUT_OF_MEMORY
        "unable to allocate enough memory to perform the requested operation"
    elseif err.code == ERROR_HARDWARE
        "an error occurred on the performance monitoring hardware"
    elseif err.code == ERROR_PARAMETER_SIZE_NOT_SUFFICIENT
        "the output buffer size is not sufficient to return all requested data"
    elseif err.code == ERROR_API_NOT_IMPLEMENTED
        "aPI is not implemented"
    elseif err.code == ERROR_MAX_LIMIT_REACHED
        "the maximum limit is reached"
    elseif err.code == ERROR_NOT_READY
        "the object is not yet ready to perform the requested operation"
    elseif err.code == ERROR_NOT_COMPATIBLE
        "the current operation is not compatible with the current state of the object"
    elseif err.code == ERROR_NOT_INITIALIZED
        "CUPTI is unable to initialize its connection to the CUDA driver"
    elseif err.code == ERROR_INVALID_METRIC_ID
        "the metric id is invalid"
    elseif err.code == ERROR_INVALID_METRIC_NAME
        "the metric name is invalid"
    elseif err.code == ERROR_QUEUE_EMPTY
        "the queue is empty"
    elseif err.code == ERROR_INVALID_HANDLE
        "invalid handle (internal?)"
    elseif err.code == ERROR_INVALID_STREAM
        "invalid stream"
    elseif err.code == ERROR_INVALID_KIND
        "invalid kind"
    elseif err.code == ERROR_INVALID_EVENT_VALUE
        "invalid event value"
    elseif err.code == ERROR_DISABLED
        "CUPTI is disabled due to conflicts with other enabled profilers"
    elseif err.code == ERROR_INVALID_MODULE
        "invalid module"
    elseif err.code == ERROR_INVALID_METRIC_VALUE
        "invalid metric value"
    elseif err.code == ERROR_HARDWARE_BUSY
        "the performance monitoring hardware is in use by other client"
    elseif err.code == ERROR_NOT_SUPPORTED
        "the attempted operation is not supported on the current system or device"
    elseif err.code == ERROR_UM_PROFILING_NOT_SUPPORTED
        "unified memory profiling is not supported on the system. Potential reason could be unsupported OS or architecture"
    elseif err.code == ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE
        "unified memory profiling is not supported on the device"
    elseif err.code == ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES
        "unified memory profiling is not supported on a multi-GPU configuration without P2P support between any pair of devices"
    elseif err.code == ERROR_UM_PROFILING_NOT_SUPPORTED_WITH_MPS
        "unified memory profiling is not supported under the Multi-Process Service (MPS) environment with CUDA 7.5"
    elseif err.code == ERROR_CDP_TRACING_NOT_SUPPORTED
        "Devices with compute capability 7.0 don't support CDP tracing"
    elseif err.code == ERROR_VIRTUALIZED_DEVICE_NOT_SUPPORTED
        "profiling on virtualized GPU is not supported"
    elseif err.code == ERROR_CUDA_COMPILER_NOT_COMPATIBLE
        "Profiling results might be incorrect for CUDA applications compiled with nvcc version older than 9.0 for devices with compute capability 6.0 and 6.1"
    elseif err.code == ERROR_INSUFFICIENT_PRIVILEGES
        """Insufficient privileges: You don't have permissions to profile GPU code.
           Please configure your system to allow all users to profile, or run Julia with
           elevated permissions: https://developer.nvidia.com/ERR_NVGPUCTRPERM#SolnAdminTag"""
    elseif err.code == ERROR_OLD_PROFILER_API_INITIALIZED
        "old profiling api's are not supported with new profiling api's"
    elseif err.code == ERROR_OPENACC_UNDEFINED_ROUTINE
        "missing definition of the OpenACC API routine in the linked OpenACC library"
    elseif err.code == ERROR_LEGACY_PROFILER_NOT_SUPPORTED
        "an unknown internal error has occurred. Legacy CUPTI Profiling is not supported on devices with Compute Capability 7.5 or higher (Turing+)"
    elseif err.code == ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED
        "CUPTI doesn't allow multiple callback subscribers. Only a single subscriber can be registered at a time."
    elseif err.code == ERROR_UNKNOWN
        "an unknown error has occurred"
    else
        "unknown status"
    end
end
## COV_EXCL_STOP
