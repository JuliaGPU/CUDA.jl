export CUTENSORNETError

struct CUTENSORNETError <: Exception
    code::cutensornetStatus_t
end

Base.convert(::Type{cutensornetStatus_t}, err::CUTENSORNETError) = err.code

Base.showerror(io::IO, err::CUTENSORNETError) =
    print(io, "CUTENSORNETError: ", description(err), " (code $(reinterpret(Int32, err.code)), $(name(err)))")

name(err) = string(err.code)

function description(err::CUTENSORNETError)
    if err.code == CUTENSORNET_STATUS_SUCCESS
        return "the operation completed successfully"
    elseif err.code == CUTENSORNET_STATUS_NOT_INITIALIZED
        return "the library was not initialized"
    elseif err.code == CUTENSORNET_STATUS_ALLOC_FAILED
        return "the resource allocation failed"
    elseif err.code == CUTENSORNET_STATUS_INVALID_VALUE
        return "an invalid value was used as an argument"
    elseif err.code == CUTENSORNET_STATUS_ARCH_MISMATCH
        return "an absent device architectural feature is required"
    elseif err.code == CUTENSORNET_STATUS_EXECUTION_FAILED
        return "the GPU program failed to execute"
    elseif err.code == CUTENSORNET_STATUS_INTERNAL_ERROR
        return "an internal operation failed"
    elseif err.code == CUTENSORNET_STATUS_NOT_SUPPORTED
        return "the API is not supported by the backend."
    elseif err.code == CUTENSORNET_STATUS_LICENSE_ERROR
        return "error checking current licensing."
    elseif err.code == CUTENSORNET_STATUS_CUBLAS_ERROR
        return "a call to CUBLAS did not succeed."
    elseif err.code == CUTENSORNET_STATUS_CUDA_ERROR
        return "some unknown CUDA error has occurred."
    elseif err.code == CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE
        return "the workspace on the device is too small to execute."
    elseif err.code == CUTENSORNET_STATUS_INSUFFICIENT_DRIVER
        return "the driver version is insufficient."
    elseif err.code == CUTENSORNET_STATUS_IO_ERROR
        return "an error occurred related to file IO."
    elseif err.code == CUTENSORNET_STATUS_CUTENSOR_VERSION_MISMATCH
        return "the dynamically linked cuTENSOR library is incompatible."
    elseif err.code == CUTENSORNET_STATUS_NO_DEVICE_ALLOCATOR
        return "drawing device memory from a mempool is requested, but the mempool is not set."
    elseif err.code == CUTENSORNET_STATUS_ALL_HYPER_SAMPLES_FAILED
        return "all hyper samples failed for one or more errors."
    else
        return "no description for this error"
    end
end
