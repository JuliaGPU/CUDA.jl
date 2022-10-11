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
        "the operation completed successfully"
    elseif err.code == CUTENSORNET_STATUS_NOT_INITIALIZED
        "the library was not initialized"
    elseif err.code == CUTENSORNET_STATUS_ALLOC_FAILED
        "the resource allocation failed"
    elseif err.code == CUTENSORNET_STATUS_INVALID_VALUE
        "an invalid value was used as an argument"
    elseif err.code == CUTENSORNET_STATUS_ARCH_MISMATCH
        "an absent device architectural feature is required"
    elseif err.code == CUTENSORNET_STATUS_EXECUTION_FAILED
        "the GPU program failed to execute"
    elseif err.code == CUTENSORNET_STATUS_INTERNAL_ERROR
        "an internal operation failed"
    elseif err.code == CUTENSORNET_STATUS_NOT_SUPPORTED
        "the API is not supported by the backend."
    elseif err.code == CUTENSORNET_STATUS_LICENSE_ERROR
        "error checking current licensing."
    elseif err.code == CUTENSORNET_STATUS_CUBLAS_ERROR
        "a call to CUBLAS did not succeed."
    elseif err.code == CUTENSORNET_STATUS_CUDA_ERROR
        "some unknown CUDA error has occurred."
    elseif err.code == CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE
        "the workspace on the device is too small to execute."
    elseif err.code == CUTENSORNET_STATUS_INSUFFICIENT_DRIVER
        "the driver version is insufficient."
    elseif err.code == CUTENSORNET_STATUS_IO_ERROR
        "an error occurred related to file IO."
    elseif err.code == CUTENSORNET_STATUS_CUTENSOR_VERSION_MISMATCH
        "the dynamically linked cuTENSOR library is incompatible."
    elseif err.code == CUTENSORNET_STATUS_NO_DEVICE_ALLOCATOR
        "drawing device memory from a mempool is requested, but the mempool is not set."
    elseif err.code == CUTENSORNET_STATUS_ALL_HYPER_SAMPLES_FAILED
        "all hyper samples failed for one or more errors (enable LOGs via export CUTENSORNET_LOG_LEVEL= > 1 for details)."
    else
        "no description for this error"
    end
end

## API call wrapper

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == CUTENSORNET_STATUS_ALLOC_FAILED
        throw(OutOfGPUMemoryError())
    else
        throw(CUTENSORNETError(res))
    end
end

macro check(ex, errs...)
    check = :(isequal(err, CUTENSORNET_STATUS_ALLOC_FAILED))
    for err in errs
        check = :($check || isequal(err, $(esc(err))))
    end

    quote
        res = @retry_reclaim err->$check $(esc(ex))
        if res != CUTENSORNET_STATUS_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end
