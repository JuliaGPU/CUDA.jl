export CUSTATEVECError

struct CUSTATEVECError <: Exception
    code::custatevecStatus_t
end

Base.convert(::Type{custatevecStatus_t}, err::CUSTATEVECError)   = err.code

Base.showerror(io::IO, err::CUSTATEVECError) =
    print(io, "CUSTATEVECError: ", description(err), " (code $(reinterpret(Int32, err.code)), $(name(err)))")

name(err)  = string(err.code)

function description(err)
    if err.code == CUSTATEVEC_STATUS_SUCCESS
        return "the operation completed successfully"
    elseif err.code == CUSTATEVEC_STATUS_NOT_INITIALIZED
        return "the library was not initialized"
    elseif err.code == CUSTATEVEC_STATUS_ALLOC_FAILED
        return "the resource allocation failed"
    elseif err.code == CUSTATEVEC_STATUS_INVALID_VALUE
        return "an invalid value was used as an argument"
    elseif err.code == CUSTATEVEC_STATUS_ARCH_MISMATCH
        return "an absent device architectural feature is required"
    elseif err.code == CUSTATEVEC_STATUS_EXECUTION_FAILED
        return "the GPU program failed to execute"
    elseif err.code == CUSTATEVEC_STATUS_INTERNAL_ERROR
        return "an internal operation failed"
    elseif err.code == CUSTATEVEC_STATUS_NOT_SUPPORTED
        return "the API is not supported by the backend."
    elseif err.code == CUSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE
        return "the workspace on the device is too small to execute."
    elseif err.code == CUSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED
        return "the sampler was called prior to preprocessing."
    elseif err.code == CUSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR
        return "the device memory pool was not set."
    elseif err.code == CUSTATEVEC_STATUS_DEVICE_ALLOCATOR_ERROR
        return "operation with the device memory pool failed"
    else
        return "no description for this error"
    end
end
