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
        "the operation completed successfully"
    elseif err.code == CUSTATEVEC_STATUS_NOT_INITIALIZED
        "the library was not initialized"
    elseif err.code == CUSTATEVEC_STATUS_ALLOC_FAILED
        "the resource allocation failed"
    elseif err.code == CUSTATEVEC_STATUS_INVALID_VALUE
        "an invalid value was used as an argument"
    elseif err.code == CUSTATEVEC_STATUS_ARCH_MISMATCH
        "an absent device architectural feature is required"
    elseif err.code == CUSTATEVEC_STATUS_EXECUTION_FAILED
        "the GPU program failed to execute"
    elseif err.code == CUSTATEVEC_STATUS_INTERNAL_ERROR
        "an internal operation failed"
    elseif err.code == CUSTATEVEC_STATUS_NOT_SUPPORTED
        "the API is not supported by the backend."
    elseif err.code == CUSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE
        "the workspace on the device is too small to execute."
    elseif err.code == CUSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED
        "the sampler was called prior to preprocessing."
    else
        "no description for this error"
    end
end

## API call wrapper

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == CUSTATEVEC_STATUS_ALLOC_FAILED
        throw(OutOfGPUMemoryError())
    else
        throw(CUSTATEVECError(res))
    end
end

macro check(ex, errs...)
    check = :(isequal(err, CUSTATEVEC_STATUS_ALLOC_FAILED))
    for err in errs
        check = :($check || isequal(err, $(esc(err))))
    end

    quote
        res = @retry_reclaim err->$check $(esc(ex))
        if res != CUSTATEVEC_STATUS_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end
