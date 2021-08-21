export CUSOLVERError

struct CUSOLVERError <: Exception
    code::cusolverStatus_t
end

Base.convert(::Type{cusolverStatus_t}, err::CUSOLVERError) = err.code

Base.showerror(io::IO, err::CUSOLVERError) =
    print(io, "CUSOLVERError: ", description(err), " (code $(reinterpret(Int32, err.code)), $(name(err)))")

name(err::CUSOLVERError) = string(err.code)

## COV_EXCL_START
function description(err)
    if err.code == CUSOLVER_STATUS_SUCCESS
        "the operation completed successfully"
    elseif err.code == CUSOLVER_STATUS_NOT_INITIALIZED
        "the library was not initialized"
    elseif err.code == CUSOLVER_STATUS_ALLOC_FAILED
        "the resource allocation failed"
    elseif err.code == CUSOLVER_STATUS_INVALID_VALUE
        "an invalid value was used as an argument"
    elseif err.code == CUSOLVER_STATUS_ARCH_MISMATCH
        "an absent device architectural feature is required"
    elseif err.code == CUSOLVER_STATUS_EXECUTION_FAILED
        "the GPU program failed to execute"
    elseif err.code == CUSOLVER_STATUS_INTERNAL_ERROR
        "an internal operation failed"
    elseif err.code == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED
        "the matrix type is not supported."
    else
        "no description for this error"
    end
end
## COV_EXCL_STOP


## API call wrapper

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == CUSOLVER_STATUS_ALLOC_FAILED
        throw(OutOfGPUMemoryError())
    else
        throw(CUSOLVERError(res))
    end
end

function initialize_api()
    CUDA.prepare_cuda_state()
end

macro check(ex, errs...)
    check = :(isequal(err, CUSOLVER_STATUS_ALLOC_FAILED))
    for err in errs
        check = :($check || isequal(err, $(esc(err))))
    end

    quote
        res = @retry_reclaim err->$check $(esc(ex))
        if res != CUSOLVER_STATUS_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end
