export CURANDError

struct CURANDError <: Exception
    code::curandStatus_t
end

Base.convert(::Type{curandStatus_t}, err::CURANDError) = err.code

Base.showerror(io::IO, err::CURANDError) =
    print(io, "CURANDError: ", description(err), " (code $(reinterpret(Int32, err.code)), $(name(err)))")

name(err::CURANDError) = string(err.code)

## COV_EXCL_START
function description(err)
    if err.code == CURAND_STATUS_SUCCESS
        "generator was created successfully"
    elseif err.code == CURAND_STATUS_VERSION_MISMATCH
        "header file and linked library version do not match"
    elseif err.code == CURAND_STATUS_NOT_INITIALIZED
        "generator not initialized"
    elseif err.code == CURAND_STATUS_ALLOCATION_FAILED
        "memory allocation failed"
    elseif err.code == CURAND_STATUS_TYPE_ERROR
        "generator is wrong type"
    elseif err.code == CURAND_STATUS_OUT_OF_RANGE
        "argument out of range"
    elseif err.code == CURAND_STATUS_LENGTH_NOT_MULTIPLE
        "length requested is not a multple of dimension"
    elseif err.code == CURAND_STATUS_DOUBLE_PRECISION_REQUIRED
        "GPU does not have double precision required by MRG32k3a"
    elseif err.code == CURAND_STATUS_LAUNCH_FAILURE
        "kernel launch failure"
    elseif err.code == CURAND_STATUS_PREEXISTING_FAILURE
        "preexisting failure on library entry"
    elseif err.code == CURAND_STATUS_INITIALIZATION_FAILED
        "initialization of CUDA failed"
    elseif err.code == CURAND_STATUS_ARCH_MISMATCH
        "architecture mismatch, GPU does not support requested feature"
    elseif err.code == CURAND_STATUS_INTERNAL_ERROR
        "internal library error"
    else
        "no description for this error"
    end
end
## COV_EXCL_STOP


## API call wrapper

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(CURANDError(res))
end

function initialize_api()
    # make sure the calling thread has an active context
    CUDAnative.initialize_context()
end

macro check(ex)
    quote
        res = @retry_reclaim CURAND_STATUS_ALLOCATION_FAILED $(esc(ex))
        if res != CURAND_STATUS_SUCCESS
            throw_api_error(res)
        end

        return
    end
end
