export CUSPARSEError

struct CUSPARSEError <: Exception
    code::cusparseStatus_t
end

Base.convert(::Type{cusparseStatus_t}, err::CUSPARSEError) = err.code

Base.showerror(io::IO, err::CUSPARSEError) =
    print(io, "CUSPARSEError: ", description(err), " (code $(reinterpret(Int32, err.code)), $(name(err)))")

name(err::CUSPARSEError) = unsafe_string(cusparseGetErrorName(err))

description(err::CUSPARSEError) = unsafe_string(cusparseGetErrorString(err))


## API call wrapper

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(CUSPARSEError(res))
end

function initialize_api()
    # make sure the calling thread has an active context
    CUDAnative.initialize_context()
end

macro check(ex)
    quote
        res = @retry_reclaim CUSPARSE_STATUS_ALLOC_FAILED $(esc(ex))
        if res != CUSPARSE_STATUS_SUCCESS
            throw_api_error(res)
        end

        return
    end
end
