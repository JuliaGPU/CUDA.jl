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
    if res == CUSPARSE_STATUS_ALLOC_FAILED
        throw(OutOfGPUMemoryError())
    else
        throw(CUSPARSEError(res))
    end
end

function initialize_api()
    CUDA.prepare_cuda_state()
end

macro check(ex, errs...)
    check = :(isequal(err, CUSPARSE_STATUS_ALLOC_FAILED))
    for err in errs
        check = :($check || isequal(err, $(esc(err))))
    end

    quote
        res = @retry_reclaim err->$check $(esc(ex))
        if res != CUSPARSE_STATUS_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end
