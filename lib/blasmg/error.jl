## API call wrapper

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(CUBLASError(res))
end

function initialize_api()
    CUDAnative.prepare_cuda_call()
end

macro check(ex)
    quote
        res = @retry_reclaim CUBLAS_STATUS_ALLOC_FAILED $(esc(ex))
        if res != CUBLAS_STATUS_SUCCESS
            throw_api_error(res)
        end

        return
    end
end
