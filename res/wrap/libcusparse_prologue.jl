# CUSPARSE uses CUDA runtime objects, which are compatible with our driver usage
const cudaStream_t = CUstream

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == CUSPARSE_STATUS_ALLOC_FAILED
        throw(OutOfGPUMemoryError())
    else
        throw(CUSPARSEError(res))
    end
end

function check(f, errs...)
    res = retry_reclaim(in((CUSPARSE_STATUS_ALLOC_FAILED, errs...))) do
        f()
    end

    if res != CUSPARSE_STATUS_SUCCESS
        throw_api_error(res)
    end

    return
end
