# CURAND uses CUDA runtime objects, which are compatible with our driver usage
const cudaStream_t = CUstream

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == CURAND_STATUS_ALLOCATION_FAILED
        throw(OutOfGPUMemoryError())
    else
        throw(CURANDError(res))
    end
end

function check(f, errs...)
    res = retry_reclaim(in((CURAND_STATUS_ALLOCATION_FAILED, errs...))) do
        f()
    end

    if res != CURAND_STATUS_SUCCESS
        throw_api_error(res)
    end

    return
end
