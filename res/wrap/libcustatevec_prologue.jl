# CUSTATEVEC uses CUDA runtime objects, which are compatible with our driver usage
const cudaStream_t = CUstream

# vector types
const int2 = Tuple{Int32,Int32}

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == CUSTATEVEC_STATUS_ALLOC_FAILED
        throw(OutOfGPUMemoryError())
    else
        throw(CUSTATEVECError(res))
    end
end

function check(f, errs...)
    res = retry_reclaim(in((CUSTATEVEC_STATUS_ALLOC_FAILED, errs...))) do
        f()
    end

    if res != CUSTATEVEC_STATUS_SUCCESS
        throw_api_error(res)
    end

    return
end
