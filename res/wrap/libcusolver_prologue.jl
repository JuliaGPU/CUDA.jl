# CUSOLVER uses CUDA runtime objects, which are compatible with our driver usage
const cudaStream_t = CUstream

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == CUSOLVER_STATUS_ALLOC_FAILED
        throw(OutOfGPUMemoryError())
    else
        throw(CUSOLVERError(res))
    end
end

@inline function check(f)
    retry_if(res) = res in (CUSOLVER_STATUS_NOT_INITIALIZED,
                            CUSOLVER_STATUS_ALLOC_FAILED,
                            CUSOLVER_STATUS_INTERNAL_ERROR)
    res = retry_reclaim(f, retry_if)

    if res != CUSOLVER_STATUS_SUCCESS
        throw_api_error(res)
    end
end
