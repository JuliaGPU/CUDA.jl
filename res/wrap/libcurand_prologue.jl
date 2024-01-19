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

@inline function check(f)
    retry_if(res) = res in (CURAND_STATUS_ALLOCATION_FAILED,
                            CURAND_STATUS_PREEXISTING_FAILURE,
                            CURAND_STATUS_INITIALIZATION_FAILED,
                            CURAND_STATUS_INTERNAL_ERROR)
    res = retry_reclaim(f, retry_if)

    if res != CURAND_STATUS_SUCCESS
        throw_api_error(res)
    end
end
