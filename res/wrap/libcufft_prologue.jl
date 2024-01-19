# CUFFT uses CUDA runtime objects, which are compatible with our driver usage
const cudaStream_t = CUstream

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == CUFFT_ALLOC_FAILED
        throw(OutOfGPUMemoryError())
    else
        throw(CUFFTError(res))
    end
end

@inline function check(f)
    retry_if(res) = res in (CUFFT_ALLOC_FAILED,
                            CUFFT_INTERNAL_ERROR)
    res = retry_reclaim(f, retry_if)

    if res != CUFFT_SUCCESS
        throw_api_error(res)
    end
end
