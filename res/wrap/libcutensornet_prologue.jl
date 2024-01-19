# CUTENSORNET uses CUDA runtime objects, which are compatible with our driver usage
const cudaStream_t = CUstream

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == CUTENSORNET_STATUS_ALLOC_FAILED
        throw(OutOfGPUMemoryError())
    else
        throw(CUTENSORNETError(res))
    end
end

@inline function check(f)
    retry_if(res) = res in (CUTENSORNET_STATUS_NOT_INITIALIZED,
                            CUTENSORNET_STATUS_ALLOC_FAILED,
                            CUTENSORNET_STATUS_INTERNAL_ERROR)
    res = retry_reclaim(f, retry_if)

    if res != CUTENSORNET_STATUS_SUCCESS
        throw_api_error(res)
    end
end
