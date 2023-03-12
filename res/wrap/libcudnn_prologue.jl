# cuDNN uses CUDA runtime objects, which are compatible with our driver usage
const cudaStream_t = CUstream

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == CUDNN_STATUS_ALLOC_FAILED
        throw(OutOfGPUMemoryError())
    else
        throw(CUDNNError(res))
    end
end

macro check(ex, errs...)
    check = :(isequal(err, CUDNN_STATUS_ALLOC_FAILED))
    for err in errs
        check = :($check || isequal(err, $(esc(err))))
    end

    quote
        res = @retry_reclaim err->$check $(esc(ex))
        if res != CUDNN_STATUS_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end
