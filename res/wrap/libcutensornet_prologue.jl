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

macro check(ex, errs...)
    check = :(isequal(err, CUTENSORNET_STATUS_ALLOC_FAILED))
    for err in errs
        check = :($check || isequal(err, $(esc(err))))
    end

    quote
        res = retry_reclaim(err -> $check) do
            $(esc(ex))
        end
        if res != CUTENSORNET_STATUS_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end
