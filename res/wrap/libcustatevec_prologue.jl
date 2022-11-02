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

macro check(ex, errs...)
    check = :(isequal(err, CUSTATEVEC_STATUS_ALLOC_FAILED))
    for err in errs
        check = :($check || isequal(err, $(esc(err))))
    end

    quote
        res = @retry_reclaim err->$check $(esc(ex))
        if res != CUSTATEVEC_STATUS_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end
