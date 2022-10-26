const uint64_t = UInt64
const uint32_t = UInt32

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == CUPTI_ERROR_OUT_OF_MEMORY
        throw(OutOfGPUMemoryError())
    else
        throw(CUPTIError(res))
    end
end

macro check(ex, errs...)
    check = :(isequal(err, CUPTI_ERROR_OUT_OF_MEMORY))
    for err in errs
        check = :($check || isequal(err, $(esc(err))))
    end

    quote
        res = @retry_reclaim err->$check $(esc(ex))
        if res != CUPTI_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end
