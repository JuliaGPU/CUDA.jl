export CUDNNError

struct CUDNNError <: Exception
    code::cudnnStatus_t
end

Base.convert(::Type{cudnnStatus_t}, err::CUDNNError) = err.code

Base.showerror(io::IO, err::CUDNNError) =
    print(io, "CUDNNError: ", name(err), " (code $(reinterpret(Int32, err.code)))")

name(err::CUDNNError) = unsafe_string(cudnnGetErrorString(err))


## API call wrapper

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(CUDNNError(res))
end

function initialize_api()
    CUDA.prepare_cuda_state()
end

macro check(ex, errs...)
    check = :(isequal(err, CUDNN_STATUS_ALLOC_FAILED))
    for err in errs
        check = :($check || isequal(err, $(esc(err))))
    end

    quote
        res = @retry_reclaim err->$check $(esc(ex))
        if res == CUDNN_STATUS_ALLOC_FAILED
            throw(OutOfGPUMemoryError())
        elseif res != CUDNN_STATUS_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end
