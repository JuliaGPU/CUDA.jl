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
    # make sure the calling thread has an active context
    CUDAnative.initialize_context()
end

macro check(ex)
    quote
        res = $(esc(ex))
        if res != CUDNN_STATUS_SUCCESS
            throw_api_error(res)
        end

        return
    end
end
