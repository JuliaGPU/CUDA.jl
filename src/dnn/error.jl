export CUDNNError

struct CUDNNError <: Exception
    code::cudnnStatus_t
    msg::AbstractString
end
Base.show(io::IO, err::CUDNNError) = print(io, "CUDNNError(code $(err.code), $(err.msg))")

function CUDNNError(status::cudnnStatus_t)
    msg = unsafe_string(cudnnGetErrorString(status))
    return CUDNNError(status, msg)
end


## API call wrapper

# API calls that are allowed without a functional context
const preinit_apicalls = Set{Symbol}([
    :cudnnGetVersion,
    :cudnnGetProperty,
    :cudnnGetCudartVersion,
    :cudnnGetErrorString,
])

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(CUDNNError(res))
end

macro check(ex)
    fun = Symbol(decode_ccall_function(ex))
    init = if !in(fun, preinit_apicalls)
        :(CUDAnative.maybe_initialize($(QuoteNode(fun))))
    end
    quote
        $init

        res = $(esc(ex))
        if res != CUDNN_STATUS_SUCCESS
            throw_api_error(res)
        end

        return
    end
end
