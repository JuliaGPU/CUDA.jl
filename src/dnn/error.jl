export CUDNNError

struct CUDNNError <: Exception
    code::cudnnStatus_t
end

Base.convert(::Type{cudnnStatus_t}, err::CUDNNError) = err.code

Base.showerror(io::IO, err::CUDNNError) =
    print(io, "CUDNNError: ", name(err), " (code $(reinterpret(Int32, err.code)))")

name(err::CUDNNError) = unsafe_string(cudnnGetErrorString(err))


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
        :(CUDAnative.maybe_initialize())
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
