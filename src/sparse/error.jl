export CUSPARSEError

struct CUSPARSEError <: Exception
    code::cusparseStatus_t
end

Base.convert(::Type{cusparseStatus_t}, err::CUSPARSEError) = err.code

Base.showerror(io::IO, err::CUSPARSEError) =
    print(io, "CUSPARSEError: ", description(err), " (code $(reinterpret(Int32, err.code)), $(name(err)))")

name(err::CUSPARSEError) = unsafe_string(cusparseGetErrorName(err))

description(err::CUSPARSEError) = unsafe_string(cusparseGetErrorString(err))


## API call wrapper

# API calls that are allowed without a functional context
const preinit_apicalls = Set{Symbol}([
    :cusparseGetVersion,
    :cusparseGetProperty,
    :cusparseGetErrorName,
    :cusparseGetErrorString,
])

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(CUSPARSEError(res))
end

macro check(ex)
    fun = Symbol(decode_ccall_function(ex))
    init = if !in(fun, preinit_apicalls)
        :(CUDAnative.maybe_initialize())
    end
    quote
        $init

        res = $(esc(ex))
        if res != CUSPARSE_STATUS_SUCCESS
            throw_api_error(res)
        end

        return
    end
end
