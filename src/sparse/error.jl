export CUSPARSEError

struct CUSPARSEError <: Exception
    code::cusparseStatus_t
    msg::AbstractString
end
Base.show(io::IO, err::CUSPARSEError) = print(io, "CUSPARSEError(code $(err.code), $(err.msg))")

function CUSPARSEError(code::cusparseStatus_t)
    msg = status_message(code)
    return CUSPARSEError(code, msg)
end

function status_message( status )
    if status == CUSPARSE_STATUS_SUCCESS
        return "success"
    end
    if status == CUSPARSE_STATUS_NOT_INITIALIZED
        return "not initialized"
    end
    if status == CUSPARSE_STATUS_ALLOC_FAILED
        return "allocation failed"
    end
    if status == CUSPARSE_STATUS_INVALID_VALUE
        return "invalid value"
    end
    if status == CUSPARSE_STATUS_ARCH_MISMATCH
        return "architecture mismatch"
    end
    if status == CUSPARSE_STATUS_MAPPING_ERROR
        return "mapping error"
    end
    if status == CUSPARSE_STATUS_EXECUTION_FAILED
        return "execution failed"
    end
    if status == CUSPARSE_STATUS_INTERNAL_ERROR
        return "internal error"
    end
    if status == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
        return "matrix type not supported"
    end
end


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
