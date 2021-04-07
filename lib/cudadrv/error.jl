# Error type and decoding functionality

export CuError


"""
    CuError(code)
    CuError(code, meta)

Create a CUDA error object with error code `code`. The optional `meta` parameter indicates
whether extra information, such as error logs, is known.
"""
struct CuError <: Exception
    code::CUresult
    meta::Any

    CuError(code, meta=nothing) = new(code, meta)
end

Base.convert(::Type{CUresult}, err::CuError) = err.code

Base.:(==)(x::CuError,y::CuError) = x.code == y.code

"""
    name(err::CuError)

Gets the string representation of an error code.

```jldoctest
julia> err = CuError(CUDA.cudaError_enum(1))
CuError(CUDA_ERROR_INVALID_VALUE)

julia> name(err)
"ERROR_INVALID_VALUE"
```
"""
function name(err::CuError)
    str_ref = Ref{Cstring}()
    cuGetErrorName(err, str_ref)
    unsafe_string(str_ref[])[6:end]
end

"""
    description(err::CuError)

Gets the string description of an error code.
"""
function description(err::CuError)
    if err.code == -1%UInt32
        "Cannot use the CUDA stub libraries."
    else
        str_ref = Ref{Cstring}()
        cuGetErrorString(err, str_ref)
        unsafe_string(str_ref[])
    end
end

function Base.showerror(io::IO, err::CuError)
    try
        print(io, "CUDA error: $(description(err)) (code $(reinterpret(Int32, err.code)), $(name(err)))")
    catch
        # we might throw before the library is initialized
        print(io, "CUDA error (code $(reinterpret(Int32, err.code)), $(err.code))")
    end

    if err.meta != nothing
        print(io, "\n")
        print(io, err.meta)
    end
end

Base.show(io::IO, ::MIME"text/plain", err::CuError) = print(io, "CuError($(err.code))")

@enum_without_prefix cudaError_enum CUDA_


## API call wrapper

# outlined functionality to avoid GC frame allocation
@noinline function initialize_api()
    prepare_cuda_state()
    return
end
@noinline function throw_api_error(res)
    if res == ERROR_OUT_OF_MEMORY
        throw(OutOfGPUMemoryError())
    else
        throw(CuError(res))
    end
end

macro check(ex)
    quote
        res = $(esc(ex))
        if res != SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end
