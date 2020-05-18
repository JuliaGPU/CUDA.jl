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

This name can often be used as a symbol in source code to get an instance of this error.
For example:

```jldoctest
julia> err = CuError(1)
CuError(1, ERROR_INVALID_VALUE)

julia> name(err)
"ERROR_INVALID_VALUE"

julia> ERROR_INVALID_VALUE
CuError(1, ERROR_INVALID_VALUE)
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
    if functional() && CuCurrentContext() !== nothing
        print(io, "CUDA error: $(description(err)) (code $(reinterpret(Int32, err.code)), $(name(err)))")
    else
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

"""
    initializer(f::Function)

Register a function to be called before making a CUDA API call that requires an initialized
context.
"""
initializer(f::Function) = (api_initializer[] = f; nothing)
const api_initializer = Union{Nothing,Function}[nothing]

# outlined functionality to avoid GC frame allocation
@noinline function initialize_api()
    hook = @inbounds api_initializer[]
    if hook !== nothing
        hook()
    end
    return
end
@noinline function throw_api_error(res)
    throw(CuError(res))
end

macro check(ex)
    quote
        res = $(esc(ex))
        if res != SUCCESS
            throw_api_error(res)
        end

        return
    end
end
