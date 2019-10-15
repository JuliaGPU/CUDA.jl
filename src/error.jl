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

    CuError(code, meta=nothing) = new(CUresult(code), meta)
end

Base.convert(::Type{CUresult}, err::CuError) = err.code

Base.:(==)(x::CuError,y::CuError) = x.code == y.code

# NOTE: `name` and `description` require CUDA to be initialized
#       (and do they work in case of error 100 or 999?)

"""
    name(err::CuError)

Gets the string representation of an error code.

This name can often be used as a symbol in source code to get an instance of this error.
For example:

```jldoctest
julia> using CUDAdrv

julia> err = CuError(1)
CuError(1, ERROR_INVALID_VALUE)

julia> name(err)
"ERROR_INVALID_VALUE"

julia> CUDAdrv.ERROR_INVALID_VALUE
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
    str_ref = Ref{Cstring}()
    cuGetErrorString(err, str_ref)
    unsafe_string(str_ref[])
end

function Base.showerror(io::IO, err::CuError)
    @printf(io, "CUDA error: %s (code #%d, %s)", description(err), Int(err.code), name(err))

    if err.meta != nothing
        print(io, "\n")
        print(io, err.meta)
    end
end

Base.show(io::IO, err::CuError) = @printf(io, "CuError(%d, %s)", err.code, name(err))

# define shorthands that give CuErryr objects
for code in instances(cudaError_enum)
    name = String(Symbol(code))
    shorthand = Symbol(name[6:end]) # strip the CUDA_ prefix
    @eval const $shorthand = CuError($code)
end


## API call wrapper

const apicall_hook = Ref{Union{Nothing,Function}}(nothing)

macro check(ex)
    # check is used in front of `ccall`s that work on a tuple (fun, lib)
    @assert Meta.isexpr(ex, :call)
    @assert ex.args[1] == :ccall
    @assert Meta.isexpr(ex.args[2], :tuple)
    fun = ex.args[2].args[1]

    quote
        # NOTE: this hook is used by CUDAnative.jl to initialize upon the first API call
        apicall_hook[] !== nothing && apicall_hook[]($fun)

        local err::CUresult
        res = $(esc(ex))
        if res != CUDA_SUCCESS
            throw(CuError(res))
        end
        res
    end
end
