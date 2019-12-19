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
    if err.code == -1%UInt32
        "ERROR_USING_STUBS"
    else
        str_ref = Ref{Cstring}()
        cuGetErrorName(err, str_ref)
        unsafe_string(str_ref[])[6:end]
    end
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
    print(io, "CUDA error: $(description(err)) (code $(reinterpret(Int32, err.code)), $(name(err)))")

    if err.meta != nothing
        print(io, "\n")
        print(io, err.meta)
    end
end

Base.show(io::IO, err::CuError) = print(io, "CuError($(err.code))")

# define shorthands that give CuErryr objects
for code in instances(cudaError_enum)
    name = String(Symbol(code))
    shorthand = Symbol(name[6:end]) # strip the CUDA_ prefix
    @eval const $shorthand = CuError($code)
end


## API call wrapper

"""
    CUDAdrv.atapicall(f::Function)

Register a function to be called before a CUDA API call is made. The function is passed a
single argument: a symbol indicating the function that will be called.

!!! warning

    For performance reasons, only a single API call hook is supported, and its use is
    currently reserved for CUDAnative.jl to perform lazy CUDA API initialization.
"""
atapicall(f::Function) = (apicall_hook[] = f; nothing)
const apicall_hook = Union{Nothing,Function}[nothing]

# outlined functionality to avoid GC frame allocation
@noinline function _atapicall(fun)
    hook = @inbounds apicall_hook[]
    if hook !== nothing
        hook(fun)
    end
    return
end
@noinline function throw_cuerror(res)
    throw(CuError(res))
end

macro check(ex)
    # check is used in front of `ccall` or `@runtime_ccall`s that work on a tuple (fun, lib)
    if Meta.isexpr(ex, :call)
        @assert ex.args[1] == :ccall
        @assert Meta.isexpr(ex.args[2], :tuple)
        fun = String(ex.args[2].args[1].value)
    elseif Meta.isexpr(ex, :macrocall)
        @assert ex.args[1] == Symbol("@runtime_ccall")
        @assert Meta.isexpr(ex.args[3], :tuple)
        fun = String(ex.args[3].args[1].value)
    else
        error("@check should prefix ccall or @runtime_ccall")
    end

    # strip any version tag (e.g. cuEventDestroy_v2 -> cuEventDestroy)
    m = match(r"_v\d+$", fun)
    if m !== nothing
        fun = fun[1:end-length(m.match)]
    end

    quote
        _atapicall($(QuoteNode(Symbol(fun))))

        res = $(esc(ex))
        if res != CUDA_SUCCESS
            throw_cuerror(res)
        end

        return
    end
end
