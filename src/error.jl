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

# define shorthands that give CuError objects
for code in instances(cudaError_enum)
    local name = String(Symbol(code))
    shorthand = Symbol(name[6:end]) # strip the CUDA_ prefix
    @eval const $shorthand = CuError($code)
end


## API call wrapper

# API calls that should not initialize the API because they are often used before that,
# e.g., to determine which device to use and initialize for.
const preinit_apicalls = Set{Symbol}([
    # error handling
    :cuGetErrorString,
    :cuGetErrorName,
    # initialization
    :cuInit,
    # version management
    :cuDriverGetVersion,
    # device management
    :cuDeviceGet,
    :cuDeviceGetAttribute,
    :cuDeviceGetCount,
    :cuDeviceGetName,
    :cuDeviceGetUuid,
    :cuDeviceTotalMem,
    :cuDeviceGetProperties,     # deprecated
    :cuDeviceComputeCapability, # deprecated
    # context management
    :cuCtxGetCurrent,
    # calls that were required before JuliaGPU/CUDAnative.jl#518
    # TODO: remove on CUDAdrv v6+
    :cuCtxPushCurrent,
    :cuDevicePrimaryCtxRetain,
])

"""
    CUDAdrv.initializer(f::Function)

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

macro checked(ex)
    # parse the function definition
    @assert Meta.isexpr(ex, :function)
    sig = ex.args[1]
    @assert Meta.isexpr(sig, :call)
    body = ex.args[2]
    @assert Meta.isexpr(body, :block)
    @assert length(body.args) == 2      # line number node and a single call

    # generate a "safe" version that performs a check
    safe_body = Expr(:block, body.args[1], :(@check $(body.args[2])))
    safe_sig = Expr(:call, esc(sig.args[1]), sig.args[2:end]...)
    safe_def = Expr(:function, safe_sig, safe_body)

    # generate a "unsafe" version that returns the error code instead
    unsafe_sig = Expr(:call, esc(Symbol("unsafe_", sig.args[1])), sig.args[2:end]...)
    unsafe_def = Expr(:function, unsafe_sig, body)

    return quote
        $safe_def
        $unsafe_def
    end
end

macro check(ex)
    fun = Symbol(decode_ccall_function(ex))
    init = if !in(fun, preinit_apicalls)
        :(initialize_api())
    end
    quote
        $init

        res = $(esc(ex))
        if res != CUDA_SUCCESS
            throw_api_error(res)
        end

        return
    end
end
