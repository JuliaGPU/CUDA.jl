# Error type and decoding functionality

export CuError


# an optional struct, used to represent e.g. optional error logs.
# this is to make CuErrors with/without additional logs compare equal
# (so that we can simply reuse `@test_throws CuError(code)`).
struct Optional{T}
    data::Union{Nothing,T}
    Optional{T}(data::Union{Nothing,T}=nothing) where {T} = new{T}(data)
end
Base.getindex(s::Optional) = s.data
function Base.isequal(a::Optional, b::Optional)
    if a.data === nothing || b.data === nothing
        return true
    else
        return isequal(a.data, b.data)
    end
end
Base.convert(::Type{Optional{T}}, ::Nothing) where T = Optional{T}()
Base.convert(::Type{Optional{T}}, x) where T = Optional{T}(convert(T, x))
Base.convert(::Type{Optional{T}}, x::Optional) where T = convert(Optional{T}, x[])


"""
    CuError(code)

Create a CUDA error object with error code `code`.
"""
struct CuError <: Exception
    code::CUresult
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
    if !functional()
        # we might throw before the library is initialized
        print(io, "CUDA error (code $(reinterpret(Int32, err.code)), $(err.code))")
    else
        print(io, "CUDA error: $(description(err)) (code $(reinterpret(Int32, err.code)), $(name(err)))")
    end
end

Base.show(io::IO, ::MIME"text/plain", err::CuError) = print(io, "CuError($(err.code))")

@enum_without_prefix cudaError_enum CUDA_
