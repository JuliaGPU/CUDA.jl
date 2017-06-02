# Forward declarations of types

## from errors.jl

const CuError_t = Cint

"""
    CuError(code::Integer)
    CuError(code::Integer, info::String)

Create a CUDA error object with error code `code`. The optional `info` parameter indicates
whether extra information, such as error logs, is known.
"""
immutable CuError <: Exception
    code::CuError_t
    info::Nullable{String}

    CuError(code) = new(code, Nullable{String}())
    CuError(code, info) = new(code, Nullable{String}(info))
end
