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
    meta::Any

    CuError(code, meta=nothing) = new(code, meta)
end
