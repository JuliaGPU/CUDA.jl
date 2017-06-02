# Forward declarations of types

## from errors.jl

const CuError_t = Cint

"""Wrapper of a CUDA result code."""
immutable CuError <: Exception
    code::CuError_t
    info::Nullable{String}

    CuError(code) = new(code, Nullable{String}())
    CuError(code, info) = new(code, Nullable{String}(info))
end
