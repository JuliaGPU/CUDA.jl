export CUSPARSEError

struct CUSPARSEError <: Exception
    code::cusparseStatus_t
end

Base.convert(::Type{cusparseStatus_t}, err::CUSPARSEError) = err.code

Base.showerror(io::IO, err::CUSPARSEError) =
    print(io, "CUSPARSEError: ", description(err), " (code $(reinterpret(Int32, err.code)), $(name(err)))")

name(err::CUSPARSEError) = unsafe_string(cusparseGetErrorName(err))

description(err::CUSPARSEError) = unsafe_string(cusparseGetErrorString(err))
