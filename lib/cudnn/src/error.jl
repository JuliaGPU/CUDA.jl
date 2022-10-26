export CUDNNError

struct CUDNNError <: Exception
    code::cudnnStatus_t
end

Base.convert(::Type{cudnnStatus_t}, err::CUDNNError) = err.code

Base.showerror(io::IO, err::CUDNNError) =
    print(io, "CUDNNError: ", name(err), " (code $(reinterpret(Int32, err.code)))")

name(err::CUDNNError) = unsafe_string(cudnnGetErrorString(err))
