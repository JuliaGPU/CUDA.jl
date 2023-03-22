struct NVPAError <: Exception
    code::NVPA_Status
end

Base.convert(::Type{NVPA_Status}, err::NVPAError) = err.code

Base.showerror(io::IO, err::NVPAError) =
    print(io, "NVPAError: $(err.code) (code $(reinterpret(Int32, err.code)))")
