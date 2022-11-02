export NVMLError

struct NVMLError <: Exception
    code::nvmlReturn_t
end

Base.convert(::Type{nvmlReturn_t}, err::NVMLError) = err.code

Base.showerror(io::IO, err::NVMLError) =
    print(io, "NVMLError: ", description(err), " (code $(reinterpret(Int32, err.code)))")

description(err::NVMLError) = unsafe_string(nvmlErrorString(err))

@enum_without_prefix nvmlReturn_enum NVML_
