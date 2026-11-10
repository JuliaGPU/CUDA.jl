export NVMLError

"""
    NVMLError

The exception the functions in this module throw when an NVML call fails. Its `code` field is
the `nvmlReturn_t` code, e.g. `ERROR_NOT_SUPPORTED` when the device doesn't support a query,
or `ERROR_NO_PERMISSION` when it needs administrator rights.
"""
struct NVMLError <: Exception
    code::nvmlReturn_t
end

Base.convert(::Type{nvmlReturn_t}, err::NVMLError) = err.code

Base.showerror(io::IO, err::NVMLError) =
    print(io, "NVMLError: ", description(err), " (code $(reinterpret(Int32, err.code)))")

description(err::NVMLError) = unsafe_string(nvmlErrorString(err))

@enum_without_prefix visibility=:public nvmlReturn_enum NVML_
