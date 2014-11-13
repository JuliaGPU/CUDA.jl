# Error type and decoding functionality

export CuError, description

# TODO: provide named instantiations for easy error comparisons

immutable CuError
    code::Int
end

const error_descriptions = Dict{Int,ASCIIString}(
    0   => "Success",
    1   => "Invalid value",
    2   => "Out of memory",
    3   => "Driver not initialized",
    4   => "Driver being shutdown",
    5   => "Profiler disabled",
    6   => "Profiler not initialized",
    7   => "Profiler already started",
    8   => "Profiler already stopped",
    100 => "No CUDA-capable device",
    101 => "Invalid device ordinal",
    200 => "Invalid kernel image",
    201 => "Invalid context",
    202 => "Context already current",
    205 => "Map operation failed",
    206 => "Unmap operation failed",
    207 => "Array mapped",
    208 => "Resource already mapped",
    209 => "No kernel image available/suitable for GPU",
    210 => "Resource already acquired",
    211 => "Resource not mapped",
    212 => "Resource not mapped as array",
    213 => "Resource not mapped as pointer",
    214 => "Uncorrectable ECC error detected",
    215 => "Unsupported limit",
    216 => "Context already in use",
    217 => "Peer access not supported",
    300 => "Invalid kernel source",
    301 => "File not found",
    302 => "Shared object symbol not found",
    303 => "Shared object initialization failed",
    304 => "OS call failed",
    400 => "Invalid handle",
    500 => "Named symbol not found",
    600 => "Not ready",
    700 => "Kernel launch failed",
    701 => "Launch out of resources",
    702 => "Launch timeout",
    703 => "Incompatible texturing mode",
    704 => "Peer access already enabled",
    705 => "Peer access not enabled",
    708 => "Primary context already active",
    709 => "Context destroyed",
    710 => "Assertion triggered failure",
    711 => "Too many peers",
    712 => "Host memory already registered",
    713 => "Host memory not registered",
    715 => "Illegal instruction",
    800 => "Operation not permitted",
    801 => "Operation not supported",
    999 => "Unknown error"
)

description(err::CuError) = error_descriptions[err.code]

Base.showerror(io::IO, err::CuError) = print(io, description(err),
                                             " (#$(err.code))")
