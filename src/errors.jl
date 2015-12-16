# Error type and decoding functionality

import Base: ==

export CuError, description

immutable CuError
    code::Int
end

==(x::CuError,y::CuError) = x.code == y.code

const return_codes = Dict{Int,Tuple{Symbol,ASCIIString}}(
    0   => (:SUCCESS, "Success"),

    1   => (:ERROR_INVALID_VALUE, "Invalid value"),
    2   => (:ERROR_OUT_OF_MEMORY, "Out of memory"),
    3   => (:ERROR_NOT_INITIALIZED, "Driver not initialized"),
    4   => (:ERROR_DEINITIALIZED, "Driver being shutdown"),
    5   => (:ERROR_PROFILER_DISABLED, "Profiler disabled"),
    6   => (:ERROR_PROFILER_NOT_INITIALIZED, "Profiler not initialized"),
    7   => (:ERROR_PROFILER_ALREADY_STARTED, "Profiler already started"),
    8   => (:ERROR_PROFILER_ALREADY_STOPPED, "Profiler already stopped"),
    100 => (:ERROR_NO_DEVICE, "No CUDA-capable device"),
    101 => (:ERROR_INVALID_DEVICE, "Invalid device ordinal"),
    200 => (:ERROR_INVALID_IMAGE, "Invalid kernel image"),
    201 => (:ERROR_INVALID_CONTEXT, "Invalid context"),
    202 => (:ERROR_CONTEXT_ALREADY_CURRENT, "Context already current"),
    205 => (:ERROR_MAP_FAILED, "Map operation failed"),
    206 => (:ERROR_UNMAP_FAILED, "Unmap operation failed"),
    207 => (:ERROR_ARRAY_IS_MAPPED, "Array mapped"),
    208 => (:ERROR_ALREADY_MAPPED, "Resource already mapped"),
    209 => (:ERROR_NO_BINARY_FOR_GPU, "No kernel image available/suitable for GPU"),
    210 => (:ERROR_ALREADY_ACQUIRED, "Resource already acquired"),
    211 => (:ERROR_NOT_MAPPED, "Resource not mapped"),
    212 => (:ERROR_NOT_MAPPED_AS_ARRAY, "Resource not mapped as array"),
    213 => (:ERROR_NOT_MAPPED_AS_POINTER, "Resource not mapped as pointer"),
    214 => (:ERROR_ECC_UNCORRECTABLE, "Uncorrectable ECC error detected"),
    215 => (:ERROR_UNSUPPORTED_LIMIT, "Unsupported limit"),
    216 => (:ERROR_CONTEXT_ALREADY_IN_USE, "Context already in use"),
    217 => (:ERROR_PEER_ACCESS_UNSUPPORTED, "Peer access not supported"),
    218 => (:ERROR_INVALID_PTX, "Invalid PTX code"),
    300 => (:ERROR_INVALID_SOURCE, "Invalid kernel source"),
    301 => (:ERROR_FILE_NOT_FOUND, "File not found"),
    302 => (:ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND, "Shared object symbol not found"),
    303 => (:ERROR_SHARED_OBJECT_INIT_FAILED, "Shared object initialization failed"),
    304 => (:ERROR_OPERATING_SYSTEM, "OS call failed"),
    400 => (:ERROR_INVALID_HANDLE, "Invalid handle"),
    500 => (:ERROR_NOT_FOUND, "Named symbol not found"),
    600 => (:ERROR_NOT_READY, "Not ready"),
    700 => (:ERROR_ILLEGAL_ADDRESS, "Illegal memory access"),
    701 => (:ERROR_LAUNCH_OUT_OF_RESOURCES, "Launch out of resources"),
    702 => (:ERROR_LAUNCH_TIMEOUT, "Launch timeout"),
    703 => (:ERROR_LAUNCH_INCOMPATIBLE_TEXTURING, "Incompatible texturing mode"),
    704 => (:ERROR_PEER_ACCESS_ALREADY_ENABLED, "Peer access already enabled"),
    705 => (:ERROR_PEER_ACCESS_NOT_ENABLED, "Peer access not enabled"),
    708 => (:ERROR_PRIMARY_CONTEXT_ACTIVE, "Primary context already active"),
    709 => (:ERROR_CONTEXT_IS_DESTROYED, "Context destroyed"),
    710 => (:ERROR_ASSERT, "Assertion triggered failure"),
    711 => (:ERROR_TOO_MANY_PEERS, "Too many peers"),
    712 => (:ERROR_HOST_MEMORY_ALREADY_REGISTERED, "Host memory already registered"),
    713 => (:ERROR_HOST_MEMORY_NOT_REGISTERED, "Host memory not registered"),
    714 => (:ERROR_HARDWARE_STACK_ERROR, "Hardware stack error"),
    715 => (:ERROR_ILLEGAL_INSTRUCTION, "Illegal instruction"),
    716 => (:ERROR_MISALIGNED_ADDRESS, "Misaligned address"),
    717 => (:ERROR_INVALID_ADDRESS_SPACE, "Invalid address space"),
    718 => (:ERROR_INVALID_PC, "Invalid program counter"),
    719 => (:ERROR_LAUNCH_FAILED, "Kernel launch failed"),
    800 => (:ERROR_NOT_PERMITTED, "Operation not permitted"),
    801 => (:ERROR_NOT_SUPPORTED, "Operation not supported"),
    999 => (:ERROR_UNKNOWN, "Unknown error")
)

name(err::CuError)        = return_codes[err.code][1]
description(err::CuError) = return_codes[err.code][2]

Base.showerror(io::IO, err::CuError) =
    @printf(io, "%s (CUDA error #%d, %s)", description(err), err.code, name(err))

for code in return_codes
    @eval $(code[2][1]) = CuError($(code[1]))
end
