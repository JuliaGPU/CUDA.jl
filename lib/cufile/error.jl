export CUFILEError

struct CUFILEError <: Exception
    code::CUfileError_t
end

Base.convert(::Type{CUfileError_t}, err::CUFILEError) = err.code

Base.showerror(io::IO, err::CUFILEError) =
    print(io, "CUFILEError: ", description(err), " (code $(reinterpret(Int32, err.code.err)), $(name(err)))")

name(err::CUFILEError) = string(err.code.err)

## COV_EXCL_START
function description(err)
    if err.code.err == CU_FILE_SUCCESS
        "cufile success"
    elseif err.code.err == CU_FILE_DRIVER_NOT_INITIALIZED
        "nvidia-fs driver is not loaded"
    elseif err.code.err == CU_FILE_DRIVER_INVALID_PROPS
        "invalid property"
    elseif err.code.err == CU_FILE_DRIVER_UNSUPPORTED_LIMIT
        "property range error"
    elseif err.code.err == CU_FILE_DRIVER_VERSION_MISMATCH
        "nvidia-fs driver version mismatch"
    elseif err.code.err == CU_FILE_DRIVER_VERSION_READ_ERROR
        "nvidia-fs driver version read error"
    elseif err.code.err == CU_FILE_DRIVER_CLOSING
        "driver shutdown in progress"
    elseif err.code.err == CU_FILE_PLATFORM_NOT_SUPPORTED
        "GPUDirect Storage not supported on current platform"
    elseif err.code.err == CU_FILE_IO_NOT_SUPPORTED
        "GPUDirect Storage not supported on current file"
    elseif err.code.err == CU_FILE_DEVICE_NOT_SUPPORTED
        "GPUDirect Storage not supported on current GPU"
    elseif err.code.err == CU_FILE_NVFS_DRIVER_ERROR
        "nvidia-fs driver ioctl error"
    elseif err.code.err == CU_FILE_CUDA_DRIVER_ERROR
        "CUDA Driver API error"
    elseif err.code.err == CU_FILE_CUDA_POINTER_INVALID
        "invalid device pointer"
    elseif err.code.err == CU_FILE_CUDA_MEMORY_TYPE_INVALID
        "invalid pointer memory type"
    elseif err.code.err == CU_FILE_CUDA_POINTER_RANGE_ERROR
        "pointer range exceeds allocated address range"
    elseif err.code.err == CU_FILE_CUDA_CONTEXT_MISMATCH
        "cuda context mismatch"
    elseif err.code.err == CU_FILE_INVALID_MAPPING_SIZE
        "access beyond maximum pinned size"
    elseif err.code.err == CU_FILE_INVALID_MAPPING_RANGE
        "access beyond mapped size"
    elseif err.code.err == CU_FILE_INVALID_FILE_TYPE
        "unsupported file type"
    elseif err.code.err == CU_FILE_INVALID_FILE_OPEN_FLAG
        "unsupported file open flags"
    elseif err.code.err == CU_FILE_DIO_NOT_SET
        "fd direct IO not set"
    elseif err.code.err == CU_FILE_INVALID_VALUE
        "invalid arguments"
    elseif err.code.err == CU_FILE_MEMORY_ALREADY_REGISTERED
        "device pointer already registered"
    elseif err.code.err == CU_FILE_MEMORY_NOT_REGISTERED
        "device pointer lookup failure"
    elseif err.code.err == CU_FILE_PERMISSION_DENIED
        "driver or file access error"
    elseif err.code.err == CU_FILE_DRIVER_ALREADY_OPEN
        "driver is already open"
    elseif err.code.err == CU_FILE_HANDLE_NOT_REGISTERED
        "file descriptor is not registered"
    elseif err.code.err == CU_FILE_HANDLE_ALREADY_REGISTERED
        "file descriptor is already registered"
    elseif err.code.err == CU_FILE_DEVICE_NOT_FOUND
        "GPU device not found"
    elseif err.code.err == CU_FILE_INTERNAL_ERROR
        "internal error"
    elseif err.code.err == CU_FILE_GETNEWFD_FAILED
        "failed to obtain new file descriptor"
    elseif err.code.err == CU_FILE_NVFS_SETUP_ERROR
        "NVFS driver initialization error"
    elseif err.code.err == CU_FILE_IO_DISABLED
        "GPUDirect Storage disabled by config on current file"
    else
        "unknown cufile error"
    end
end
## COV_EXCL_STOP

## API call wrapper

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(CUFILEError(res))
end

macro check(ex, errs...)
    quote
        res = $(esc(ex))
        if res.err != CU_FILE_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end
