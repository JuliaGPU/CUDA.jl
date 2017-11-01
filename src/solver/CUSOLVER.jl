module CUSOLVER

using ..CuArrays
const cudaStream_t = Ptr{Void}

using ..CuArrays: libcusolver, _getindex

import Base.one
import Base.zero

include("libcusolver_types.jl")

immutable CUSOLVERError <: Exception
    msg::AbstractString
    status::UInt32

    function CUSOLVERError(status)
        new(status,statusmessage(status))
    end
end

function statusmessage( status )
    if status == CUSOLVER_STATUS_SUCCESS
        return "cusolver success"
    elseif status == CUSOLVER_STATUS_NOT_INITIALIZED
        return "cusolver not initialized"
    elseif status == CUSOLVER_STATUS_ALLOC_FAILED
        return "cusolver allocation failed"
    elseif status == CUSOLVER_STATUS_INVALID_VALUE
        return "cusolver invalid value"
    elseif status == CUSOLVER_STATUS_ARCH_MISMATCH
        return "cusolver architecture mismatch"
    elseif status == CUSOLVER_STATUS_EXECUTION_FAILED
        return "cusolver execution failed"
    elseif status == CUSOLVER_STATUS_INTERNAL_ERROR
        return "cusolver internal error"
    elseif status == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED
        return "cusolver matrix type not supported"
    end
end

function statuscheck( status )
    if status == CUSOLVER_STATUS_SUCCESS
        return nothing
    end
    warn("CUSOLVER error triggered from:")
    Base.show_backtrace(STDOUT, backtrace())
    println()
    throw(CUSOLVERError( status ))
end

include("libcusolver.jl")

const cusolverDnhandle = cusolverDnHandle_t[0]

function __init__()
  cusolverDnCreate(cusolverDnhandle)
  atexit(() -> cusolverDnDestroy(cusolverDnhandle[1]))
end

include("dense.jl")

end
