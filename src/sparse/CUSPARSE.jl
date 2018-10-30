module CUSPARSE

import CUDAdrv
using CUDAdrv: CuContext, CuDevice
using CUDAnative

using ..CuArrays
const cudaStream_t = Ptr{Nothing}

using ..CuArrays: libcusparse, configured, _getindex

using SparseArrays
using LinearAlgebra

import Base.one
import Base.zero

const SparseChar = Char
import Base.one
import Base.zero

export CuSparseMatrixCSC, CuSparseMatrixCSR,
       CuSparseMatrixHYB, CuSparseMatrixBSR,
       CuSparseMatrix, AbstractCuSparseMatrix,
       CuSparseVector

include("util.jl")
include("libcusparse_types.jl")
include("error.jl")
include("libcusparse.jl")

const libcusparse_handles = Dict{CuContext,cusparseHandle_t}()
const libcusparse_handle = Ref{cusparseHandle_t}()

function __init__()
    configured || return

    # initialize the library when we switch devices
    callback = (dev::CuDevice, ctx::CuContext) -> begin
        libcusparse_handle[] = get!(libcusparse_handles, ctx) do
            @debug "Initializing CUSPARSE for $dev"
            handle = Ref{cusparseHandle_t}()
            cusparseCreate(handle)
            handle[]
        end
    end
    push!(CUDAnative.device!_listeners, callback)

    # deinitialize when exiting
    atexit() do
        libcusparse_handle[] = C_NULL

        for (ctx, handle) in libcusparse_handles
            if CUDAdrv.isvalid(ctx)
                cusparseDestroy(handle)
            end
        end
    end
end

include("sparse.jl")
include("highlevel.jl")

end
