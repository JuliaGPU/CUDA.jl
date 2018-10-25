module CUSOLVER

import CUDAdrv
using CUDAdrv: CuContext, CuDevice
using CUDAnative

using ..CuArrays
const CuStream_t = Ptr{Nothing}

using ..CuArrays: libcusolver, active_context, _getindex

using LinearAlgebra

import Base.one
import Base.zero

include("libcusolver_types.jl")

const _dense_handles = Dict{CuContext,cusolverDnHandle_t}()
const _dense_handle = Ref{cusolverDnHandle_t}(C_NULL)

function dense_handle()
    if _dense_handle[] == C_NULL
        @assert isassigned(active_context) # some other call should have initialized CUDA
        _dense_handle[] = get!(_dense_handles, active_context[]) do
            handle = cusolverDnCreate()
            atexit(()->cusolverDnDestroy(handle))
            handle
        end
    end

    return _dense_handle[]
end

include("error.jl")
include("libcusolver.jl")
include("dense.jl")
include("highlevel.jl")

end
